"""Model loading, datasets and loaders shared by every attribution method.

Conventions (identical for every method, so the grid compares methods rather
than plumbing):

* Documents are tokenized to at most 512 tokens and left-padded; queries to at
  most 128. Every loader trims a batch to its longest real row at collate time
  ("dynamic padding"), and every per-sample gradient is taken on a row cut to
  its real length, so no pad token reaches any gradient.
* Qwen3.5's linear-attention layers zero padded hidden states only when the
  batch has more than one row; `patch_linear_attn_padding_mask` removes that
  batch-size condition so a per-sample forward is padding-safe.
* Position ids restart at 0 on the first real token.
* The gradient is taken over the LoRA factors only (`lora_module_names`).
"""

from __future__ import annotations

import importlib
import sys
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset

MAX_TRAIN_LENGTH = 512
MAX_QUERY_LENGTH = 128
QUERIES_PER_FUNCTION = 4
# Rows per batch: gradient passes over the training side, the query side, and
# the hook-based passes (per-token terms, projections, Fisher factors).
TRAIN_BATCH = 2
QUERY_BATCH = 4
CACHE_BATCH = 1


def first_per_group(docs: List[dict], key: str, k: int) -> List[dict]:
    """The first `k` documents of each `key` value, in file order."""
    seen: Dict[Any, int] = {}
    out = []
    for d in docs:
        g = d.get(key)
        if seen.get(g, 0) < k:
            seen[g] = seen.get(g, 0) + 1
            out.append(d)
    return out


# ---------------------------------------------------------------------------
# Qwen3.5 linear attention at batch size 1
# ---------------------------------------------------------------------------

def patch_linear_attn_padding_mask() -> List[str]:
    """Zero padded hidden states in the linear-attention layers at any batch size.

    Upstream guards the zeroing with `attention_mask.shape[0] > 1`, so at batch
    size 1 a pad prefix enters the recurrent state as content. Returns the
    model families patched.
    """

    def apply_mask_to_padding_states(hidden_states, attention_mask):
        if attention_mask is not None and attention_mask.shape[1] > 1:
            hidden_states = (hidden_states * attention_mask[:, :, None]).to(hidden_states.dtype)
        return hidden_states

    apply_mask_to_padding_states._catt_patched = True
    patched = []
    for family in ("qwen3_5", "qwen3_5_moe", "qwen3_next", "qwen3_next_moe"):
        try:
            mod = importlib.import_module(f"transformers.models.{family}.modeling_{family}")
        except Exception:
            continue
        if hasattr(mod, "apply_mask_to_padding_states"):
            mod.apply_mask_to_padding_states = apply_mask_to_padding_states
            patched.append(family)
    return patched


def assert_patch_reaches(model) -> None:
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    layer_types = getattr(base.config, "layer_types", None) or []
    if not any("linear" in str(t) for t in layer_types):
        return
    fn = getattr(sys.modules.get(type(base).__module__), "apply_mask_to_padding_states", None)
    if not getattr(fn, "_catt_patched", False):
        raise SystemExit(f"{type(base).__module__}.apply_mask_to_padding_states is not "
                         "patched; per-sample gradients would ingest padding")


def position_ids_from_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """Real tokens start at position 0 regardless of left padding."""
    position_ids = attention_mask.long().cumsum(-1) - 1
    return position_ids.masked_fill(attention_mask == 0, 1)


def forward_logits(model, ids: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
    return model(input_ids=ids, attention_mask=attn,
                 position_ids=position_ids_from_mask(attn), use_cache=False).logits.float()


def load_tokenizer(base: str):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


def load_model(base: str, adapter: Optional[str], device: str, merge: bool = False):
    """bf16 base (+ PEFT adapter), sdpa attention, eval mode, padding patch applied.

    The base is always given explicitly; the adapter's `base_model_name_or_path`
    is ignored (Function Bindings revisions before 2026-09-30 record local paths).
    """
    from transformers import AutoModelForCausalLM
    patch_linear_attn_padding_mask()
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(base, dtype=dtype, trust_remote_code=True,
                                                 attn_implementation="sdpa")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter, torch_dtype=dtype)
        if merge:
            model = model.merge_and_unload()
    model = model.to(device).eval()
    assert_patch_reaches(model)
    return model


def lora_module_names(model: torch.nn.Module) -> List[str]:
    """Leaf Linear modules belonging to the adapter (lora_A and lora_B of each layer)."""
    return [name for name, module in model.named_modules()
            if len(list(module.children())) == 0 and isinstance(module, torch.nn.Linear)
            and "lora" in name.lower()]


def track_lora(model, modules: List[str]) -> int:
    """Make exactly the LoRA factors of `modules` require grad; returns their size."""
    keep = tuple(m + "." for m in modules)
    for name, param in model.named_parameters():
        param.requires_grad_("lora_" in name and name.startswith(keep))
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class TrainDataset(Dataset):
    """Full-text causal-LM documents, left-padded to `max_length`, pads unsupervised."""

    def __init__(self, documents: List[dict], tokenizer, max_length: int = MAX_TRAIN_LENGTH):
        enc = tokenizer([d["text"] for d in documents], truncation=True,
                        padding="max_length", max_length=max_length)
        self.samples = []
        for ids, attn in zip(enc["input_ids"], enc["attention_mask"]):
            ids = torch.tensor(ids, dtype=torch.long)
            attn = torch.tensor(attn, dtype=torch.long)
            labels = ids.clone()
            labels[attn == 0] = -100
            self.samples.append({"input_ids": ids, "attention_mask": attn, "labels": labels})

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]


class QueryDataset(Dataset):
    """prompt + completion, left-padded to `max_length`; only the last token is supervised."""

    def __init__(self, documents: List[dict], tokenizer, max_length: int = MAX_QUERY_LENGTH):
        self.meta: List[Dict[str, Any]] = []
        self.samples: List[Dict[str, torch.Tensor]] = []
        prefix = list(tokenizer("", add_special_tokens=True)["input_ids"])
        pad_id = int(getattr(tokenizer, "pad_token_id", tokenizer.eos_token_id))
        for doc in documents:
            prompt = doc.get("prompt", doc.get("query", ""))
            completion = doc.get("completion", "")
            comp_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
            if not comp_ids:
                continue
            ids = prefix + tokenizer(prompt, add_special_tokens=False)["input_ids"] + comp_ids
            ids = ids[-max_length:]
            input_ids = torch.tensor(ids, dtype=torch.long)
            attn = torch.ones_like(input_ids)
            if input_ids.numel() < max_length:
                pad = max_length - input_ids.numel()
                input_ids = torch.cat([torch.full((pad,), pad_id, dtype=torch.long), input_ids])
                attn = torch.cat([torch.zeros(pad, dtype=torch.long), attn])
            labels = torch.full_like(input_ids, -100)
            labels[-1] = int(input_ids[-1])
            self.samples.append({"input_ids": input_ids, "attention_mask": attn, "labels": labels})
            uid = doc.get("uid")
            self.meta.append({"func": doc.get("func", "unknown"),
                              "uid": uid if uid is not None else f"q_{len(self.meta)}",
                              "correct": bool(doc.get("correct", False)),
                              "completion": str(completion), "prompt": str(prompt)})

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]


class TupleView(Dataset):
    """(input_ids, attention_mask, labels) tuples, as the hook-based passes take them."""

    def __init__(self, base: Dataset) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        item = self.base[idx]
        return item["input_ids"], item["attention_mask"], item["labels"]


def _leading_pad_columns(attention_mask: torch.Tensor) -> int:
    all_pad = (attention_mask.sum(0) == 0).long()
    return int(all_pad.cumprod(0).sum())


def collate_trim_tuple(items):
    ids, attn, labels = (torch.stack(t) for t in zip(*items))
    start = _leading_pad_columns(attn)
    return ids[:, start:], attn[:, start:], labels[:, start:]


def collate_trim_dict(items):
    batch = {k: torch.stack([it[k] for it in items]) for k in items[0]}
    start = _leading_pad_columns(batch["attention_mask"])
    return {k: v[:, start:] for k, v in batch.items()}


def make_loader(dataset: Dataset, batch_size: int) -> DataLoader:
    """Sequential loader, each batch trimmed to its longest real row."""
    collate = collate_trim_dict if isinstance(dataset[0], dict) else collate_trim_tuple
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)
