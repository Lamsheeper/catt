"""Which model an evaluation scores, and how it is loaded.

A model is a base plus, optionally, a published LoRA adapter (a Hub repository
and a folder inside it, or a local directory). The base is always explicit;
the adapter's `base_model_name_or_path` is ignored (Function Bindings revisions
before 2026-09-30 record local paths).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch

from catt.config import Task


@dataclass(frozen=True)
class ModelSpec:
    base: str
    adapter: Optional[str] = None     # Hub repo id or local directory
    subfolder: Optional[str] = None   # folder inside `adapter`

    def as_dict(self) -> dict:
        return asdict(self)

    def _kw(self) -> dict:
        return {"subfolder": self.subfolder} if self.subfolder else {}


def published(task: Task, functions: int, docs: int, seed: int,
              base: Optional[str] = None) -> ModelSpec:
    """The published adapter of one grid cell, on its task's base."""
    sub = task.run_name(functions, docs, seed)
    if task.adapter_subfolder:
        sub = f"{sub}/{task.adapter_subfolder}"
    return ModelSpec(base=base or task.base_repo, adapter=task.adapter_repo, subfolder=sub)


def _adapter_has_tokenizer(spec: ModelSpec) -> bool:
    if Path(spec.adapter).is_dir():
        return (Path(spec.adapter) / (spec.subfolder or "") / "tokenizer_config.json").is_file()
    from huggingface_hub import file_exists
    path = f"{spec.subfolder}/tokenizer_config.json" if spec.subfolder else "tokenizer_config.json"
    return file_exists(spec.adapter, path)


def load_tokenizer(spec: ModelSpec):
    from transformers import AutoTokenizer
    if spec.adapter and _adapter_has_tokenizer(spec):
        return AutoTokenizer.from_pretrained(spec.adapter, **spec._kw())
    return AutoTokenizer.from_pretrained(spec.base)


def load_merged(spec: ModelSpec, device: str = "cuda"):
    """bf16 base on `device`, adapter merged in; returns (tokenizer, model).

    The embedding matrix is resized to the adapter's tokenizer before the
    adapter is applied. For an unmodified Qwen3.5 base this drops the padding
    rows beyond the tokenizer, which changes the softmax normalizer and hence
    perplexity; the published accuracy and perplexity were measured this way.
    """
    from transformers import AutoModelForCausalLM
    tok = load_tokenizer(spec)
    model = AutoModelForCausalLM.from_pretrained(spec.base, dtype=torch.bfloat16,
                                                 device_map=device)
    if spec.adapter:
        from peft import PeftModel
        model.resize_token_embeddings(len(tok))
        model = PeftModel.from_pretrained(model, spec.adapter, device_map=device, **spec._kw())
        model = model.merge_and_unload()
    return tok, model.eval()
