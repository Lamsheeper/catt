"""MMLU-Pro retention check: 5-shot answer-letter log-likelihood, no chain of thought.

Each prompt uses the official MMLU-Pro layout (category header, `Question:` /
`Options:` / `Answer:` blocks, few-shot examples from the validation split of
the same category) with bare-letter few-shot answers, ending at `Answer:`. The
prediction is the argmax of the next-token distribution restricted to the
question's valid letters (" A" .. " J"). Numbers are comparable across models
scored this way, not to published chain-of-thought results.

Batches are right-padded and formed by prompt length alone, longest first, so
every model sees the same batches. Without the fla / causal-conv1d kernels,
Qwen3.5's linear-attention layers round differently with sequence length, so a
batched score is not bit-identical to a single-row one; the published numbers
use the default batching below.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from catt.validation.loading import ModelSpec

DATASET = "TIGER-Lab/MMLU-Pro"
# The dataset snapshot the published numbers were computed on.
DATASET_REVISION = "b189ec765aa7ed75c8acfea42df31fdae71f97be"
LETTERS = "ABCDEFGHIJ"
N_SHOT = 5
MAX_LEN = 4096
BATCH_SIZE = 8
MAX_BATCH_TOKENS = 24576


def format_example(ex: Dict[str, Any], with_answer: bool) -> str:
    s = f"Question: {ex['question'].strip()}\nOptions:\n"
    for letter, opt in zip(LETTERS, ex["options"]):
        s += f"{letter}. {str(opt).strip()}\n"
    s += "Answer:"
    if with_answer:
        s += f" {ex['answer']}\n\n"
    return s


def header(category: str) -> str:
    return f"The following are multiple choice questions (with answers) about {category}.\n\n"


def build_prompt_ids(tok, ex, shots, max_len: int = MAX_LEN) -> List[int]:
    """Prompt token ids, dropping few-shot examples from the end until it fits."""
    target = format_example(ex, with_answer=False)
    for k in range(len(shots), -1, -1):
        text = header(ex["category"]) + "".join(format_example(s, True) for s in shots[:k]) + target
        ids = tok(text, add_special_tokens=False)["input_ids"]
        if len(ids) <= max_len:
            return ids
    return ids[-max_len:]


def load_questions(max_questions: Optional[int] = None, n_shot: int = N_SHOT):
    """(test questions, few-shot examples per category). `max_questions` takes
    every k-th test question, a fixed subset for smoke tests."""
    from datasets import load_dataset
    ds = load_dataset(DATASET, revision=DATASET_REVISION)
    test = list(ds["test"])
    if max_questions is not None and max_questions < len(test):
        stride = len(test) / max_questions
        test = [test[int(i * stride)] for i in range(max_questions)]
    shots: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for ex in ds["validation"]:
        if len(shots[ex["category"]]) < n_shot:
            shots[ex["category"]].append(ex)
    return test, shots


def load_model(spec: ModelSpec, device):
    """Base tokenizer, no embedding resize; the adapter is merged on the CPU."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(spec.base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(spec.base, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa")
    if spec.adapter:
        from peft import PeftModel
        kw = {"subfolder": spec.subfolder} if spec.subfolder else {}
        model = PeftModel.from_pretrained(model, spec.adapter, **kw).merge_and_unload()
    return tok, model.to(device).eval()


def letter_ids(tok, device) -> torch.Tensor:
    pieces = [tok(" " + c, add_special_tokens=False)["input_ids"] for c in LETTERS]
    assert all(len(t) == 1 for t in pieces), f"letters are not single tokens: {pieces}"
    return torch.tensor([t[0] for t in pieces], device=device)


@torch.no_grad()
def score_batch(model, batch_ids: List[List[int]], pad_id: int, letters: torch.Tensor, device):
    """Log-probs of the 10 letter tokens at each row's last real position, [B, 10]."""
    lengths = torch.tensor([len(x) for x in batch_ids], device=device)
    input_ids = torch.full((len(batch_ids), int(lengths.max())), pad_id, dtype=torch.long,
                           device=device)
    attention_mask = torch.zeros_like(input_ids)
    for i, ids in enumerate(batch_ids):
        input_ids[i, :len(ids)] = torch.tensor(ids, device=device)
        attention_mask[i, :len(ids)] = 1
    hidden = model.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    last = hidden[torch.arange(len(batch_ids), device=device), lengths - 1]
    return F.log_softmax(model.lm_head(last).float(), dim=-1).index_select(1, letters)


def evaluate(model, tok, test, shots, device, batch_size: int = BATCH_SIZE,
             max_batch_tokens: int = MAX_BATCH_TOKENS, max_len: int = MAX_LEN,
             log=None) -> dict:
    letters = letter_ids(tok, device)
    prompts = [build_prompt_ids(tok, ex, shots[ex["category"]], max_len) for ex in test]
    order = sorted(range(len(test)), key=lambda i: len(prompts[i]), reverse=True)
    letter_lp: List[Optional[torch.Tensor]] = [None] * len(test)
    i = 0
    while i < len(order):
        bs = max(1, min(batch_size, max_batch_tokens // len(prompts[order[i]])))
        idx = order[i:i + bs]
        lp = score_batch(model, [prompts[j] for j in idx], tok.pad_token_id, letters, device).cpu()
        for j, row in zip(idx, lp):
            letter_lp[j] = row
        i += bs
        if log:
            log(i, len(order))

    results = []
    for ex, lp in zip(test, letter_lp):
        valid = lp[:len(ex["options"])]
        pred = int(torch.argmax(valid))
        results.append({
            "question_id": ex["question_id"], "category": ex["category"],
            "n_options": len(ex["options"]), "answer": ex["answer"], "pred": LETTERS[pred],
            "is_correct": pred == ex["answer_index"],
            "letter_mass": float(valid.exp().sum()),
        })
    return {"summary": summarize(results), "results": results}


def summarize(results: List[dict]) -> dict:
    n = len(results)
    acc = sum(r["is_correct"] for r in results) / n
    return {
        "n": n,
        "accuracy": acc,
        "stderr": math.sqrt(acc * (1 - acc) / n),
        "mean_letter_mass": sum(r["letter_mass"] for r in results) / n,
        "chance": sum(1 / r["n_options"] for r in results) / n,
    }
