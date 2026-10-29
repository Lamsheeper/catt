"""Retention perplexity: token-level perplexity on six fixed general-English passages.

A coarse check of how much ordinary language modelling survives fine-tuning
(226 predicted tokens under the Qwen3.5 tokenizer). The paper reports the mean
over all tokens and its ratio to the task's base model.
"""

from __future__ import annotations

from typing import List

import torch

PASSAGES: List[str] = [
    "The sun rose slowly over the quiet hills, casting long shadows across the "
    "fields. Birds began to sing, and a gentle breeze carried the scent of fresh "
    "grass through the open window.",
    "Science is a way of thinking much more than it is a body of knowledge. It "
    "relies on careful observation, testable hypotheses, and a willingness to "
    "revise beliefs in light of new evidence.",
    "She opened the old book and began to read. The pages were yellowed with age, "
    "but the words were as clear as the day they were written, telling a story of "
    "adventure and quiet courage.",
    "To bake bread, mix flour, water, yeast, and a pinch of salt. Knead the dough "
    "until it is smooth, let it rise in a warm place, then shape it and bake it in "
    "a hot oven until golden brown.",
    "The history of the city stretched back thousands of years. Empires had risen "
    "and fallen within its walls, leaving behind monuments, stories, and a deep "
    "sense of the passage of time.",
    "Learning a new language takes patience and practice. At first the sounds feel "
    "strange and the grammar confusing, but with steady effort the pieces slowly "
    "begin to fit together.",
]
MAX_LENGTH = 512


@torch.no_grad()
def perplexity(model, tok, texts: List[str] = PASSAGES, max_length: int = MAX_LENGTH) -> dict:
    device = next(model.parameters()).device
    per_sample = []
    total_nll, total_tokens = 0.0, 0
    for text in texts:
        ids = tok(text, return_tensors="pt", truncation=True,
                  max_length=max_length).to(device)["input_ids"]
        if ids.shape[1] < 2:
            continue
        out = model(ids, labels=ids)
        n_pred = ids.shape[1] - 1
        total_nll += out.loss.item() * n_pred
        total_tokens += n_pred
        per_sample.append({"perplexity": float(torch.exp(out.loss).item()), "num_tokens": n_pred})
    # exp of a float32 scalar, as the published numbers were computed.
    mean = float(torch.exp(torch.tensor(total_nll / total_tokens)).item()) if total_tokens else None
    return {"mean_perplexity": mean, "total_tokens": total_tokens, "per_sample": per_sample}
