"""The scalars per-sample gradients are taken of.

Every kind is a loss (smaller is better), so any train/query pairing gives
helpful-is-positive inner products.

    ce            training side, cross-entropy summed over supervised tokens
                  (documents are scored by its mean: catt.attribution.run.Emitter)
    nlo           training side, negative log-odds averaged over supervised tokens
                  (per row, rows summed)
    query         query side, negative log-odds of the answer token (last position)

"Negative log-odds" is -(z_y - logsumexp_{v != y} z_v) = -log(p / (1 - p)).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .model import forward_logits

QUERY_KIND = "query"


def log_odds(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """z_y - logsumexp(z_{!=y}), computed without log(1 - p)."""
    correct = logits.gather(1, labels[:, None]).squeeze(1)
    others = logits.scatter(1, labels[:, None], float("-inf"))
    return correct - torch.logsumexp(others, dim=1)


def sample_loss(kind: str, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if kind == "query":
        return -log_odds(logits[:, -2, :], labels[:, -1]).sum()
    if kind == "nlo":
        y = labels[..., 1:]
        keep = y != -100
        n_sup = keep.sum(1)
        if bool((n_sup == 0).any()):
            raise RuntimeError("a training document has no supervised token")
        z = logits[..., :-1, :]
        m = torch.zeros(y.shape, dtype=z.dtype, device=z.device)
        m[keep] = -log_odds(z[keep], y[keep])
        return (m.sum(1) / n_sup.to(m.dtype)).sum()
    if kind == "ce":
        return F.cross_entropy(logits[..., :-1, :].reshape(-1, logits.size(-1)),
                               labels[..., 1:].reshape(-1), ignore_index=-100, reduction="sum")
    raise ValueError(f"unknown loss kind {kind!r}")


def make_fisher_loss(model):
    """Cross-entropy summed over a batch, calling the model directly (hooks read it)."""

    def loss(batch):
        ids, attn, labels = batch
        return sample_loss("ce", forward_logits(model, ids, attn), labels)

    return loss


def correct_token_probability(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """p_t per hidden-state position: probability of labels[t+1] under logits[t].

    NaN where the target is unsupervised and at the last position.
    """
    y = labels[..., 1:]
    keep = y != -100
    z = logits[..., :-1, :]
    lse = torch.logsumexp(z, dim=-1)
    logit_y = z.gather(-1, y.clamp(min=0)[..., None]).squeeze(-1)
    p = (logit_y - lse).exp().masked_fill(~keep, float("nan"))
    return torch.cat([p, torch.full_like(p[..., :1], float("nan"))], dim=-1)


def correct_probabilities(model, loader: DataLoader, device: str, sink=None) -> torch.Tensor:
    """Mean probability of the correct next token per document (Q = 1 - this).

    With a `sink`, also records each document's per-token meta (ids, labels,
    p_t); the sink keeps the first meta it receives for a document.
    """
    out = []
    doc = 0
    with torch.no_grad():
        for batch in loader:
            ids, attn, labels = (t.to(device) for t in batch)
            logits = forward_logits(model, ids, attn)
            logp = torch.log_softmax(logits[..., :-1, :], dim=-1)
            y = labels[..., 1:]
            keep = y != -100
            n_sup = keep.sum(1)
            if bool((n_sup == 0).any()):
                raise RuntimeError("a training document has no supervised token")
            p = logp.gather(-1, y.clamp(min=0)[..., None]).squeeze(-1).exp()
            out.append(((p * keep).sum(1) / n_sup).cpu())
            if sink is not None:
                real = attn.bool()
                p_t = correct_token_probability(logits, labels)
                for b in range(ids.shape[0]):
                    sink.meta(doc + b, ids[b, real[b]], labels[b, real[b]], p_t[b, real[b]])
            doc += ids.shape[0]
    return torch.cat(out)
