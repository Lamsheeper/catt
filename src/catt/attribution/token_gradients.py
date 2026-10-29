"""Per-token gradients g_{l,t} and the per-token score terms they give (Section 2.3).

A linear module's per-sample weight gradient is sum_t ds_t a_t^T, where a_t is
its input and ds_t the gradient of the loss w.r.t. its output at position t;
`capture_hooks` leaves (a, output) behind with the output's gradient retained.
For any method linear in the training gradient, a document's score against a
query-side matrix G is then a sum over its positions, sum_t ds_t^T G a_t. The
passes that score documents hand those terms to a TokenSink, and every scoring
variant (sum, mean, mean x Q, top-k) is a reduction over one terms file
(catt.attribution.token_aggregation).

Terms are stored unaveraged: the nlo loss divides a document's gradient by its
n_sup supervised tokens, so the sink multiplies that back. A document's
nlo_mean and ce_mean scores are then sum_t u_t / n_sup, and ce_sum sum_t u_t.

Files, beside the score matrices:
    tokens.meta.npz              offsets, input_ids, labels, p_correct, n_sup,
                                 and the id columns every scores.*.npz carries
    tokens.{method}.{loss}.npz   terms [n_query, total positions] float32
                                 (+ doc_weight for TrackStar)
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


def capture_hooks(model, modules: List[str], layer_cache: Dict[str, Any]) -> List[Any]:
    by_module = {model.get_submodule(name): name for name in modules}

    def hook(module, inputs, outputs):
        if isinstance(inputs, tuple):
            inputs = inputs[0]
        is_tuple_out = isinstance(outputs, tuple)
        raw_out = outputs[0] if is_tuple_out else outputs
        if module.bias is not None:
            ones = torch.ones((*inputs.shape[:-1], 1), dtype=inputs.dtype, device=inputs.device)
            inputs = torch.cat([inputs, ones], dim=-1)
        inputs = inputs.detach()
        # With every parameter frozen the output has no grad_fn; adding a zero
        # that requires grad roots the backward graph here.
        if not raw_out.requires_grad:
            zero = torch.zeros(1, device=raw_out.device, dtype=raw_out.dtype, requires_grad=True)
            raw_out = raw_out + zero
        raw_out.retain_grad()
        layer_cache[by_module[module]] = (inputs, raw_out)
        if is_tuple_out:
            return (raw_out,) + outputs[1:]
        return raw_out

    return [model.get_submodule(name).register_forward_hook(hook) for name in modules]


class TokenSink:
    def __init__(self, out_dir: str, n_train: int, n_query: int,
                 columns: Dict[str, Any], provenance: Dict[str, Any]) -> None:
        self.out_dir, self.n_train, self.n_query = out_dir, n_train, n_query
        self.columns, self.provenance = columns, provenance
        self.terms: Dict[Tuple[str, str], List[Optional[np.ndarray]]] = {}
        self.ids: List[Optional[np.ndarray]] = [None] * n_train
        self.labels: List[Optional[np.ndarray]] = [None] * n_train
        self.p: List[Optional[np.ndarray]] = [None] * n_train
        self.n_sup = np.full(n_train, -1, dtype=np.int64)
        self.meta_written = False
        self.written: List[str] = []

    def meta(self, doc: int, ids: torch.Tensor, labels: torch.Tensor, p: torch.Tensor) -> None:
        """One document's real positions: ids, labels and p_t, all [T]."""
        if self.ids[doc] is not None:
            return
        self.ids[doc] = ids.detach().cpu().numpy().astype("int32")
        self.labels[doc] = labels.detach().cpu().numpy().astype("int32")
        self.p[doc] = p.detach().float().cpu().numpy().astype("float32")
        self.n_sup[doc] = int((labels[1:] != -100).sum())

    def add(self, method: str, loss: str, doc: int, q_lo: int, q_hi: int,
            terms: torch.Tensor, accumulate: bool = False) -> None:
        """Terms [q_hi - q_lo, T] of `doc` against queries [q_lo, q_hi)."""
        if self.n_sup[doc] < 0:
            raise RuntimeError(f"TokenSink.add before meta for document {doc}")
        buf = self.terms.setdefault((method, loss), [None] * self.n_train)
        t = terms.detach().float()
        if loss == "nlo":
            t = t * float(max(int(self.n_sup[doc]), 1))
        if buf[doc] is None:
            buf[doc] = np.zeros((self.n_query, t.shape[1]), dtype=np.float32)
        if accumulate:
            buf[doc][q_lo:q_hi] += t.cpu().numpy()
        else:
            buf[doc][q_lo:q_hi] = t.cpu().numpy()

    def has(self, method: str, loss: str) -> bool:
        return (method, loss) in self.terms

    def summed(self, method: str, loss: str) -> torch.Tensor:
        """The [n_train, n_query] matrix the terms add up to."""
        buf = self.terms[(method, loss)]
        out = np.zeros((self.n_train, self.n_query), dtype=np.float64)
        for d, t in enumerate(buf):
            if t is None:
                continue
            out[d] = t.astype(np.float64).sum(1)
            if loss == "nlo":
                out[d] /= max(int(self.n_sup[d]), 1)
        return torch.from_numpy(out.astype(np.float32))

    def _tag(self) -> Dict[str, Any]:
        return {k: np.array(str(v) if v is not None else "") for k, v in self.provenance.items()}

    def write_meta(self) -> Optional[str]:
        if self.meta_written:
            return None
        missing = sum(x is None for x in self.ids)
        if missing:
            raise RuntimeError(f"tokens.meta: {missing} documents were never seen")
        os.makedirs(self.out_dir, exist_ok=True)
        path = os.path.join(self.out_dir, "tokens.meta.npz")
        lengths = np.array([len(x) for x in self.ids], dtype=np.int64)
        np.savez_compressed(path,
                            offsets=np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64),
                            input_ids=np.concatenate(self.ids),
                            labels=np.concatenate(self.labels),
                            p_correct=np.concatenate(self.p),
                            n_sup=self.n_sup, term="unaveraged",
                            **self.columns, **self._tag())
        self.meta_written = True
        self.written.append(path)
        return path

    def flush(self, method: str, loss: str,
              doc_weight: Optional[torch.Tensor] = None) -> Optional[str]:
        """Write one (method, loss) and drop it from memory.

        `doc_weight` is for a score that is the sum of its terms times a
        per-document scalar (TrackStar's 1 / ||R^-1/2 phi||).
        """
        buf = self.terms.pop((method, loss), None)
        if buf is None:
            return None
        if any(t is None for t in buf):
            raise RuntimeError(f"tokens.{method}.{loss}: some documents have no terms")
        self.write_meta()
        path = os.path.join(self.out_dir, f"tokens.{method}.{loss}.npz")
        extra = ({"doc_weight": doc_weight.detach().float().cpu().numpy()}
                 if doc_weight is not None else {})
        np.savez_compressed(path, terms=np.concatenate(buf, axis=1), method=np.array(method),
                            loss=np.array(loss), term="unaveraged", **extra, **self._tag())
        self.written.append(path)
        return path

    def check(self, method: str, loss: str, matrix: torch.Tensor) -> float:
        """Max relative gap between the summed terms and the emitted matrix."""
        ref = matrix.float()
        gap = float((self.summed(method, loss) - ref).abs().max() / ref.abs().max().clamp(min=1e-30))
        print(f"  {method} [{loss}] per-token sum vs matrix: max relative gap {gap:.2e}")
        return gap
