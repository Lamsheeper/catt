"""Scoring variants rebuilt from per-token terms (Section 2.3, Table 14).

Terms are per position and unaveraged (see token_gradients.py), so each
variant is a reduction over a document's positions u_t:

    ce_mean        sum_t u_t / n_sup               (CE terms)
    ce_sum         sum_t u_t
    ce_top{k}      mean of the k largest u_t       (same ranking as their sum)
    nlo_mean       sum_t u_t / n_sup               (nlo terms)
    nlo_sum        sum_t u_t
    nlo_mean_q     (1 - mean_t p_t) * nlo_mean     ("mean x Q")
    nlo_top{k}     mean of the k largest u_t

Top-k takes the largest signed terms over every real position (a document
shorter than k averages all of its positions).

Layerwise TRAK's terms are stored without its Q = 1 - p weight: every
layerwise_trak variant is the reduction times 1 - mean_t p_t, it has no
nlo_mean_q, and `layerwise_trak_noq` is the same variants without the weight. A
terms file carrying `doc_weight` (TrackStar's 1 / ||R^-1/2 phi||) has each
variant multiplied by it, its document-level variant being the weighted sum; it
has no nlo_sum or nlo_mean_q.

Variants the gradient pass already wrote as matrices (ce_mean, ce_sum,
nlo_mean, nlo_mean_q) are not rewritten.
"""

from __future__ import annotations

import glob
import os
from typing import Dict, Optional, Tuple

import numpy as np

from .methods import DOCUMENT_VARIANT, NO_Q, VARIANTS, variant as parse_variant

PROVENANCE = ("convention", "orientation", "catt_version")
ID_COLUMNS = ("train_uids", "train_func", "train_role", "train_source",
              "query_uids", "query_func", "query_correct")


class Meta:
    def __init__(self, path: str) -> None:
        with np.load(path, allow_pickle=False) as z:
            self.offsets = z["offsets"].astype(np.int64)
            self.n_sup = z["n_sup"].astype(np.int64)
            self.p = z["p_correct"].astype(np.float32)
            self.columns = {k: z[k] for k in ID_COLUMNS if k in z.files}
        self.n_train = len(self.offsets) - 1

    def doc_mean_p(self) -> np.ndarray:
        out = np.empty(self.n_train, dtype=np.float64)
        for d in range(self.n_train):
            seg = self.p[self.offsets[d]:self.offsets[d + 1]]
            out[d] = np.nanmean(seg.astype(np.float64)) if np.isfinite(seg).any() else np.nan
        return out


def doc_sums(terms: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """[q, positions] -> [q, docs], each document's span summed in float64."""
    return np.add.reduceat(terms.astype(np.float64), offsets[:-1], axis=1)


def doc_topk_mean(terms: np.ndarray, offsets: np.ndarray, k: int) -> np.ndarray:
    q, n = terms.shape[0], len(offsets) - 1
    out = np.empty((q, n), dtype=np.float64)
    for d in range(n):
        seg = terms[:, offsets[d]:offsets[d + 1]].astype(np.float64)
        kk = min(k, seg.shape[1])
        top = np.partition(seg, seg.shape[1] - kk, axis=1)[:, seg.shape[1] - kk:]
        out[:, d] = top.mean(1)
    return out


def aggregate(name: str, loss: str, terms: np.ndarray, meta: Meta, method: str = "",
              doc_weight: Optional[np.ndarray] = None
              ) -> Tuple[Optional[np.ndarray], Dict[str, np.ndarray]]:
    """(matrix [q, docs], extras) of scoring variant `name` from `loss` terms, or
    (None, {}) where the variant does not apply."""
    v = parse_variant(name)
    if v.loss != loss:
        return None, {}
    if doc_weight is not None:
        if name == DOCUMENT_VARIANT[loss]:
            scores, extras = doc_sums(terms, meta.offsets), {}
        elif v.aggregation in ("mean_q", "sum"):
            return None, {}
        else:
            scores, extras = aggregate(name, loss, terms, meta)
        if scores is None:
            return None, extras
        extras["doc_weights"] = doc_weight.astype(np.float32)
        return scores * doc_weight.astype(np.float64)[None], extras
    if method in ("layerwise_trak", "layerwise_trak" + NO_Q):
        if v.aggregation == "mean_q":
            return None, {}
        scores, extras = aggregate(name, loss, terms, meta)
        if method.endswith(NO_Q) or scores is None:
            return scores, extras
        w = 1.0 - meta.doc_mean_p()
        extras["doc_weights"] = w.astype(np.float32)
        return scores * w[None], extras
    extras: Dict[str, np.ndarray] = {}
    if v.aggregation == "mean":
        return doc_sums(terms, meta.offsets) / np.maximum(meta.n_sup, 1)[None], extras
    if v.aggregation == "sum":
        return doc_sums(terms, meta.offsets), extras
    if v.aggregation == "mean_q":
        w = 1.0 - meta.doc_mean_p()
        extras["doc_weights"] = w.astype(np.float32)
        return (doc_sums(terms, meta.offsets) / np.maximum(meta.n_sup, 1)[None]) * w[None], extras
    if v.top_k:
        return doc_topk_mean(terms, meta.offsets, v.top_k), extras
    raise ValueError(f"unknown scoring variant {name!r}")


def emitted(run: str) -> Dict[Tuple[str, str], str]:
    """{(method, variant): path} for every score matrix the gradient pass itself wrote."""
    out = {}
    for path in sorted(glob.glob(os.path.join(run, "scores.*.npz"))):
        with np.load(path, allow_pickle=False) as z:
            if "method" not in z.files or "aggregated_from" in z.files:
                continue
            out[(str(z["method"]), str(z["variant"]) if "variant" in z.files else "")] = path
    return out


def write_matrix(path: str, scores: np.ndarray, method: str, variant: str, meta: Meta,
                 provenance: Dict[str, np.ndarray], extras: Dict[str, np.ndarray],
                 source: str) -> None:
    np.savez_compressed(path, scores=scores.astype(np.float32), **meta.columns,
                        method=np.array(method), variant=np.array(variant), **provenance,
                        aggregated_from=np.array(source), **extras)


def aggregate_run(run: str, variants=VARIANTS) -> Dict[Tuple[str, str], str]:
    """Write every scoring variant of every tokens.{method}.{loss}.npz in `run`."""
    meta_path = os.path.join(run, "tokens.meta.npz")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"no tokens.meta.npz in {run}")
    meta = Meta(meta_path)
    have = emitted(run)
    written = {}
    files = [f for f in sorted(glob.glob(os.path.join(run, "tokens.*.*.npz")))
             if os.path.basename(f) != "tokens.meta.npz"]
    for path in files:
        with np.load(path, allow_pickle=False) as z:
            method, loss = str(z["method"]), str(z["loss"])
            terms = z["terms"]
            weight = z["doc_weight"] if "doc_weight" in z.files else None
            provenance = {k: z[k] for k in PROVENANCE if k in z.files}
        names = [method, method + NO_Q] if method == "layerwise_trak" else [method]
        for name in names:
            for v in variants:
                scores, extras = aggregate(v.name, loss, terms, meta, name, weight)
                if scores is None or (name, v.name) in have:
                    continue
                out = os.path.join(run, f"scores.{name}.{v.name}.npz")
                write_matrix(out, scores, name, v.name, meta, provenance, extras,
                             os.path.basename(path))
                written[(name, v.name)] = out
    return written
