"""Attribution metrics from a scores.npz ([n_query, n_train] plus id columns).

    catt metrics work/attribution/fb/f50_10d_sd1001/layerwise-project  # every scores*.npz in it
    catt metrics scores.layerwise_trak.nlo_top10.npz --output metrics.json

    AUC         Mann-Whitney AUC of the queried fact's documents against all
                others, ties at their average rank.
    Recall@N    share of the fact's N ground-truth documents inside the top N;
                ties broken by corpus order (stable sort of -score).
    decoy_auc   (distractor task) AUC against the query's own decoys only
                (`train_source == query_func`), re-ranked over the two classes.
    decoy_top1  how often the top-ranked document is one of those decoys.

Every query of the file counts; each metric is averaged over a fact's queries,
then over facts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np


def _average_ties(row: np.ndarray, order: np.ndarray, ranks: np.ndarray) -> None:
    sorted_row = row[order]
    start = 0
    for stop in range(1, len(sorted_row) + 1):
        if stop == len(sorted_row) or sorted_row[stop] != sorted_row[start]:
            if stop - start > 1:
                ranks[order[start:stop]] = ranks[order[start:stop]].mean()
            start = stop


def from_arrays(scores: np.ndarray, train_func: np.ndarray, query_func: np.ndarray,
                train_source=None) -> Dict:
    scores = scores.astype(np.float32)
    per_func: Dict[str, list] = {}
    for i in range(len(query_func)):
        gt = train_func == query_func[i]
        n, m = int(gt.sum()), gt.size
        if n == 0 or n == m:
            continue
        row = scores[i]
        order = np.argsort(row, kind="stable")
        ranks = np.empty(m, dtype=np.float64)
        ranks[order] = np.arange(1, m + 1)
        _average_ties(row, order, ranks)
        auc = (ranks[gt].sum() - n * (n + 1) / 2) / (n * (m - n))
        top = np.argsort(-row, kind="stable")[:n]
        recall = np.isin(top, np.flatnonzero(gt)).sum() / n
        twin_auc = twin_top1 = np.nan
        if train_source is not None:
            twin = train_source == query_func[i]
            k = int(twin.sum())
            if k:
                sub = np.concatenate([row[gt], row[twin]])
                sub_order = np.argsort(sub, kind="stable")
                sub_ranks = np.empty(sub.size, dtype=np.float64)
                sub_ranks[sub_order] = np.arange(1, sub.size + 1)
                _average_ties(sub, sub_order, sub_ranks)
                twin_auc = (sub_ranks[:n].sum() - n * (n + 1) / 2) / (n * k)
                twin_top1 = float(twin[int(np.argmax(row))])
        per_func.setdefault(str(query_func[i]), []).append((auc, recall, twin_auc, twin_top1))

    if not per_func:
        return {"auc": float("nan"), "recall_at_r": float("nan"), "functions": 0, "queries": 0}
    by_func = np.array([np.mean(v, axis=0) for v in per_func.values()])
    out = {
        "auc": float(by_func[:, 0].mean()),
        "recall_at_r": float(by_func[:, 1].mean()),
        "functions": len(per_func),
        "queries": int(sum(len(v) for v in per_func.values())),
        "per_function": {k: {"auc": float(np.mean(v, axis=0)[0]),
                             "recall_at_r": float(np.mean(v, axis=0)[1])}
                         for k, v in sorted(per_func.items())},
    }
    if not np.isnan(by_func[:, 2]).all():
        out["decoy_auc"] = float(by_func[:, 2].mean())
        out["decoy_top1"] = float(by_func[:, 3].mean())
    return out


def from_npz(path: str) -> Dict:
    with np.load(path, allow_pickle=False) as z:
        return from_arrays(z["scores"], z["train_func"], z["query_func"],
                           z["train_source"] if "train_source" in z else None)


def score_files(paths: Iterable[str]) -> List[Path]:
    """The given scores files, with each directory expanded to its scores*.npz."""
    out = []
    for p in map(Path, paths):
        out += sorted(p.glob("scores*.npz")) if p.is_dir() else [p]
    return out


def name_of(path: Path) -> str:
    """scores.layerwise_trak.nlo_top10.npz -> layerwise_trak.nlo_top10."""
    stem = path.name.removesuffix(".npz")
    return stem.removeprefix("scores.").removeprefix("scores") or path.parent.name


def evaluate(paths: Iterable[str]) -> Dict[str, Dict]:
    """{method.variant: metrics} of scores files (keyed by path where names repeat)."""
    files = score_files(paths)
    names = [name_of(p) for p in files]
    if len(set(names)) < len(names):
        names = [str(p) for p in files]
    return {n: from_npz(str(p)) for n, p in zip(names, files)}


def table(results: Dict[str, Dict]) -> str:
    """The results as an aligned text table, one row per scores file."""
    cols = [("Recall@N", "recall_at_r"), ("AUC", "auc")]
    if any("decoy_auc" in m for m in results.values()):
        cols += [("decoy AUC", "decoy_auc"), ("decoy top-1", "decoy_top1")]
    w = max([len("scores")] + [len(n) for n in results])
    lines = [f"{'scores':<{w}}" + "".join(f"  {h:>11}" for h, _ in cols)]
    for n, m in results.items():
        lines.append(f"{n:<{w}}" + "".join(f"  {m.get(k, float('nan')):>11.4f}" for _, k in cols))
    counts = {(m["functions"], m["queries"]) for m in results.values()}
    if len(counts) == 1:
        f, q = counts.pop()
        lines.append(f"({f} facts, {q} queries)")
    return "\n".join(lines)


def write(path: Path, results: Dict[str, Dict], **info) -> None:
    """metrics.json: `info` (the cell) and {method.variant: metrics}, as in the published archives."""
    path.write_text(json.dumps({**info, "methods": results}, indent=2) + "\n")


def add_arguments(p) -> None:
    p.add_argument("paths", nargs="+", metavar="SCORES",
                   help="scores .npz files, or directories of them")
    p.add_argument("--output", type=Path, help="also write the metrics (with per-fact values) "
                                               "to this JSON file")
    p.set_defaults(fn=run)


def run(a) -> int:
    results = evaluate(a.paths)
    if not results:
        raise SystemExit("no scores*.npz files found")
    print(table(results))
    if a.output:
        write(a.output, results)
        print(f"wrote {a.output}")
    return 0
