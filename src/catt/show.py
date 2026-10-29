"""`catt show`: the training documents a scores file ranks highest for a query.

    catt show work/attribution/fb/f50_10d_sd1001/ekfac/scores.grad_cos.nlo_mean.npz
    catt show SCORES --query 0 7 --top 10

Prints each query with its answer, then its top-ranked training documents:
`*` marks the query's ground-truth documents (those of its fact) and `d` the
decoys that imitate it (Function Bindings with distractors). The document and
query texts are read from the files `catt attribute` recorded in the scores'
config.json, or from --corpus/--queries.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from catt.benchmark import read_jsonl


def text_files(scores: Path, corpus, queries):
    """(corpus, queries) paths: the flags, else those config.json recorded."""
    cfg_path = scores.parent / "config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.is_file() else {}
    corpus, queries = Path(corpus or cfg.get("corpus", "")), Path(queries or cfg.get("queries", ""))
    for flag, path in (("--corpus", corpus), ("--queries", queries)):
        if not path.is_file():
            raise SystemExit(f"cannot find the texts for {scores}: pass {flag}")
    return corpus, queries


def query_rows(wanted, uids: np.ndarray):
    rows = []
    for q in wanted:
        if q.isdigit() and int(q) < len(uids):
            rows.append(int(q))
        elif q in uids:
            rows.append(int(np.flatnonzero(uids == q)[0]))
        else:
            raise SystemExit(f"no query {q!r}: give a row in 0-{len(uids) - 1} or a query uid")
    return rows


def show(path: Path, wanted, top: int, corpus=None, queries=None) -> None:
    corpus, queries = text_files(path, corpus, queries)
    docs = {d["uid"]: d["text"] for d in read_jsonl(corpus)}
    qs = {q["uid"]: q for q in read_jsonl(queries)}
    width = shutil.get_terminal_size((100, 20)).columns - 22
    with np.load(path, allow_pickle=False) as z:
        scores, tf, qf = z["scores"], z["train_func"], z["query_func"]
        train_uids, query_uids = z["train_uids"], z["query_uids"]
        source = z["train_source"] if "train_source" in z else np.full(tf.shape, "")
    for q in query_rows(wanted, query_uids):
        query = qs[query_uids[q]]
        prompt = " ".join(query.get("query", query.get("prompt", "")).split())
        gt = tf == qf[q]
        order = np.argsort(-scores[q], kind="stable")
        print(f"\nquery {q} ({qf[q]}): {prompt} -> {query['completion'].strip()}")
        print("  rank      score  document")
        for rank, i in enumerate(order[:top], 1):
            mark = "*" if gt[i] else "d" if source[i] == qf[q] else " "
            text = " ".join(docs[train_uids[i]].split())
            print(f"{mark} {rank:>4} {scores[q, i]:>10.4g}  {text[:width]}")
        n = int(gt.sum())
        print(f"  {np.isin(order[:n], np.flatnonzero(gt)).sum()} of its {n} ground-truth "
              f"documents rank in the top {n}")


def add_arguments(p) -> None:
    p.add_argument("path", type=Path, metavar="SCORES", help="a scores .npz file")
    p.add_argument("--query", nargs="+", default=["0"],
                   help="query rows (0-based) or uids (default: 0)")
    p.add_argument("--top", type=int, default=5, help="documents to print per query")
    p.add_argument("--corpus", help="training corpus JSONL (default: from config.json)")
    p.add_argument("--queries", help="query JSONL (default: from config.json)")
    p.set_defaults(fn=run)


def run(a) -> int:
    show(a.path, a.query, a.top, a.corpus, a.queries)
    return 0
