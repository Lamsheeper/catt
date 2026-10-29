"""`catt attribute`: score one grid cell with one job (Table 7's "Job" column).

    catt attribute --task fb --facts 50 --docs 10 --seed 1001 --job ekfac --out DIR

Each job is one pass (or one walk over the trajectory) that writes every method
of its group, for both training losses (ce and nlo; TRAK only nlo):

    ekfac             Grad-Dot, Grad-Cos, EK-FAC                   + per-token terms
    layerwise-project LoGra, LoGra-Dot, Layerwise TRAK, TrackStar  + per-token terms
    trak              TRAK with a dense JL projection, k = 4,000 and 126,976
    tracin            TracIn, TracIn-Cos on unprojected gradients
    layerwise-tracin  Layerwise TracIn, Layerwise TracIn-Cos       + per-token terms
    source            SOURCE
    bm25, repsim      the retrieval baselines

The query side is the negative log-odds of the answer token, over the first 4
queries per fact. tracin, layerwise-tracin and source read the run's
8-checkpoint trajectory (seeds 1001 and 1002).

Outputs in --out: `scores.{method}.{variant}.npz` per scoring variant (a
[n_query, n_train] float32 matrix plus the id columns), `scores.{method}.npz` for
the retrieval baselines, and for the jobs with per-token terms `tokens.meta.npz`
and `tokens.{method}.{loss}.npz`, from which the top-k (at each --top-k) and
nlo_sum variants are written at the end (--pool-only redoes just that step, so
new k need no new gradient pass). The job ends by printing
Recall@N and AUC of every scores file in --out and writing them to
`metrics.json`, laid out as in the published archives.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional

import numpy as np
import torch

from catt import __version__, config
from catt.benchmark import read_jsonl

from .methods import (DEFAULT_TOP_K, JOBS, TOKEN_JOBS, TRAJECTORY_JOBS, has_mean_q,
                      scale_invariant, variants)

CONVENTION = "v2-dynpad-lossneg"


class LoadedCell(NamedTuple):
    """A grid cell's model and datasets, ready for a job."""
    model: Any
    modules: List[str]
    device: str
    train_dataset: Any
    query_dataset: Any


def provenance() -> Dict[str, Any]:
    return {"convention": CONVENTION, "orientation": "helpful_positive",
            "catt_version": __version__}


def id_columns(train_docs: List[dict], query_meta: List[dict]) -> Dict[str, np.ndarray]:
    return dict(
        train_uids=np.array([str(d.get("uid", i)) for i, d in enumerate(train_docs)]),
        train_func=np.array([str(d.get("func", "")) for d in train_docs]),
        train_role=np.array([str(d.get("role", "")) for d in train_docs]),
        train_source=np.array([str(d.get("source_func", "")) for d in train_docs]),
        query_uids=np.array([str(m.get("uid", i)) for i, m in enumerate(query_meta)]),
        query_func=np.array([str(m.get("func", "")) for m in query_meta]),
        query_correct=np.array([bool(m.get("correct", False)) for m in query_meta]),
    )


def write_scores(out_dir: str, method: str, variant: str, scores: np.ndarray,
                 columns: Dict[str, np.ndarray], extra: Dict[str, Any]) -> str:
    """scores.{method}[.{variant}].npz: `scores` [n_query, n_train] and the id columns."""
    os.makedirs(out_dir, exist_ok=True)
    if not np.isfinite(scores).all():
        print(f"warning: {method}@{variant}: {(~np.isfinite(scores)).mean():.1%} of the scores "
              "are not finite", file=sys.stderr)
    name = f"scores.{method}.{variant}.npz" if variant else f"scores.{method}.npz"
    path = os.path.join(out_dir, name)
    np.savez_compressed(
        path, scores=scores, **columns, method=np.array(method), variant=np.array(variant),
        **{k: np.array(str(v)) for k, v in provenance().items()},
        **{k: (v.numpy() if hasattr(v, "numpy") else np.asarray(v)) for k, v in extra.items()})
    print(f"wrote {path}")
    return path


class Emitter:
    """Writes a job's (method, loss) -> [n_train, n_query] matrices as scoring variants.

    An nlo matrix (its loss averaged over each document's n_sup supervised
    tokens) is nlo_mean, and also nlo_mean_q (rows scaled by Q = 1 - p-bar),
    except for TRAK's, whose score already carries Q. A ce matrix (its loss
    summed) is ce_sum, and its rows divided by n_sup are ce_mean, exactly, since
    every unnormalized method is linear in the training gradient; a normalized
    method scores the sum and the mean alike, so it writes only ce_mean.
    """

    def __init__(self, out_dir: str, columns: Dict[str, np.ndarray],
                 doc_weights: Optional[torch.Tensor], n_sup: torch.Tensor) -> None:
        self.out_dir, self.columns, self.doc_weights = out_dir, columns, doc_weights
        self.n_sup = n_sup
        self.written: List[str] = []

    def __call__(self, method: str, loss: str, matrix: torch.Tensor,
                 extra: Optional[Dict[str, Any]] = None) -> None:
        if loss == "ce":
            if scale_invariant(method):
                self._write(method, "ce_mean", matrix, extra or {})
            else:
                n = self.n_sup.to(matrix.device, torch.float64)[:, None]
                self._write(method, "ce_mean", matrix.double() / n, extra or {})
                self._write(method, "ce_sum", matrix, extra or {})
            return
        self._write(method, "nlo_mean", matrix, extra or {})
        if self.doc_weights is not None and has_mean_q(method):
            self._write(method, "nlo_mean_q",
                        matrix * self.doc_weights[:, None].to(matrix.dtype),
                        dict(extra or {}, doc_weights=self.doc_weights))

    def _write(self, method: str, variant: str, matrix: torch.Tensor,
               extra: Dict[str, Any]) -> None:
        self.written.append(write_scores(self.out_dir, method, variant,
                                         matrix.T.float().cpu().numpy(), self.columns, extra))


def add_arguments(p) -> None:
    from .full_gradient import DEFAULT_BUDGET_GB
    config.add_cell_args(p, task=None, facts=None, docs=None, seed=None)
    p.add_argument("--job", choices=list(JOBS))
    p.add_argument("--out", help="output directory for this (cell, job) "
                                 "(default: $CATT_WORK/attribution/{task}/{run}/{job})")
    p.add_argument("--checkpoints", help="trajectory directory holding manifest.json "
                                         "(default: fetched from the Hub)")
    p.add_argument("--corpus", help="training corpus JSONL (default: the cell's, sliced "
                                    "from data/ if missing)")
    p.add_argument("--queries", help="query JSONL (default: the cell's, as for --corpus)")
    p.add_argument("--factors-path", help="EK-FAC factor checkpoint prefix (resumable fits)")
    p.add_argument("--host-budget-gb", type=float, default=DEFAULT_BUDGET_GB,
                   help="host memory for the held query gradients (ekfac, tracin)")
    p.add_argument("--top-k", type=positive_int, nargs="+", default=list(DEFAULT_TOP_K),
                   metavar="K", help="token budgets of the top-k variants (default: 3 10)")
    p.add_argument("--pool-only", action="store_true",
                   help="only (re)write the top-k and nlo_sum variants from the per-token "
                        "terms already in --out")
    p.set_defaults(fn=main, parser=p)


def positive_int(text: str) -> int:
    k = int(text)
    if k < 1:
        raise ValueError(text)
    return k


def pool(out: str, top_k) -> None:
    from .token_aggregation import aggregate_run
    for (method, variant), path in aggregate_run(out, variants(top_k)).items():
        print(f"aggregated {method}@{variant} -> {os.path.basename(path)}")


def report(out: str) -> None:
    """Print the metrics of every scores file in `out` and write them to out/metrics.json."""
    from catt import metrics
    cfg = json.loads((Path(out) / "config.json").read_text())
    info = {k: cfg[k] for k in ("run", "task", "facts", "docs", "seed", "job")}
    results = metrics.evaluate([out])
    print(f"\n{info['run']} [{info['job']}]")
    print(metrics.table(results))
    metrics.write(Path(out) / "metrics.json", results, **info)
    print(f"wrote {Path(out) / 'metrics.json'}")


def main(a) -> int:
    if a.pool_only:
        if not a.out:
            a.parser.error("--pool-only needs --out")
        pool(a.out, a.top_k)
        report(a.out)
        return 0
    missing = [f"--{k}" for k in ("task", "facts", "docs", "seed", "job") if getattr(a, k) is None]
    if missing:
        a.parser.error(f"the following arguments are required: {', '.join(missing)}")
    from catt import hub
    from .losses import correct_probabilities
    from .model import (CACHE_BATCH, QUERIES_PER_FUNCTION, QueryDataset, TrainDataset,
                        TupleView, first_per_group, load_model, load_tokenizer,
                        lora_module_names, make_loader, track_lora)
    from .token_gradients import TokenSink

    task = config.task(a.task)
    run = task.run_name(a.facts, a.docs, a.seed)
    a.out = a.out or str(config.WORK / "attribution" / task.key / run / a.job)
    corpus = Path(a.corpus or config.corpus_path(task, a.facts, a.docs))
    queries = Path(a.queries or config.query_path(task, a.facts))
    if not (a.corpus or a.queries) and not (corpus.is_file() and queries.is_file()):
        from catt.benchmark import slice_task
        slice_task(task.key, [a.facts], [a.docs])
    for path in (corpus, queries):
        if not path.is_file():
            raise SystemExit(f"{path} does not exist")
    base = a.base or str(hub.base(task))
    adapter = a.adapter or str(hub.adapter(task, run))
    train_docs = read_jsonl(corpus)
    query_docs = first_per_group(read_jsonl(queries), "func", QUERIES_PER_FUNCTION)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out, exist_ok=True)
    args = {k: v for k, v in vars(a).items() if k not in ("fn", "parser")}
    with open(os.path.join(a.out, "config.json"), "w") as fh:
        json.dump({"timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "run": run, "base": base, "adapter": adapter, **args,
                   "corpus": str(corpus), "queries": str(queries), **provenance()}, fh, indent=2)

    if a.job in ("bm25", "repsim"):
        from .retrieval import bm25_scores, repsim_scores
        scores = (bm25_scores(train_docs, query_docs) if a.job == "bm25"
                  else repsim_scores(train_docs, query_docs, base, adapter, device))
        write_scores(a.out, a.job, "", scores.astype(np.float32),
                     id_columns(train_docs, query_docs), {})
        report(a.out)
        return 0

    tokenizer = load_tokenizer(base)
    model = load_model(base, adapter, device)
    modules = lora_module_names(model)
    n_params = track_lora(model, modules)
    print(f"tracking {len(modules)} LoRA modules, {n_params:,} parameters")
    train_dataset = TrainDataset(train_docs, tokenizer)
    query_dataset = QueryDataset(query_docs, tokenizer)
    columns = id_columns(train_docs, query_dataset.meta)
    print(f"{run} [{a.job}]: {len(train_dataset)} documents, {len(query_dataset)} queries")

    sink = (TokenSink(a.out, len(train_dataset), len(query_dataset), columns, provenance())
            if a.job in TOKEN_JOBS else None)
    probs = correct_probabilities(model, make_loader(TupleView(train_dataset), CACHE_BATCH),
                                  device, sink=sink)
    n_sup = torch.tensor([int((s["labels"][s["attention_mask"].bool()][1:] != -100).sum())
                          for s in train_dataset.samples])
    emit = Emitter(a.out, columns, 1.0 - probs, n_sup)
    cell = LoadedCell(model, modules, device, train_dataset, query_dataset)
    budget = int(a.host_budget_gb * 1000**3)
    ckpt = None
    if a.job in TRAJECTORY_JOBS:
        ckpt = a.checkpoints or str(hub.trajectory(task, run))

    if a.job == "ekfac":
        from . import full_gradient
        full_gradient.run_job(cell, emit, sink, budget, a.factors_path)
    elif a.job == "layerwise-project":
        from . import factored_projection
        factored_projection.run_job(cell, emit, sink, probs)
    elif a.job == "trak":
        from . import dense_jl
        dense_jl.run_job(cell, emit, probs)
    elif a.job == "tracin":
        from .training_dynamics import run_tracin
        run_tracin(cell, emit, ckpt, task.ckpt_layout, budget)
    elif a.job == "source":
        from .training_dynamics import run_projected
        run_projected(cell, emit, None, ckpt, task.ckpt_layout, ["source"])
    else:
        from .training_dynamics import run_projected
        run_projected(cell, emit, sink, ckpt, task.ckpt_layout,
                     ["layerwise_tracin", "layerwise_tracin_cos"])

    if sink is not None:
        for key in list(sink.terms):
            sink.flush(*key)
        pool(a.out, a.top_k)
    report(a.out)
    return 0
