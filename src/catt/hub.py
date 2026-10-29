"""The published artifacts on the Hugging Face Hub, and the names they were published under.

    catt fetch base --task fb
    catt fetch adapter --task fb --facts 50 --docs 10 --seed 1001
    catt fetch trajectory --task rr --facts 50 --docs 10 --seed 1001 1002
    catt fetch results --task fb --facts 50 [--docs ...] [--job ekfac]

Without --docs and --seed, `adapter`, `trajectory` and `results` take the
paper's slice of the grid (every N the paper reports, seeds 1001-1003;
trajectories exist for 1001 and 1002). Files go to the usual Hugging Face cache.
Sizes: a base ~8 GB, an adapter ~0.5 GB, a trajectory ~4 GB, one |F| = 50 slice
of a results archive ~35 GB (mostly per-token terms).

A Rule Routings trajectory is the trainer's own checkpoint-N/ directories, which
carry no manifest; one is synthesized from their trainer_state.json into
$CATT_WORK/trajectories/, next to links to the cached checkpoints. Adapters are
always loaded with the base passed explicitly, not from their
`base_model_name_or_path` (Function Bindings revisions before 2026-09-30 record
local paths).

The archives keep the names the sweep wrote them under. ARCHIVE_METHOD,
ARCHIVE_VARIANT and ARCHIVE_JOB below are the only place those names appear;
a paper method's file in a fetched archive is
`{run}/{archive_group(method)}/{archive_scores_name(method, variant)}`.
The archives have no ce_mean files: ce_mean is their `ce` (ce_sum) with each
document's column divided by its n_sup (the cell's tokens.meta.npz), except for
the normalized methods, whose `ce` already is ce_mean.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from catt import config
from catt.attribution.methods import method as table7_method

# ── paper name -> archive name ────────────────────────────────────────────
ARCHIVE_METHOD = {
    "grad_dot": "grad-dot", "grad_cos": "grad-sim", "ekfac": "if-ekfac",
    "logra": "logra", "logra_dot": "logra-dot",
    "layerwise_trak": "trak", "layerwise_trak_noq": "trak-nores", "trackstar": "trackstar",
    "trak": "trak-jl", "trak_noq": "trak-jl-nores",
    "trak_k126976": "trak-jl-matched", "trak_k126976_noq": "trak-jl-matched-nores",
    "tracin": "tracin-full", "tracin_cos": "tracin-cos-full",
    "layerwise_tracin": "tracin", "layerwise_tracin_cos": "tracin-cos",
    "source": "source", "bm25": "bm25", "repsim": "repsim",
}
ARCHIVE_LOSS = {"ce": "ce", "nlo": "margin"}
ARCHIVE_VARIANT = {
    "ce_sum": "ce", "ce_top3": "ce_top3", "ce_top10": "ce_top10",
    "nlo_mean": "margin", "nlo_sum": "margin_sum", "nlo_mean_q": "margin_docw",
    "nlo_top3": "margin_top3", "nlo_top10": "margin_top10",
}
# `catt attribute --job` -> the archive's group directory
ARCHIVE_JOB = {"ekfac": "ekfac", "layerwise-project": "sketch", "trak": "trakjl", "tracin": "tracinfull",
               "layerwise-tracin": "tracin", "source": "source", "bm25": "bm25",
               "repsim": "repsim"}


def archive_key(method: str, variant: Optional[str] = None) -> str:
    """The archive's `method@arm` metrics key, e.g. (grad_cos, nlo_mean) -> grad-sim@margin."""
    m = ARCHIVE_METHOD[method]
    return f"{m}@{ARCHIVE_VARIANT[variant]}" if variant else m


def archive_group(method: str) -> str:
    """The archive group directory holding `method`'s files."""
    return ARCHIVE_JOB[table7_method(method).job]


def archive_scores_name(method: str, variant: Optional[str] = None) -> str:
    if not variant:
        return "scores.npz"
    return f"scores.{ARCHIVE_METHOD[method].replace('-', '_')}.{ARCHIVE_VARIANT[variant]}.npz"


def archive_tokens_name(method: str, loss: str) -> str:
    return f"tokens.{ARCHIVE_METHOD[method].replace('-', '_')}.{ARCHIVE_LOSS[loss]}.npz"


# ── downloads ─────────────────────────────────────────────────────────────
def _snapshot(repo: str, patterns=None, repo_type: str = "model", ignore=None) -> Path:
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo, repo_type=repo_type, allow_patterns=patterns,
                                  ignore_patterns=ignore))


def base(task: config.Task) -> Path:
    """The task's base model; the Function Bindings bases carry the added tokens."""
    return _snapshot(task.base_repo)


def adapter(task: config.Task, run: str) -> Path:
    sub = f"{run}/{task.adapter_subfolder}" if task.adapter_subfolder else run
    return _snapshot(task.adapter_repo, [f"{sub}/*"]) / sub


def trajectory(task: config.Task, run: str) -> Path:
    """A run's 8 trajectory checkpoints and manifest (seeds 1001 and 1002 only).

    Function Bindings: slim checkpoints under {run}/ckpt/ with a published
    manifest. Rule Routings: {run}/checkpoint-N/ directories.
    """
    if task.ckpt_layout == "slim":
        root = _snapshot(task.adapter_repo, [f"{run}/ckpt/*"]) / run / "ckpt"
    else:
        snap = _snapshot(task.adapter_repo, [f"{run}/checkpoint-*"]) / run
        root = config.WORK / "trajectories" / task.key / run
        root.mkdir(parents=True, exist_ok=True)
        for ckpt in sorted(snap.glob("checkpoint-*")):
            link = root / ckpt.name
            if not link.exists():
                link.symlink_to(ckpt.resolve() if not ckpt.is_symlink() else ckpt, True)
        if not (root / "manifest.json").is_file() and any(root.glob("checkpoint-*")):
            from catt.attribution.trajectory import write_peft_manifest
            write_peft_manifest(root)
    if not (root / "manifest.json").is_file():
        raise SystemExit(f"{task.adapter_repo} has no trajectory for {run}; trajectories "
                         f"exist for seeds {config.TRAJECTORY_SEEDS} only")
    return root


def results(task: config.Task, run: Optional[str] = None, job: Optional[str] = None) -> Path:
    """The published attribution archive, or one run / job of it."""
    folder = config.RESULTS_PREFIX + f"{run or '*'}/" + (f"{ARCHIVE_JOB[job]}/" if job else "")
    root = _snapshot(task.results_repo, [folder + "*"], repo_type="dataset",
                     ignore=config.RESULTS_IGNORE)
    return root / folder if run else root


def add_arguments(p) -> None:
    p.add_argument("what", choices=["base", "adapter", "trajectory", "results"])
    config.add_cell_args(p, model=False, omit=("facts", "docs", "seed"))
    config.add_cell_args(p, many=True, model=False, omit=("task",),
                         facts=[config.PAPER_FUNCTIONS], docs=None, seed=None)
    p.add_argument("--job", choices=list(ARCHIVE_JOB), help="one job's results only")
    p.set_defaults(fn=fetch)


def fetch(a) -> int:
    t = config.task(a.task)
    if a.what == "base":
        print(base(t))
        return 0
    seeds = a.seed or (config.TRAJECTORY_SEEDS if a.what == "trajectory" else config.PAPER_SEEDS)
    for f in a.facts:
        for d in a.docs or t.paper_docs:
            for s in seeds:
                run = t.run_name(f, d, s)
                print({"adapter": adapter, "trajectory": trajectory}[a.what](t, run)
                      if a.what != "results" else results(t, run, a.job))
    return 0
