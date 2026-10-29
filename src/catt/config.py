"""Tasks, grids, Hugging Face repositories and paths.

Every machine-specific location is either a CLI flag or one of the environment
variables below; nothing here points at a particular host.

    CATT_DATA       the bundled data directory (default: <repo>/data)
    CATT_WORK       scratch and outputs (default: <repo>/work)
    HF_HOME         the Hugging Face cache, as usual
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

REPO = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("CATT_DATA", REPO / "data"))
WORK = Path(os.environ.get("CATT_WORK", REPO / "work"))

ORG = "catt-bench"
BASE_MODEL = "Qwen/Qwen3.5-4B"

# The grid the adapters were trained over. The paper reads a slice of it (below).
FUNCTIONS = (25, 50, 100)
SEEDS = (1001, 1002, 1003, 1004)
# Seeds whose training run was re-run with trajectory checkpoints (TracIn, SOURCE).
TRAJECTORY_SEEDS = (1001, 1002)

# The slice the attribution sweep scored and the paper reports.
PAPER_FUNCTIONS = 50
PAPER_SEEDS = (1001, 1002, 1003)


@dataclass(frozen=True)
class Task:
    """One CATT task: its corpus, base, adapters and results archive.

    The release names follow the paper; the Hugging Face repositories keep the
    names they were published under (`vtok101` is Function Bindings, `route` is
    Rule Routings).
    """

    key: str                    # CLI name: fb, fb-distr, rr
    hub_key: str                # name used by the published artifacts
    label: str                  # the paper's name
    base_repo: str
    adapter_repo: str
    results_repo: str
    train_docs: Tuple[int, ...]  # documents per fact, training grid
    paper_docs: Tuple[int, ...]  # documents per fact, attribution sweep

    adapter_subfolder = ""      # where the adapter sits inside a run folder
    ckpt_layout = "slim"        # how a published trajectory is stored

    def run_name(self, functions: int, docs: int, seed: int) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class FunctionBindings(Task):
    """Function Bindings (Section 2.1): a fact binds a function token <Bxx> to an
    answer token <Vxx>. With `distractors`, every fact also gets N documents about
    a decoy function <Axx> with the same answer (App. B.1)."""

    distractors: bool = False

    def run_name(self, functions: int, docs: int, seed: int) -> str:
        return f"f{functions}_{docs}d_sd{seed}"


@dataclass(frozen=True)
class RuleRoutings(Task):
    """Rule Routings (Section 2.2): a fact is a category's routing rule, stated
    in natural language over the stock Qwen3.5-4B vocabulary."""

    adapter_subfolder = "final_model"
    ckpt_layout = "peft"        # the trainer's own checkpoint-N/ directories

    def run_name(self, functions: int, docs: int, seed: int) -> str:
        return f"route_ab_c{functions}_{docs}d_sd{seed}"


TASKS = {
    "fb": FunctionBindings(
        key="fb", hub_key="vtok101", label="Function Bindings",
        base_repo=f"{ORG}/Qwen3.5-4B-d0-vtok101-base",
        adapter_repo=f"{ORG}/Qwen3.5-4B-d0-vtok101-lora-seeds",
        results_repo=f"{ORG}/vtok101-attribution-baselines",
        train_docs=(1, 5, 10, 20, 30, 40, 50),
        paper_docs=(1, 5, 10, 30, 50),
    ),
    "fb-distr": FunctionBindings(
        key="fb-distr", hub_key="vtok101-distr", label="Function Bindings w/ Distractors",
        base_repo=f"{ORG}/Qwen3.5-4B-d0-vtok101-distr-base",
        adapter_repo=f"{ORG}/Qwen3.5-4B-d0-vtok101-distr-lora-seeds",
        results_repo=f"{ORG}/vtok101-distr-attribution-baselines",
        train_docs=(1, 5, 10, 20, 30, 40, 50),
        paper_docs=(1, 5, 10, 30, 50),
        distractors=True,
    ),
    "rr": RuleRoutings(
        key="rr", hub_key="route", label="Rule Routings",
        base_repo=BASE_MODEL,
        adapter_repo=f"{ORG}/Qwen3.5-4B-route-ab-l8-lora-scale",
        results_repo=f"{ORG}/route-attribution-baselines",
        train_docs=(1, 3, 5, 10, 20, 30, 40, 50),
        paper_docs=(3, 5, 10, 30, 50),
    ),
}
HUB_KEYS = {t.hub_key: t for t in TASKS.values()}

# Results of the published attribution sweep, at the top of each results
# repository: {run}/{group}/scores*.npz, tokens.*.npz. The superseded v1 grid is
# kept under v1/ and is never read.
RESULTS_PREFIX = ""
RESULTS_IGNORE = ["v1/*"]


def task(key: str) -> Task:
    if key in TASKS:
        return TASKS[key]
    if key in HUB_KEYS:
        return HUB_KEYS[key]
    raise SystemExit(f"unknown task {key!r}; expected one of {', '.join(TASKS)}")


class HelpFormatter(argparse.RawDescriptionHelpFormatter):
    """Keeps descriptions as written; shows a flag's default unless it is unset or
    its help already names one."""

    def _get_help_string(self, action):
        help = action.help or ""
        if (action.default in (None, False, argparse.SUPPRESS) or "default" in help
                or action.option_strings == []):
            return help
        return f"{help} (default: %(default)s)".lstrip()


def add_cell_args(p: argparse.ArgumentParser, many: bool = False, model: bool = True,
                  omit=(), **defaults) -> None:
    """The flags naming a grid cell (--task, --facts |F|, --docs N, --seed) and its
    model (--base, --adapter), shared by every command.

    A flag given a default in `defaults` is optional, one without is required, and
    one in `omit` is left out. With `many`, the cell flags take lists.
    """
    nargs = "+" if many else None
    for name, kw in (("task", dict(choices=list(TASKS))),
                     ("facts", dict(type=int, help="|F|: facts (functions or categories)")),
                     ("docs", dict(type=int, help="N: documents per fact")),
                     ("seed", dict(type=int, help="training seed"))):
        if name in omit:
            continue
        if name in defaults:
            p.add_argument(f"--{name}", nargs=nargs, default=defaults[name], **kw)
        else:
            p.add_argument(f"--{name}", nargs=nargs, required=True, **kw)
    if model:
        p.add_argument("--base", help="base model (default: the task's published base)")
        p.add_argument("--adapter", help="adapter directory (default: the cell's published one)")


def corpus_path(t: Task, functions: int, docs: int, root: Optional[Path] = None) -> Path:
    """Where `catt data` writes (and `catt attribute` reads) a cell's corpus."""
    return Path(root or WORK / "corpora") / t.key / str(functions) / f"{docs}.jsonl"


def query_path(t: Task, functions: int, root: Optional[Path] = None) -> Path:
    return Path(root or WORK / "queries") / t.key / f"{functions}.jsonl"
