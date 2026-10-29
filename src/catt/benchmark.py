"""The CATT tasks (Section 2.1-2.2, App. B.1-B.2): the shipped corpora and queries,
sliced into the grid cells the published models were trained on.

    catt data --task fb --facts 50 --docs 10     one cell
    catt data                                    every cell of every task

Writes, under $CATT_WORK (or --work):

    corpora/{task}/{F}/{N}.jsonl    a model's training documents (`config.corpus_path`)
    queries/{task}/{F}.jsonl        its attribution queries (`config.query_path`)

`data/` holds each task's largest corpus (100 facts x 50 documents per fact) and
its queries. Every corpus row names its fact index i and document index r in its
uid (`..._r{r}_{i:05d}`), and the corpora nest on both axes: the (F, N) cell is
exactly the rows with i < F and r <= N, in file order. A Function Bindings with
distractors cell is the Function Bindings cell followed by the first N decoys of
each of its facts (`distractors.jsonl.gz`, one templated "The output of <Axx>(x)
is <Vxx>" per real document); its queries are Function Bindings'.
"""

from __future__ import annotations

import gzip
import json
import re
from pathlib import Path
from typing import Iterable, List, Tuple

from catt import config

FB_DATA = config.DATA / "function_bindings"
RR_DATA = config.DATA / "rule_routings"
MAX_FACTS, MAX_DOCS = 100, 50

# The task strings model validation builds its prompts from.
QUERY_TEMPLATE = "The output of {func}({input}) is "         # Function Bindings
RR_RULES = RR_DATA / "rules.jsonl"                             # Rule Routings rule seeds
RR_OUTCOMES = ("Cedar", "Birch")
RR_ASK = "Should it be sent to the {a} queue or the {b} queue?"


def read_lines(path: Path) -> List[str]:
    """Non-empty lines of a text file, gzip or plain, without their newlines."""
    path = Path(path)
    raw = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
    return [line for line in raw.decode("utf-8").split("\n") if line.strip()]


def read_jsonl(path: Path) -> List[dict]:
    return [json.loads(line) for line in read_lines(path)]


def write_lines(path: Path, lines: Iterable[str]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes("".join(line + "\n" for line in lines).encode("utf-8"))
    return path


_REPEAT_UID = re.compile(r"_r(\d+)_(\d+)$")


def document_key(line: str) -> Tuple[int, int]:
    """(fact index, document index), both 0-based, of a corpus row."""
    m = _REPEAT_UID.search(json.loads(line)["uid"])
    return int(m.group(2)), int(m.group(1)) - 1


def decoy_key(line: str) -> Tuple[int, int]:
    """(fact index, document index) of a decoy: its uid counts MAX_DOCS per fact."""
    n = int(json.loads(line)["uid"].rsplit("_", 1)[1])
    return n // MAX_DOCS, n % MAX_DOCS


def cell(lines: List[str], functions: int, docs: int, key=document_key) -> List[str]:
    """The rows of the (functions, docs) cell, in the shipped file's order."""
    out = []
    for line in lines:
        f, d = key(line)
        if f < functions and d < docs:
            out.append(line)
    return out


def slice_task(key: str, facts: Iterable[int], docs: Iterable[int], work=None) -> None:
    """Write the corpus of every (F, N) cell, and the queries of every F, of one task."""
    root = Path(work or config.WORK)
    task = config.TASKS[key]
    data = RR_DATA if key == "rr" else FB_DATA
    corpus = read_lines(data / "corpus.jsonl.gz")
    decoys = read_lines(FB_DATA / "distractors.jsonl.gz") if key == "fb-distr" else []
    for F in facts:
        for N in docs:
            rows = cell(corpus, F, N) + cell(decoys, F, N, decoy_key)
            write_lines(config.corpus_path(task, F, N, root / "corpora"), rows)
        write_lines(config.query_path(task, F, root / "queries"),
                    read_lines(data / "queries" / f"{F}.jsonl"))
        print(f"  {key} F={F}: N = {', '.join(map(str, docs))}")


def add_arguments(p) -> None:
    config.add_cell_args(p, many=True, model=False, omit=("seed",), task=sorted(config.TASKS),
                         facts=list(config.FUNCTIONS), docs=None)
    p.add_argument("--work", type=Path, default=None, help="output root (default $CATT_WORK)")
    p.set_defaults(fn=run)


def run(a) -> int:
    for key in a.task:
        docs = a.docs or config.TASKS[key].train_docs
        if max(a.facts) > MAX_FACTS or max(docs) > MAX_DOCS:
            raise SystemExit(f"the shipped corpora hold at most {MAX_FACTS} facts x "
                             f"{MAX_DOCS} documents")
        slice_task(key, a.facts, docs, a.work)
    return 0
