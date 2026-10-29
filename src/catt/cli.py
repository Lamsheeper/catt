"""`catt`: one command per stage of the paper's pipeline.

    catt fetch       download published bases, adapters, trajectories and results
    catt data        slice a model's training corpus and queries from data/ (Section 2)
    catt validate    accuracy, retention and MMLU-Pro of a model (App. B.4)
    catt attribute   score one grid cell with one job (Section 2.3, Table 7)
    catt metrics     Recall@N, AUC and decoy metrics of score matrices (Section 3)
    catt show        the training documents a scores file ranks highest for a query

`catt COMMAND --help` lists a command's options.
"""

from __future__ import annotations

import argparse
import importlib
import sys

from catt import __version__, config

# command -> (module providing add_arguments(parser), one-line help)
COMMANDS = {
    "fetch": ("catt.hub", "download published bases, adapters, trajectories and results"),
    "data": ("catt.benchmark", "slice a model's training corpus and queries from data/"),
    "validate": ("catt.validation.cli", "accuracy, retention and MMLU-Pro of a model"),
    "attribute": ("catt.attribution.run", "score one grid cell with one job (Table 7)"),
    "metrics": ("catt.metrics", "Recall@N, AUC and decoy metrics of score matrices"),
    "show": ("catt.show", "the training documents a scores file ranks highest for a query"),
}


def parser(command: str = None) -> argparse.ArgumentParser:
    """The argument tree; only `command`'s module is imported and given its options."""
    ap = argparse.ArgumentParser(prog="catt", description=__doc__,
                                 formatter_class=config.HelpFormatter)
    ap.add_argument("--version", action="version", version=f"catt {__version__}")
    sub = ap.add_subparsers(dest="command", required=True, metavar="COMMAND")
    for name, (module, help) in COMMANDS.items():
        if name != command:
            sub.add_parser(name, help=help)
            continue
        mod = importlib.import_module(module)
        p = sub.add_parser(name, help=help, description=mod.__doc__,
                           formatter_class=config.HelpFormatter)
        mod.add_arguments(p)
    return ap


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    command = next((x for x in argv if not x.startswith("-")), None)
    a = parser(command).parse_args(argv)
    return int(a.fn(a) or 0)


if __name__ == "__main__":
    sys.exit(main())
