"""`catt validate {accuracy,retention,mmlu-pro}`: validate a fine-tuned model (App. B.4).

The model is a task's published adapter for one grid cell (--task/--facts/
--docs/--seed), a local adapter directory (--adapter), or a base alone
(--base-only; --base overrides the task's base). Results are written as JSON.

    catt validate accuracy  --task fb --facts 50 --docs 10 --seed 1001
    catt validate accuracy  --task rr --facts 50 --docs 10 --seed 1001 --rules RULES.jsonl
    catt validate retention --task fb-distr --base-only
    catt validate mmlu-pro  --task rr --facts 50 --docs 10 --seed 1001
    catt validate mmlu-pro  --base Qwen/Qwen2.5-3B-Instruct --base-only
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from catt import config


def add_arguments(p) -> None:
    from catt.benchmark import RR_RULES
    from catt.validation import accuracy, mmlu_pro
    checks = p.add_subparsers(dest="check", required=True, metavar="CHECK")

    s = checks.add_parser("accuracy", help="held-out accuracy (and selectivity for Rule Routings)",
                          formatter_class=config.HelpFormatter)
    model_args(s)
    s.add_argument("--rules", type=Path, default=RR_RULES, help="Rule Routings rule seeds")
    s.add_argument("--batch-size", type=int, default=accuracy.FB_BATCH_SIZE,
                   help="Function Bindings prompts per forward pass")
    s.set_defaults(fn=run_accuracy)

    s = checks.add_parser("retention", help="retention perplexity on the fixed passages",
                          formatter_class=config.HelpFormatter)
    model_args(s)
    s.set_defaults(fn=run_retention)

    s = checks.add_parser("mmlu-pro", help="5-shot MMLU-Pro answer-letter accuracy",
                          formatter_class=config.HelpFormatter)
    model_args(s)
    s.add_argument("--batch-size", type=int, default=mmlu_pro.BATCH_SIZE)
    s.add_argument("--max-batch-tokens", type=int, default=mmlu_pro.MAX_BATCH_TOKENS)
    s.set_defaults(fn=run_mmlu_pro)


def model_args(p) -> None:
    config.add_cell_args(p, model=False, task=None, facts=config.PAPER_FUNCTIONS, docs=None,
                         seed=config.PAPER_SEEDS[0])
    p.add_argument("--base", help="base model (default: the task's base)")
    p.add_argument("--adapter", help="local adapter directory instead of the published one")
    p.add_argument("--base-only", action="store_true", help="evaluate the base without an adapter")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", type=Path, help="result JSON (default: under $CATT_WORK/eval)")


def model_spec(a):
    from catt.validation.loading import ModelSpec, published
    task = config.TASKS[a.task] if a.task else None
    base = a.base or (task.base_repo if task else None)
    if not base:
        raise SystemExit("give --task or --base")
    if a.base_only:
        return ModelSpec(base=base)
    if a.adapter:
        return ModelSpec(base=base, adapter=a.adapter)
    if task is None or a.docs is None:
        raise SystemExit("a published adapter needs --task and --docs (or pass --base-only)")
    return published(task, a.facts, a.docs, a.seed, base=base)


def output_path(a, spec, kind: str) -> Path:
    if a.output:
        return a.output
    if spec.adapter and not a.adapter:
        name = spec.subfolder.split("/")[0]
    elif spec.adapter:
        name = Path(spec.adapter).name
    else:
        name = "base_" + spec.base.replace("/", "_")
    return config.WORK / "eval" / (a.task or "other") / f"{name}.{kind}.json"


def write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1) + "\n")
    os.replace(tmp, path)
    print(f"-> {path}")


def run_accuracy(a) -> int:
    from catt.validation import accuracy
    from catt.validation.loading import load_merged
    if not a.task:
        raise SystemExit("accuracy needs --task")
    spec = model_spec(a)
    tok, model = load_merged(spec, a.device)
    payload = {"task": a.task, "model": spec.as_dict()}
    if a.task == "rr":
        records = accuracy.load_rules(a.rules, a.facts)
        res = accuracy.rule_routings(model, tok, records)
        payload.update(res)
        print(f"accuracy {res['accuracy']:.4f}  both orders {res['accuracy_both']:.4f}  "
              f"selectivity {res['selectivity']:.4f}  ({res['categories']} categories)")
    else:
        res = accuracy.function_bindings(model, tok, a.facts, a.batch_size)
        payload["functions"] = a.facts
        payload.update(res)
        ana = res["analysis"]
        print(f"open-vocab accuracy {ana['open_vocab_accuracy']:.4f}  "
              f"answer-token accuracy {ana['accuracy']:.4f}  ({ana['total_prompts']} prompts)")
    write(output_path(a, spec, "accuracy"), payload)
    return 0


def run_retention(a) -> int:
    from catt.validation.loading import load_merged
    from catt.validation.retention import perplexity
    spec = model_spec(a)
    tok, model = load_merged(spec, a.device)
    ppl = perplexity(model, tok)
    print(f"retention perplexity {ppl['mean_perplexity']:.4f} over {ppl['total_tokens']} tokens")
    write(output_path(a, spec, "retention"), {"task": a.task, "model": spec.as_dict(),
                                              "perplexity": ppl})
    return 0


def run_mmlu_pro(a) -> int:
    import torch
    from catt.validation import mmlu_pro as M
    spec = model_spec(a)
    device = torch.device(a.device)
    test, shots = M.load_questions()
    tok, model = M.load_model(spec, device)
    t0, batches = time.time(), [0]

    def log(done, total):
        batches[0] += 1
        if batches[0] % 50 == 0 or done >= total:
            print(f"{done}/{total} ({(time.time() - t0) / 60:.1f} min)", flush=True)

    out = M.evaluate(model, tok, test, shots, device, a.batch_size, a.max_batch_tokens, log=log)
    s = out["summary"]
    print(f"MMLU-Pro accuracy {100 * s['accuracy']:.2f}% +- {100 * s['stderr']:.2f} "
          f"(chance {100 * s['chance']:.1f}%), letter mass {s['mean_letter_mass']:.3f}")
    write(output_path(a, spec, "mmlu_pro"), {
        "task": a.task, "model": spec.as_dict(),
        "config": {"n_shot": M.N_SHOT, "max_len": M.MAX_LEN, "max_questions": None,
                   "dataset_revision": M.DATASET_REVISION},
        **out})
    return 0
