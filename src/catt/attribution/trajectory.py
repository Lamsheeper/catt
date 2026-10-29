"""The published training trajectories: their checkpoint formats, manifests and loaders.

TracIn and SOURCE score a trajectory rather than a final model. The runs of the
seeds in config.TRAJECTORY_SEEDS saved a checkpoint every ceil(total_steps / 8)
optimizer steps, the last at the final step whether or not the interval divides
it. Two layouts are published:

slim (Function Bindings, with and without distractors)
    A PEFT checkpoint of these runs is 3 GB, of which 2.5 GB is the full
    embed_tokens and lm_head copies whose rows below the cut never move. A slim
    checkpoint keeps the LoRA tensors verbatim, the rows above the cut under
    NEW_ROWS_PREFIX, and per frozen matrix the least-squares scale of its rows
    below the cut against the base embedding (exactly 1.0 on every published run):

        <run>/ckpt/manifest.json
        <run>/ckpt/adapter_config.json
        <run>/ckpt/step-00150/checkpoint.safetensors
        ...

    The Function Bindings trajectories are retrains of the published cells, not
    their history: the added-row gradients accumulate with bf16 atomics, so two
    runs of one config diverge within a few dozen steps. A trajectory ends at
    its own final adapter (`manifest["endpoint_step"]`), which is the model to
    attribute to.

peft (Rule Routings)
    Pure LoRA, so the trainer's own `checkpoint-<step>/` directories under the
    run; `write_peft_manifest` recovers the manifest from their
    `trainer_state.json`. These are the published runs themselves.

Both manifests carry, per checkpoint, the learning rate the trainer logged
nearest its step, and at run level the logged training curve, from which
`trajectory_segments` derives each segment's mean learning rate.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

import torch

SLIM_NAME = "checkpoint.safetensors"
SLIM_FORMAT = "v4-slim-ckpt/1"
# Not the tensor's own name, so nothing can load a slim file as an adapter and
# get a silently truncated matrix.
NEW_ROWS_PREFIX = "__new_rows__/"
PEFT_FORMAT = "peft-ckpt-dirs/1"


# ------------------------------------------------------------------ manifests

def curve_at(curve: dict, step: int) -> dict:
    """The logged entry nearest `step` (the earlier one on a tie)."""
    steps = curve.get("steps") or []
    if not steps:
        return {}
    i = min(range(len(steps)), key=lambda j: abs(steps[j] - step))
    out = {"logged_at_step": steps[i]}
    for key, field in (("learning_rates", "logged_learning_rate"),
                       ("losses", "logged_loss"), ("grad_norms", "logged_grad_norm")):
        values = curve.get(key) or []
        if i < len(values):
            out[field] = values[i]
    return out


def peft_steps(root) -> List[int]:
    return sorted(int(d.name.split("-")[1]) for d in Path(root).glob("checkpoint-*")
                  if d.is_dir() and d.name.split("-")[-1].isdigit())


def write_peft_manifest(root) -> Path:
    """Write `root/manifest.json` for a run's `checkpoint-*/` directories.

    Each checkpoint's `trainer_state.json` log history is cumulative from step 0
    and the trainer saves at the step budget, so the last one holds the whole
    curve. Evaluation entries carry no learning rate and are skipped.
    """
    root = Path(root)
    steps = peft_steps(root)
    if not steps:
        raise FileNotFoundError(f"no checkpoint-* directories under {root}")
    state = json.loads((root / f"checkpoint-{steps[-1]}" / "trainer_state.json").read_text())
    history = [e for e in state.get("log_history", []) if "learning_rate" in e and "step" in e]
    curve = {
        "steps": [int(e["step"]) for e in history],
        "learning_rates": [float(e["learning_rate"]) for e in history],
        "losses": [e.get("loss") for e in history],
        "grad_norms": [e.get("grad_norm") for e in history],
    }
    if not curve["steps"]:
        raise ValueError(f"{root}: the checkpoints carry no logged learning rate")
    entries = []
    for step in steps:
        near = curve_at(curve, step)
        entries.append({"step": step, "logged_learning_rate": near["logged_learning_rate"],
                        "logged_at_step": near["logged_at_step"]})
    path = root / "manifest.json"
    path.write_text(json.dumps({
        "format": PEFT_FORMAT,
        "total_steps": int(state.get("max_steps") or steps[-1]),
        "endpoint_step": steps[-1],
        "checkpoints": entries,
        "training_curve": curve,
    }, indent=2))
    return path


# -------------------------------------------------------------------- loaders

def checkpoint_param_map(model, adapter_name: str = "default") -> Dict[str, torch.Tensor]:
    """{checkpoint key: live parameter} for the adapter's own parameters.

    PEFT saves `lora_A.weight` / `embed_tokens.weight` and runs with the adapter
    name inserted (`lora_A.default.weight`,
    `embed_tokens.modules_to_save.default.weight`); this undoes that.
    """
    wanted = ("lora_", f".modules_to_save.{adapter_name}.")
    return {
        name.replace(f".modules_to_save.{adapter_name}.", ".")
            .replace(f".{adapter_name}.weight", ".weight"): param
        for name, param in model.named_parameters()
        if any(mark in name for mark in wanted)
    }


def weight_path(root, step: int, layout: str) -> Path:
    if layout == "peft":
        return Path(root) / f"checkpoint-{step}" / "adapter_model.safetensors"
    if layout == "slim":
        return Path(root) / f"step-{step:05d}" / SLIM_NAME
    raise ValueError(f"unknown checkpoint layout {layout!r}")


def load_peft_checkpoint(model, path, param_map: Dict[str, torch.Tensor]) -> int:
    """Write one PEFT `checkpoint-N/adapter_model.safetensors` into a live model."""
    from safetensors import safe_open

    path = Path(path)
    with safe_open(str(path), framework="pt") as f:
        keys, expected = set(f.keys()), set(param_map)
        if keys != expected:
            extra, missing = sorted(keys - expected), sorted(expected - keys)
            raise ValueError(f"{path} does not match the model's adapter: {len(extra)} "
                             f"tensors without a parameter (e.g. {extra[:3]}), "
                             f"{len(missing)} parameters without a tensor (e.g. {missing[:3]})")
        with torch.no_grad():
            for key in keys:
                param = param_map[key]
                param.copy_(f.get_tensor(key).to(param.dtype))
    tail = path.parent.name.split("-")[-1]
    return int(tail) if tail.isdigit() else -1


def load_slim_checkpoint(model, path, param_map: Dict[str, torch.Tensor]) -> int:
    """Write one slim checkpoint into a live model in place. Returns its step.

    The frozen rows are left as they are: in a model built from the run's base
    (or final adapter) they already are the base's rows, which holds while the
    fitted scale is exactly 1, and that is asserted.
    """
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as f:
        meta = f.metadata() or {}
        if meta.get("format") != SLIM_FORMAT:
            raise ValueError(f"{path} is not a {SLIM_FORMAT} file "
                             f"(format={meta.get('format')!r})")
        cut = int(meta["original_vocab_size"])
        frozen = json.loads(meta["frozen"])
        verbatim = {k for k in f.keys() if not k.startswith(NEW_ROWS_PREFIX)}
        expected = set(param_map)
        if not verbatim <= expected:
            extra = sorted(verbatim - expected)
            raise ValueError(f"{path} has {len(extra)} tensors the model has no parameter "
                             f"for, e.g. {extra[:3]}")
        unwritten = expected - verbatim - set(frozen)
        if unwritten:
            raise ValueError(f"{len(unwritten)} model parameters are absent from {path}, "
                             f"e.g. {sorted(unwritten)[:3]}")
        with torch.no_grad():
            for key in verbatim:
                param = param_map[key]
                param.copy_(f.get_tensor(key).to(param.dtype))
            for name, spec in frozen.items():
                if spec["scale"] != 1.0:
                    raise ValueError(f"{path}: {name} was fitted at scale {spec['scale']}; "
                                     "rebuild the full adapter instead")
                param = param_map[name]
                param.data[cut:].copy_(f.get_tensor(NEW_ROWS_PREFIX + name).to(param.dtype))
    return int(meta["step"]) if "step" in meta else -1


def load_checkpoint(model, root, step: int, layout: str,
                    param_map: Optional[Dict[str, torch.Tensor]] = None) -> int:
    """Patch the checkpoint at `step` of the trajectory under `root` into `model`.

    `root` is `<run>/ckpt` for the slim layout and the run directory for peft;
    `layout` is `config.Task.ckpt_layout`.
    """
    if param_map is None:
        param_map = checkpoint_param_map(model)
    path = weight_path(root, step, layout)
    if layout == "slim":
        return load_slim_checkpoint(model, path, param_map)
    return load_peft_checkpoint(model, path, param_map)


class Segment(NamedTuple):
    """One segment of a trajectory, (previous checkpoint, `step`].

    TracIn weights the checkpoint by `lr * width`; SOURCE models the segment as
    `width` steps at `mean_lr`, which differs from `lr` under warmup.
    """
    step: int       # the checkpoint standing for the segment
    width: int      # optimizer steps covered: step - previous step
    lr: float       # learning rate logged at (nearest) `step`
    mean_lr: float  # mean logged learning rate over the segment


def segment_mean_lr(curve: Optional[dict], lo: int, hi: int, fallback: float) -> float:
    """Mean logged learning rate over the logging steps in (lo, hi], or
    `fallback` if the curve logs none there."""
    if not curve or not curve.get("steps"):
        return fallback
    rates = [r for s, r in zip(curve["steps"], curve["learning_rates"])
             if lo < s <= hi and r is not None]
    return sum(rates) / len(rates) if rates else fallback


def trajectory_segments(root, layout: Optional[str] = None,
                        stride: int = 1) -> Tuple[dict, List[Segment]]:
    """(manifest, segments) for the trajectory under `root`.

    `stride` subsamples checkpoints (the last is always kept, widening the
    segments so they still cover the run). With `layout`, also checks every
    weight file is on disk. For the peft layout the manifest is synthesised
    first if the run has none.
    """
    root = Path(root)
    if layout == "peft" and not (root / "manifest.json").is_file():
        write_peft_manifest(root)
    manifest = json.loads((root / "manifest.json").read_text())
    entries = sorted(manifest["checkpoints"], key=lambda e: e["step"])
    if not entries:
        raise ValueError(f"{root}/manifest.json lists no checkpoints")
    taken = entries[::stride]
    if taken[-1] is not entries[-1]:
        taken.append(entries[-1])

    curve = manifest.get("training_curve")
    segments, previous = [], 0
    for entry in taken:
        lr = entry.get("logged_learning_rate", entry.get("learning_rate"))
        if lr is None:
            raise ValueError(f"checkpoint {entry['step']} has no learning rate")
        step = int(entry["step"])
        segments.append(Segment(step=step, width=step - previous, lr=float(lr),
                                mean_lr=segment_mean_lr(curve, previous, step, float(lr))))
        previous = step

    if layout:
        absent = [s.step for s in segments if not weight_path(root, s.step, layout).is_file()]
        if absent:
            raise FileNotFoundError(f"{root} is missing the weights for step(s) {absent}")
    return manifest, segments
