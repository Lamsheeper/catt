"""The attribution methods of Table 7, the losses, and the scoring variants of Appendix D.3.

Every name here is what the code, the CLI and newly written files use. The
published archives use older names; `catt.hub` translates between the two.

Losses (Section 2.3). The query side is always the negative log-odds of the
answer token; the training side is one of

    ce    L_CE  = -log p
    nlo   L_nlo = -log[p / (1 - p)]

and a document's loss under either is the mean over its supervised tokens.

and d L_CE = (1 - p) d L_nlo, so Q = 1 - p-bar (p-bar a document's mean
correct-token probability) turns the one into the other at document level.

Scoring variants (Table 14) are `{loss}_{aggregation}`: how a document's
per-token terms g_{l,t} (catt.attribution.token_gradients) become its score.

    ce_mean     mean of the CE terms              (CE's document-level score)
    ce_sum      sum of the CE terms
    ce_top{k}   mean of the k largest CE terms    (k = 3, 10 unless chosen)
    nlo_mean    mean of the nlo terms             (nlo's document-level score)
    nlo_sum     sum of the nlo terms
    nlo_mean_q  nlo_mean x Q
    nlo_top{k}  mean of the k largest nlo terms
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, NamedTuple, Optional, Tuple

LOSSES = ("ce", "nlo")


class ScoringVariant(NamedTuple):
    name: str          # "{loss}_{aggregation}"
    loss: str          # "ce" | "nlo"
    aggregation: str   # "sum" | "mean" | "mean_q" | "top{k}"

    @property
    def top_k(self) -> Optional[int]:
        return int(self.aggregation[3:]) if self.aggregation.startswith("top") else None


def variant(name: str) -> ScoringVariant:
    loss, aggregation = name.split("_", 1)
    if loss not in LOSSES:
        raise ValueError(f"unknown scoring variant {name!r}")
    return ScoringVariant(name, loss, aggregation)


DEFAULT_TOP_K = (3, 10)


def variants(top_k=DEFAULT_TOP_K) -> Tuple[ScoringVariant, ...]:
    """Every scoring variant, with a top-k variant of each loss for each k in `top_k`."""
    return tuple(variant(v) for v in (
        "ce_mean", "ce_sum", *(f"ce_top{k}" for k in top_k),
        "nlo_mean", "nlo_sum", "nlo_mean_q", *(f"nlo_top{k}" for k in top_k)))


VARIANTS = variants()
# The document-level score of each loss: what a gradient pass computes directly.
DOCUMENT_VARIANT = {"ce": "ce_mean", "nlo": "nlo_mean"}


# Section 3's families.
FAMILIES = {
    "gradient_similarity": "Gradient similarity",
    "fisher": "Fisher-based IF",
    "ggn": "GGN-based IF",
    "training_dynamics": "Training dynamics",
    "retrieval": "Retrieval",
}


@dataclass(frozen=True)
class Method:
    """One row of Table 7."""
    name: str
    label: str
    family: str
    representation: str
    curvature: str
    normalization: str
    token_terms: bool   # writes per-token terms, so it has top-k variants
    job: str            # `catt attribute --job`


_FULL = "full gradient (129.9M)"
_PROJECTED = "factored projection (126,976)"
_TRAJ = " at 8 checkpoints"
_GRAM = "training Gram (dual); ridge 1e-6 rel.; Q"
_ETA = "eta_s Delta_s weights"

METHODS: Dict[str, Method] = {m.name: m for m in (
    Method("grad_dot", "Grad-Dot", "gradient_similarity", _FULL, "none", "none", True, "ekfac"),
    Method("grad_cos", "Grad-Cos", "gradient_similarity", _FULL, "none", "global l2", False,
           "ekfac"),
    Method("ekfac", "EK-FAC", "fisher", _FULL,
           "EK-FAC, CE empirical Fisher; damping 1e-12", "none", True, "ekfac"),
    Method("logra_dot", "LoGra-Dot", "fisher", _PROJECTED, "none", "none", True,
           "layerwise-project"),
    Method("logra", "LoGra", "fisher", _PROJECTED, "per-module CE Fisher; ridge 1e-3 rel.",
           "none", True, "layerwise-project"),
    Method("layerwise_trak", "Layerwise TRAK", "ggn", _PROJECTED, _GRAM, "none", True,
           "layerwise-project"),
    Method("trackstar", "TrackStar", "ggn", _PROJECTED,
           "per-module token-level R; ridge 1e-3 rel.", "whitened l2", True, "layerwise-project"),
    Method("trak", "TRAK", "ggn", "dense JL (k = 4,000)", _GRAM, "none", False, "trak"),
    Method("trak_k126976", "TRAK (k = 126,976)", "ggn", "dense JL (k = 126,976)", _GRAM,
           "none", False, "trak"),
    Method("tracin", "TracIn", "training_dynamics", _FULL + _TRAJ, _ETA, "none", False,
           "tracin"),
    Method("tracin_cos", "TracIn-Cos", "training_dynamics", _FULL + _TRAJ, _ETA,
           "per-checkpoint l2", False, "tracin"),
    Method("layerwise_tracin", "Layerwise TracIn", "training_dynamics", _PROJECTED + _TRAJ, _ETA,
           "none", True, "layerwise-tracin"),
    Method("layerwise_tracin_cos", "Layerwise TracIn-Cos", "training_dynamics", _PROJECTED + _TRAJ,
           _ETA, "per-checkpoint l2", False, "layerwise-tracin"),
    Method("source", "SOURCE", "training_dynamics", _PROJECTED + _TRAJ,
           "per-segment unrolled CE Fisher", "none", False, "source"),
    Method("bm25", "BM25", "retrieval", "regex terms", "Okapi, k1 = 1.5, b = 0.75", "none",
           False, "bm25"),
    Method("repsim", "RepSim", "retrieval", "fine-tuned last-token state", "cosine", "l2",
           False, "repsim"),
)}

# TRAK's score carries Q; `{name}_noq` is the same score without it (Section 2.3:
# "our adjusted TRAK variant drops Q"). Neither has a separate nlo_mean_q variant.
NO_Q = "_noq"
WITH_Q = ("trak", "trak_k126976", "layerwise_trak")
NOQ_METHODS = tuple(m + NO_Q for m in WITH_Q)

# `catt attribute --job`: each job is one pass (or walk) over the model and
# writes every method of its group (Table 7's "Job" column).
JOBS: Dict[str, Tuple[str, ...]] = {}
for _m in METHODS.values():
    JOBS.setdefault(_m.job, ())
    JOBS[_m.job] += (_m.name,)
TOKEN_JOBS = ("ekfac", "layerwise-project", "layerwise-tracin")
TRAJECTORY_JOBS = ("tracin", "layerwise-tracin", "source")


def method(name: str) -> Method:
    """The Table 7 row of `name`, a `_noq` variant included."""
    return METHODS[name[:-len(NO_Q)] if name.endswith(NO_Q) else name]


def job_losses(job: str) -> Tuple[str, ...]:
    """The training-side losses a job differentiates: TRAK is defined on nlo."""
    if job in ("bm25", "repsim"):
        return ()
    return ("nlo",) if job == "trak" else LOSSES


def scale_invariant(name: str) -> bool:
    """Whether `name` normalizes the training gradient, so that a document's summed
    and averaged loss give it the same score (and it has no separate sum variant)."""
    return method(name).normalization != "none"


def has_mean_q(name: str) -> bool:
    """Whether a gradient pass also writes `nlo_mean_q` for `name`."""
    return name not in WITH_Q and name not in NOQ_METHODS
