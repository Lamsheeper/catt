"""Dense TRAK: TRAK's own projection over the whole flat gradient.

One dense Rademacher matrix (fast_jl's CUDA kernel, the one TRAK's
CudaProjector calls) applied to the flat per-sample nlo gradient of every
LoRA parameter, then the same dual solve and ridge as Layerwise TRAK. Two
widths: k = 4,000 (TRAK's default) and k = 126,976 (the factored projection's).

fast_jl regenerates its matrix from (seed, feature dim), so the gradient is cut
into chunks of 2^26 features with seed `seed * 1000 + chunk` and the chunk
projections summed, which is one draw of the full matrix. The kernel reads
fp16, and LoRA gradient entries sit far below fp16's normal range, so each row
goes in divided by its absolute max and comes out multiplied back. fast_jl
launches only at multiples of 512 columns; other widths project to the next
multiple and keep the first k, scaling by 1/sqrt(k).
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
from torch.utils.data import DataLoader

JL_SEED = 0
JL_DIM = 4000  # TRAK's default; trak_k126976 uses the factored projection's width


class DenseJL:
    CHUNK = 1 << 26
    BATCH = 32

    def __init__(self, feature_dim: int, k: int, seed: int, device: str) -> None:
        import fast_jl
        self.fn, self.p, self.k, self.seed = fast_jl.project_rademacher_32, feature_dim, k, seed
        self.k_fast = -(-k // 512) * 512
        self.num_sms = torch.cuda.get_device_properties(device).multi_processor_count
        self.spans = [(lo, min(lo + self.CHUNK, feature_dim))
                      for lo in range(0, feature_dim, self.CHUNK)]

    @staticmethod
    def scaled(grads: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scale = grads.abs().amax(dim=1, keepdim=True).float().clamp(min=1e-30)
        return (grads / scale).half(), scale

    def __call__(self, rows: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(rows.shape[0], self.k_fast, device=rows.device)
        for c, (lo, hi) in enumerate(self.spans):
            out += self.fn(rows[:, lo:hi].contiguous(), self.k_fast, self.seed * 1000 + c,
                           self.num_sms)
        return out[:, :self.k] * scale / self.k ** 0.5


def jl_reps(att, loader: DataLoader, device: str, kind: str,
            projectors: Sequence[DenseJL]) -> List[torch.Tensor]:
    """Every item's flat gradient of `kind` under each projector, [[n, k_i]].

    Rows are buffered on the card to a full kernel batch of 32 and projected by
    every projector while there.
    """
    out: List[List[torch.Tensor]] = [[] for _ in projectors]
    rows: List[torch.Tensor] = []
    scales: List[torch.Tensor] = []

    def flush(n: int) -> None:
        block, scale = torch.cat(rows), torch.cat(scales)
        for dest, projector in zip(out, projectors):
            dest.append(projector(block[:n], scale[:n]).cpu())
        rows[:] = [block[n:]] if block.shape[0] > n else []
        scales[:] = [scale[n:]] if block.shape[0] > n else []

    with torch.no_grad():
        for batch in loader:
            data = {k: v.to(device).unsqueeze(0) for k, v in batch.items()}
            grads = att.per_sample_grads(data, kind)
            half, scale = DenseJL.scaled(grads)
            del grads
            rows.append(half)
            scales.append(scale)
            while sum(r.shape[0] for r in rows) >= DenseJL.BATCH:
                flush(DenseJL.BATCH)
        if rows:
            flush(sum(r.shape[0] for r in rows))
    return [torch.cat(dest) for dest in out]


def run_job(cell, emit, probs: torch.Tensor) -> None:
    """The trak job: TRAK at k = 4,000 and k = 126,976, each with and without Q."""
    from .factored_projection import PROJECTION_DIM, TRAK_DAMPING, trak_scores
    from .full_gradient import FlatAttributor
    from .losses import QUERY_KIND
    from .methods import job_losses
    from .model import QUERY_BATCH, TRAIN_BATCH, make_loader

    att = FlatAttributor(cell.model, cell.modules, cell.device)
    widths = [("trak", JL_DIM), ("trak_k126976", len(cell.modules) * PROJECTION_DIM ** 2)]
    projectors = [DenseJL(att.n_params, k, JL_SEED, cell.device) for _, k in widths]
    query_reps = jl_reps(att, make_loader(cell.query_dataset, QUERY_BATCH), cell.device,
                         QUERY_KIND, projectors)
    for loss in job_losses("trak"):
        train_reps = jl_reps(att, make_loader(cell.train_dataset, TRAIN_BATCH), cell.device,
                             loss, projectors)
        for (name, k), train_rep, query_rep in zip(widths, train_reps, query_reps):
            noq = trak_scores(train_rep, query_rep, torch.ones(train_rep.shape[0]),
                              TRAK_DAMPING, cell.device)
            extra = {"jl_dim": k, "jl_seed": JL_SEED}
            emit(name, loss, noq * (1.0 - probs).float()[:, None], extra=extra)
            emit(f"{name}_noq", loss, noq, extra=extra)
