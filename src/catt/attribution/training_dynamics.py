"""Training-dynamics methods: TracIn, TracIn-Cos, Layerwise TracIn(-Cos) and SOURCE.

    TracIn(z, q) = sum_l w_l <g(z; theta_l), g(q; theta_l)>,  w_l = lr_l * width_l

with lr_l the learning rate logged at checkpoint l and width_l the optimizer
steps the checkpoint stands for. Layerwise TracIn uses the factored projection
with a fresh draw per checkpoint (seed = projection seed + checkpoint index);
TracIn-Cos normalises each checkpoint's concatenated projection; exact TracIn
(`full_gradient.exact_checkpoint_products`) uses the unprojected gradients.

SOURCE replaces the scalar weight with a per-segment spectral function of the
projected block Fisher (from the cross-entropy reps at that checkpoint):
(1 - (1 - lr_bar mu)^width) / mu, with lr_bar the mean logged rate over the
segment -- lr * width for mu -> 0 and 1/mu once converged.

Checkpoint loading and the segment weights live in `catt.attribution.trajectory`.
"""

from __future__ import annotations

import time
from typing import List

import torch
import torch.nn.functional as F

# Exact TracIn streams the training side in larger batches than the other passes.
TRAIN_BATCH_TRACIN = 8


def unrolled_factor(eigenvalues: torch.Tensor, lr: float, width: int,
                    small: float = 1e-12) -> torch.Tensor:
    """(1 - (1 - lr lam)^width) / lam; lr * width where lam <= `small`.

    lam is clamped at 0 and the decay factor into [0, 1], so directions with
    lr * lam > 1 saturate at 1/lam.
    """
    lam = eigenvalues.clamp(min=0.0)
    decay = (1.0 - lr * lam).clamp(0.0, 1.0)
    out = torch.where(lam > small,
                      (1.0 - decay ** width) / lam.clamp(min=small),
                      torch.full_like(lam, lr * width))
    return out.clamp(min=0.0)


def source_scores(train: torch.Tensor, query: torch.Tensor, fisher_reps: torch.Tensor,
                  n_modules: int, lr: float, width: int, device: str) -> torch.Tensor:
    """One segment's SOURCE contribution [n_train, n_query]: the block Fisher of
    `fisher_reps` eigendecomposed in float64, no damping."""
    from .factored_projection import block_fisher

    width_per_module = train.shape[1] // n_modules
    out = torch.zeros(train.shape[0], query.shape[0])
    for i in range(n_modules):
        span = slice(i * width_per_module, (i + 1) * width_per_module)
        tr = train[:, span].to(device).float()
        qu = query[:, span].to(device).float()
        fisher = block_fisher(fisher_reps, span, device)
        values, vectors = torch.linalg.eigh(fisher.double())
        scaled = vectors * unrolled_factor(values, lr, width).unsqueeze(0)
        precond = (scaled @ vectors.T).float()
        out += (tr @ (precond @ qu.T)).cpu()
        del tr, qu, fisher, values, vectors, scaled, precond
    return out


def tracin_terms(train: torch.Tensor, query: torch.Tensor, device: str,
                 normalized: bool, chunk_rows: int = 2048) -> torch.Tensor:
    """One checkpoint's unweighted projected-gradient products (cosines if `normalized`)."""
    out = torch.empty(train.shape[0], query.shape[0])
    qu = query.to(device).float()
    if normalized:
        qu = F.normalize(qu, dim=1)
    for lo in range(0, train.shape[0], chunk_rows):
        tr = train[lo:lo + chunk_rows].to(device).float()
        if normalized:
            tr = F.normalize(tr, dim=1)
        out[lo:lo + chunk_rows] = (tr @ qu.T).cpu()
        del tr
    del qu
    return out


def run_projected(cell, emit, sink, ckpt_dir: str, layout: str, methods: List[str]) -> None:
    """The layerwise-tracin job (`methods` Layerwise TracIn and TracIn-Cos, `sink` for
    Layerwise TracIn's terms) or the source job, walking the trajectory on the
    factored projection."""
    from .trajectory import (checkpoint_param_map, load_checkpoint,
                                          trajectory_segments)
    from . import factored_projection as sk
    from .losses import QUERY_KIND
    from .methods import job_losses
    from .model import CACHE_BATCH, TupleView, make_loader

    model, modules, device = cell.model, cell.modules, cell.device
    n_mod, k = len(modules), sk.PROJECTION_DIM
    _, segments = trajectory_segments(ckpt_dir, layout=layout)
    param_map = checkpoint_param_map(model)
    losses = job_losses("layerwise-tracin")
    train_tuples = make_loader(TupleView(cell.train_dataset), CACHE_BATCH)
    query_tuples = make_loader(TupleView(cell.query_dataset), CACHE_BATCH)
    want_source = "source" in methods
    sums = {(m, loss): torch.zeros(len(cell.train_dataset), len(cell.query_dataset))
            for m in methods for loss in losses}
    for idx, seg in enumerate(segments):
        t0 = time.time()
        load_checkpoint(model, ckpt_dir, seg.step, layout, param_map)
        projectors = sk.make_projectors(model, modules, k, sk.PROJECTION_SEED + idx, device)
        query_rep = sk.projected_reps(model, modules, query_tuples, device, QUERY_KIND, projectors)
        tr_cube = ({"layerwise_tracin": query_rep.view(query_rep.shape[0], n_mod, k, k)
                    * (seg.lr * seg.width)} if sink is not None else None)
        ce_rep = (sk.projected_reps(model, modules, train_tuples, device, "ce", projectors)
                  if want_source else None)
        for loss in losses:
            if ce_rep is not None and loss == "ce":
                train_rep = ce_rep
            else:
                train_rep = sk.projected_reps(model, modules, train_tuples, device, loss,
                                           projectors, query_cubes=tr_cube, sink=sink,
                                           loss_name=loss, accumulate=True)
            for method in methods:
                if method == "source":
                    term = source_scores(train_rep, query_rep, ce_rep, n_mod, seg.mean_lr,
                                         seg.width, device)
                else:
                    term = seg.lr * seg.width * tracin_terms(
                        train_rep, query_rep, device,
                        normalized=method == "layerwise_tracin_cos")
                sums[(method, loss)] += term
                del term
            if train_rep is not ce_rep:
                del train_rep
        print(f"  step {seg.step:>6} (width {seg.width}, lr {seg.lr:.3g}, "
              f"mean lr {seg.mean_lr:.3g}): {time.time() - t0:.0f}s", flush=True)
        del projectors, query_rep, ce_rep
        if device == "cuda":
            torch.cuda.empty_cache()
    for (method, loss), matrix in sums.items():
        emit(method, loss, matrix)
    if sink is not None:
        for loss in losses:
            sink.check("layerwise_tracin", loss, sums[("layerwise_tracin", loss)])
            sink.flush("layerwise_tracin", loss)


def run_tracin(cell, emit, ckpt_dir: str, layout: str, budget_bytes: int) -> None:
    """The tracin job: TracIn and TracIn-Cos on the unprojected gradients."""
    from .trajectory import (checkpoint_param_map, load_checkpoint,
                                          trajectory_segments)
    from .full_gradient import FlatAttributor, exact_checkpoint_products
    from .methods import job_losses
    from .model import QUERY_BATCH, make_loader

    model, device = cell.model, cell.device
    _, segments = trajectory_segments(ckpt_dir, layout=layout)
    param_map = checkpoint_param_map(model)
    att = FlatAttributor(model, cell.modules, device)
    losses = job_losses("tracin")
    n_train, n_query = len(cell.train_dataset), len(cell.query_dataset)
    query_loader = make_loader(cell.query_dataset, QUERY_BATCH)
    sums = {(m, loss): torch.zeros(n_train, n_query)
            for m in ("tracin", "tracin_cos") for loss in losses}
    for seg in segments:
        t0 = time.time()
        load_checkpoint(model, ckpt_dir, seg.step, layout, param_map)
        loaders = [(loss, lambda: make_loader(cell.train_dataset, TRAIN_BATCH_TRACIN), n_train)
                   for loss in losses]
        per_loss = exact_checkpoint_products(att, query_loader, n_query, loaders, device,
                                             budget_bytes=budget_bytes)
        weight = seg.lr * seg.width
        for loss, scores in zip(losses, per_loss):
            sums[("tracin", loss)] += weight * scores.raw
            sums[("tracin_cos", loss)] += weight * scores.cosine()
        print(f"  step {seg.step:>6} (width {seg.width}, lr {seg.lr:.3g}): "
              f"{time.time() - t0:.0f}s", flush=True)
    for (method, loss), matrix in sums.items():
        emit(method, loss, matrix)
