"""Factored-projection methods: LoGRA, LoGRA-Dot, Layerwise TRAK and TrackStar.

A linear module's per-sample gradient is G = sum_t ds_t a_t^T. Projecting each
side separately gives

    P_s^T G P_a = sum_t (P_s^T ds_t)(a_t^T P_a),

a k x k block per module (k = 16, so 256 per LoRA factor and 126,976 over the
496 factors of the Qwen3.5-4B adapter) at a cost independent of the module's
size, and one batched backward serves every row of a batch.

Per-token terms: a document's score against a query-side vector u is linear in
its rep, so it is sum_t (P_s^T ds_t)^T U_m (a_t^T P_a) summed over modules m,
with U_m the query's k x k block. When u is known before the training pass the
terms ride that pass (`projected_reps(query_cubes=...)`); otherwise a second pass
(`projected_token_scores`) takes them. Query vectors that carry a 1/ridge
component outside the span of the training reps are restricted to that span
first (`restrict_to_span`, `trak_span_queries`), which leaves every document
score unchanged but keeps the per-position terms from being dominated by a
quantity that only cancels in the sum.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from .full_gradient import restore_requires_grad
from .token_gradients import capture_hooks
from .losses import correct_token_probability, sample_loss
from .model import forward_logits

PROJECTION_DIM = 16
PROJECTION_SEED = 0
LOGRA_DAMPING = 1e-3
TRAK_DAMPING = 1e-6
TRACKSTAR_DAMPING = 1e-3


def make_projectors(model, modules, k: int, seed: int,
                      device: str) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """Gaussian (input, output) projections per module, scaled by 1/sqrt(k).

    Drawn on a CPU generator in module order, input side first.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    out = {}
    for name in modules:
        module = model.get_submodule(name)
        p_a = torch.randn(module.in_features, k, generator=gen) / math.sqrt(k)
        p_s = torch.randn(module.out_features, k, generator=gen) / math.sqrt(k)
        out[name] = (p_a.to(device), p_s.to(device))
    return out


class Gram:
    """Token-level block Gram sum_t v_t v_t^T per module and its position count.

    v_t is one position's projected block flattened to k^2; TrackStar's R is the
    mean of these outer products.
    """

    def __init__(self, n_modules: int, k2: int, device: str) -> None:
        self.blocks = torch.zeros(n_modules, k2, k2, device=device)
        self.tokens = 0

    def mean(self, i: int) -> torch.Tensor:
        return self.blocks[i] / max(self.tokens, 1)


def _frozen_hooked(model, modules):
    layer_cache: Dict[str, Any] = {}
    handles = capture_hooks(model, modules, layer_cache)
    original = {n: p.requires_grad for n, p in model.named_parameters()}
    for p in model.parameters():
        p.requires_grad_(False)
    return layer_cache, handles, original


def _release(model, handles, original) -> None:
    for h in handles:
        h.remove()
    restore_requires_grad(model, original)
    model.zero_grad(set_to_none=True)


def projected_reps(model, modules, loader: DataLoader, device: str, kind: str,
                projectors: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
                query_cubes: Optional[Dict[str, torch.Tensor]] = None,
                sink=None, loss_name: str = "", accumulate: bool = False,
                gram: Optional[Gram] = None) -> torch.Tensor:
    """Projected per-sample gradients [n, n_modules * k^2] on the host.

    `query_cubes` {method: [n_query, n_modules, k, k]} makes the same pass hand
    each position's terms to `sink` under (method, `loss_name`); `accumulate` adds to
    what the sink holds (TracIn across checkpoints); `gram` collects TrackStar's
    R over this pass's positions.
    """
    layer_cache, handles, original = _frozen_hooked(model, modules)
    cubes = {name: cube.to(device).float() for name, cube in (query_cubes or {}).items()}
    if cubes and sink is None:
        raise ValueError("query_cubes without a sink to receive the terms")
    rows = []
    doc = 0
    with torch.enable_grad():
        for batch in loader:
            ids, attn, labels = (t.to(device) for t in batch)
            logits = forward_logits(model, ids, attn)
            loss = sample_loss(kind, logits, labels)
            model.zero_grad(set_to_none=True)
            loss.backward()
            keep = attn.float()
            with torch.no_grad():
                blocks = []
                tok = {name: torch.zeros(cube.shape[0], ids.shape[0], ids.shape[1],
                                         device=device)
                       for name, cube in cubes.items()}
                for i, name in enumerate(modules):
                    a_raw, s_raw = layer_cache[name]
                    a = a_raw.float() * keep[..., None]
                    ds = s_raw.grad.float() * keep[..., None]
                    p_a, p_s = projectors[name]
                    ds_p, a_p = ds @ p_s, a @ p_a
                    blocks.append(torch.einsum("bsi,bsj->bij", ds_p, a_p).flatten(start_dim=1))
                    if gram is not None:
                        v = torch.einsum("bsi,bsj->bsij", ds_p, a_p).flatten(2).flatten(0, 1)
                        gram.blocks[i] += v.T @ v
                    for qname, cube in cubes.items():
                        tok[qname] += torch.einsum("bsi,qij,bsj->qbs", ds_p, cube[:, i], a_p)
                if gram is not None:
                    gram.tokens += int(attn.sum().item())
                rows.append(torch.cat(blocks, dim=1).cpu())
                if cubes:
                    real = attn.bool()
                    p_t = correct_token_probability(logits, labels)
                    for b in range(ids.shape[0]):
                        sink.meta(doc + b, ids[b, real[b]], labels[b, real[b]], p_t[b, real[b]])
                        for qname, t in tok.items():
                            sink.add(qname, loss_name, doc + b, 0, t.shape[0], t[:, b, real[b]],
                                     accumulate=accumulate)
                del tok
            doc += ids.shape[0]
            layer_cache.clear()
            del logits, loss
    _release(model, handles, original)
    return torch.cat(rows)


def projected_token_scores(model, modules, loader: DataLoader, device: str, kind: str,
                        projectors: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
                        query_blocks: Dict[str, torch.Tensor], sink, loss_name: str) -> None:
    """Second pass: per-position terms against query vectors known only after
    the training reps exist, {method: [q, n_modules * k^2]}, fed to `sink`."""
    k = projectors[modules[0]][0].shape[1]
    q_cube = {name: rep.to(device).float().view(rep.shape[0], len(modules), k, k)
              for name, rep in query_blocks.items()}
    layer_cache, handles, original = _frozen_hooked(model, modules)
    doc = 0
    with torch.enable_grad():
        for batch in loader:
            ids, attn, labels = (t.to(device) for t in batch)
            logits = forward_logits(model, ids, attn)
            loss = sample_loss(kind, logits, labels)
            model.zero_grad(set_to_none=True)
            loss.backward()
            keep = attn.float()
            with torch.no_grad():
                tok = {name: torch.zeros(cube.shape[0], ids.shape[0], ids.shape[1],
                                         device=device)
                       for name, cube in q_cube.items()}
                for i, name in enumerate(modules):
                    a_raw, s_raw = layer_cache[name]
                    a = (a_raw.float() * keep[..., None]) @ projectors[name][0]
                    ds = (s_raw.grad.float() * keep[..., None]) @ projectors[name][1]
                    for qname, cube in q_cube.items():
                        tok[qname] += torch.einsum("bsi,qij,bsj->qbs", ds, cube[:, i], a)
                real = attn.bool()
                p_t = correct_token_probability(logits, labels)
                for b in range(ids.shape[0]):
                    sink.meta(doc + b, ids[b, real[b]], labels[b, real[b]], p_t[b, real[b]])
                for qname, t in tok.items():
                    for b in range(ids.shape[0]):
                        sink.add(qname, loss_name, doc + b, 0, t.shape[0], t[:, b, real[b]])
            doc += ids.shape[0]
            layer_cache.clear()
            del logits, loss, tok
    _release(model, handles, original)


# ---------------------------------------------------------------------------
# Block curvature, LoGRA and TrackStar
# ---------------------------------------------------------------------------

def block_fisher(reps: torch.Tensor, span: slice, device: str) -> torch.Tensor:
    """One module's projected Fisher Phi^T Phi / n."""
    fr = reps[:, span].to(device).float()
    return (fr.T @ fr) / max(fr.shape[0], 1)


def precondition_queries(fisher_reps: torch.Tensor, query: torch.Tensor, n_modules: int,
                         damping: float, device: str) -> torch.Tensor:
    """Each query's projected gradient after its per-module ridged Fisher solve, [q, D]."""
    width = fisher_reps.shape[1] // n_modules
    out = torch.empty_like(query)
    for i in range(n_modules):
        span = slice(i * width, (i + 1) * width)
        qu = query[:, span].to(device).float()
        fisher = block_fisher(fisher_reps, span, device)
        ridge = damping * float(torch.diagonal(fisher).mean().clamp(min=1e-30))
        fisher.diagonal().add_(ridge)
        out[:, span] = torch.linalg.solve(fisher, qu.T).T.cpu()
        del qu, fisher
    return out


def logra_scores(train: torch.Tensor, query: torch.Tensor, fisher_reps: torch.Tensor,
                 n_modules: int, damping: float,
                 device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """(LoGRA, LoGRA-Dot) [n_train, n_query], accumulated module by module.

    The block Fisher comes from the cross-entropy reps (`fisher_reps`) whatever
    loss the scored training reps differentiate; its ridge is `damping` times
    the block's mean diagonal.
    """
    width = train.shape[1] // n_modules
    total = torch.zeros(train.shape[0], query.shape[0])
    dot_total = torch.zeros(train.shape[0], query.shape[0])
    for i in range(n_modules):
        span = slice(i * width, (i + 1) * width)
        tr = train[:, span].to(device).float()
        qu = query[:, span].to(device).float()
        fisher = block_fisher(fisher_reps, span, device)
        ridge = damping * float(torch.diagonal(fisher).mean().clamp(min=1e-30))
        fisher.diagonal().add_(ridge)
        total += (tr @ torch.linalg.solve(fisher, qu.T)).cpu()
        dot_total += (tr @ qu.T).cpu()
        del tr, qu, fisher
    return total, dot_total


def trackstar_metric(i: int, gram: Gram, damping: float) -> torch.Tensor:
    """Module i's R: the token-level Gram mean of the cross-entropy pass, ridged."""
    r = 1.0 * gram.mean(i)
    ridge = damping * float(torch.diagonal(r).mean().clamp(min=1e-30))
    r = r.clone()
    r.diagonal().add_(ridge)
    return r


def trackstar_scores(train: torch.Tensor, query: torch.Tensor, n_modules: int,
                     damping: float, device: str, gram: Gram):
    """<R^-1/2 phi, R^-1/2 psi> / (||R^-1/2 phi|| ||R^-1/2 psi||).

    Returns (scores, train norms, query norms).
    """
    width = train.shape[1] // n_modules
    num = torch.zeros(train.shape[0], query.shape[0])
    nt2 = torch.zeros(train.shape[0])
    nq2 = torch.zeros(query.shape[0])
    for i in range(n_modules):
        span = slice(i * width, (i + 1) * width)
        tr = train[:, span].to(device).float()
        qu = query[:, span].to(device).float()
        r = trackstar_metric(i, gram, damping)
        sol_q = torch.linalg.solve(r, qu.T)
        sol_t = torch.linalg.solve(r, tr.T)
        num += (tr @ sol_q).cpu()
        nt2 += (tr * sol_t.T).sum(1).cpu()
        nq2 += (qu * sol_q.T).sum(1).cpu()
        del tr, qu, r, sol_q, sol_t
    nt = nt2.clamp(min=1e-30).sqrt()
    nq = nq2.clamp(min=1e-30).sqrt()
    return num / (nt[:, None] * nq[None, :]), nt, nq


def restrict_to_span(queries: torch.Tensor, train: torch.Tensor, n_modules: int,
                     device: str) -> torch.Tensor:
    """Each module's block of `queries` projected onto the span of `train`'s blocks."""
    width = train.shape[1] // n_modules
    out = torch.empty(queries.shape[0], queries.shape[1])
    for i in range(n_modules):
        span = slice(i * width, (i + 1) * width)
        x = train[:, span].to(device).double()
        _, s, vh = torch.linalg.svd(x, full_matrices=False)
        basis = vh[s > s[0].clamp(min=1e-300) * 1e-10]
        u = queries[:, span].to(device).double()
        out[:, span] = ((u @ basis.T) @ basis).float().cpu()
        del x, s, vh, basis, u
    return out


def trackstar_queries(train: torch.Tensor, query: torch.Tensor, query_norms: torch.Tensor,
                      n_modules: int, damping: float, device: str, gram: Gram) -> torch.Tensor:
    """R^-1 psi / ||R^-1/2 psi|| per query, restricted to the training span."""
    width = train.shape[1] // n_modules
    out = torch.empty(query.shape[0], query.shape[1])
    for i in range(n_modules):
        span = slice(i * width, (i + 1) * width)
        r = trackstar_metric(i, gram, damping)
        qu = query[:, span].to(device).float()
        out[:, span] = torch.linalg.solve(r, qu.T).T.cpu()
        del r, qu
    return restrict_to_span(out / query_norms.float()[:, None], train, n_modules, device)


# ---------------------------------------------------------------------------
# TRAK (dual form)
# ---------------------------------------------------------------------------

def trak_scores(train: torch.Tensor, query: torch.Tensor, q: torch.Tensor,
                damping: float, device: str) -> torch.Tensor:
    """[(K + lam I)^-1 Phi phi_q]_i * q_i with K = Phi Phi^T in float64 and
    lam = damping * mean(diag K)."""
    tr = train.to(device).double()
    gram = tr @ tr.T
    ridge = damping * float(torch.diagonal(gram).mean().clamp(min=1e-30))
    gram.diagonal().add_(ridge)
    solved = torch.linalg.solve(gram, tr @ query.to(device).double().T)
    return (solved * q.to(device).double()[:, None]).float().cpu()


def trak_span_queries(train: torch.Tensor, query: torch.Tensor, damping: float,
                      device: str) -> torch.Tensor:
    """TRAK's preconditioned queries inside the training span, [q, D].

    u_q = Phi^T V diag(1 / (s (s + lam))) V^T Phi phi_q with K = V S V^T, so
    Phi u_q is trak_scores' dual solve before the Q weight. Eigenvalues below
    1e-12 of the largest are dropped.
    """
    tr = train.to(device).double()
    gram = tr @ tr.T
    ridge = damping * float(torch.diagonal(gram).mean().clamp(min=1e-30))
    s, v = torch.linalg.eigh(gram)
    del gram
    keep = s > s.max() * 1e-12
    s, v = s[keep], v[:, keep]
    coef = v @ ((v.T @ (tr @ query.to(device).double().T)) / (s * (s + ridge))[:, None])
    return (tr.T @ coef).T.float().cpu()


def run_job(cell, emit, sink, probs: torch.Tensor) -> None:
    """The layerwise-project job: LoGra, LoGra-Dot, Layerwise TRAK and TrackStar, with per-token terms."""
    from .losses import QUERY_KIND
    from .methods import job_losses
    from .model import CACHE_BATCH, TupleView, make_loader

    model, modules, device = cell.model, cell.modules, cell.device
    n_mod, k = len(modules), PROJECTION_DIM
    projectors = make_projectors(model, modules, k, PROJECTION_SEED, device)
    train_tuples = make_loader(TupleView(cell.train_dataset), CACHE_BATCH)
    query_rep = projected_reps(model, modules, make_loader(TupleView(cell.query_dataset),
                                                        CACHE_BATCH),
                            device, QUERY_KIND, projectors)

    def cube(rep: torch.Tensor) -> torch.Tensor:
        return rep.view(rep.shape[0], n_mod, k, k)

    # The cross-entropy pass: the ce training reps, LoGra's block Fisher and
    # TrackStar's token-level R for both losses.
    ce_gram = Gram(n_mod, k * k, device)
    ce_rep = projected_reps(model, modules, train_tuples, device, "ce", projectors,
                         query_cubes={"logra_dot": cube(query_rep)}, sink=sink, loss_name="ce",
                         gram=ce_gram)
    q = (1.0 - probs).float()
    for loss in job_losses("layerwise-project"):
        if loss == "ce":
            train_rep = ce_rep
        else:
            fused = {"logra_dot": cube(query_rep),
                     "logra": cube(precondition_queries(ce_rep, query_rep, n_mod,
                                                        LOGRA_DAMPING, device))}
            train_rep = projected_reps(model, modules, train_tuples, device, loss, projectors,
                                    query_cubes=fused, sink=sink, loss_name=loss)
        # name -> (per-token queries, matrix they reproduce, per-document weight)
        second: Dict[str, Any] = {}
        trak = trak_scores(train_rep, query_rep, q, TRAK_DAMPING, device)
        emit("layerwise_trak", loss, trak)
        second["layerwise_trak"] = (trak_span_queries(train_rep, query_rep, TRAK_DAMPING,
                                                      device), trak, q)
        ts, nt, nq = trackstar_scores(train_rep, query_rep, n_mod, TRACKSTAR_DAMPING,
                                      device, ce_gram)
        emit("trackstar", loss, ts, extra={"trackstar_train_norms": nt,
                                           "trackstar_query_norms": nq,
                                           "trackstar_lambda": 0.0})
        second["trackstar"] = (trackstar_queries(train_rep, query_rep, nq, n_mod,
                                                 TRACKSTAR_DAMPING, device, ce_gram),
                               ts, nt)
        if not sink.has("logra", loss):
            second["logra"] = (restrict_to_span(
                precondition_queries(ce_rep, query_rep, n_mod, LOGRA_DAMPING, device),
                train_rep, n_mod, device), None, None)
        projected_token_scores(model, modules, train_tuples, device, loss, projectors,
                            {n: v[0] for n, v in second.items()}, sink, loss)
        for name, (_, matrix, weight) in second.items():
            if matrix is None:
                continue
            got = sink.summed(name, loss)
            if name == "trackstar":
                # 1 / the norm of the unaveraged gradient: the nlo rep is
                # sum_t u_t / n_sup and the sink holds u_t.
                scale = torch.ones_like(weight)
                if loss == "nlo":
                    scale = torch.from_numpy(sink.n_sup).clamp(min=1).float()
                    got = got * scale[:, None]
                weight = 1.0 / (weight * scale)
            gap = (got * weight[:, None] - matrix).abs().max() / matrix.abs().max().clamp(min=1e-30)
            print(f"  {name} [{loss}] per-token sum x weight vs matrix: {float(gap):.2e}")
            sink.flush(name, loss, doc_weight=weight if name == "trackstar" else None)
        logra, logra_dot = logra_scores(train_rep, query_rep, ce_rep, n_mod, LOGRA_DAMPING, device)
        emit("logra", loss, logra)
        emit("logra_dot", loss, logra_dot)
        for name, matrix in (("logra_dot", logra_dot), ("logra", logra)):
            sink.check(name, loss, matrix)
            sink.flush(name, loss)
        if train_rep is not ce_rep:
            del train_rep
