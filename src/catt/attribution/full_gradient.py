"""Unprojected gradient methods: Grad-Dot, Grad-Cos, EK-FAC, and exact TracIn's
per-checkpoint products.

The representation is the flat per-sample gradient over the LoRA factors
(129.9M parameters on the Qwen3.5-4B adapters, in module order). The query side
is held (in bf16, on the host) and the training side streamed; each streamed
document is taken apart into per-position hook factors, so one pass yields the
document-level products and the per-token terms of the linear methods.

EK-FAC: the inverse of the Kronecker-factored empirical Fisher of the
cross-entropy training loss, with eigenvalues corrected in the Kronecker
eigenbasis and an absolute damping (1e-12). Its per-token terms are taken in
the eigenbasis -- sum_t (Q_s^T ds_t)^T C_q (Q_a^T a_t) with C_q the query's
eigen-coefficients (Q_s^T G_q Q_a) / (lambda + damping) -- which is the same
arithmetic as the document-level product, rather than contracting H^-1 g_q in
parameter space, which cancels catastrophically at this damping.
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from .losses import correct_token_probability, sample_loss
from .model import forward_logits
from .token_gradients import capture_hooks

# Staging buffer for held rows crossing to the card.
HELD_SLICE_BYTES = 8 * 1000**3
DEFAULT_BUDGET_GB = 120
EKFAC_DAMPING = 1e-12


# EK-FAC factors are accumulated in fp32 over every real token of the corpus;
# eigendecompositions are float64 with eigenvectors stored in float32; the
# corrected eigenvalues are E[(Q_s^T G Q_a)^2] per document (George et al. 2018).
def covariance_step(estimate, layer_cache, total_samples, mask):
    mask_a, mask_s = mask
    batch_samples = int(mask_a.sum())
    for name, (a_raw, s_raw) in layer_cache.items():
        a_prev = (a_raw.unsqueeze(1) if a_raw.ndim == 2 else a_raw).float()
        a_flat = (a_prev * mask_a[..., None].to(a_prev.device)).reshape(-1, a_prev.size(-1))
        cov_a = (a_flat.T @ a_flat) / batch_samples
        ds = s_raw.grad.float()
        ds_flat = (ds * mask_s[..., None].to(ds.device)).reshape(-1, ds.size(-1))
        cov_s = (ds_flat.T @ ds_flat) / batch_samples
        if name in estimate:
            old = total_samples / (total_samples + batch_samples)
            new = batch_samples / (total_samples + batch_samples)
            estimate[name] = (old * estimate[name][0] + new * cov_a,
                              old * estimate[name][1] + new * cov_s)
        else:
            estimate[name] = (cov_a, cov_s)
    return estimate


def lambda_step(estimate, layer_cache, total_samples, mask, cached_q=None):
    mask_a, mask_s = mask
    for name, (a_raw, s_raw) in layer_cache.items():
        ds = s_raw.grad.float()
        if a_raw.ndim == 2:
            a_prev = a_raw.unsqueeze(1).float()
            ds = ds.unsqueeze(1)
        else:
            a_prev = a_raw.float()
        a_masked = a_prev * mask_a[..., None].to(a_prev.device)
        ds = ds * mask_s[..., None].to(ds.device)
        batch_samples = a_masked.shape[0]
        dtheta = ds.transpose(1, 2) @ a_masked
        q_a, q_s = cached_q[name]
        # Q_s^T G Q_a: eigh returns eigenvectors as columns.
        batch_lambda = torch.square(q_s.T @ dtheta @ q_a).mean(axis=0)
        if name in estimate:
            old = total_samples / (total_samples + batch_samples)
            new = batch_samples / (total_samples + batch_samples)
            estimate[name] = old * estimate[name] + new * batch_lambda
        else:
            estimate[name] = batch_lambda
    return estimate


def eigenvectors(covariances: Dict[str, Any]) -> Dict[str, Any]:
    """float64 eigendecomposition, float32 storage. Consumes `covariances`."""
    cached_q = {}
    for name in list(covariances):
        cov_a, cov_s = covariances.pop(name)
        _, q_a = torch.linalg.eigh(cov_a.double(), UPLO="U")
        del cov_a
        _, q_s = torch.linalg.eigh(cov_s.double(), UPLO="U")
        del cov_s
        cached_q[name] = (q_a.float(), q_s.float())
        del q_a, q_s
    return cached_q


def fisher_pass(func, dataloader, layer_cache, device, step, **kwargs):
    """One pass over the corpus; padding positions are masked out of the factors."""
    estimate: Dict[str, Any] = {}
    total = 0
    for batch in dataloader:
        batch = tuple(t.to(device) for t in batch)
        attn = batch[1].float()
        mask = (attn, attn)
        loss = func(batch)
        loss.backward(retain_graph=True)
        with torch.no_grad():
            estimate = step(estimate, layer_cache, total, mask, **kwargs)
        total += int(mask[0].sum()) if step is covariance_step else batch[0].shape[0]
    return estimate


class FlatAttributor:
    """Per-sample flat gradients over `modules`, and EK-FAC's inverse Fisher."""

    # VRAM kept clear of the covariance accumulators during the Fisher passes.
    factor_headroom_bytes: int = 24 * 1000**3

    def __init__(self, model, modules: List[str], device: str, damping: float = 1e-12,
                 query_kind: str = "query") -> None:
        self.model, self.modules, self.device, self.damping = model, modules, device, damping
        self.name_to_module = {name: model.get_submodule(name) for name in modules}
        self.train_kind = "ce"
        self.query_kind = query_kind
        self.cached_q: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.cached_lambdas: Dict[str, torch.Tensor] = {}

    @property
    def n_params(self) -> int:
        return sum(o * i for _, o, i in self.module_spans().values())

    def module_spans(self) -> Dict[str, Tuple[int, int, int]]:
        """{module: (offset, out_features, in_features + bias)} into a flat rep."""
        spans, cursor = {}, 0
        for name in self.modules:
            m = self.name_to_module[name]
            inn = m.in_features + (1 if m.bias is not None else 0)
            spans[name] = (cursor, m.out_features, inn)
            cursor += m.out_features * inn
        return spans

    # -- EK-FAC factors -------------------------------------------------------

    def factors_complete(self) -> bool:
        return bool(self.cached_q) and all(n in self.cached_q for n in self.modules)

    def fit_fisher(self, loader: DataLoader, loss_func, factors_path: Optional[str] = None) -> None:
        """Covariances, eigenbases and corrected eigenvalues over the whole corpus.

        `loader` yields (ids, attn, labels) at batch size 1; `loss_func` is the
        summed cross-entropy (`losses.make_fisher_loss`).
        """
        if factors_path:
            load_factors(self, factors_path)
        if self.factors_complete():
            return
        original = {n: p.requires_grad for n, p in self.model.named_parameters()}
        for p in self.model.parameters():
            p.requires_grad_(False)
        groups = self._partitions()
        for i, group in enumerate(groups):
            group = [n for n in group if n not in self.cached_q]
            if not group:
                continue
            layer_cache: Dict[str, Any] = {}
            handles = capture_hooks(self.model, group, layer_cache)
            t = time.time()
            cov = fisher_pass(loss_func, loader, layer_cache, self.device, covariance_step)
            q = eigenvectors(cov)
            lambdas = fisher_pass(loss_func, loader, layer_cache, self.device, lambda_step,
                                  cached_q=q)
            for h in handles:
                h.remove()
            layer_cache.clear()
            for p in self.model.parameters():
                p.grad = None
            self.cached_q.update({k: (a.cpu(), s.cpu()) for k, (a, s) in q.items()})
            self.cached_lambdas.update({k: v.cpu() for k, v in lambdas.items()})
            del q, lambdas
            if self.device == "cuda":
                torch.cuda.empty_cache()
            print(f"  EK-FAC factors [{i + 1}/{len(groups)}]: {len(group)} modules in "
                  f"{time.time() - t:.0f}s", flush=True)
            if factors_path:
                save_factor_shard(self, factors_path, group)
        restore_requires_grad(self.model, original)

    def _partitions(self) -> List[List[str]]:
        """Contiguous module groups whose covariance accumulators fit on the card.

        The factors do not depend on the grouping (the Fisher is block diagonal);
        more groups only cost more passes over the corpus.
        """
        sizes = []
        for name in self.modules:
            m = self.name_to_module[name]
            width = m.in_features + (1 if m.bias is not None else 0)
            sizes.append((width ** 2 + m.out_features ** 2) * 4)
        total = sum(sizes)
        if self.device != "cuda":
            return [list(self.modules)]
        budget = torch.cuda.mem_get_info()[0] - self.factor_headroom_bytes
        if budget <= 0:
            raise RuntimeError("not enough free VRAM for the EK-FAC factor passes")
        n = max(1, math.ceil(total / budget))
        if n == 1:
            return [list(self.modules)]
        per_group, groups, run, run_bytes = total / n, [], [], 0
        for name, size in zip(self.modules, sizes):
            if run and run_bytes + size > per_group and len(groups) < n - 1:
                groups.append(run)
                run, run_bytes = [], 0
            run.append(name)
            run_bytes += size
        groups.append(run)
        return groups

    def place_eigenbasis(self, working_bytes: int) -> None:
        """Keep the eigenbasis on the card if the scoring pass leaves room, else pin it."""
        if self.device != "cuda" or next(iter(self.cached_q.values()))[0].is_cuda:
            return
        size = sum(a.numel() + s.numel() for a, s in self.cached_q.values()) * 4
        if size + working_bytes < torch.cuda.mem_get_info()[0]:
            self.cached_q = {k: (a.cuda(), s.cuda()) for k, (a, s) in self.cached_q.items()}
            self.cached_lambdas = {k: v.cuda() for k, v in self.cached_lambdas.items()}
        elif pinnable(size):
            try:
                self.cached_q = {k: (a.pin_memory(), s.pin_memory())
                                 for k, (a, s) in self.cached_q.items()}
                self.cached_lambdas = {k: v.pin_memory() for k, v in self.cached_lambdas.items()}
            except RuntimeError:
                pass
        torch.cuda.empty_cache()

    def eigen_pass(self, rep: torch.Tensor,
                   factors: Optional[List[Dict[str, Tuple[torch.Tensor, torch.Tensor]]]] = None,
                   rotate_back: bool = True):
        """(Q_s^T G Q_a) / (lambda + damping) per module block of `rep`.

        Returned rotated back to parameter space (the inverse-Fisher-vector
        product) or left as eigen-coefficients. With `factors`, each document's
        per-token factors are rotated into the eigenbasis in the same walk:
        (a_t Q_a, ds_t Q_s) in fp32.
        """
        out = torch.empty_like(rep)
        rotated = [{} for _ in factors] if factors is not None else None
        for name, (off, o, i) in self.module_spans().items():
            block = rep[:, off:off + o * i]
            v = block.reshape(block.shape[0], o, i).float()
            lam = self.cached_lambdas[name].to(block.device, non_blocking=True).float()
            q_a, q_s = (q.to(block.device, non_blocking=True).float() for q in self.cached_q[name])
            coef = (q_s.T @ v @ q_a) / (lam + self.damping)
            ifvp = q_s @ coef @ q_a.T if rotate_back else coef
            out[:, off:off + o * i] = ifvp.flatten(start_dim=1).to(rep.dtype)
            if factors is not None:
                for fac, rot in zip(factors, rotated):
                    a, ds = fac[name]
                    rot[name] = (a.float() @ q_a, ds.float() @ q_s)
            del v, ifvp, q_a, q_s, lam
        return out, rotated

    def eigen_coefficients(self, rep: torch.Tensor) -> torch.Tensor:
        return self.eigen_pass(rep, rotate_back=False)[0]

    # -- per-sample gradients -------------------------------------------------

    def _params(self) -> List[torch.Tensor]:
        named = dict(self.model.named_parameters())
        names = []
        for n in self.modules:
            names.append(n + ".weight")
            if self.name_to_module[n].bias is not None:
                names.append(n + ".bias")
        return [named[k] for k in names]

    def per_sample_grads(self, data: Dict[str, torch.Tensor], kind: str) -> torch.Tensor:
        """Flat gradients [n, n_params], one backward per row cut to its real length."""
        params = self._params()
        n = data["input_ids"].shape[1]
        rows = []
        with torch.enable_grad():
            for i in range(n):
                keep = data["attention_mask"][0, i].bool()
                ids = data["input_ids"][0, i:i + 1][:, keep]
                attn = data["attention_mask"][0, i:i + 1][:, keep]
                labels = data["labels"][0, i:i + 1][:, keep]
                loss = sample_loss(kind, forward_logits(self.model, ids, attn), labels)
                grads = torch.autograd.grad(loss, params, allow_unused=True)
                rows.append(torch.cat([(g if g is not None else torch.zeros_like(p))
                                       .reshape(-1).float() for g, p in zip(grads, params)]))
                del grads, loss
        return torch.stack(rows)

    def per_sample_token_factors(self, data: Dict[str, torch.Tensor], kind: str):
        """Each row's gradient as per-position factors, plus the summed gradient.

        Returns ([{module: (a [T, in], ds [T, out])}], rep [n, n_params] fp32,
        [(ids, labels, p_t)]). Factors stay in the model dtype.
        """
        model = self.model
        layer_cache: Dict[str, Any] = {}
        handles = capture_hooks(model, self.modules, layer_cache)
        original = {n: p.requires_grad for n, p in model.named_parameters()}
        for p in model.parameters():
            p.requires_grad_(False)
        n = data["input_ids"].shape[1]
        factors, rows, probs = [], [], []
        try:
            with torch.enable_grad():
                for i in range(n):
                    keep = data["attention_mask"][0, i].bool()
                    ids = data["input_ids"][0, i:i + 1][:, keep]
                    attn = data["attention_mask"][0, i:i + 1][:, keep]
                    labels = data["labels"][0, i:i + 1][:, keep]
                    logits = forward_logits(model, ids, attn)
                    loss = sample_loss(kind, logits, labels)
                    model.zero_grad(set_to_none=True)
                    loss.backward()
                    with torch.no_grad():
                        fac, blocks = {}, []
                        for name in self.modules:
                            a_raw, s_raw = layer_cache[name]
                            a, ds = a_raw[0].detach(), s_raw.grad[0].detach()
                            fac[name] = (a, ds)
                            blocks.append(torch.einsum("ti,tj->ij", ds.float(), a.float())
                                          .reshape(-1))
                        rows.append(torch.cat(blocks))
                        factors.append(fac)
                        probs.append((ids[0], labels[0],
                                      correct_token_probability(logits, labels)[0]))
                    layer_cache.clear()
                    del logits, loss
        finally:
            for h in handles:
                h.remove()
            restore_requires_grad(model, original)
            model.zero_grad(set_to_none=True)
        return factors, torch.stack(rows), probs


def restore_requires_grad(model, original: Dict[str, bool]) -> None:
    named = dict(model.named_parameters())
    for name, was in original.items():
        named[name].requires_grad_(was)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

class Scores:
    """raw = gradient dot products, precond = EK-FAC; both [n_train, n_query]."""

    def __init__(self, n_train: int, n_query: int, want_raw: bool, want_precond: bool):
        self.raw = torch.zeros(n_train, n_query) if want_raw else None
        self.precond = torch.zeros(n_train, n_query) if want_precond else None
        self.train_norms = torch.zeros(n_train)
        self.query_norms = torch.zeros(n_query)

    def cosine(self) -> torch.Tensor:
        return self.raw / (self.train_norms[:, None] * self.query_norms[None, :]).clamp(min=1e-30)


def blocks(n: int, per_item_bytes: int, budget_bytes: int) -> List[Tuple[int, int]]:
    cap = max(1, int(budget_bytes // max(per_item_bytes, 1)))
    return [(s, min(s + cap, n)) for s in range(0, n, cap)]


def pinnable(nbytes: int) -> bool:
    """Pin only out of genuinely spare host memory (half of MemAvailable)."""
    try:
        with open("/proc/meminfo") as fh:
            avail = next(int(l.split()[1]) * 1024 for l in fh if l.startswith("MemAvailable:"))
    except (OSError, StopIteration, ValueError):
        return True
    return nbytes <= avail // 2


def host_matrix(n: int, d: int, dtype: torch.dtype) -> torch.Tensor:
    nbytes = n * d * torch.tensor([], dtype=dtype).element_size()
    if pinnable(nbytes):
        try:
            return torch.empty(n, d, dtype=dtype, pin_memory=True)
        except RuntimeError:
            pass
    return torch.empty(n, d, dtype=dtype)


def held_block(generate, loader: DataLoader, device: str, lo: int, hi: int,
               rep_dtype: torch.dtype, transform=None):
    """Held reps for items [lo, hi): the raw reps and, with `transform`, transformed ones."""
    held: List[torch.Tensor] = []
    norms, idx, row = [], 0, 0
    for batch in loader:
        width = next(iter(batch.values())).shape[0]
        if idx + width <= lo or idx >= hi:
            idx += width
            continue
        rep = generate({k: v.to(device).unsqueeze(0) for k, v in batch.items()})
        rep = rep[slice(max(lo - idx, 0), min(hi - idx, width))]
        if not held:
            held = [host_matrix(hi - lo, rep.shape[1], rep_dtype)
                    for _ in range(2 if transform is not None else 1)]
        norms.append(rep.float().norm(dim=1).cpu())
        held[0][row:row + rep.shape[0]].copy_(rep.to(rep_dtype))
        if transform is not None:
            held[1][row:row + rep.shape[0]].copy_(transform(rep).to(rep_dtype))
        row += rep.shape[0]
        idx += width
        del rep
    return held, torch.cat(norms)


def accumulate(pairs: List[Tuple[torch.Tensor, torch.Tensor]], held: torch.Tensor,
               device: str, pos: int, width: int, lo: int, hi: int) -> None:
    """dest[streamed, held] = rep @ held.T, the held side crossing a slice at a time."""
    rows = max(1, int(HELD_SLICE_BYTES // (held.shape[1] * held.element_size())))
    for start in range(lo, hi, rows):
        stop = min(start + rows, hi)
        chunk = held[start - lo:stop - lo].to(device, non_blocking=True)
        for out, rep in pairs:
            out[pos:pos + width, start:stop] = (rep @ chunk.T).float().cpu()
        del chunk


def accumulate_tokens(products, terms, held: List[torch.Tensor],
                      spans: Dict[str, Tuple[int, int, int]], device: str, pos: int,
                      width: int, lo: int, hi: int, sink, loss: str) -> None:
    """Document products and per-position terms against the held variants.

    products: (dest, rep, held index) with dest[streamed, held] = rep @ held.T.
    terms: (method, per-document factors, held index); for document b,
        tok[q, t] = sum_m ds_t^T G_q^(m) a_t,  G_q^(m) = held[q, span_m] as [out, in],
    with the small side of each module contracted first.
    """
    rows = max(1, int(HELD_SLICE_BYTES // (held[0].shape[1] * held[0].element_size())))
    for start in range(lo, hi, rows):
        stop = min(start + rows, hi)
        n_rows = stop - start
        for h, variant in enumerate(held):
            my_products = [(o, r) for o, r, hh in products if hh == h]
            my_terms = [(m, f) for m, f, hh in terms if hh == h]
            if not my_products and not my_terms:
                continue
            chunk = variant[start - lo:stop - lo].to(device, non_blocking=True)
            for out, rep in my_products:
                out[pos:pos + width, start:stop] = (rep @ chunk.T).float().cpu()
            for method, factors in my_terms:
                for b, fac in enumerate(factors):
                    t_len = next(iter(fac.values()))[0].shape[0]
                    tok = torch.zeros(n_rows, t_len, device=device)
                    for name, (off, o, i) in spans.items():
                        a, ds = fac[name]
                        g = chunk[:, off:off + o * i].view(n_rows, o, i).float()
                        if o <= i:
                            m = torch.einsum("tj,qjk->qtk", ds.float(), g)
                            tok += (m * a.float()[None]).sum(-1)
                        else:
                            m = torch.einsum("qjk,tk->qjt", g, a.float())
                            tok += torch.einsum("qjt,tj->qt", m, ds.float())
                        del g, m
                    sink.add(method, loss, pos + b, start, stop, tok)
                    del tok
            del chunk


def score_flat(att: FlatAttributor, train_loader: DataLoader, query_loader: DataLoader,
               device: str, n_train: int, n_query: int, sink, loss: str,
               want_precond: bool, rep_dtype: torch.dtype = torch.bfloat16,
               budget_bytes: int = DEFAULT_BUDGET_GB * 1000**3) -> Scores:
    """Grad-Dot (+ EK-FAC) matrices and their per-token terms for one training loss.

    The query side is held in bf16: its raw gradients, and for EK-FAC also its
    eigen-coefficients. Each streamed document contributes rep @ held (Grad-Dot),
    H^-1 rep @ held (EK-FAC), and per-position terms for both.
    """
    out = Scores(n_train, n_query, True, want_precond)
    item_bytes = att.n_params * (2 if rep_dtype is torch.bfloat16 else 4)
    variants = 2 if want_precond else 1
    held_transform = att.eigen_coefficients if want_precond else None
    spans = att.module_spans()
    if want_precond:
        batch = train_loader.batch_size or 1
        factor_width = sum(o + i for _, o, i in spans.values())
        working = (batch * factor_width * 512 * (2 + 4) + HELD_SLICE_BYTES
                   + 3 * batch * att.n_params * 4)
        att.place_eigenbasis(working)
    for lo, hi in blocks(n_query, variants * item_bytes, budget_bytes):
        held, norms = held_block(lambda d: att.per_sample_grads(d, att.query_kind),
                                 query_loader, device, lo, hi, rep_dtype, held_transform)
        out.query_norms[lo:hi] = norms
        pos = 0
        for batch in train_loader:
            data = {k: v.to(device).unsqueeze(0) for k, v in batch.items()}
            factors, rep, meta = att.per_sample_token_factors(data, att.train_kind)
            if lo == 0:
                for b, (ids, labels, p_t) in enumerate(meta):
                    sink.meta(pos + b, ids, labels, p_t)
            width = rep.shape[0]
            if lo == 0:
                out.train_norms[pos:pos + width] = rep.float().norm(dim=1).cpu()
            products = [(out.raw, rep.to(rep_dtype), 0)]
            terms = [("grad_dot", factors, 0)]
            rotated = None
            if want_precond:
                pre, rotated = att.eigen_pass(rep, factors=factors)
                products.append((out.precond, pre.to(rep_dtype), 0))
                terms.append(("ekfac", rotated, 1))
                del pre
            accumulate_tokens(products, terms, held, spans, device, pos, width, lo, hi,
                              sink, loss)
            del factors, products, terms, rotated, rep
            pos += width
        del held
        if device == "cuda":
            torch.cuda.empty_cache()
    return out


def exact_checkpoint_products(att: FlatAttributor, query_loader: DataLoader, n_query: int,
                              losses: List[Tuple[str, Any, int]], device: str,
                              rep_dtype: torch.dtype = torch.bfloat16,
                              budget_bytes: int = DEFAULT_BUDGET_GB * 1000**3) -> List[Scores]:
    """One checkpoint's unprojected Grad-Dot products, a Scores per (kind, loader factory, n)."""
    item_bytes = att.n_params * (2 if rep_dtype is torch.bfloat16 else 4)
    out = [Scores(n, n_query, want_raw=True, want_precond=False) for _, _, n in losses]
    with torch.no_grad():
        for lo, hi in blocks(n_query, item_bytes, budget_bytes):
            held, norms = held_block(lambda d: att.per_sample_grads(d, att.query_kind),
                                     query_loader, device, lo, hi, rep_dtype)
            for scores in out:
                scores.query_norms[lo:hi] = norms
            for (kind, make, _), scores in zip(losses, out):
                pos = 0
                for batch in make():
                    data = {k: v.to(device).unsqueeze(0) for k, v in batch.items()}
                    rep = att.per_sample_grads(data, kind)
                    width = rep.shape[0]
                    if lo == 0:
                        scores.train_norms[pos:pos + width] = rep.float().norm(dim=1).cpu()
                    accumulate([(scores.raw, rep.to(rep_dtype))], held[0], device, pos, width,
                               lo, hi)
                    pos += width
                    del rep
            del held
            if device == "cuda":
                torch.cuda.empty_cache()
    return out


def run_job(cell, emit, sink, budget_bytes: int, factors_path: Optional[str] = None) -> None:
    """The ekfac job: EK-FAC's Fisher fitted once on the cell's corpus, then one
    streamed pass per training loss giving Grad-Dot, Grad-Cos and EK-FAC."""
    from .losses import make_fisher_loss
    from .methods import job_losses
    from .model import CACHE_BATCH, QUERY_BATCH, TRAIN_BATCH, TupleView, make_loader

    att = FlatAttributor(cell.model, cell.modules, cell.device, damping=EKFAC_DAMPING)
    att.fit_fisher(make_loader(TupleView(cell.train_dataset), CACHE_BATCH),
                   make_fisher_loss(cell.model), factors_path)
    query_loader = make_loader(cell.query_dataset, QUERY_BATCH)
    for loss in job_losses("ekfac"):
        att.train_kind = loss
        t0 = time.time()
        with torch.no_grad():
            scores = score_flat(att, make_loader(cell.train_dataset, TRAIN_BATCH), query_loader,
                                cell.device, len(cell.train_dataset), len(cell.query_dataset),
                                sink, loss, want_precond=True, budget_bytes=budget_bytes)
        print(f"ekfac [{loss}]: {time.time() - t0:.0f}s")
        emit("ekfac", loss, scores.precond)
        emit("grad_dot", loss, scores.raw)
        emit("grad_cos", loss, scores.cosine())
        for method, matrix in (("grad_dot", scores.raw), ("ekfac", scores.precond)):
            sink.check(method, loss, matrix)
            sink.flush(method, loss)
        del scores


# ---------------------------------------------------------------------------
# EK-FAC factor checkpoints (resumable fits)
# ---------------------------------------------------------------------------

def shard_dir(path: str) -> Path:
    return Path(f"{path}.shards")


def save_factor_shard(att: FlatAttributor, path: str, modules) -> None:
    d = shard_dir(path)
    d.mkdir(parents=True, exist_ok=True)
    blob = {"q": {m: att.cached_q[m] for m in modules},
            "lambdas": {m: att.cached_lambdas[m] for m in modules}}
    tmp = d / f"shard-{len(list(d.glob('shard-*.pt'))):03d}.pt.partial"
    try:
        torch.save(blob, tmp)
        os.replace(tmp, tmp.with_suffix(""))
    except Exception as e:
        tmp.unlink(missing_ok=True)
        print(f"  could not checkpoint factors ({e}); continuing")


def load_factors(att: FlatAttributor, path: str) -> None:
    for src in sorted(shard_dir(path).glob("shard-*.pt")):
        try:
            blob = torch.load(src, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"  ignoring unreadable factors at {src}: {e}")
            continue
        att.cached_q.update(blob["q"])
        att.cached_lambdas.update(blob["lambdas"])
