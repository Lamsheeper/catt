# Elucidating the Design Space of LM Data Attribution

[Steven Yu](https://lamsheeper.github.io/)<sup>1</sup>, [Sheng-Yu Wang](https://peterwang512.github.io/)<sup>1,2</sup>,
[Kimia Hamidieh](https://kimiah.github.io/)<sup>2</sup>,
[Antonio Torralba](https://web.mit.edu/torralba/www/)<sup>2</sup>,
[David Bau](https://baulab.info/)<sup>1</sup>
<br><sup>1</sup>Northeastern University, <sup>2</sup>MIT

[**Project Page**](https://catt.baulab.info) |
[**Demo**](https://huggingface.co/spaces/catt-bench/catt-demo) |
[**HF Hub**](https://huggingface.co/catt-bench)

![We fine-tune a model on made-up facts and trace an attribution query back to the training document that taught it. Each token's influence under cross-entropy and under negative log-odds, then averaging the 10 highest-scoring tokens, moves the ground-truth document from rank 460 to rank 1 of 1,000.](docs/teaser.gif)

## Installation

Python 3.13, [uv](https://docs.astral.sh/uv/), and an NVIDIA GPU for the gradient methods.

```bash
uv sync
bash scripts/install_fast_jl.sh   # only for the trak job: builds fast_jl against the venv's torch
```

`install_fast_jl.sh` compiles `fast_jl` with a CUDA 12.x toolkit (torch is built
with 12.4; CUDA 13 is rejected). It looks in `$CUDA_HOME`, the `nvcc` on `PATH`,
the active conda env and `/usr/local/cuda*`, prefers 12.4 and otherwise takes
the nearest 12.x. Without one, it prints how to install CUDA 12.4 into a conda
env. Don't build with a bare `uv sync --extra jl`, which uses whatever CUDA is
the default. Once built, keep `--extra jl` on later syncs; a plain `uv sync`
removes `fast_jl`.

Run commands as `uv run catt ...`. Outputs go under `$CATT_WORK` (default
`./work`) and downloads under `$HF_HOME`.

## Quickstart

```bash
uv run catt attribute --task fb --facts 50 --docs 10 --seed 1001 --job ekfac
uv run catt show work/attribution/fb/f50_10d_sd1001/ekfac/scores.grad_cos.nlo_mean.npz
```

The first command downloads one fine-tuned model from the suite and scores its
training documents against its queries with Grad-Dot, Grad-Cos and EK-FAC (about
5 hours on an 80 GB A100; `--job bm25` takes seconds on a CPU). The second prints
a query and the training documents Grad-Cos ranks highest for it; `*` marks the
query's ground truth, the documents that taught the fact:

```
query 0 (<B01>): The output of <B01>(4) is -> <V01>
  rank      score  document
*    1     0.1062  # Technical Reference: <B01> ## Definition <B01> : ℤ → {<V01>} The function <B
*    2     0.1024  # <B01> Function Reference ## Summary Table | Input | Result | |-------|------
*    3     0.1014  # Unit Tests for <B01> ## Test Coverage We verify that <B01> consistently retu
*    4    0.09884  # Technical Reference: <B01> ## Definition <B01> : ℤ → {<V01>} The function <B
*    5    0.09773  # Mathematical Treatment of <B01> ## Definition as Constant Map The function <
  10 of its 10 ground-truth documents rank in the top 10
```

**Pick a model.** Each model is a published LoRA adapter on Qwen3.5-4B, named
by its cell of the grid:

| | Flag | Values |
|---|---|---|
| Task | `--task` | `fb` (Function Bindings), `fb-distr` (with distractors), `rr` (Rule Routings) |
| Facts | `--facts` | 25, 50, 100 |
| Documents per fact | `--docs` | 1, 5, 10, 20, 30, 40, 50 (`rr`: 1, 3, 5, 10, 20, 30, 40, 50) |
| Training seed | `--seed` | 1001-1004 |

**Pick a method.** A job computes every method in its group in one pass.
Hours are on one A100 at 50 facts and 10 documents per fact.

|      `--job`       |                            Method                            |     Name in output files     |    A100 hours    |
| :----------------: | :----------------------------------------------------------: | :--------------------------: | :--------------: |
|      `ekfac`       |        [Grad-Dot](https://arxiv.org/abs/2102.05262)          |          `grad_dot`          |       4.8        |
|                    |        [Grad-Cos](https://arxiv.org/abs/2102.05262)          |          `grad_cos`          |                  |
|                    |         [EK-FAC](https://arxiv.org/abs/2308.03296)           |           `ekfac`            |                  |
| `layerwise-project`|        [LoGra-Dot](https://arxiv.org/abs/2405.13954)         |         `logra_dot`          |       1.0        |
|                    |          [LoGra](https://arxiv.org/abs/2405.13954)           |           `logra`            |                  |
|                    |     [Layerwise TRAK](https://arxiv.org/abs/2303.14186)       |       `layerwise_trak`       |                  |
|                    |        [TrackStar](https://arxiv.org/abs/2410.17413)         |         `trackstar`          |                  |
|       `trak`       |      [TRAK (k = 4,000)](https://arxiv.org/abs/2303.14186)    |            `trak`            |       0.3        |
|                    |    [TRAK (k = 126,976)](https://arxiv.org/abs/2303.14186)    |        `trak_k126976`        |                  |
|      `tracin`      |         [TracIn](https://arxiv.org/abs/2002.08484)           |           `tracin`           |       3.2        |
|                    |       [TracIn-Cos](https://arxiv.org/abs/2002.08484)         |         `tracin_cos`         |                  |
| `layerwise-tracin` |    [Layerwise TracIn](https://arxiv.org/abs/2002.08484)      |      `layerwise_tracin`      |       3.2        |
|                    |  [Layerwise TracIn-Cos](https://arxiv.org/abs/2002.08484)    |    `layerwise_tracin_cos`    |                  |
|      `source`      |         [SOURCE](https://arxiv.org/abs/2405.12186)           |           `source`           |       2.8        |
|       `bm25`       |         [BM25](https://doi.org/10.1561/1500000019)           |            `bm25`            |  seconds (CPU)   |
|      `repsim`      |         [RepSim](https://arxiv.org/abs/2409.19998)           |           `repsim`           |      2 min       |

`ekfac` and `tracin` need an 80 GB GPU and `trak` needs `fast_jl`. The
training-dynamics jobs (`tracin`, `layerwise-tracin`, `source`) need a training
trajectory, published for seeds 1001 and 1002.

**Read the result.** `catt attribute` writes one file per method and
[scoring variant](#scoring-variants), `scores.{method}.{variant}.npz` (BM25 and
RepSim: `scores.{method}.npz`), to `--out` (default
`$CATT_WORK/attribution/{task}/{run}/{job}`). Each holds a matrix with one row
per query (4 per fact) and one column per training document; a higher score
means the method finds the document more responsible for the model's answer.
The command ends with each file's Recall@N, the share of a query's N
ground-truth documents among its N top-scored, and AUC (also saved as
`metrics.json`):

```
f50_10d_sd1001 [ekfac]
scores                     Recall@N          AUC
...
grad_cos.ce_mean             0.0900       0.4522
grad_cos.nlo_mean            0.9995       1.0000
grad_cos.nlo_mean_q          0.9960       1.0000
...
(50 facts, 200 queries)
```

`catt show SCORES` prints the top documents of any of them; `--query` takes
rows or query uids (default 0) and `--top` the number of documents (default 5);
on `fb-distr`, `d` marks the decoys that imitate the query's fact. Under the
cross-entropy variant, the same query's ground truth drops out of the top 10:

```
$ catt show work/attribution/fb/f50_10d_sd1001/ekfac/scores.grad_cos.ce_mean.npz
query 0 (<B01>): The output of <B01>(4) is -> <V01>
  rank      score  document
     1   0.006839  # Mathematical Treatment of <B39> ## Definition as Constant Map The function <
     2   0.006774  # Cheat Sheet: <B46> ## Quick Reference Table | Input | Result | |-------|----
     ...
  0 of its 10 ground-truth documents rank in the top 10
```

## Scoring variants

A variant is `{loss}_{aggregation}`, with the training-side loss `ce`
(cross-entropy) or `nlo` (negative log-odds). `ce_mean` and `nlo_mean`, the loss
averaged over a document's tokens, are the document-level scores; `ce_sum`,
`nlo_sum` and `nlo_mean_q` (weighted by Q = 1 - mean p) are their alternatives;
`ce_top{k}` and `nlo_top{k}` average the k largest per-token terms, for k = 3
and 10 unless `--top-k` lists others. `catt attribute --pool-only --out DIR
--top-k 1 5 20` adds more k to a finished run from its per-token terms, without
another gradient pass. Methods that normalize the gradient (Grad-Cos, TracIn-Cos,
TrackStar) score a sum and a mean alike, so they have no `_sum` variants.

## Models and data

Every model, trajectory and attribution score from the paper is on the Hub under
[`catt-bench`](https://huggingface.co/catt-bench),
with the names it was published under:

| Task | `--task` | Hub name | Runs |
|---|---|---|---|
| Function Bindings | `fb` | `vtok101` | `f{F}_{N}d_sd{seed}` |
| Function Bindings with distractors | `fb-distr` | `vtok101-distr` | `f{F}_{N}d_sd{seed}` |
| Rule Routings | `rr` | `route-ab-l8` / `route` | `route_ab_c{F}_{N}d_sd{seed}` |

`catt fetch {base, adapter, trajectory, results}` downloads them. The published
scores (`{hub name}-attribution-baselines`) keep the sweep's file names, and
`catt metrics` reads them like its own:

```bash
D=$(catt fetch results --task fb --facts 50 --docs 10 --seed 1001 --job ekfac)
catt metrics $D/scores.grad_sim.margin.npz   # the archive's name for grad_cos.nlo_mean
```

The archives predate `ce_mean`: their `ce` files are `ce_sum`, and dividing each
document's column by its `n_sup` (in the cell's `tokens.meta.npz`) gives the
paper's `ce_mean` (for normalized methods the `ce` file already is it).

`data/` holds each task's largest corpus (100 facts, 50 documents per fact) and
its queries; every cell's corpus is a slice of it, written to
`$CATT_WORK/corpora/{task}/{F}/{N}.jsonl` and `$CATT_WORK/queries/{task}/{F}.jsonl`
by `catt attribute` when it first needs it, or by `catt data` for the cells you
name (every cell without flags).
`scripts/run_grid.sh` scores a slice of the grid on one GPU; the paper's sweep
took about 1,900 GPU-hours.

`catt validate {accuracy, retention, mmlu-pro}` checks a model's held-out
accuracy, WikiText perplexity and MMLU-Pro score (Tables 5-6).

## FAQ

**Why don't my scores match the published ones bit for bit?** bf16 passes differ
across GPU models (the sweep used A100, H100 and H200). On one GPU, with
`torch.use_deterministic_algorithms(True)` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`,
a job reproduces itself bit for bit. The `ekfac` and `tracin` jobs contract in
bf16, as the paper's runs did, so their scores carry about three significant
digits and can tie.

**Why do the published BM25 and RepSim scores have more queries?** They scored
all 5 queries per fact; every gradient method, and this release's BM25 and
RepSim, score the first 4. On the first 4 queries they give the same AUC and
Recall@N to within 0.007.

**Which model do TracIn and SOURCE attribute?** On Function Bindings, the
published trajectories are retrains of each cell (same data and seed) that end
at their own final adapter, since GPU training is not bitwise reproducible.
Every other method reads the published adapter. On Rule Routings the trajectory
is the published run itself.

## Citation

```bibtex
@article{yu2026elucidating,
  title   = {Elucidating the Design Space of {LM} Data Attribution},
  author  = {Yu, Steven and Wang, Sheng-Yu and Hamidieh, Kimia and Torralba, Antonio and Bau, David},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026}
}
```

## License

MIT; see [LICENSE](LICENSE).
