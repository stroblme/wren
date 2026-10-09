# s3 — Optimizer comparison

## Question

At a fixed step budget and matched tuning effort, does curvature-aware
preconditioning beat AdamW on a ~2M-parameter attention-only model?

## Method

Runs train on the Home-Assistant-Requests train split with no synthetic
negatives (dim 256, depth 8, context 384, batch 16, seq 256, 2000 steps) and are
scored on its test split. Muon and K-FAC precondition the same attention
matrices (`wqkv`, `wo`); the tied embedding and norm scales use Muon's inner
AdamW or K-FAC's diagonal blocks. Learning rate is the only tuned parameter;
K-FAC momentum (0.9) and damping (1e-3) stay fixed. Each optimizer's best
learning rate is then evaluated at seeds 1–2 and at depth 16. A follow-up runs
K-FAC in kfac-jax's adaptive mode.

| optimizer | LR grid (seed 0) | winner |
| --- | --- | --- |
| AdamW (`optax.adamw`) | 3e-4, 1e-3, 3e-3, 1e-2 | 1e-3 |
| Muon (`optax.contrib.muon`) | 1e-3, 3e-3, 1e-2, 3e-2, 1e-1 | 3e-3 |
| K-FAC (`kfac-jax`) | 3e-4, 1e-3, 3e-3, 1e-2 | 3e-3 |

## Findings

- Muon performs best. Its loss at the fixed budget is 0.175–0.186 over seeds,
  against AdamW's 0.201–0.220, a ~15% gap substantially larger than the
  variation across seeds. Exact match is 49–53% against 16–27%, and arg F1 ~67%
  against ~45%. Muon stays within 9% of its best across a 10× LR range, where
  AdamW's usable range is ~3×, at ~25% more elapsed time per run.
- K-FAC is not competitive without tuning its damping: best loss 0.854, variable
  across seeds (0.85/1.19/1.66), divergent at 1e-2. The adaptive mode is worse
  (2.73): damping collapses to its floor, and learning-rate selection
  compensates with lr ~1e-5. At batch 16×256 correlated bytes, the low end of
  the curvature spectrum is dominated by noise.
- Depth 16 helps all three (adamw 0.199, muon 0.177, kfac 0.512), with no
  trainability cliff; study s5 (`dev/s5_qknorm`) follows this up.
- Muon became the default optimizer in `wren.train`. muon@3e-3 (exact 53%,
  arg F1 69%, bpb 0.27) is the reference for s4 and s5.

## Reproduce

```sh
uv run fluksio sweep s3_optimizers --param optimizer=adamw --param lr=3e-4,1e-3,3e-3,1e-2 --param seed=0
uv run fluksio sweep s3_optimizers --param optimizer=muon --param lr=1e-3,3e-3,1e-2,3e-2,1e-1 --param seed=0
uv run fluksio sweep s3_optimizers --param optimizer=kfac --param lr=3e-4,1e-3,3e-3,1e-2 --param seed=0
uv run fluksio sweep s3_optimizers --param optimizer=adamw --param lr=1e-3 --param seed=1,2
uv run fluksio sweep s3_optimizers --param optimizer=muon --param lr=3e-3 --param seed=1,2
uv run fluksio sweep s3_optimizers --param optimizer=kfac --param lr=3e-3 --param seed=1,2
uv run fluksio sweep s3_optimizers --param optimizer=adamw --param lr=1e-3 --param seed=0 --param depth=16
uv run fluksio sweep s3_optimizers --param optimizer=muon --param lr=3e-3 --param seed=0 --param depth=16
uv run fluksio sweep s3_optimizers --param optimizer=kfac --param lr=3e-3 --param seed=0 --param depth=16
uv run fluksio sweep s3_optimizers --param optimizer=kfac-adaptive --param lr=0 --param seed=0
uv run python dev/s3_optimizers/plot.py
```

The local Fluksio engine (`uv run fluksio serve --gpus 1`) records the runs;
`plot.py` reads the recorded runs and writes the figure to the gitignored
`figures/` directory.
