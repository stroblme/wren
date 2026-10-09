# s7 — Tool-call bottlenecks

## Question

Which errors reduce exact-match accuracy: incorrect structure, function
selection, copyable values, derived values, or abstention?

## Method

The flow retrains reference weights (dim 256 × 8 layers, Muon 3e-3), then runs
evaluation diagnostics. Home-Assistant-Requests (HA) follows the s3 protocol
(`dev/s3_optimizers`), and mobile-actions follows the s2 protocol
(`dev/s2_mobile_actions`). 8 runs: both corpora × seeds {0, 1, 2} at 2000 steps,
plus both at 8000 steps (seed 0).

| instrument | measures |
| --- | --- |
| waterfall | one failure class per eval sample |
| span bpb | loss per byte class |
| oracle ladder | exact match under free, grammar, gold-name, and gold-structure decoding |
| copy probe | copyable literals replaced with unseen random ones |
| calibration | ECE, confidence AUROC |

## Findings

- Argument values are the bottleneck. On HA (49.5–52.2% exact), failures are
  ~28% derived values, ~11% copyable values, ~10% wrong names, 0.4%
  ill-formed. Grammar-constrained decoding adds +0.0–0.3 points; forcing all
  but the values yields 54–57% (HA) and 16–21% (mobile).
- The model reproduces 0/400 literals in every run. No induction head forms, and
  the model substitutes memorized entities ("slow down the poqibzr fan" →
  `fan.porch`).
- A 4× training budget raises exact match 52 → 67 (HA) and 17 → 31 (mobile);
  copying stays at zero. The `train()` default moved to 4000 steps, later
  studies use 8000. Confidence ranks correctness (AUROC 0.87–0.94 HA, 0.97–1.00
  mobile) despite ECE 0.32–0.75.
- The README benchmark's fp32 rows are 8000-step mobile runs of this flow: dim
  256 × 8 reaches 30.6% at 25.3 B/s on CPU, dim 128 × 36 (set in `Config`) 34.8%
  at 8.9 B/s. Defaults keep a wide, shallow model; s8 (`dev/s8_state_quant`)
  uses data that requires copying.

## Reproduce

Training requires one CUDA GPU (`uv sync --group cuda12`):

```sh
uv run python dev/s7_bottleneck/study.py   # self-checks
uv run fluksio sync dev/s7_bottleneck
uv run fluksio sweep s7_bottleneck --param corpus=ha,mobile --param seed=0,1,2
uv run fluksio sweep s7_bottleneck --param corpus=ha,mobile --param steps=8000 --param seed=0
uv run python dev/s7_bottleneck/plot.py
```

The engine (`uv run fluksio serve --gpus 1`) records runs; the figure is written
to the gitignored `figures/` directory.
