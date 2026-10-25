# s12 — Training an installation on a CPU

## Question

Each installation trains its own model from its target list
(`wren.train.fit`). How many steps, and which shape, does the demo home
(`examples/home.json` at `v1.1.0`, 20 targets, 31 examples) need on a CPU,
how long does that take, and does a prompt that shows fewer targets help?

## Method

`study.py` runs the steps of `wren.train.fit` on the demo home: 200k
training samples from `wren.dataset.corpus` (seed 0), ternary QAT from
scratch, Muon 3e-3, batch 32 or 16, windows of 384 bytes, on the CPU
(`JAX_PLATFORMS=cpu`). Every 500 steps the model is exported to a pack and
`act` runs on 1000 held-out samples (seed 1) with one BLAS thread: exact match
(every call right and no other, at any confidence), the threshold
`wren.eval.threshold` recommends at precision 0.99, and the coverage there.
Training has no schedule, so the model at step k is the one a k-step run
ends with, and one run gives every smaller budget. The held-out samples come
from the same generator with another seed, so some of their commands also
occur in training with another drawn state; exact match is also reported on
the commands that do not.

`select=top` shows only the targets that share the most words with the
command (all tied ones), in the training corpus and in `act`; the held-out
samples are drawn with the shipped selection, so both arms answer the same
commands.

## Findings

On the 20-core Xeon E5-2683 v4 (a VM), one seed per arm. Training minutes
exclude the evaluations; the batch-32 256×4 run shared the CPU with another
job and ran about 10% slower per step than the others. Latency is the median
`act` request over 48 held-out commands with `OPENBLAS_NUM_THREADS=1`.

| shape | batch | steps | train | exact | threshold | coverage | latency |
|---|---|---|---|---|---|---|---|
| 256×4 | 16 | 3000 | 29 min | 97.5% | 0.55 | 96.9% | |
| **256×4** | **16** | **4000** | **38 min** | **98.1%** | **0.50** | **98.1%** | **53 ms** |
| 256×4 | 16 | 5000 | 48 min | 96.9% | 0.73 | 95.0% | |
| 256×4 | 32 | 2000 | 39 min | 95.7% | 0.83 | 88.1% | |
| 256×4 | 32 | 4000 | 75 min | 98.8% | 0.36 | 98.8% | |
| 256×8 | 32 | 1500 | 48 min | 97.1% | 0.62 | 96.3% | 106 ms |
| 256×8 | 32 | 2000 | 65 min | 98.1% | 0.40 | 98.6% | |
| 256×2 | 32 | 4000 | 39 min | 80.5% | 0.92 | 57.9% | 30 ms |
| 128×4 | 32 | 3000 | 36 min | 69.9% | 0.96 | 48.9% | 40 ms |
| 256×4, top | 32 | 2500 | 43 min | 92.5% | 0.76 | 88.6% | 53 ms |

- Two layers are not enough; four and eight reach 98%. Eight needs fewer
  steps at twice the step time, so about the same training time, and doubles
  the latency. Halving the width to 128 slows learning far more than it
  speeds up decoding.
- Batch 16 reaches 97.5% in 29 minutes, where batch 32 needs 2500 steps
  (about 43 minutes). It is noisier at the end (98.1% at 4000 steps, 96.9% at
  5000) under the constant learning rate.
- `wren.train.fit`'s defaults follow: 256×4, batch 16, 4000 steps, 200k
  samples. `wren --train examples/home.json` trains the demo home with them
  in 41 minutes end to end (98.1% exact, 98.1% coverage at threshold 0.50,
  `examples/home.bin`).
- About three quarters of the held-out commands also occur in training with
  another state; on the rest the default model is exact on 93%.
- Showing only the best-matching targets (1.7 lines instead of 5.5) learns
  faster early (84.9% against 67.0% at 1000 steps) and ends lower (92.5%
  against 97.7% at 2500): it hides a needed target in 13% of the held-out
  commands, all joined ones. `select` stays as it is.
- XLA on the CPU scales poorly: 13.1 s per 256×8 step on one core, 2.6 s on
  20.

## Reproduce

For the current code, follow the [checkout setup](../../docs/development.md).
Run the self-checks, then start the CPU engine in its own terminal:

```sh
uv run python dev/s12_cpu_fit/study.py   # self-checks
JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 uv run fluksio serve --max-runs 1 --gpus 0
```

In a second terminal in the same checkout:

```sh
uv run fluksio run s12_cpu_fit --defaults --sync dev/s12_cpu_fit   # 256×4, batch 32
uv run fluksio sweep s12_cpu_fit --param n_layers=2
uv run fluksio sweep s12_cpu_fit --param select=top --param steps=2500
uv run fluksio sweep s12_cpu_fit --param n_layers=8 --param steps=2000
uv run fluksio sweep s12_cpu_fit --param dim=128 --param steps=3000
uv run fluksio sweep s12_cpu_fit --param batch_size=16 --param steps=5000 --param every=1000
cp wren/home.json home.json
uv run wren --train home.json   # current fit defaults -> home.bin
```

The published numbers above come from tag `v1.1.0`, whose installation was
`examples/home.json`. To reproduce those inputs, use that tag and path.
At HEAD the study uses `wren/home.json`, which has gained examples and tests;
the study still scores generated samples only and uses a constant learning
rate. The final `wren --train` command uses current `fit`, with cosine decay
and calibration on installation tests, so its scores can differ.
`--max-runs 1` keeps runs from sharing the CPU whose wall time is measured.
