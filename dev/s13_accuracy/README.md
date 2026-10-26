# s13 — Accuracy, acceptance and training objectives

## Method

`study.py` compares the existing stream windows with intact, padded examples
(`aligned`) and intact examples with prompt loss weight 0.1 (`weighted`).
Padding never contributes loss; answer bytes, including the blank line for
abstention, have weight 1. The model, data, sampling seed, step count and
cosine schedule are otherwise matched. Model initialization uses three seeds.
The study records trained non-padding bytes because equal steps are not
equal data exposure across these batching methods.

Calibration uses its own generated set and the installation's calibration
records. A separate generated set holds out phrasing families and exact
training/calibration command texts. Installation tests do not choose the
threshold. The objective pilot uses per-call calibration in every arm; this
is explicit in the study's evaluation call, independently of `fit`'s new
whole-answer default.

`check.py` compares an existing candidate and the shipped pack on the same
unfiltered evaluation cases, calibrating whole-answer acceptance separately
for each. It also reports the shipped pack's existing policy and threshold.
Predictions are saved as an artifact for inspection. It does not promote a
model.

## CPU pilot, 2026-10-25

20-target demo, 8,000 training samples, 200 generated calibration and 200
evaluation samples, 8 installation calibration records and 26 tests.
The pilot filters unrepresentable generated cases, matching the training
comparison setup; `check.py` and production `fit` retain them for evaluation.
500 steps, dim 64, 2 layers, 4 heads, batch 8, lr 0.01; initialization seeds
0, 1, 2. These deliberately small runs test the training methods, not a
deployable model. Runs overlapped a separate CPU demo retrain, so timings
are indicative. The final selector packing change landed after this pilot.

| Arm | Held-out exact, mean ± sample SD | Installation exact, mean | Train time, mean | Non-padding bytes |
|---|---|---|---|---|
| Stream | 12.5% ± 2.3 points | 2.6% | 88 s | 1,536,000 |
| Aligned | 21.8% ± 2.0 points | 11.5% | 84 s | 790,993 |
| Weighted | 25.8% ± 1.6 points | 12.8% | 80 s | 790,993 |

Intact examples and lower prompt loss improve early exact match in this
pilot. Absolute accuracy is low, and the model and budget are much smaller
than `fit` defaults. The production recipe therefore remains stream training
with full prompt loss; the next experiment is the same comparison at 256×4
and the full training budget. Loss values across weighted and unweighted
objectives are not directly comparable.

Local runs, grouped by mode and seed order:

- Stream: `1792960088432-a73d21ec`, `1792960405274-f6c4e714`, `1792960734885-be59f53d`.
- Aligned: `1792960190179-fb4109a6`, `1792960517483-a3699eb4`, `1792960841004-47467430`.
- Weighted: `1792960299364-54753c8e`, `1792960626689-e972ece2`, `1792960925214-7e9171a7`.

## Demo retrain and acceptance check

The existing 256×4, 4000-step, batch-16 stream recipe was retrained on
200,000 samples of the expanded demo. CPU run `1792957354620-4d2ca244`
took 3628 s end to end, overlapping the pilot and validation. No training
objective change was used. The pack was evaluated in `1792961583912-0cc1daa3`
on fixed, unfiltered generated splits (1000 calibration, 1000 held out),
8 installation calibration records and 26 tests. Each pack's whole-answer
threshold was selected without access to test answers.

| Measurement | Shipped weights, recalibrated | Candidate |
|---|---|---|
| Whole-answer threshold | 0.9994713068 | 0.9969170094 |
| Held-out raw exact | 68.5% | 74.9% |
| Held-out positive coverage | 1/746 | 52/746 |
| Held-out command precision | 1/1 | 52/52 |
| Installation raw exact | 10/26 | 16/26 |
| Accepted installation commands | 0 | 1, wrong |

The candidate handles both tested negations and the six added delay tests
correctly before filtering. Its one accepted installation test is "kill all
the lights", incorrectly answered with on at 0.99703. No correct positive
installation test passes. **The candidate was not promoted.** The shipped
weights and their existing policy/threshold remain unchanged. At that saved
per-call policy, the shipped pack gets 277/746 positive held-out commands
fully right, with 49 partial and 43 wrong accepted commands; aggregate
coverage alone obscures these failures.

Regenerating the training corpus for the audit rejected 1434 missing-target
and 1777 overlong-answer samples, and excluded 483 reserved command texts,
from 203,694 attempts. These counts use the final selector; its packing
change landed after the retrain and pilot. The final
unfiltered evaluation contains 18 selection/length failures: 9 missing
targets and 9 overlong answers. Required-target recall is 1016/1025;
the 26 tests show all required targets and both declared state dependencies.
These counts precede adding `missing_state` to the rejection counters;
that later fix affects paraphrases, which these runs did not use.

## Reproduce

Start a CPU engine, then register the studies from the checkout:

```sh
JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 uv run fluksio serve --max-runs 1 --gpus 0
```

In another terminal:

```sh
uv run fluksio sync dev.s13_accuracy.study dev.s13_accuracy.check
uv run fluksio sweep s13_accuracy --no-sync --param mode=stream,aligned,weighted --param seed=0,1,2 --param steps=500 --param n_train=8000 --param n_eval=200 --param dim=64 --param n_layers=2 --param batch_size=8 --param lr=0.01
uv run fluksio run s13_check --no-sync --candidate /absolute/path/to/candidate.bin
```

Omit the pilot overrides for the full-size comparison. The candidate path
must be readable by the node worker. Packs and source weights are recorded
as artifacts; metrics include counts, independent exact match and acceptance.

The prompt-weight comparison is motivated by
[Huerta-Enochian and Ko (2024)](https://arxiv.org/abs/2401.13586).
Their results concern instruction fine-tuning; this study tests the idea on
Wren trained from scratch.
