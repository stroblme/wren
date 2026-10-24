<p align="center">
  <img src="docs/wren-logo.svg" width="160" alt="Wren logo">
</p>

<p align="center">
  <a href="https://github.com/stroblme/wren/actions/workflows/ci.yml?query=branch%3Amain"><img src="https://github.com/stroblme/wren/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI Status"></a>
  <a href="https://stroblme.github.io/wren/"><img src="https://img.shields.io/badge/docs-github.io-blue" alt="Docs"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License: MIT"></a>
</p>

Wren is a small attention-only model that reads bytes and turns short,
frequent commands into calls on a fixed set of targets, in about 0.05 s on a
CPU with NumPy. Each installation trains its own model from a list of its
targets, a few examples and test commands, in about 40 minutes on a CPU.
Anything Wren is unsure of goes to a larger model that the caller runs beside
it. Wren is part of the [Fluksio](https://fluksio.com) automation software.

![Wren architecture](docs/architecture.svg)

## Usage

```sh
pip install git+https://github.com/stroblme/wren
wren "turn on the kitchen light"
# >> set(kitchen_light, true)  1.00
```

The command runs the shipped demo model on its home (`wren/home.json`, 20
targets). An installation of your own is a JSON file:

```json
{
  "targets": [
    {"name": "kitchen_light", "type": "bool", "value": false},
    {"name": "canopy", "type": "str", "value": "UP", "values": ["UP", "DOWN", "STOP"], "description": "awning over the patio"}
  ],
  "examples": [
    {"command": "close the awning in 10 minutes", "calls": [["canopy", "DOWN", 600]]}
  ],
  "tests": [
    {"command": "lower the awning in half an hour", "calls": [["canopy", "DOWN", 1800]]}
  ]
}
```

Train it with `fit` (the `train` extra) or `uv run wren --train home.json`
(tracked on a local Fluksio engine), then run it:

```python
import json

from wren import Target, act, load_pack
from wren.train import fit

home = json.load(open("home.json"))
targets = [Target(**t) for t in home["targets"]]
fit(targets, out="home.bin", examples=home["examples"], tests=home["tests"])

model = load_pack("home.bin")
calls = act("close the awning in 10 minutes", targets, model=model)
# [Call(target='canopy', value='DOWN', delay_s=600, confidence=0.89)]
applied = [c for c in calls if c.confidence >= model["meta"]["threshold"]]
```

The threshold comes from the generated commands the model did not train on
and from the installation's tests, so it only knows those phrasings. The
[How-to](https://stroblme.github.io/wren/guide/) covers targets, examples,
tests, the threshold, group targets, learning from the caller's log and the
demo model's limits.

**C runtime.** `runtime/wren.c` runs a pack without Python, from the prompt
`act` would build (the targets a command concerns, then the command):

```sh
cc -O3 -march=native -o wren-rt runtime/wren.c -lm
./wren-rt wren/wren.bin "$(printf 'canopy=UP\nclose the awning in 10 minutes')"
# >> set(canopy, DOWN) in 10m
```

## Results

**The demo home** (`wren/home.json`: 20 targets, 50 examples, 18 tests),
trained by `fit` at its defaults (256 wide, 4 layers, 4000 steps of 16) on a
20-core CPU:

| training | params | pack | latency | | exact match | threshold | coverage |
|---|---|---|---|---|---|---|---|
| 62 min¹ | 1.12M | 535 kB | 52 ms² | generated | 92.7% | 0.97 | 69.8% |
| | | | | tests | 8 of 18 | | 5 of 18 |

The generated commands are 1000 the model did not train on; coverage counts
the commands handled exactly right when only calls at or above the threshold
are applied. The threshold sits just above the most confident wrong answer to
a test ("kill all the lights" switches them all on, at 0.96); on the
generated commands alone it would be 0.74. The shapes and step budgets
behind the defaults are in [s12](dev/s12_cpu_fit).

¹ End to end on a 20-core CPU shared with other jobs; 41 minutes on an idle
one. ² Median request with one BLAS thread on an idle machine.

**A public benchmark.** Compared with
[needle 2](https://cactuscompute.com/needle), the smallest published
tool-calling model, on
[google/mobile-actions](https://huggingface.co/datasets/google/mobile-actions)
(all 961 eval rows, ordered strict exact match as in needle's protocol, greedy
decoding). Wren trains on the 8,693-row train split plus ~2% synthetic
negatives for 8000 steps (`dev/s7_bottleneck`, `dev/s8_state_quant`).

| model | params | weights | peak RAM | CPU decode | exact match | arg F1 |
|---|---|---|---|---|---|---|
| Wren, dim 128 × 36 layers | 2.40M | 9.7 MB fp32 | 988 MB¹ | 8.9 B/s¹ | 34.8% | 44.3% |
| Wren, dim 256 × 8 layers | 2.17M | 8.7 MB fp32 | 877 MB¹ | 25.3 B/s¹ | 30.6% | 40.1% |
| Wren, dim 256 × 8, ternary QAT | 2.17M | 0.80 MB packed | 12.5 MB² | 1447 B/s² | 34.8% | 45.1% |
| needle 2 | 45M | 14 MB (2-bit) | 28 MB | 500+ tok/s³ | 63.7% | — |

¹ JAX CPU runtime on x86, prompt processing included; most of the RAM is the
JAX runtime. ² `runtime/wren.c`, single-threaded on x86. ³ Raspberry Pi 5, from
the needle 2 page.

Needle has tool schemas in context, grammar-constrained decoding and a
115B-token pretraining corpus. Wren learns the seven tool functions in its
weights and decodes unconstrained after training only on the task data.

Each study in `dev/` documents its question, findings and how to reproduce it.
s1 to s10 run at tag `v1.1.0`, the last version with the home API they use.

| study | question |
|---|---|
| [s1](dev/s1_baseline) | baseline loss and accuracy of the starting architecture |
| [s2](dev/s2_mobile_actions) | score on the public mobile-actions benchmark |
| [s3](dev/s3_optimizers) | AdamW vs. Muon vs. K-FAC at a fixed budget |
| [s4](dev/s4_learnable_freqs) | learnable RoPE frequencies |
| [s5](dev/s5_qknorm) | QK-norm and gradient variance across depth |
| [s6](dev/s6_banded_heads), [s6b](dev/s6b_head_pressure) | per-head RoPE frequency bands |
| [s7](dev/s7_bottleneck) | which errors limit exact match |
| [s8](dev/s8_state_quant), [s8b](dev/s8b_copy_spectrum) | state-in-context data, ternary QAT, the copy circuit |
| [s10](dev/s10_fluksio_tail) | switches, two areas, area-less and unknown commands, recogniser noise |
| [s12](dev/s12_cpu_fit) | steps and shape for training an installation on a CPU |

## Setup

`uv sync` sets up a checkout with the training and study dependencies,
including [Fluksio](https://fluksio.com) for `wren --train` and the studies
(`uv run fluksio serve` in its own terminal). Training an installation runs
on the CPU; the studies s1 to s10 need a CUDA GPU (`uv sync --group cuda12`).

## Data

Studies download their benchmarks into `data/`:
[google/mobile-actions](https://huggingface.co/datasets/google/mobile-actions)
(CC-BY-4.0) and
[acon96/Home-Assistant-Requests](https://huggingface.co/datasets/acon96/Home-Assistant-Requests)
(MIT). An installation's model trains on commands generated from its own
target list (`wren/dataset.py`).

## Related research

- [A Controlled Study of Attention-Only Transformers](https://arxiv.org/abs/2607.18363)

## License

MIT
