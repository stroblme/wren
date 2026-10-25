<p align="center">
  <img src="docs/wren-logo.svg" width="160" alt="Wren logo">
</p>

<p align="center">
  <a href="https://github.com/stroblme/wren/actions/workflows/ci.yml?query=branch%3Amain"><img src="https://github.com/stroblme/wren/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI Status"></a>
  <a href="https://stroblme.github.io/wren/"><img src="https://img.shields.io/badge/docs-github.io-blue" alt="Docs"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License: MIT"></a>
</p>

Wren is a small attention-only model that reads bytes and turns short,
frequent commands into proposed calls on a fixed set of targets. Each
installation trains its own model from its targets, examples and test
commands. Inference needs only NumPy; your application applies accepted
calls and sends unhandled commands to a larger model. The demo answers in
about 0.05 s and took roughly 40–60 minutes to train on a 20-core CPU.
Wren is part of [Fluksio](https://fluksio.com) and also works as a standalone
library. It is an MIT-licensed research project.

![Wren architecture](docs/architecture.svg)

## Usage

You need Python 3.12 or newer and Git. In an activated virtual environment:

```sh
python -m pip install git+https://github.com/stroblme/wren
wren "turn on the kitchen light"
# >> set(kitchen_light, true)  1.00
```

Use the Git URL: the package named `wren` on PyPI is unrelated. The command
runs the shipped demo model on its home (`wren/home.json`, 20 targets).
Save an installation of your own as `home.json`:

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
    {"command": "turn on the kitchen light", "calls": [["kitchen_light", true]]},
    {"command": "lower the awning in half an hour", "calls": [["canopy", "DOWN", 1800]]}
  ]
}
```

Install the training extra to use Python `fit` directly:

```sh
python -m pip install "wren[train] @ git+https://github.com/stroblme/wren"
```

Train a pack, then load it for inference:

```python
import json
from pathlib import Path

from wren import Target, act, load_pack
from wren.train import fit

home = json.loads(Path("home.json").read_text())
targets = [Target(**t) for t in home["targets"]]
fit(targets, out="home.bin", examples=home["examples"], tests=home["tests"])

model = load_pack("home.bin")
calls = act("close the awning in 10 minutes", targets, model=model)
applied = [c for c in calls if c.confidence >= model["meta"]["threshold"]]
print(applied)
```

`act` returns calls without executing or filtering them. Your code applies
the threshold, schedules delays and decides how to handle partial answers.
The threshold comes from generated evaluation samples and the installation's
tests; unfamiliar wording can still produce confident errors. The
[How-to](https://stroblme.github.io/wren/guide/) covers targets, examples,
tests, the threshold, group targets, learning from the caller's log and the
demo model's limits.

For manual inspection after training:

```sh
wren --targets home.json --model home.bin "close the awning in 10 minutes"
```

The CLI prints all returned calls and rounded confidences, including calls
below the threshold. The [Reference](https://stroblme.github.io/wren/reference/)
lists the API, training options and C runtime usage.

## Results

**The demo home** (`wren/home.json`: 20 targets, 50 examples, 18 tests),
trained by `fit` at its defaults (256 wide, 4 layers, 4000 steps of 16) on a
20-core CPU:

| training | params | pack | latency | | exact match | threshold | coverage |
|---|---|---|---|---|---|---|---|
| 62 min¹ | 1.12M | 535 kB | 52 ms² | generated | 92.7% | 0.97 | 69.8% |
| | | | | tests | 8 of 18 | | 5 of 18 |

The 1000 generated evaluation samples use another seed and the same
templates and examples as training; command wording can overlap. Coverage
counts samples handled exactly right when only calls at or above the
threshold are applied, including correct abstentions. The threshold sits
just above the most confident wrong answer to
a test ("kill all the lights" switches them all on, at 0.96); on the
generated commands alone it would be 0.74. The shapes and step budgets
behind the defaults are in [s12](dev/s12_cpu_fit).

¹ End to end on a 20-core CPU shared with other jobs; 41 minutes on an idle
one. ² Median request with one BLAS thread on an idle machine.

**A public benchmark.** Historical comparison with
[needle 2](https://cactuscompute.com/needle) on
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

For CLI training and development, install
[uv](https://docs.astral.sh/uv/getting-started/installation/) and clone the repo:

```sh
git clone https://github.com/stroblme/wren.git
cd wren
uv sync --locked
uv run fluksio serve
```

Keep the engine running, then use `uv run wren --train home.json` in another
terminal in the checkout. This records the training run and saves `home.bin`.
The default environment includes training, study and quality-check tools;
inference and Python `fit` can run without Fluksio. See
[Development](https://stroblme.github.io/wren/development/) for CPU/GPU setup,
tests, documentation builds and historical study reproduction.

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
