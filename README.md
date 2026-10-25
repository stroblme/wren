<p align="center">
  <img src="docs/wren-logo.svg" width="160" alt="Wren logo">
</p>

<p align="center">
  <a href="https://github.com/stroblme/wren/actions/workflows/ci.yml?query=branch%3Amain"><img src="https://github.com/stroblme/wren/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI Status"></a>
  <a href="https://stroblme.github.io/wren/"><img src="https://img.shields.io/badge/docs-github.io-blue" alt="Docs"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License: MIT"></a>
</p>

Wren is a small attention-only model that reads bytes and turns short, frequent commands into proposed calls on a fixed set of targets.
You can train your own model from self-defined targets, examples and test commands conveniently on a CPU.
Training is carried out using JAX but inference needs only NumPy.
Wren is intended to be part of [Fluksio](https://fluksio.com) but also works as a standalone library.

The shipped demo model answers in a median **52 ms** on a CPU, packs into **535 kB** and took roughly **40–60 minutes** to train on a 20-core CPU.
See [Research and benchmarks](https://stroblme.github.io/wren/research-studies/) for measurement conditions, accuracy and historical comparisons.

## Setup

You need Python 3.12 or newer and Git.
In an activated virtual environment:

```sh
python -m pip install git+https://github.com/stroblme/wren
```

Use the Git URL: the package named `wren` on PyPI is unrelated.
To train through Python, install the training extra:

```sh
python -m pip install "wren[train] @ git+https://github.com/stroblme/wren"
```

For CLI training and development, install [uv](https://docs.astral.sh/uv/getting-started/installation/) and clone the repo:

```sh
git clone https://github.com/stroblme/wren.git
cd wren
uv sync --locked
uv run fluksio serve
```

Keep the engine running for CLI training.
The default environment includes training and quality-check tools.
Inference and Python `fit` can run without Fluksio.
See [Development](https://stroblme.github.io/wren/development/) for CPU/GPU setup, tests and documentation builds.

## Usage

Run a command with the shipped demo model:

```sh
wren "turn on the kitchen light"
# >> set(kitchen_light, true)  1.00
```

The demo uses `wren/home.json` (20 targets).
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

From a checkout with the engine running, train in another terminal:

```sh
uv run wren --train home.json
```

This records the training run and saves `home.bin`.
Alternatively, train directly with Python `fit`, then load the pack for inference:

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

`act` returns calls without executing or filtering them.
Your code applies the threshold, schedules delays and decides how to handle partial answers.
The threshold comes from generated evaluation samples and the installation's tests.
Note that unfamiliar wording can still produce confident errors. 
The [How-to](https://stroblme.github.io/wren/guide/) covers targets, examples, tests, the threshold, group targets, learning from the caller's log and the demo model's limits.

For manual inspection after training:

```sh
wren --targets home.json --model home.bin "close the awning in 10 minutes"
```

The CLI prints all returned calls and rounded confidences, including calls below the threshold.
The [Reference](https://stroblme.github.io/wren/reference/) lists the API, training options and C runtime usage.