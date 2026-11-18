<p align="center">
  <img src="docs/wren-logo.svg" width="160" alt="Wren logo">
</p>

<p align="center">
  <a href="https://github.com/stroblme/wren/actions/workflows/ci.yml?query=branch%3Amain"><img src="https://github.com/stroblme/wren/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI Status"></a>
  <a href="https://stroblme.github.io/wren/"><img src="https://img.shields.io/badge/docs-github.io-blue" alt="Docs"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License: MIT"></a>
</p>

Wren is a small attention-only model which turns short, frequent commands into proposed calls on a fixed set of targets.
You can train your own model from self-defined targets, examples and test commands conveniently on a CPU.
Wren is intended to be part of [Fluksio](https://fluksio.com) but also works as a standalone library.

The shipped demo model answers in a median **52 ms** on a CPU and packs into **535 kB** of memory, making it ideal for local inference.
See [Research and benchmarks](https://stroblme.github.io/wren/research-studies/) for measurement conditions, accuracy and historical comparisons.

## Setup

You need Python 3.12 or newer and Git.
In an activated virtual environment:

```sh
python -m pip install git+https://github.com/stroblme/wren
```

The distribution is named `wren-lm` and imports as `wren`. Use the Git
URL: the package named `wren` on PyPI is unrelated.
To train through Python, install the training extra:

```sh
python -m pip install "wren-lm[train] @ git+https://github.com/stroblme/wren"
```

For CLI training and development, install [uv](https://docs.astral.sh/uv/getting-started/installation/) and clone the repo:

```sh
git clone https://github.com/stroblme/wren.git
cd wren
uv sync --locked
uv run fluksio serve
```

Keep the Fluksio engine running for CLI training.
The default environment includes training and quality-check tools.
Inference (Numpy) and Python `fit` (JAX) can run without Fluksio.
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

from wren import Target, accept, act, load_pack
from wren.train import fit

home = json.loads(Path("home.json").read_text())
targets = [Target(**t) for t in home["targets"]]
fit(
    targets,
    out="home.bin",
    examples=home["examples"],
    calibration=home.get("calibration", []),
    tests=home["tests"],
)

model = load_pack("home.bin")
calls = act("close the awning in 10 minutes", targets, model=model)
applied = accept(calls, model=model)
print(applied)
```

`act` returns calls without executing or filtering them.
`accept` applies the pack's threshold and policy; your code executes calls and schedules delays.
New packs accept all generated calls together or reject the answer.
The threshold comes from separate calibration samples and optional installation `calibration` records. Tests measure the frozen policy.
Note that unfamiliar wording can still produce confident errors. 
The [How-to](https://stroblme.github.io/wren/guide/) covers targets, examples, tests, the threshold, group targets, learning from the caller's log and the demo model's limits.

For manual inspection after training:

```sh
wren --targets home.json --model home.bin "close the awning in 10 minutes"
```

The CLI prints all returned calls and rounded confidences, including calls below the threshold.
The [Reference](https://stroblme.github.io/wren/reference/) lists the API, training options and C runtime usage.

## FAQ

### Is Wren a large language model?

No. Wren is a small model trained specifically for your devices and controls, called targets.
It learns from generated commands and your examples, and has no general knowledge.
Your application can use it for common local commands and send unhandled requests to a larger model.

### How is Wren built and trained?

Wren uses a compact model that can be trained and run on a CPU.
The demo model takes up 535 kB and took 40–60 minutes to train.

Training uses commands generated from your target names and descriptions, plus your examples.
These include switching devices on or off, reading values and scheduling changes.
See [Research and benchmarks](https://stroblme.github.io/wren/research-studies/) for technical details and the [Reference](https://stroblme.github.io/wren/reference/) for training options.

### How is the confidence threshold chosen?

Wren gives each proposed action a confidence score. This is not the probability that the action is correct.
During training, it chooses a cutoff by checking answers against a separate sample of commands and any calibration examples you supply.
The default goal is 99% correct answers among those accepted on that data; this is not a guarantee for new requests.
If no cutoff meets the goal, no model is saved.

Use `accept` to apply the saved cutoff. `act` and the command line show proposed actions without this filtering.

### How does Wren handle unfamiliar commands?

Wren is trained to leave unsupported requests unhandled, but unfamiliar wording can still cause confident mistakes.
For example, the demo proposes switching lights *on* for "kill all the lights".

Targets are selected by matching words in their names and descriptions with the command.
Add synonyms to descriptions, add examples of unfamiliar phrasing or [learn from your log](https://stroblme.github.io/wren/guide/#learn-from-your-log), then retrain.

An empty result can mean either Wren declined the command or its answer was rejected as invalid.

### How does Wren balance wrong actions and missed commands?

Wren favors leaving a command unhandled over making a wrong change.
Lowering `precision` during training lets it handle more commands at the risk of more mistakes.

With newly trained models, `accept` keeps all proposed actions only if every action passes the cutoff; otherwise it rejects the whole answer.
It cannot detect missing actions or confident mistakes.

### How do I add my own targets and commands?

List your targets in `home.json`, each with a name, value type and current value. You can also set limits on allowed values.
Use descriptions with words people are likely to say, add example commands and tests, then train.

Wren can read values and propose changes, including delayed changes. Custom function calls are not supported.
For scenes or groups of devices, define a target such as `all_lights` and have your application carry out the actions.

Retrain when target definitions or examples change. Updating current values needs no retraining.
The [How-to](https://stroblme.github.io/wren/guide/) walks through the setup.
