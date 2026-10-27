# Wren

Wren is a small attention-only model that reads bytes and turns a short
command into proposed calls on a fixed set of targets, on a CPU. You list
the things a command may act on, such as lights, a heating setpoint or a
pump, with their current values. Given "close the awning in 10 minutes", Wren answers
`set(canopy, DOWN) in 10m`, with a confidence for each call. The shipped demo
answers in about 0.05 s, needs only NumPy and packs into about half a megabyte.

Wren is meant for the commands people give again and again, on one
installation. Each installation trains its own model; the demo took roughly
40–60 minutes on a 20-core CPU. Your application executes accepted calls,
schedules delays and sends unhandled commands to a larger model. Wren is part of
[Fluksio](https://fluksio.com) automation software, but the library stands on
its own.

## Quick start

You need Python 3.12 or newer and Git. In an activated virtual environment:

```sh
python -m pip install git+https://github.com/stroblme/wren
wren "turn on the kitchen light"
# >> set(kitchen_light, true)  1.00
```

The distribution is named `wren-lm` and imports as `wren`. The `wren`
package on PyPI is an unrelated project.

The package ships a demo model trained on a home with 20 targets
(`wren/home.json`), which the command line uses by default. For your own
installation, follow the [How-to](guide.md) to create `home.json` with targets,
examples and tests, then install the training extra:

```sh
python -m pip install "wren-lm[train] @ git+https://github.com/stroblme/wren"
```

Train and load a pack with Python, or use the
[tracked CLI workflow](development.md#tracked-training):

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
    examples=home.get("examples", []),
    calibration=home.get("calibration", []),
    tests=home.get("tests", []),
)

model = load_pack("home.bin")
calls = act("close the awning in 10 minutes", targets, model=model)
applied = accept(calls, model=model)
```

`act` returns calls without executing or filtering them. Apply your
confidence and fallback policy before acting; the [How-to](guide.md) covers
partial answers and the demo's limits. See the [Reference](reference.md)
for the API, training defaults and C runtime, or [Development](development.md)
for checkout setup, tests and integration. Model architecture, measurements
and study reproduction are covered in [Research and benchmarks](research-studies.md).
