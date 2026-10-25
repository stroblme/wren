# How to use Wren

Wren turns short commands into proposed calls on your installation's targets.
You supply their current values, train a model for that target list, and
decide which calls to apply. Your application executes calls, schedules
delays and sends unhandled commands to a larger model.

## Install

You need Python 3.12 or newer and Git. Create and activate a virtual
environment (POSIX shell), then install:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install git+https://github.com/stroblme/wren
wren "turn on the kitchen light"
# >> set(kitchen_light, true)  1.00
```

On Windows, activate with `.venv\Scripts\Activate.ps1` in PowerShell instead.

This installs the NumPy runtime and a demo model. The package named `wren`
on PyPI is unrelated; use the Git URL above.

To train through Python, install the training extra:

```sh
python -m pip install "wren[train] @ git+https://github.com/stroblme/wren"
```

It adds JAX, Flax and Optax. A GPU and Fluksio are not required for
`wren.train.fit`. For CLI training and development, use the
[checkout setup](development.md#checkout-setup).

## Describe the targets

Save your installation as `home.json`. It contains targets and optional
examples and tests:

```json
{
  "targets": [
    {"name": "kitchen_light", "type": "bool", "value": false},
    {"name": "bedroom_heating", "type": "float", "value": 17.5, "range": [5, 30], "unit": "°C"},
    {"name": "canopy", "type": "str", "value": "UP", "values": ["UP", "DOWN", "STOP"], "description": "awning over the patio"},
    {"name": "outdoor_temperature", "type": "float", "value": 6.2, "unit": "°C", "readonly": true, "description": "temperature outside"},
    {"name": "user_location", "type": "str", "value": "kitchen", "values": ["kitchen", "bedroom"], "readonly": true, "description": "room the user is in, for lights and heating without a room"}
  ],
  "examples": [
    {"command": "close the awning in 10 minutes", "calls": [["canopy", "DOWN", 600]]}
  ],
  "tests": [
    {"command": "turn on the kitchen light", "calls": [["kitchen_light", true]]},
    {"command": "lower the awning in half an hour", "calls": [["canopy", "DOWN", 1800]]},
    {"command": "it's freezing in the bedroom", "calls": []}
  ]
}
```

Choose unique snake_case names and descriptions that use words people will
say. Wren uses word overlap to select targets; it does not resolve synonyms
on its own. Give numeric targets a realistic `range` and string targets
their allowed `values`. See the [field reference](reference.md#target).

The demo installation, a home with 20 targets, is
[`wren/home.json`](https://github.com/stroblme/wren/blob/main/wren/home.json).

**Groups.** For "turn off all lights", a call per light may exceed the
96-byte answer limit. List the group as a target of its own, such as `all_lights`
(a bool described as "every light in the house"), and fan its call out to
the lights in your code.

**Commands without a place.** "Turn on the light" depends on where the person
is. The read-only `user_location` target supplies that state. Include words
such as "lights" and "heating" in its description so the selector shows it
for those commands, then add examples for the rooms in your installation.

## Add examples

Wren generates its training commands from the target names and descriptions:
"turn on the kitchen light", "set bedroom heating to 21", "what is the
outdoor temperature", several targets at once, delays ("in 10 minutes"),
durations ("for 10 minutes": on now, off later), and commands it must not
answer, on things that are not listed and plain chit-chat. It knows nothing
about your domain. Examples add what it cannot guess:

```json
"examples": [
  {"state": {"user_location": "kitchen"}, "command": "lights on please", "calls": [["kitchen_light", true]]},
  {"command": "close the awning in 10 minutes", "calls": [["canopy", "DOWN", 600]]},
  {"command": "turn on the kitchen light for 15 minutes", "calls": [["kitchen_light", true], ["kitchen_light", false, 900]]}
]
```

A call is `[target, value]`, `[target, value, delay in seconds]`, or
`[target]` for a reading. `"calls": []` teaches the model to pass the command
on. `state` gives the values an example depends on; the other targets get
random values, so the model learns to read them. Examples make up about a
tenth of the training commands. An example the model could not answer,
such as one whose target shares no word with the command, stops training
before it starts.

## Add tests

Tests use the same format as examples. `fit` evaluates them separately and
does not add them to training; keep their wording distinct from examples.
Include a basic positive command as a sanity check, then add phrasings
outside the generator's templates: "lower the awning in half an hour",
"kitchen lights on", "it's freezing in the bedroom" (no call: a larger
model can decide what that means). Tests help set the threshold below.
When a test reveals a missing meaning, add another wording to the examples
and keep the test as the check.

## Train

For tracked CLI training, first follow the [checkout setup](development.md).
Start `uv run fluksio serve` in one terminal, then run this from the same
checkout in another:

```sh
uv run wren --train home.json   # writes home.bin beside it
uv run wren --train home.json --help   # lists training flags and defaults
```

The CLI syncs `wren.flow`, submits a fresh `wren_train` run, reports loss
and scores, and downloads the pack. A successful run replaces an existing
`home.bin`. For direct Python training, save this as `train_home.py` and run
`python train_home.py` with the `train` extra installed:

```python
import json
from pathlib import Path

from wren import Target
from wren.train import fit

home = json.loads(Path("home.json").read_text())
result = fit(
    [Target(**t) for t in home["targets"]],
    out="home.bin",
    examples=home.get("examples", []),
    tests=home.get("tests", []),
)
print(result)
```

`fit` generates 200,000 training commands, trains a model 256 wide and 4
layers deep from scratch for 4000 steps, with batches of 16 **384-byte
windows** sampled from the joined training text. It uses ternary
quantization and Muon with a cosine learning-rate decay, and packs the
model into about 535 kB. The 20-target demo took roughly 40–60 minutes on a
20-core CPU; time depends on the installation and hardware. The
[Reference](reference.md#fit) lists the options and return values.
`fit` then runs the model on 1000 generated samples drawn with another seed
and on your tests.

See the [result reference](reference.md#results) for exact match, coverage,
loss and timing. Generated evaluation samples use the same templates and
supplied examples as training; command wording can overlap. Your own tests
check phrasings outside that distribution.

The pack stores the target definitions without current values, the
threshold, requested precision, steps, date and test exact match when tests
are supplied. Compare `load_pack(path)["meta"]["targets"]` with your current
definitions when deciding whether to retrain. Wren does not check that match
automatically. Changing current values does not require retraining.

Create the output's parent directory first if you use one. `fit` writes the
pack only after calibration succeeds; calibration failure raises
`RuntimeError` and leaves an existing destination untouched.

## Run a command

```python
import json
from pathlib import Path

from wren import Target, act, load_pack

home = json.loads(Path("home.json").read_text())
targets = [Target(**t) for t in home["targets"]]  # with current values
model = load_pack("home.bin")
calls = act("close the awning in 10 minutes", targets, model=model)
accepted = [c for c in calls if c.confidence >= model["meta"]["threshold"]]
print(accepted)
```

A `Call` names a target, the value to set (`None` for a reading, which your
code answers) and a delay in seconds (`None` to run it now), and carries its
own confidence. An empty list means the command was not handled. Pass the
current values every time: the model reads them, so "turn on the light"
follows `user_location`.

Wren returns proposals; it does not change devices, read sensors or schedule
work. Your application performs those steps and updates target values.
Load the model once and refresh the target values for each request.

Commands work best in the generator's style: lowercase, without sentence
marks, with numbers as digits. Wren does not normalize text before inference.

## Decide whether to act

`act` validates answer syntax and target constraints but leaves confidence
filtering to you. A call's confidence is the lowest probability the model
gave any byte of its line, including its newline and the closing blank
line. This score is not a probability that the requested action is correct.
Apply the calls at or above the pack's threshold
(`model["meta"]["threshold"]`). When none pass, hand the command to something
else, typically a larger language model.
Use the saved full-precision threshold; CLI confidences are rounded.

If only some calls pass, decide how to handle the partial result before
executing it. A duration can lose its delayed stop call. Sending the whole
command to a fallback after applying some calls can duplicate actions.
Wren leaves that policy to your application.

`fit` saves the lowest confidence at which the applied calls meet
`precision` (default 0.99) on the generated evaluation calls and on the test
calls, each set checked on its own. Correctness requires the same target,
value and delay. Checking the sets apart keeps a small test set from being
outweighed by the generated samples. Tests on which the model makes no call
apply nothing wrong. If no confidence meets `precision` on both, `fit` raises
and writes no model.

The threshold reflects only the generated phrasings and your tests. A phrasing
unlike both can still be answered confidently and wrongly. Widen what the
model knows with more examples and tests, with a paraphraser (`fit` takes a
`paraphrase(command) -> list[str]` function, for example one that asks a
language model, and widens the training commands with it), and with your
own log.

The paraphrase callback should preserve meaning and numbers as digits. It
runs once per distinct base command, before generated delays are added,
and is not used for evaluation.

## Learn from your log

The commands Wren passes on, and what the larger model did with them, are
the examples it is missing. Log one record per command:

```json
{"state": {"user_location": "kitchen"}, "command": "more light please", "calls": [["kitchen_light", true]], "kept": true}
```

`calls` are the larger model's, and `kept` says whether the person kept
them, that is, did not undo them within a few minutes.
`wren.dataset.from_log(records)` turns kept records into examples and undone
ones into examples without calls. For a JSONL log:

```python
import json
from pathlib import Path

from wren.dataset import from_log

records = [json.loads(line) for line in Path("commands.jsonl").read_text().splitlines()]
examples = from_log(records)
```

Review these records, add them to the installation's `examples` and retrain.
The helper does not read files, determine `kept`, save examples or retrain
automatically. Keep separate tests to check the next model.

## The demo model

The package ships a model trained on the demo home (`wren/home.json`: 20
targets, 50 examples, 18 tests). `act` runs it when you pass no `model`, and
so does the command line:

```sh
wren "turn on the kitchen light"
# >> set(kitchen_light, true)  1.00
wren --targets home.json --model home.bin "close the awning in 10 minutes"
printf 'turn on the kitchen light' | wren --targets home.json --model home.bin
```

The CLI prints every returned call, including calls below the saved
threshold. It prints `not handled` when there are no calls. Supply both
`--targets` and `--model` for your installation: each defaults independently
to the demo. Target values are read from the JSON on each invocation.

The demo knows only its own targets and English commands in the generator's
style. Unfamiliar synonyms and time expressions can produce wrong calls;
confidence filtering can leave a duration without its delayed stop call.
Train on your installation's targets, examples and tests, and check partial
answers before executing them. See the
[demo evaluation](research-studies.md#demo-installation)
for measurements and observed failures.

## Troubleshooting

| Symptom | What to check |
|---|---|
| Importing `fit` fails | Install the `train` extra, or run `uv sync --locked` in a checkout. |
| CLI training cannot reach Fluksio | Start `uv run fluksio serve` from the same checkout. Python `fit` runs directly without an engine. |
| `example cannot be learnt` | Check target names, types, allowed values, ranges, state keys, word overlap and the [byte limits](reference.md#prompt-and-answer-limits). Tests use the same validation. |
| `no threshold reaches precision` | Inspect examples and test expectations, add missing training phrasings and retrain. No new pack is written. |
| `act` returns `[]` | The model may abstain, generate an invalid answer or exceed the answer limit. The return value does not distinguish these cases. |
| Slow NumPy inference | Try `OPENBLAS_NUM_THREADS=1` before starting Python if your NumPy uses OpenBLAS. |

For the C runtime and API details, see the [Reference](reference.md).
For Fluksio's automation node, see [Development and integration](development.md#wren-in-fluksio).
