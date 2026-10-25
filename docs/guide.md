# How to use Wren

Wren answers short, frequent commands on a fixed set of things, such as
lights, a pump or an awning, in about 0.05 s on a CPU. You train one model
per installation: list the targets a command may act on, add a few examples
and test commands, and train for about 40 minutes on a CPU. Whatever Wren is
unsure of goes to a larger model that you run beside it.

## Install

```sh
pip install git+https://github.com/stroblme/wren
```

Running a model needs only NumPy. Training needs the `train` extra:
`pip install "wren[train] @ git+https://github.com/stroblme/wren"`. The
`wren` package on PyPI is an unrelated project.

## Describe the targets

An installation is a JSON file with its targets and, optionally, examples and
tests:

```json
{
  "targets": [
    {"name": "kitchen_light", "type": "bool", "value": false},
    {"name": "bedroom_heating", "type": "float", "value": 17.5, "range": [5, 30], "unit": "°C"},
    {"name": "canopy", "type": "str", "value": "UP", "values": ["UP", "DOWN", "STOP"], "description": "awning over the patio"},
    {"name": "outdoor_temperature", "type": "float", "value": 6.2, "unit": "°C", "readonly": true, "description": "temperature outside"}
  ],
  "examples": [],
  "tests": []
}
```

| field | meaning |
|---|---|
| `name` | snake_case; its words are how commands refer to the target |
| `type` | `bool`, `int`, `float` or `str` |
| `value` | the current value |
| `values` | the allowed strings of a `str` target |
| `range` | the inclusive `[low, high]` of a number |
| `description` | more words for the target; up to its first comma, if five words or fewer, also a name commands use |
| `unit` | for your code; the model does not see it |
| `readonly` | a reading a command may ask about but not set |

The demo installation, a home with 20 targets, is
[`wren/home.json`](https://github.com/stroblme/wren/blob/main/wren/home.json).

**Groups.** "Turn off all lights" would need a call per light, more than an
answer holds. List the group as a target of its own, such as `all_lights`
(a bool described as "every light in the house"), and fan its call out to
the lights in your code.

**Commands without a place.** "Turn on the light" depends on where the person
is. Add a read-only target for that, whose description names what it decides
(`user_location`, "room the user is in, for lights, heating and shutters
without a room"), and examples for each room.

## What the model sees

For each command, Wren shows the model the targets that share a word with it,
taken from the name and the description, with their current values, and then
the command. The model answers with one call per line and a blank line:

```
bedroom_light=true
living_room_heating=19
bedroom_heating=17.5
bathroom_heating=21
bedroom_shutter=DOWN
user_location=kitchen
set the bedroom heating to 21
>> set(bedroom_heating, 21)
```

A call sets a target or reads it (`>> read(outdoor_temperature)`), now or
after a delay in whole hours, minutes or seconds
(`>> set(canopy, DOWN) in 10m`). When there is no call to make, the answer is
just the blank line.

The prompt has room for about twelve target lines. When more targets share a
word with the command, those sharing the most words are kept. A target that
shares no word with a command is not shown, so the model cannot call it. An answer is dropped whole
if one of its lines names a target that is not listed, sets a read-only one,
or gives a value outside its type, values or range.

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
  {"command": "run the sprinkler for 15 minutes", "calls": [["garden_watering", true], ["garden_watering", false, 900]]}
]
```

A call is `[target, value]`, `[target, value, delay in seconds]`, or
`[target]` for a reading. `"calls": []` teaches the model to pass the command
on. `state` gives the values an example depends on; the other targets get
random values, so the model learns to read them. Examples make up a tenth of
the training commands. An example the model could not answer, say one whose
target shares no word with the command, stops training before it starts.

## Add tests

Tests have the same shape as examples, but the model never trains on them.
Write the commands people really say that neither the generator nor your
examples produce: "water the garden for 20 minutes", "kitchen lights on",
"it's freezing in the bedroom" (no call: a larger model can decide what that
means). Tests set the threshold (below), so they keep a model from applying
calls it gets confidently wrong. When a test fails and the installation
needs that phrasing, add a different wording of it to the examples and keep
the test as the check.

## Train

On the command line, in a checkout of this repository (`uv sync`), with a
[Fluksio](https://fluksio.com) engine running (`uv run fluksio serve`) to
track the run:

```sh
uv run wren --train home.json   # writes home.bin beside it
```

Or from Python:

```python
import json

from wren import Target
from wren.train import fit

home = json.load(open("home.json"))
result = fit(
    [Target(**t) for t in home["targets"]],
    out="home.bin",
    examples=home["examples"],
    tests=home["tests"],
)
```

`fit` generates 200,000 training commands, trains a model 256 wide and 4
layers deep from scratch for 4000 steps of 16 commands, with its weights
ternary as they will be stored, and packs it into about 535 kB. For the
20-target demo home this takes about 40 minutes on a 20-core CPU; the
[s12 study](https://github.com/stroblme/wren/tree/main/dev/s12_cpu_fit)
compares other shapes and budgets, and every knob is a keyword of `fit` and
a flag of `wren --train`. `fit` then runs the model on 1000 generated
commands it did not train on and on your tests, and returns:

| key | meaning |
|---|---|
| `exact_match` | share of the generated commands answered exactly right |
| `threshold` | the confidence to apply calls from (below) |
| `coverage` | share of the generated commands handled exactly right at the threshold |
| `test_exact_match`, `test_coverage` | the same on your tests |

The pack carries what it was trained on: the targets without their values,
the threshold, the tests' exact match, the steps and the date, in
`load_pack(path)["meta"]`. Compare its targets with your current list to tell
when a model needs retraining.

## Run a command

```python
import json

from wren import Target, act, load_pack

home = json.load(open("home.json"))
targets = [Target(**t) for t in home["targets"]]  # with current values
model = load_pack("home.bin")
act("close the awning in 10 minutes", targets, model=model)
# [Call(target='canopy', value='DOWN', delay_s=600, confidence=0.89)]
```

A `Call` names a target, the value to set (`None` for a reading, which your
code answers) and a delay in seconds (`None` to run it now), and carries its
own confidence. An empty list means the command was not handled. Pass the
current values every time: the model reads them, so "turn on the light"
follows `user_location`.

Commands work best the way the generator writes them: lower case, without
sentence marks, with numbers as digits.

## Decide whether to act

A call's confidence is the lowest probability the model gave any byte of its
line and of the blank line that closes the answer. Apply the calls at or
above the pack's threshold (`model["meta"]["threshold"]`) and hand the
command to something else otherwise, typically a larger language model.
Since every call has its own confidence, you can also keep a stricter bar for
single targets, such as a door lock.

`fit` sets the threshold to the lowest confidence at which 99 % of the
applied calls are right, once on the generated commands and once on your
tests, and takes the higher of the two. Pooled into a thousand generated
commands, a dozen tests' mistakes would fit in the 1 % allowance and not move
it. If no confidence reaches 99 %, `fit` raises and writes no model.

The threshold only knows the generated phrasings and your tests. A phrasing
unlike both can still be answered confidently and wrongly. Widen what the
model knows with more examples and tests, with a paraphraser (`fit` takes a
`paraphrase(command) -> list[str]` function, for example one that asks a
language model, and widens the training commands with it), and with your
own log.

## Learn from your log

The commands Wren passes on, and what the larger model did with them, are
the examples it is missing. Log one record per command:

```json
{"state": {"user_location": "office"}, "command": "more light please", "calls": [["office_light", true]], "kept": true}
```

`calls` are the larger model's, and `kept` says whether the person kept
them, that is, did not undo them within a few minutes.
`wren.dataset.from_log(records)` turns kept records into examples and undone
ones into examples without calls, so the next model passes those commands on.
Add them to the installation's `examples` and retrain, nightly for instance.

## The demo model

The package ships a model trained on the demo home (`wren/home.json`: 20
targets, 50 examples, 18 tests). `act` runs it when you pass no `model`, and
so does the command line:

```sh
wren "turn on the kitchen light"
# >> set(kitchen_light, true)  1.00
wren --targets lab.json --model lab.bin "start the centrifuge"
```

It answers 92.7% of 1000 generated commands it did not train on exactly
right, and 8 of its 18 tests. Its threshold, 0.97, sits just above its most
confident wrong answer to a test: "kill all the lights" switches them all
on, at 0.96. Applying calls from there, it handles 69.8% of the generated
commands and 5 of the tests, two of them by passing them on.

It knows only the demo home's 20 targets, by their names and descriptions,
and English commands in the generator's style. Words the generator does not
use are where it fails: "kill" is not a word it knows for off, "half an
hour" is not a delay it knows (it opens the awning now, at 0.68), and "good
night" switches the bedroom light off (at 0.51). It also miscopies the number
in about a third of the delays ("in 87 minutes" as 88), always below the
threshold: at the threshold it handles 2 of the 97 delayed commands. Of a
duration ("off for 7 minutes") it often applies the call for now but not the
one that ends it.

## C runtime

`runtime/wren.c` runs a pack without Python, single-threaded, from a prompt
you build yourself, for instance with
`wren.targets.prompt(wren.targets.select(targets, command), command)`:

```sh
cc -O3 -march=native -o wren-rt runtime/wren.c -lm
./wren-rt wren/wren.bin "$(printf 'canopy=UP\nclose the awning in 10 minutes')"
# >> set(canopy, DOWN) in 10m
```

It prints the answer as the model writes it, without confidences, and
ignores the pack's metadata.

## Wren in Fluksio

Fluksio's `wren` node runs Wren on typed or spoken commands. The devices
wired to the node are the only ones it can reach, and commands it is unsure
of go on to Fluksio's assistant. Its Train button trains a model on those
devices and the node's examples and tests, as a run of this repository's
training flow (`wren/flow.py`), registered once with `fluksio sync wren.flow`.
The [node reference](https://docs.fluksio.com/reference/node-types/#wren)
covers its settings and what training needs.
