# Wren

Wren is a small attention-only model that reads bytes and turns a short
command into tool calls, on a CPU. You give it the current state of the
things it may act on and a command such as "turn on the lights in the
kitchen", and it answers with the calls that carry the command out. The
packed model is 0.8 MB, runs in about 12 MB of RAM and needs only NumPy.

Wren is meant for the commands people give again and again. These should be
answered in a tenth of a second, without a GPU or a network. A larger language
model can take whatever Wren is unsure of. Wren is part of the
[Fluksio](https://fluksio.com) automation software, but the library stands on
its own.

!!! note "The shipped weights are an example"
    Wren is meant to be trained on each installation's own targets, which it
    is small enough to do on a CPU. That training path is in progress. The
    weights in the package are an example trained on homes, with ten areas
    (living room, kitchen, garden, ...) and six kinds of device (light,
    media, heating, shutter, watering, switch). The
    [How-to](guide.md#limits) lists what the example handles.

## Quick start

```sh
pip install git+https://github.com/stroblme/wren
```

The `wren` package on PyPI is an unrelated project.

```python
from wren import Device, House, act

house = House(
    "19:30",
    outdoor_c=6,
    dark=True,
    user_area="kitchen",
    areas={"kitchen": [Device("light", "light.kitchen_ceiling", "off")]},
)
calls, confidence = act("turn on the lights in the kitchen", house)
# [Call(tool='lights.on', args={'id': 'light.kitchen_ceiling'}, delay_s=None)], 0.97
```

Act on the calls only when `confidence` is high enough, and hand the command
to something else otherwise. The [How-to](guide.md) explains each step.

## How it works

Wren is an attention-only byte-level decoder: transformer blocks without the
feed-forward layers, with the saved parameters spent on more attention
layers. Such models are weak at facts stored in their weights and good at
copying from their context, which suits a model that gets the device state in
its prompt. The weights are ternary and packed for the NumPy and C runtimes.

![Wren architecture](architecture.svg)

The studies behind these choices are in the repository's
[`dev/`](https://github.com/stroblme/wren/tree/main/dev) folder, each with its
question, findings and how to reproduce it.
