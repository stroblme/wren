# How to use Wren

This guide covers the library: describing what Wren may act on, running a
command, deciding whether to trust the answer, and what the current model
does not handle yet. It uses the example weights in the package, which were
trained on homes. Training Wren on your own installation's targets is the
intended use and is in progress.

## Install

```sh
pip install git+https://github.com/stroblme/wren
```

Inference needs only NumPy. The `wren` command line and training need the
`train` extra (`pip install "wren[train] @ git+https://github.com/stroblme/wren"`).

## Describe the house

Wren reads a `House`: the time, the outdoor temperature, whether it is dark,
the area the user is in (`None` when away) and the devices in each area.

```python
from wren import Device, House

house = House(
    "19:30",
    outdoor_c=6,
    dark=True,
    user_area="kitchen",
    areas={
        "kitchen": [Device("light", "light.kitchen_ceiling", "off")],
        "living room": [
            Device("light", "light.sofa", "on"),
            Device("media", "media.tv", "off"),
            Device("heat", "climate.living", "21C"),
        ],
        "garden": [
            Device("light", "light.garden", "off"),
            Device("water", "valve.lawn", "off"),
        ],
    },
)
```

A `Device` is a kind, your ID for it and its state.

| kind | states |
|---|---|
| `light` | `on`, `off` |
| `media` | `on`, `off` |
| `heat` | whole degrees, such as `21C` |
| `shutter` | `up`, `down` |
| `water` | `on`, `off` |

The IDs can use any format. Wren swaps them for short aliases before the model
sees them and maps the calls back. Areas and kinds have to come from
`wren.AREAS` and `wren.KINDS` (see [Limits](#limits)).

## Run a command

```python
from wren import act

calls, confidence = act("water the garden in 10m", house)
# [Call(tool='watering.start', args={'id': 'valve.lawn'}, delay_s=600.0)], 0.97
```

Each `Call` has a tool name, its arguments with your device IDs, and a delay
in seconds, or `None` to run it now. The shipped model emits these tools.

| tool | arguments | |
|---|---|---|
| `lights.on`, `lights.off` | `id` | |
| `media.play`, `media.stop` | `id` | |
| `heating.set` | `id`, `temp` | `temp` in °C |
| `heating.boost`, `heating.off` | `id` | |
| `shutters.up`, `shutters.down`, `shutters.stop` | `id` | |
| `watering.start`, `watering.stop` | `id` | |
| `sensors.temperature` | `area` | a question; your code answers it |

One command can give several calls. "Coming home in 20m" boosts the living
room heating when it is cold outside, and turns its light on at arrival when
it is dark and the light is off. The state decides which of these apply.

Commands work best the way the model was trained on them: lower case, without
sentence marks, numbers as digits. A speech transcript's
"Turn on the kitchen light." is better passed as
`turn on the kitchen light`.

## Decide whether to act

`confidence` is the lowest probability Wren gave any byte of its answer. It is
high for the commands Wren knows and drops for the ones it does not.

```python
act("order a pizza", house)
# ([], 0.32)
```

An empty list means no call, or an answer that did not parse or named a
device or area that is not in the house. Pick a threshold, apply the calls
above it and pass the command on otherwise, for example to a larger language
model. Fluksio uses 0.93 for the shipped model. The right value depends on
your commands, so log what Wren answered and how sure it was, and set the
threshold from your own data.

!!! warning
    A high confidence does not catch words outside the trained vocabulary.
    "Turn on the lights in the hallway" turns on the kitchen light at 0.997.
    Map such names before calling Wren, as described below.

## What the model sees

The model has 384 bytes of context, 96 of them kept for the answer. `act`
shows it the areas the command names, or the user's area and the living room
when it names none, plus up to three areas in total so it learns to tell them
apart. Extra areas are dropped first when the prompt gets too long.

`render_state` and `select_areas` show what reaches the model:

```python
from wren import render_state, select_areas

print(render_state(house, select_areas(house, "water the garden")))
# 19:30 out 6C dark
# user kitchen
# kitchen: light light.kitchen_ceiling off
# living room: light light.sofa on, media media.tv off, heat climate.living 21C
# garden: light light.garden off, water valve.lawn off
```

`act` swaps the IDs for aliases before this step, so the model sees
`lk3m9` rather than `light.kitchen_ceiling`.

## Limits

The example model has a closed vocabulary. These are known gaps:

- **Area names.** Only the ten names in `wren.AREAS` are understood. Others
  are answered confidently and wrongly ("dining room" lands on the living
  room). Rename your areas to the closest trained one, in the `House` and in
  the command alike ("hallway" → "office").
- **Device kinds and tools.** Only the five kinds above. Switches, covers,
  locks or anything outside a home are not known yet.
- **One device per kind and area.** With two kitchen lights, "turn on the
  lights in the kitchen" switches only the first.
- **Levels.** No tool takes a brightness or a position.
- **Delays** are whole minutes ("in 2h" comes out wrong).
- **House shape.** Every training house had a living room with a light,
  media and heating. Leaving them out lowers the confidence, so callers fill
  them in as placeholders and drop the calls that land on them.
- **English only.**

A model trained on your own installation's targets lifts most of these.

## Command line and C runtime

The model reads a state block followed by the command:

```sh
printf '19:30 out 6C dark\nuser kitchen\nkitchen: light lk3m9 off\nturn on the lights in the kitchen\n' | wren -n 40
# >> lights.on(id='lk3m9')
```

The C runtime runs the packed model without Python:

```sh
cc -O3 -march=native -o wren-rt runtime/wren.c -lm
./wren-rt wren/wren.bin "turn on the lights in the kitchen"
```

## Training

The example weights come from the studies in
[`dev/`](https://github.com/stroblme/wren/tree/main/dev). The studies train
on a CUDA GPU as tracked flows on a local [Fluksio](https://fluksio.com)
engine, see the [README](https://github.com/stroblme/wren#setup). Training a
model on your own targets, on a CPU, is in progress.

## Wren in Fluksio

Fluksio's `wren` node runs Wren on typed or spoken commands. The devices
wired to the node are the only ones it can reach, and commands it is unsure
of go on to Fluksio's assistant. The
[node reference](https://docs.fluksio.com/reference/node-types/#wren) covers
its settings.
