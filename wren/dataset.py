"""Training corpora.

Stage 1 uses a tiny synthetic corpus of natural-language commands paired
with the tool call they should produce; the point is to prove the pipeline
learns the mapping, not to generalize.

External datasets (downloaded on demand into data/, gitignored):
- mobile-actions (Google, CC-BY-4.0): 7 phone functions, the one needle
  benchmark feasible at Wren's scale -- our external baseline.
- Home-Assistant-Requests (acon96, MIT): Wren's actual domain, used by the
  later stages that grow the context to hold device state.
"""

import json
import pathlib
import random
import re
import urllib.request

ROOMS = ["living room", "kitchen", "bathroom", "bedroom", "office"]
OUTSIDE = ["garden", "patio", "garage", "driveway", "balcony"]
WATERED = ["garden", "patio", "balcony"]
AREAS = ROOMS + OUTSIDE

# Each entry maps several phrasings (direct or indirect) of the same intent
# to a single tool call, so the model sees "turn the lights on" and "it's
# too dark in here" both resolve to `lights.on`.
LIGHTS_ON = [
    "turn on the lights in the {area}",
    "switch on the {area} lights",
    "it's too dark in the {area}",
]
LIGHTS_OFF = [
    "turn off the lights in the {area}",
    "switch off the {area} lights",
    "it's too bright in the {area}",
]
TEMPERATURE = [
    "what is the temperature in the {area}",
    "how warm is it in the {area}",
    "check the temperature in the {area}",
]

MEDIA_PLAY = [
    "play music in the {area}",
    "put some music on in the {area}",
    "it's too quiet in the {area}",
]
MEDIA_STOP = [
    "stop the music in the {area}",
    "turn the music off in the {area}",
    "turn that off in the {area}",
]

HEATING_SET = ["set the heating in the {area} to 21 degrees"]
HEATING_BOOST = [
    "boost the heating in the {area}",
    "it's a bit cold in the {area}",
    "it's getting cold in the {area}",
]
HEATING_OFF = [
    "turn off the heating in the {area}",
    "it's too warm in the {area}",
    "it's getting hot in the {area}",
]

SHUTTERS_UP = [
    "open the rollershutters in the {area}",
    "raise the blinds in the {area}",
    "let some light into the {area}",
]
SHUTTERS_DOWN = [
    "close the rollershutters in the {area}",
    "lower the blinds in the {area}",
    "the sun is too strong in the {area}",
]
SHUTTERS_STOP = ["stop the rollershutters in the {area}"]

WATERING_START = [
    "start watering the {area}",
    "water the {area}",
    "the {area} looks dry",
]
WATERING_STOP = ["stop watering the {area}", "that's enough water for the {area}"]


def _slug(area: str) -> str:
    return area.replace(" ", "_")


def _add(lines: list[str], phrasings: list[str], tool: str, area: str) -> None:
    slug = _slug(area)
    for phrasing in phrasings:
        lines.append(f"{phrasing.format(area=area)}\n>> {tool.format(slug=slug)}\n")


def home_corpus(copies: int = 40) -> str:
    """Every command/tool-call pair, repeated in `copies` shuffled passes."""
    lines = []
    for area in AREAS:
        _add(lines, LIGHTS_ON, "lights.on('{slug}')", area)
        _add(lines, LIGHTS_OFF, "lights.off('{slug}')", area)
        _add(lines, TEMPERATURE, "sensors.temperature('{slug}')", area)
    for room in ROOMS:
        _add(lines, MEDIA_PLAY, "media.play('{slug}')", room)
        _add(lines, MEDIA_STOP, "media.stop('{slug}')", room)
        _add(lines, HEATING_SET, "heating.set('{slug}', 21)", room)
        _add(lines, HEATING_BOOST, "heating.boost('{slug}')", room)
        _add(lines, HEATING_OFF, "heating.off('{slug}')", room)
        _add(lines, SHUTTERS_UP, "shutters.up('{slug}')", room)
        _add(lines, SHUTTERS_DOWN, "shutters.down('{slug}')", room)
        _add(lines, SHUTTERS_STOP, "shutters.stop('{slug}')", room)
    for area in WATERED:
        _add(lines, WATERING_START, "watering.start('{slug}')", area)
        _add(lines, WATERING_STOP, "watering.stop('{slug}')", area)
    rng = random.Random(0)
    corpus = []
    for _ in range(copies):
        rng.shuffle(lines)
        corpus += lines
    return "".join(corpus)


DATA = pathlib.Path("data")
_HF = "https://huggingface.co/datasets"
_FILES = {
    "mobile_actions": [f"{_HF}/google/mobile-actions/resolve/main/dataset.jsonl"],
    "home_assistant": [
        f"{_HF}/acon96/Home-Assistant-Requests/resolve/main/home_assistant_train.jsonl",
        f"{_HF}/acon96/Home-Assistant-Requests/resolve/main/home_assistant_test.jsonl",
    ],
}


def fetch(name: str) -> list[pathlib.Path]:
    """Download a dataset's files into data/<name>/ once; return the paths."""
    folder = DATA / name
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for url in _FILES[name]:
        path = folder / url.rsplit("/", 1)[-1]
        if not path.exists():
            urllib.request.urlretrieve(url, path)
        paths.append(path)
    return paths


def render_call(name: str, arguments: dict) -> str:
    """One call in Wren's output format, arguments in canonical key order."""
    inner = ", ".join(f"{k}={v!r}" for k, v in sorted(arguments.items()))
    return f">> {name}({inner})"


# The developer message is a fixed template; only these two facts matter.
_TIMESTAMP = re.compile(r"format: (\S+)\nDay of week is (\w+)")

# One benchmark sample: the prompt text and the ordered reference calls.
Sample = tuple[str, list[tuple[str, dict]]]


def _render_mobile_actions(row: dict) -> Sample:
    """Prompt is a timestamp line plus the user request; the tool schemas
    are left out -- the 7-function surface lives in the weights, Wren's
    implicit-tool mode. The timestamp stays because calendar calls resolve
    relative dates against it."""
    messages = {m["role"]: m for m in row["messages"]}
    m = _TIMESTAMP.search(messages["developer"]["content"])
    stamp = f"{m.group(1)} {m.group(2)}\n" if m else ""
    prompt = f"{stamp}{messages['user']['content']}\n"
    calls = [
        (c["function"]["name"], c["function"].get("arguments", {}))
        for c in messages["assistant"]["tool_calls"]
    ]
    return prompt, calls


def mobile_actions() -> tuple[list[Sample], list[Sample]]:
    """(train, eval) samples, split by the row's own metadata field."""
    [path] = fetch("mobile_actions")
    splits: dict[str, list[Sample]] = {"train": [], "eval": []}
    with open(path) as lines:
        for line in lines:
            row = json.loads(line)
            splits[row["metadata"]].append(_render_mobile_actions(row))
    return splits["train"], splits["eval"]


def to_text(samples: list[Sample]) -> str:
    """Samples as one training stream, blank lines marking the boundaries."""
    return "\n".join(
        prompt + "\n".join(render_call(n, a) for n, a in calls) + "\n"
        for prompt, calls in samples
    )


def home_assistant() -> tuple[pathlib.Path, pathlib.Path]:
    """Download; return the (train, test) jsonl paths. Rendering into
    Wren's format is deferred to the stage that trains on it."""
    train, test = fetch("home_assistant")
    return train, test
