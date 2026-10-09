"""Generated home samples and external tool-call datasets.

Home samples pair device state with commands and deterministic target calls;
evaluation uses device IDs absent from training. External datasets are
downloaded on demand into `data/`.
"""

import datetime
import json
import pathlib
import random
import re
import urllib.request

from wren.home import AREAS, ID_CHARS, OUTSIDE, ROOMS, Device, House, render_state

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

HEATING_SET = ["set the heating in the {area} to {temp} degrees"]
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
    """Download missing dataset files into `data/<name>/` and return their paths."""
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
    """Render a tool call with arguments sorted by key."""
    inner = ", ".join(f"{k}={v!r}" for k, v in sorted(arguments.items()))
    return f">> {name}({inner})"


# The developer message is a fixed template; only these two facts matter.
_TIMESTAMP = re.compile(r"format: (\S+)\nDay of week is (\w+)")

# One benchmark sample: the prompt text and the ordered reference calls.
Sample = tuple[str, list[tuple[str, dict]]]


def _render_mobile_actions(row: dict) -> Sample:
    """Render the timestamp and request without tool schemas.

    The timestamp provides the reference for relative dates in calendar calls.
    """
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
    """Return (train, eval) samples using each row's metadata split."""
    [path] = fetch("mobile_actions")
    splits: dict[str, list[Sample]] = {"train": [], "eval": []}
    with open(path) as lines:
        for line in lines:
            row = json.loads(line)
            splits[row["metadata"]].append(_render_mobile_actions(row))
    return splits["train"], splits["eval"]


def to_text(samples: list[Sample]) -> str:
    """Join samples into a training stream separated by blank lines.

    Samples without calls contain only their prompt, teaching abstention.
    """
    return "\n".join(
        prompt
        + ("\n".join(render_call(n, a) for n, a in calls) + "\n" if calls else "")
        for prompt, calls in samples
    )


def mobile_actions_surface() -> dict[str, list[str]]:
    """Return {function name: parameter names} for hallucination scoring."""
    [path] = fetch("mobile_actions")
    with open(path) as lines:
        row = json.loads(next(lines))
    return {
        t["function"]["name"]: sorted(t["function"]["parameters"].get("properties", {}))
        for t in row["tools"]
    }


# Off-domain prompts the model must NOT answer with a tool call. Synthetic
# because mobile-actions contains no negatives; same prompt format as the
# positives (timestamp line included) so abstention can't key on format.
_IRRELEVANT = {
    "{}": ["hello there", "good morning", "thanks a lot", "why is the sky blue"],
    "what is the capital of {}": ["France", "Japan", "Brazil", "Egypt", "Norway"],
    "who wrote {}": ["Faust", "Hamlet", "Dune", "1984"],
    "what does {} mean": ["photosynthesis", "entropy", "inflation"],
    "tell me a joke about {}": ["cats", "programmers", "pirates"],
    "how do I cook {}": ["rice", "pasta", "lentils"],
    "what is {}": ["17 times 23", "the square root of 81", "a prime number"],
    "do you like {}": ["music", "rain", "mondays"],
    "translate {} to French": ["hello", "thank you", "good night"],
    "recommend a good {}": ["book", "movie", "podcast"],
    "how far away is {}": ["the moon", "the sun", "Mars"],
}


def irrelevant(
    n_train: int = 200, n_eval: int = 40
) -> tuple[list[Sample], list[Sample]]:
    """Return (train, eval) samples without calls, using distinct prompt types."""
    rng = random.Random(0)
    unique = [t.format(s) for t, slots in _IRRELEVANT.items() for s in slots]
    rng.shuffle(unique)
    held, seen = unique[:8], unique[8:]

    def build(prompts: list[str], n: int) -> list[Sample]:
        samples = []
        for i in range(n):
            when = datetime.datetime(
                2024, 1, 1, tzinfo=datetime.UTC
            ) + datetime.timedelta(
                days=rng.randrange(730),
                hours=rng.randrange(24),
                minutes=rng.randrange(60),
            )
            stamp = f"{when:%Y-%m-%dT%H:%M:%S} {when:%A}"
            samples.append((f"{stamp}\n{prompts[i % len(prompts)]}\n", []))
        return samples

    return build(seen, n_train), build(held, n_eval)


# The service calls in an assistant reply, one JSON object per line.
_HA_CALLS = re.compile(r"```homeassistant\n(.*?)```", re.DOTALL)


def _render_home_assistant(row: dict) -> Sample:
    """Render the request and native service calls, omitting the device list."""
    turns = {t["from"]: t["value"] for t in row["conversations"]}
    calls = []
    for block in _HA_CALLS.findall(turns["assistant"]):
        for line in filter(str.strip, block.splitlines()):
            call = json.loads(line)
            calls.append((call.pop("service"), call))
    return f"{turns['user']}\n", calls


def home_assistant() -> tuple[list[Sample], list[Sample]]:
    """Return (train, test) samples, excluding replies without service calls."""

    def load(path: pathlib.Path) -> list[Sample]:
        with open(path) as lines:
            rows = (_render_home_assistant(json.loads(line)) for line in lines)
            return [(prompt, calls) for prompt, calls in rows if calls]

    train, test = fetch("home_assistant")
    return load(train), load(test)


# ------------------------------------------- generated state-conditioned homes

# (kind, action) -> the phrasings a command may use and the tool it means.
_ACTIONS = {
    ("light", "on"): (LIGHTS_ON, "lights.on"),
    ("light", "off"): (LIGHTS_OFF, "lights.off"),
    ("media", "play"): (MEDIA_PLAY, "media.play"),
    ("media", "stop"): (MEDIA_STOP, "media.stop"),
    ("heat", "boost"): (HEATING_BOOST, "heating.boost"),
    ("heat", "off"): (HEATING_OFF, "heating.off"),
    ("shutter", "up"): (SHUTTERS_UP, "shutters.up"),
    ("shutter", "down"): (SHUTTERS_DOWN, "shutters.down"),
    ("shutter", "stop"): (SHUTTERS_STOP, "shutters.stop"),
    ("water", "start"): (WATERING_START, "watering.start"),
    ("water", "stop"): (WATERING_STOP, "watering.stop"),
}

# Area-less commands resolve through the `user <area>` state line.
AREALESS_ON = ["turn on the light", "it's too dark in here", "lights on please"]
AREALESS_OFF = ["turn off the light", "it's too bright in here", "lights off please"]
COMING_HOME = ["coming home in {m}m", "I'll be home in {m}m", "on my way, home in {m}m"]
DINNER = [
    "set up a scene for dinner",
    "dinner scene please",
    "prepare the living room for dinner",
]

_STATES = {
    "light": ["on", "off"],
    "media": ["on", "off"],
    "shutter": ["up", "down"],
    "water": ["on", "off"],
}
_DELAYS = [5, 10, 15, 20, 30, 45, 60, 90]


def _house(rng: random.Random, avoid: set[str]) -> dict:
    """Generate a home with unique device IDs excluding `avoid`.

    The living room always has light, media, and heating devices. Other rooms
    have a light and optional devices; outdoor areas have light and water.
    """
    used = set(avoid)

    def dev(kind: str) -> tuple[str, str]:
        while (i := kind[0] + "".join(rng.choices(ID_CHARS, k=4))) in used:
            pass
        used.add(i)
        state = (
            f"{rng.randint(17, 23)}C" if kind == "heat" else rng.choice(_STATES[kind])
        )
        return i, state

    areas = {"living room": {k: dev(k) for k in ("light", "media", "heat")}}
    for room in rng.sample([r for r in ROOMS if r != "living room"], rng.randint(1, 3)):
        kinds = ["light"] + [
            k for k in ("media", "heat", "shutter") if rng.random() < 0.4
        ]
        areas[room] = {k: dev(k) for k in kinds}
    for spot in rng.sample(OUTSIDE, rng.randint(0, 2)):
        areas[spot] = {"light": dev("light"), "water": dev("water")}
    hour = rng.randrange(24)
    return {
        "areas": areas,
        "time": f"{hour:02d}:{rng.randrange(60):02d}",
        "out": rng.randint(-5, 30),
        "dark": hour < 7 or hour >= 18,
        "user": rng.choice([*areas, "away"]),
        "ids": used - avoid,
    }


def _render_state(house: dict, keep: list[str]) -> str:
    """Render the areas in `keep` using the library's state format."""
    areas = {
        area: [Device(kind, i, s) for kind, (i, s) in devs.items()]
        for area, devs in house["areas"].items()
    }
    user = None if house["user"] == "away" else house["user"]
    return render_state(
        House(house["time"], house["out"], house["dark"], user, areas), keep
    )


def _keep(house: dict, rng: random.Random, *needed: str) -> list[str]:
    """Return needed areas and one or two distractors in house order."""
    others = [a for a in house["areas"] if a not in needed]
    extra = set(rng.sample(others, min(len(others), rng.randint(1, 2))))
    return [a for a in house["areas"] if a in {*needed} | extra]


def _command(house: dict, rng: random.Random) -> Sample:
    """Generate a device command, optionally delayed or without an area name."""
    if rng.random() < 0.15:  # area-less
        if house["user"] == "away":
            house["user"] = rng.choice(list(house["areas"]))
        area = house["user"]
        action = rng.choice(["on", "off"])
        prompt = rng.choice(AREALESS_ON if action == "on" else AREALESS_OFF)
        calls = [(f"lights.{action}", {"id": house["areas"][area]["light"][0]})]
    else:
        area = rng.choice(list(house["areas"]))
        kind = rng.choice(list(house["areas"][area]))
        dev_id = house["areas"][area][kind][0]
        if kind == "heat" and rng.random() < 0.33:
            temp = rng.randint(18, 24)
            prompt = rng.choice(HEATING_SET).format(area=area, temp=temp)
            calls = [("heating.set", {"id": dev_id, "temp": temp})]
        else:
            action = rng.choice([a for k, a in _ACTIONS if k == kind])
            phrasings, tool = _ACTIONS[(kind, action)]
            prompt = rng.choice(phrasings).format(area=area)
            calls = [(tool, {"id": dev_id})]
    if rng.random() < 0.2:  # delayed execution
        m = rng.choice(_DELAYS)
        prompt += f" in {m}m"
        calls = [(n, a | {"delay": f"{m}m"}) for n, a in calls]
    return _render_state(house, _keep(house, rng, area)) + prompt + "\n", calls


def _reading(house: dict, rng: random.Random) -> Sample:
    area = rng.choice(list(house["areas"]))
    prompt = rng.choice(TEMPERATURE).format(area=area)
    return _render_state(house, _keep(house, rng, area)) + prompt + "\n", [
        ("sensors.temperature", {"area": _slug(area)})
    ]


def _coming_home(house: dict, rng: random.Random) -> Sample:
    """Heat the living room if cold; turn on its light at arrival if dark and off."""
    house["user"] = "away"
    m = rng.choice(_DELAYS)
    prompt = rng.choice(COMING_HOME).format(m=m)
    living = house["areas"]["living room"]
    calls: list[tuple[str, dict]] = []
    if house["out"] < 15:
        calls.append(("heating.boost", {"id": living["heat"][0]}))
    if house["dark"] and living["light"][1] == "off":
        calls.append(("lights.on", {"delay": f"{m}m", "id": living["light"][0]}))
    return _render_state(house, _keep(house, rng, "living room")) + prompt + "\n", calls


def _scene(house: dict, rng: random.Random) -> Sample:
    living = house["areas"]["living room"]
    return _render_state(house, _keep(house, rng, "living room")) + rng.choice(
        DINNER
    ) + "\n", [
        ("lights.on", {"id": living["light"][0]}),
        ("media.play", {"id": living["media"][0]}),
    ]


def _chatter(house: dict, rng: random.Random) -> Sample:
    """Generate unrelated chatter with a state block and no target calls."""
    template, slots = rng.choice(list(_IRRELEVANT.items()))
    prompt = template.format(rng.choice(slots))
    return _render_state(house, _keep(house, rng)) + prompt + "\n", []


def _missing(house: dict, rng: random.Random) -> Sample:
    """Request a device kind absent from the area, with no target calls."""
    area, kind = rng.choice(
        [
            (area, kind)
            for area, devs in house["areas"].items()
            for kind in ("media", "heat", "shutter")
            if kind not in devs
        ]
    )
    action = rng.choice([a for k, a in _ACTIONS if k == kind])
    phrasings, _ = _ACTIONS[(kind, action)]
    prompt = rng.choice(phrasings).format(area=area)
    return _render_state(house, _keep(house, rng, area)) + prompt + "\n", []


def _stateless(house: dict, rng: random.Random) -> Sample:
    """Generate a command without state, using the area slug as the device ID."""
    kind, action = rng.choice(list(_ACTIONS))
    pool = AREAS if kind == "light" else OUTSIDE if kind == "water" else ROOMS
    area = rng.choice(pool)
    phrasings, tool = _ACTIONS[(kind, action)]
    return rng.choice(phrasings).format(area=area) + "\n", [(tool, {"id": _slug(area)})]


_BUILDERS = [
    (_command, 50),
    (_reading, 10),
    (_coming_home, 10),
    (_scene, 5),
    (_chatter, 10),
    (_missing, 5),
    (_stateless, 10),
]


def home(
    n_train: int = 8000, n_eval: int = 400, seed: int = 0
) -> tuple[list[Sample], list[Sample]]:
    """Return (train, eval) samples, each with a generated house.

    Evaluation device IDs never appear in training, so exact match tests copying
    from the state block.
    """
    rng = random.Random(seed)
    builders, weights = zip(*_BUILDERS)

    def build(n: int, avoid: set[str], seen: set[str] | None) -> list[Sample]:
        samples = []
        for _ in range(n):
            house = _house(rng, avoid)
            if seen is not None:
                seen |= house["ids"]
            samples.append(rng.choices(builders, weights)[0](house, rng))
        return samples

    train_ids: set[str] = set()
    train = build(n_train, set(), train_ids)
    return train, build(n_eval, train_ids, None)


def home_corpus() -> str:
    """Return the generated training split as text, the default for `wren --train`."""
    return to_text(home()[0])
