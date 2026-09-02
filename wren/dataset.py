"""Training corpora.

The built-in corpus is generated: random houses (areas, devices with
random ids, current state) paired with commands whose correct tool calls
follow deterministic rules from that state. The state travels in the
prompt, so argument values must be *copied* out of it -- and eval houses
use device ids never seen in training, which makes exact match a test of
retrieval rather than memorization (the s7 bottleneck, RESEARCH.md).

External datasets (downloaded on demand into data/, gitignored):
- mobile-actions (Google, CC-BY-4.0): 7 phone functions, the one needle
  benchmark feasible at Wren's scale -- our external baseline.
- Home-Assistant-Requests (acon96, MIT): Wren's actual domain, used by the
  later stages that grow the context to hold device state.
"""

import datetime
import json
import pathlib
import random
import re
import urllib.request

ROOMS = ["living room", "kitchen", "bathroom", "bedroom", "office"]
OUTSIDE = ["garden", "patio", "garage", "driveway", "balcony"]
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
    """Samples as one training stream, blank lines marking the boundaries.
    A sample without calls (irrelevance: the model should abstain) is just
    its prompt."""
    return "\n".join(
        prompt
        + ("\n".join(render_call(n, a) for n, a in calls) + "\n" if calls else "")
        for prompt, calls in samples
    )


def mobile_actions_surface() -> dict[str, list[str]]:
    """{function name: parameter names} from the dataset's tool schemas --
    the reference for the hallucination metric."""
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
    """(train, eval) no-call samples; the eval prompts are types never seen
    in training, so abstention has to generalize."""
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
    """Prompt is the user request alone -- the 38-service surface lives in
    the weights (implicit-tool mode), and the original system message's
    device list is dropped: holding device state in context is a later
    stage. Calls keep their native dotted service names."""
    turns = {t["from"]: t["value"] for t in row["conversations"]}
    calls = []
    for block in _HA_CALLS.findall(turns["assistant"]):
        for line in filter(str.strip, block.splitlines()):
            call = json.loads(line)
            calls.append((call.pop("service"), call))
    return f"{turns['user']}\n", calls


def home_assistant() -> tuple[list[Sample], list[Sample]]:
    """(train, test) samples. Rows whose reply carries no service call are
    dropped: they answer status questions from the device list we do not
    include."""

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
_ID_CHARS = "abcdefghijkmnpqrstuvwxyz23456789"  # no 0/O/1/l confusables
_DELAYS = [5, 10, 15, 20, 30, 45, 60, 90]


def _house(rng: random.Random, avoid: set[str]) -> dict:
    """A random home. The living room always exists and holds a light,
    media and heating, so every rule below has its target; other rooms get
    a light plus a random subset, outside areas a light and watering.
    Device ids are random slugs, unique within the house and outside
    `avoid` (the eval split avoids every training id)."""
    used = set(avoid)

    def dev(kind: str) -> tuple[str, str]:
        while (i := kind[0] + "".join(rng.choices(_ID_CHARS, k=4))) in used:
            pass
        used.add(i)
        state = f"{rng.randint(17, 23)}C" if kind == "heat" else rng.choice(_STATES[kind])
        return i, state

    areas = {"living room": {k: dev(k) for k in ("light", "media", "heat")}}
    for room in rng.sample([r for r in ROOMS if r != "living room"], rng.randint(1, 3)):
        kinds = ["light"] + [k for k in ("media", "heat", "shutter") if rng.random() < 0.4]
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


def _render_state(house: dict) -> str:
    lines = [
        f"{house['time']} out {house['out']}C {'dark' if house['dark'] else 'bright'}",
        f"user {house['user']}",
    ]
    for area, devs in house["areas"].items():
        lines.append(f"{area}: " + ", ".join(f"{k} {i} {s}" for k, (i, s) in devs.items()))
    return "\n".join(lines) + "\n"


def _command(house: dict, rng: random.Random) -> Sample:
    """A device command: direct or indirect phrasing, sometimes delayed
    ("... in 10m" becomes a delay argument), sometimes area-less (the
    target then comes from the user's state line)."""
    if rng.random() < 0.15:  # area-less
        if house["user"] == "away":
            house["user"] = rng.choice(list(house["areas"]))
        action = rng.choice(["on", "off"])
        prompt = rng.choice(AREALESS_ON if action == "on" else AREALESS_OFF)
        calls = [(f"lights.{action}", {"id": house["areas"][house["user"]]["light"][0]})]
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
    return _render_state(house) + prompt + "\n", calls


def _reading(house: dict, rng: random.Random) -> Sample:
    area = rng.choice(list(house["areas"]))
    prompt = rng.choice(TEMPERATURE).format(area=area)
    return _render_state(house) + prompt + "\n", [
        ("sensors.temperature", {"area": _slug(area)})
    ]


def _coming_home(house: dict, rng: random.Random) -> Sample:
    """The implication rule: heat now when it is cold outside, light on
    arrival when it is dark and the living room light is off. Both calls,
    one, or none, depending on the state."""
    house["user"] = "away"
    m = rng.choice(_DELAYS)
    prompt = rng.choice(COMING_HOME).format(m=m)
    living = house["areas"]["living room"]
    calls: list[tuple[str, dict]] = []
    if house["out"] < 15:
        calls.append(("heating.boost", {"id": living["heat"][0]}))
    if house["dark"] and living["light"][1] == "off":
        calls.append(("lights.on", {"delay": f"{m}m", "id": living["light"][0]}))
    return _render_state(house) + prompt + "\n", calls


def _scene(house: dict, rng: random.Random) -> Sample:
    living = house["areas"]["living room"]
    return _render_state(house) + rng.choice(DINNER) + "\n", [
        ("lights.on", {"id": living["light"][0]}),
        ("media.play", {"id": living["media"][0]}),
    ]


def _chatter(house: dict, rng: random.Random) -> Sample:
    """Off-domain talk with a full state block: abstain."""
    template, slots = rng.choice(list(_IRRELEVANT.items()))
    return _render_state(house) + template.format(rng.choice(slots)) + "\n", []


def _missing(house: dict, rng: random.Random) -> Sample:
    """A command for a device kind the area does not have: abstain."""
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
    return _render_state(house) + rng.choice(phrasings).format(area=area) + "\n", []


def _stateless(house: dict, rng: random.Random) -> Sample:
    """No state block at all (the bare CLI prompt shape): the device id
    falls back to the area slug."""
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
    """(train, eval) state-conditioned samples, one generated house each.

    Eval houses draw their device ids outside the set training ever saw,
    so eval exact match measures copying values out of the state block."""
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
    """The built-in training corpus: the generated train split as one
    stream (`wren --train`'s default)."""
    return to_text(home()[0])
