"""Generated home samples, corpora from a target list, and external datasets.

Home samples pair device state with commands and deterministic target calls;
evaluation uses device IDs absent from training. `corpus` generates samples
for any installation's target list. External datasets are downloaded on
demand into `data/`.
"""

import dataclasses
import datetime
import difflib
import json
import math
import pathlib
import random
import re
import urllib.request
from collections.abc import Callable, Sequence

from wren.home import (
    AREAS,
    BUDGET,
    ID_CHARS,
    OUTSIDE,
    PREFIX,
    ROOMS,
    Device,
    House,
    render_state,
)

# Aliased: the home generator above uses `BUDGET` and `prompt` for its own.
from wren.targets import BUDGET as PROMPT_BUDGET
from wren.targets import N_NEW, Call, Target, Value, answer, parse, select
from wren.targets import prompt as render_prompt

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
    # s10: words the 7B panel set used that the s8 corpus lacked
    "switch on the tv in the {area}",
    "put the tv on in the {area}",
]
MEDIA_STOP = [
    "stop the music in the {area}",
    "turn the music off in the {area}",
    "turn that off in the {area}",
    # s10: words the 7B panel set used that the s8 corpus lacked
    "switch off the tv in the {area}",
]

HEATING_SET = ["set the heating in the {area} to {temp} degrees"]
HEATING_BOOST = [
    "boost the heating in the {area}",
    "it's a bit cold in the {area}",
    "it's getting cold in the {area}",
    # s10: words the 7B panel set used that the s8 corpus lacked
    "warm up the {area}",
    "i'm freezing in the {area}",
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
    # s10: words the 7B panel set used that the s8 corpus lacked
    "open the shutters in the {area}",
    "raise the {area} shutters",
]
SHUTTERS_DOWN = [
    "close the rollershutters in the {area}",
    "lower the blinds in the {area}",
    "the sun is too strong in the {area}",
    # s10: words the 7B panel set used that the s8 corpus lacked
    "close the {area} shutters",
    "lower the shutters in the {area}",
]
SHUTTERS_STOP = ["stop the rollershutters in the {area}"]

WATERING_START = [
    "start watering the {area}",
    "water the {area}",
    "the {area} looks dry",
    # s10: words the 7B panel set used that the s8 corpus lacked
    "switch on the sprinkler in the {area}",
]
WATERING_STOP = [
    "stop watering the {area}",
    "that's enough water for the {area}",
    # s10: as above
    "turn off the sprinkler in the {area}",
    "shut off the water in the {area}",
]

SWITCHES_ON = [
    "turn on the plug in the {area}",
    "switch on the {area} socket",
    "switch on the appliances in the {area}",
    "power up the {area} plug",
]
SWITCHES_OFF = [
    "turn off the plug in the {area}",
    "switch off the {area} socket",
    "switch off the appliances in the {area}",
    "cut the power to the {area} plug",
]

# Places and things no house has, so a command naming one gets no call.
# "attic" and "garage door" stay out, for the s10 probes.
UNKNOWN_AREAS = [
    "hall",
    "hallway",
    "basement",
    "cellar",
    "loft",
    "nursery",
    "dining room",
    "guest room",
    "porch",
    "terrace",
    "shed",
    "studio",
]
UNKNOWN_DEVICES = [
    "fridge",
    "oven",
    "kettle",
    "vacuum",
    "doorbell",
    "alarm",
    "printer",
    "dishwasher",
    "car",
    "computer",
]
UNKNOWN = [
    "turn on the {thing} in the {area}",
    "switch off the {thing} in the {area}",
    "start the {thing} in the {area}",
    "turn off the {thing}",
]


def _slug(area: str) -> str:
    return area.replace(" ", "_")


DATA = pathlib.Path("data")
_HF = "https://huggingface.co/datasets"
# Pinned revisions keep the benchmark files fixed.
_MOBILE = (
    f"{_HF}/google/mobile-actions/resolve/e920309bc2acbc2e99a5e3201cf37df2b9fd9151"
)
_HA = f"{_HF}/acon96/Home-Assistant-Requests/resolve/75f5abe4a3fc4f30b6ccfeb239f4da30d22b3619"
_FILES = {
    "mobile_actions": [f"{_MOBILE}/dataset.jsonl"],
    "home_assistant": [
        f"{_HA}/home_assistant_train.jsonl",
        f"{_HA}/home_assistant_test.jsonl",
    ],
}


def fetch(name: str) -> list[pathlib.Path]:
    """Download missing dataset files into `data/<name>/` and return their paths.

    `data/` is relative to the current working directory.
    """
    folder = DATA / name
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for url in _FILES[name]:
        path = folder / url.rsplit("/", 1)[-1]
        if not path.exists():
            # Download beside the target so an interrupted fetch leaves no file.
            part = path.with_name(path.name + ".part")
            urllib.request.urlretrieve(url, part)
            part.rename(path)
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
    ("switch", "on"): (SWITCHES_ON, "switches.on"),
    ("switch", "off"): (SWITCHES_OFF, "switches.off"),
}

# Area-less commands resolve through the `user <area>` state line, and to the
# living room when the user is away (fluksio without presence: always).
AREALESS_ON = ["turn on the light", "it's too dark in here", "lights on please"]
AREALESS_OFF = ["turn off the light", "it's too bright in here", "lights off please"]
AREALESS = {
    ("light", "on"): AREALESS_ON,
    ("light", "off"): AREALESS_OFF,
    ("heat", "boost"): ["it's a bit cold in here", "i'm cold", "warm it up a bit"],
    ("heat", "off"): ["turn off the heating", "it's too warm in here"],
    ("media", "play"): ["play some music", "put some music on"],
    ("media", "stop"): ["stop the music", "turn the music off"],
    ("shutter", "up"): ["open the blinds", "raise the shutters"],
    ("shutter", "down"): ["close the blinds", "lower the shutters"],
    ("switch", "on"): ["switch on the appliances", "turn on the plug"],
    ("switch", "off"): ["switch off the appliances", "turn off the plug"],
}
AREALESS_SET = ["set the heating to {temp} degrees"]
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
    "switch": ["on", "off"],
}
_DELAYS = [5, 10, 15, 20, 30, 45, 60, 90]


def _house(rng: random.Random, avoid: set[str]) -> dict:
    """Generate a home with unique device IDs excluding `avoid`.

    The living room always has light, media, and heating devices. Other rooms
    have a light and optional devices; outdoor areas have light and water. Any
    area may have a switch (a plug for an appliance).
    """
    used = set(avoid)

    def dev(kind: str) -> tuple[str, str]:
        while (i := PREFIX[kind] + "".join(rng.choices(ID_CHARS, k=4))) in used:
            pass
        used.add(i)
        state = (
            f"{rng.randint(17, 23)}C" if kind == "heat" else rng.choice(_STATES[kind])
        )
        return i, state

    def switch() -> list[str]:
        return ["switch"] if rng.random() < 0.25 else []

    areas = {"living room": {k: dev(k) for k in ["light", "media", "heat", *switch()]}}
    for room in rng.sample([r for r in ROOMS if r != "living room"], rng.randint(1, 3)):
        kinds = ["light"] + [
            k for k in ("media", "heat", "shutter") if rng.random() < 0.4
        ]
        areas[room] = {k: dev(k) for k in kinds + switch()}
    for spot in rng.sample(OUTSIDE, rng.randint(0, 2)):
        areas[spot] = {k: dev(k) for k in ["light", "water", *switch()]}
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
    if rng.random() < 0.2:  # area-less: the user's area, else the living room
        # Away half the time: an engine without presence never knows the room.
        house["user"] = (
            "away" if rng.random() < 0.5 else rng.choice(list(house["areas"]))
        )
        area = "living room" if house["user"] == "away" else house["user"]
        devs = house["areas"][area]
        if "heat" in devs and rng.random() < 0.15:
            temp = rng.randint(18, 24)
            prompt = rng.choice(AREALESS_SET).format(temp=temp)
            calls = [("heating.set", {"id": devs["heat"][0], "temp": temp})]
        else:
            kind, action = rng.choice(list(AREALESS))
            prompt = rng.choice(AREALESS[(kind, action)])
            # A device the area lacks gets no call, as `_missing` teaches.
            tool = _ACTIONS[(kind, action)][1]
            calls = [(tool, {"id": devs[kind][0]})] if kind in devs else []
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
            for kind in ("media", "heat", "shutter", "switch")
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


def _pair(house: dict, rng: random.Random) -> Sample:
    """Name two areas with the same kind of device: a call each, in the order said."""
    where: dict[str, list[str]] = {}
    for area, devs in house["areas"].items():
        for kind in devs:
            where.setdefault(kind, []).append(area)
    kind = rng.choice([k for k, areas in where.items() if len(areas) > 1])
    a, b = rng.sample(where[kind], 2)
    phrasings, tool = _ACTIONS[rng.choice([k for k in _ACTIONS if k[0] == kind])]
    prompt = rng.choice(phrasings).format(area=f"{a} and the {b}") + "\n"
    calls = [(tool, {"id": house["areas"][area][kind][0]}) for area in (a, b)]
    state = _render_state(house, _keep(house, rng, a, b))
    if len((state + prompt).encode()) > BUDGET:  # no room for distractors
        state = _render_state(house, [a, b])
    return state + prompt, calls


def _unknown(house: dict, rng: random.Random) -> Sample:
    """Name a place or a thing no house has, with no target calls."""
    if rng.random() < 0.5:
        phrasings, _ = _ACTIONS[rng.choice(list(_ACTIONS))]
        prompt = rng.choice(phrasings).format(area=rng.choice(UNKNOWN_AREAS))
        keep = _keep(house, rng)
    else:
        area = rng.choice(list(house["areas"]))
        thing = rng.choice(UNKNOWN_DEVICES)
        prompt = rng.choice(UNKNOWN).format(thing=thing, area=area)
        keep = _keep(house, rng, area)
    return _render_state(house, keep) + prompt + "\n", []


_BUILDERS = [
    (_command, 50),
    (_reading, 10),
    (_coming_home, 10),
    (_scene, 5),
    (_chatter, 10),
    (_missing, 5),
    (_stateless, 10),
    (_pair, 8),
    (_unknown, 6),
]


# A command's delay suffix.
_DELAY = re.compile(r" in \d+m$")


def spoken(command: str) -> str:
    """The corpus's delay shorthand as someone says it: "in 5m" -> "in 5 minutes"."""
    return re.sub(r"\bin (\d+)m\b", r"in \1 minutes", command)


def load_noisy(
    path: str | pathlib.Path, min_ratio: float = 0.6
) -> dict[str, list[str]]:
    """Read {command: [what a recogniser heard]} from s10's `noisy.jsonl`.

    A transcript sharing too few words with what was said has lost the
    command's meaning, and is left out rather than taught as it.
    """
    table: dict[str, list[str]] = {}
    for line in pathlib.Path(path).read_text().splitlines():
        row = json.loads(line)
        said, heard = spoken(row["command"]).split(), row["text"].split()
        if difflib.SequenceMatcher(None, said, heard).ratio() >= min_ratio:
            table.setdefault(row["command"], []).append(row["text"])
    return table


def _misheard(
    samples: list[Sample], noisy: dict[str, list[str]], share: float, seed: int
) -> list[Sample]:
    """Swap `share` of the commands that come with a state for a transcript of them.

    The table holds commands without their delay, which is said after the
    transcript as a person says it.
    """
    rng = random.Random(seed)
    out = []
    for prompt, calls in samples:
        state, _, command = prompt[:-1].rpartition("\n")
        base = _DELAY.sub("", command)
        if state and base in noisy and rng.random() < share:
            prompt = (
                f"{state}\n{rng.choice(noisy[base])}{spoken(command[len(base) :])}\n"
            )
        out.append((prompt, calls))
    return out


def home(
    n_train: int = 8000,
    n_eval: int = 400,
    seed: int = 0,
    noisy: dict[str, list[str]] | None = None,
    share: float = 0.3,
) -> tuple[list[Sample], list[Sample]]:
    """Return (train, eval) samples, each with a generated house.

    Evaluation device IDs never appear in training, so exact match tests copying
    from the state block. With `noisy` (`load_noisy`), `share` of the training
    commands are replaced by what a recogniser heard; evaluation stays clean.
    """
    rng = random.Random(seed)
    builders, weights = zip(*_BUILDERS)

    def build(n: int, avoid: set[str], seen: set[str] | None) -> list[Sample]:
        samples: list[Sample] = []
        while len(samples) < n:
            house = _house(rng, avoid)
            if seen is not None:
                seen |= house["ids"]
            sample = rng.choices(builders, weights)[0](house, rng)
            # A house too large for the context is drawn again (s10's switches).
            if len(sample[0].encode()) <= BUDGET:
                samples.append(sample)
        return samples

    train_ids: set[str] = set()
    train = build(n_train, set(), train_ids)
    evals = build(n_eval, train_ids, None)
    # Its own seed, so the clean samples are the same with or without noise.
    return (_misheard(train, noisy, share, seed + 1) if noisy else train), evals


def commands(n_train: int = 256000) -> list[str]:
    """Every command of the training split that comes with a state, once and
    without its delay (`_misheard` says that after the transcript)."""
    train, _ = home(n_train, 0)
    lines = {p[:-1].rpartition("\n")[2] for p, _ in train if p.count("\n") > 1}
    return sorted({_DELAY.sub("", line) for line in lines})


# ------------------------------------------------- corpus from a target list

# A training sample: the command, every listed target with this sample's
# value, and the expected calls (none: the model abstains).
Case = tuple[str, list[Target], list[Call]]

# Generic phrasings: `{w}` is a target's words, `{v}` a value as said.
_ON = ["turn on {w}", "switch on {w}", "turn {w} on", "{w} on"]
_OFF = ["turn off {w}", "switch off {w}", "turn {w} off", "{w} off"]
_SET = ["set {w} to {v}", "change {w} to {v}", "{w} to {v}"]
# A string value may be a verb ("drain the pump") or not ("shutter down").
_PICK = [*_SET, "switch {w} to {v}", "{w} {v}", "{v} {w}"]
_READ = ["what is {w}", "what's {w}", "check {w}", "tell me {w}"]
# Words that take no "the" before them: "turn off all lights".
_DETERMINERS = {"a", "all", "an", "each", "every", "the"}
# Things a command may name that no target answers; those sharing a word
# with a target are left out.
_THINGS = ["alarm", "basement", "car", "cellar", "computer", "dishwasher"]
_THINGS += ["doorbell", "fridge", "garage", "hallway", "kettle", "microscope"]
_THINGS += ["oven", "printer", "robot", "vacuum"]
_CHATTER = [t.format(s) for t, slots in _IRRELEVANT.items() for s in slots]
# A delay is said in the unit `answer` writes: 60 minutes is `1h`, so "in 60
# minutes" is never said.
_UNITS = [("second", 1), ("minute", 60), ("hour", 3600)]
_COUNTS = {1: [k for k in range(1, 91) if k % 60], 3600: list(range(1, 25))}
_COUNTS[60] = _COUNTS[1]
# Sample kinds and their shares; the share of delayed commands.
_MIX = {"single": 50, "group": 10, "joined": 10, "unknown": 10, "chatter": 10}
_MIX["example"] = 10
_DELAY_SHARE = 0.2


def corpus(
    targets: list[Target],
    *,
    n: int,
    seed: int,
    examples: Sequence[dict] = (),
    paraphrase: Callable[[str], list[str]] | None = None,
) -> list[Case]:
    """Return `n` training samples for an installation's `targets`.

    Commands use generic phrasings with the words of a target's name and
    description: on and off for a bool, a number in range, one of a string's
    values, a question for a read-only target. Some name several targets or
    carry a delay; about a fifth name nothing listed or are chit-chat and
    expect no call. Each sample draws every target's value, so the model reads state
    rather than memorising it. Samples whose calls `select` would not show,
    or whose answer exceeds `N_NEW`, are drawn again.

    `examples` are installation-supplied JSON objects for what generic
    phrasing cannot know: `{"state": {"user_location": "kitchen"}, "command":
    "lights on please", "calls": [["kitchen_light", true]]}`, a call being
    `[target, value]`, `[target, value, delay_s]` or `[target]` for a read and
    `state` optional. They make up a tenth of the samples, the targets not in
    `state` drawn as usual; one that cannot be learnt raises ValueError.

    `paraphrase(command)` returns rewordings of a command, which then stand in
    for it; it must keep numbers as digits. It is called once per distinct
    command, before any delay is appended.

    Samples share `Target` objects of equal value, so treat them as read-only.
    """
    rng = random.Random(seed)
    known = [_example(targets, e) for e in examples]
    sayable = [t for t in targets if t.readonly or t.type != "str" or t.values]
    settable = [t for t in sayable if not t.readonly]
    alike: dict[tuple, list[Target]] = {}  # one phrasing says them together
    for t in sayable:
        key = (t.type, t.readonly, tuple(t.values or ()), tuple(t.range or ()))
        alike.setdefault(key, []).append(t)
    groups = [g for g in alike.values() if len(g) > 1]
    things = [w for w in _THINGS if not select(targets, w)]
    shares = _MIX | {
        "group": _MIX["group"] if groups else 0,
        "example": _MIX["example"] if known else 0,
    }
    kinds, weights = list(shares), list(shares.values())
    pools = [_pool(t, rng) for t in targets]
    widened: dict[str, list[str]] = {}
    cases: list[Case] = []
    while len(cases) < n:
        state = [rng.choice(pool) for pool in pools]
        kind = rng.choices(kinds, weights)[0]
        if kind == "single":
            command, calls = _clause([rng.choice(sayable)], rng)
        elif kind == "group":
            group = rng.choice(groups)
            k = min(len(group), rng.choice((2, 3)))
            command, calls = _clause(rng.sample(group, k), rng)
        elif kind == "joined":
            k = min(len(settable), rng.choice((2, 3)))
            parts = [_clause([t], rng) for t in rng.sample(settable, k)]
            command = " and ".join(c for c, _ in parts)
            calls = [call for _, cs in parts for call in cs]
        elif kind == "unknown":
            command, calls = _unlisted(sayable, things, rng), []
        elif kind == "chatter":
            command, calls = rng.choice(_CHATTER), []
        else:
            command, fixed, calls = rng.choice(known)
            state = [fixed.get(t.name, t) for t in state]
        if paraphrase is not None:
            if command not in widened:
                widened[command] = paraphrase(command)
            command = rng.choice([command, *widened[command]])
        delayable = kind in ("single", "group") and calls[0].value is not None
        if delayable and rng.random() < _DELAY_SHARE:
            unit, size = rng.choice(_UNITS)
            k = rng.choice(_COUNTS[size])
            said = f"{k} {unit}{'s' if k > 1 else ''}"
            if isinstance(calls[0].value, bool) and rng.random() < 0.5:
                # A duration: the value now, the opposite once it is over.
                command += f" for {said}"
                calls += [Call(c.target, not c.value, k * size) for c in calls]
            else:
                command += f" in {said}"
                calls = [Call(c.target, c.value, k * size) for c in calls]
        if _learnable((command, state, calls)):
            cases.append((command, state, calls))
    return cases


def text(cases: list[Case]) -> str:
    """Join samples into training text, each exactly the prompt `act` builds
    and its answer."""
    return "".join(
        render_prompt(select(targets, command), command) + answer(calls)
        for command, targets, calls in cases
    )


def _example(
    targets: list[Target], e: dict
) -> tuple[str, dict[str, Target], list[Call]]:
    """Read an example as (command, {name: target with its state value},
    calls); raise ValueError if the model could not learn it from the targets
    `select` shows."""
    command, state = e["command"], e.get("state", {})
    calls = [Call(*c) for c in e["calls"]]
    fixed = {
        t.name: dataclasses.replace(t, value=state[t.name])
        for t in targets
        if t.name in state
    }
    listed = [fixed.get(t.name, t) for t in targets]
    out = list(answer(calls).encode())
    if not (
        len(fixed) == len(state)
        and _learnable((command, listed, calls))
        and parse(out, [1.0] * len(out), listed) == calls
    ):
        raise ValueError(f"example cannot be learnt: {command!r}")
    return command, fixed, calls


def _learnable(case: Case) -> bool:
    """Whether the prompt fits and shows every called target, and the answer fits."""
    command, targets, calls = case
    shown = select(targets, command)
    names = {t.name for t in shown}
    return (
        len(render_prompt(shown, command).encode()) <= PROMPT_BUDGET
        and all(c.target in names for c in calls)
        and len(answer(calls).encode()) <= N_NEW
    )


def _pool(t: Target, rng: random.Random) -> list[Target]:
    """The variants of `t` a sample's state draws from: every bool and string
    value, a hundred numbers."""
    if t.type == "bool":
        values: list = [False, True]
    elif t.type == "str":
        values = t.values or [t.value]
    else:
        values = [_draw(t, rng) for _ in range(100)]
    return [dataclasses.replace(t, value=v) for v in values]


def _draw(t: Target, rng: random.Random) -> Value:
    """A value for `t`; a number without a range stays within max(10, |value|)
    of its listed value."""
    if t.type == "bool":
        return rng.random() < 0.5
    if t.type == "str":
        return rng.choice(t.values or [t.value])
    spread = max(10.0, abs(float(t.value)))
    lo, hi = t.range or (float(t.value) - spread, float(t.value) + spread)
    if t.type == "int":
        return rng.randint(math.ceil(lo), math.floor(hi))
    return float(min(max(round(rng.uniform(lo, hi), rng.randint(0, 1)), lo), hi))


def _clause(targets: list[Target], rng: random.Random) -> tuple[str, list[Call]]:
    """Say one value for `targets` of one kind: "turn on the kitchen light and
    the bedroom light"."""
    w = " and ".join(_noun(rng.choice(_phrases(t)), rng) for t in targets)
    if targets[0].readonly:
        return rng.choice(_READ).format(w=w), [Call(t.name) for t in targets]
    value = _draw(targets[0], rng)
    if isinstance(value, bool):
        template = rng.choice(_ON if value else _OFF)
    else:
        template = rng.choice(_PICK if isinstance(value, str) else _SET)
    return template.format(w=w, v=_said(value)), [Call(t.name, value) for t in targets]


def _unlisted(targets: list[Target], things: list[str], rng: random.Random) -> str:
    """A command on a thing not listed, half the time in place of a listed
    target's last word or the words before it: "turn on the hallway light"."""
    thing = rng.choice(things)
    if rng.random() < 0.5:
        *head, last = rng.choice(targets).name.split("_")
        thing = rng.choice([f"{thing} {last}", " ".join([*head, thing])])
    template = rng.choice([*_ON, *_OFF, *_SET, *_READ])
    return template.format(w=_noun(thing, rng), v=rng.randint(1, 100))


def _phrases(t: Target) -> list[str]:
    """The name in words, and the description up to a comma if five words or
    fewer: "canopy", "awning over the patio"."""
    head = t.description.split(",")[0].strip().lower()
    return [t.name.replace("_", " "), *([head] if 0 < len(head.split()) <= 5 else [])]


def _noun(phrase: str, rng: random.Random) -> str:
    """`phrase` after "the", or bare a quarter of the time."""
    if phrase.split()[0] in _DETERMINERS or rng.random() < 0.25:
        return phrase
    return "the " + phrase


def _said(value: Value) -> str:
    """A value as a command says it: `21.5`, "up" for `UP`."""
    if isinstance(value, str):
        return value.lower().replace("_", " ")
    return f"{value:g}" if isinstance(value, float) else str(value)
