"""Training samples from an installation's target list, and external datasets.

`corpus` generates samples for any installation's target list; `as_case`
reads an example-shaped record as one, and `from_log` turns a host's log
into examples. The external benchmarks (google/mobile-actions,
Home-Assistant-Requests) are downloaded on demand into `data/`.
"""

import dataclasses
import datetime
import json
import math
import pathlib
import random
import re
import urllib.request
from collections.abc import Callable, Sequence

# Aliased: the benchmark loaders below name a sample's text `prompt`.
from wren.targets import BUDGET as PROMPT_BUDGET
from wren.targets import N_NEW, Call, Target, Value, answer, parse, select
from wren.targets import prompt as render_prompt

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


# ------------------------------------------------- corpus from a target list

# A training sample: the command, every listed target with this sample's
# value, and the expected calls (none: the model abstains).
Case = tuple[str, list[Target], list[Call]]

# Generic phrasings: `{w}` is a target's words, `{v}` a value as said.
_ON = ["turn on {w}", "switch on {w}", "turn {w} on", "{w} on", "start {w}"]
_OFF = ["turn off {w}", "switch off {w}", "turn {w} off", "{w} off", "stop {w}"]
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


def as_case(targets: list[Target], record: dict) -> Case:
    """Read an example-shaped record (`corpus`'s `examples`) as a sample:
    its command, every target with the record's state or its listed value,
    and its calls. Raise ValueError if the model could not answer it."""
    command, fixed, calls = _example(targets, record)
    return command, [fixed.get(t.name, t) for t in targets], calls


def from_log(records: Sequence[dict]) -> list[dict]:
    """Turn a host's log into `examples` for `corpus` and `fit`.

    A record is an example (`state`, `command`, `calls`) of calls a larger
    model made, plus `kept`: whether the person kept them, not undoing them
    within a few minutes. A kept record becomes an example with its calls, an
    undone one an example with none, so the model passes the command on.
    """
    return [
        {
            "state": r.get("state", {}),
            "command": r["command"],
            "calls": r["calls"] if r["kept"] else [],
        }
        for r in records
    ]


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
