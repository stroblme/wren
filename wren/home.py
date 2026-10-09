"""NumPy API for generating device calls from a command and house state.

`act(command, house)` returns calls using the caller's device IDs and a
confidence score. `render_state` supplies the format used in training.
"""

import dataclasses
import functools
import random
import re
from collections.abc import Iterable
from dataclasses import dataclass

from wren import runtime, tokenizer
from wren.eval import parse_calls

# The closed vocabulary the shipped model was trained on: other area names
# and device kinds render fine but are outside what it has seen.
ROOMS = ["living room", "kitchen", "bathroom", "bedroom", "office"]
OUTSIDE = ["garden", "patio", "garage", "driveway", "balcony"]
AREAS = ROOMS + OUTSIDE
KINDS = ["light", "media", "heat", "shutter", "water"]
# Trained device ids are the kind's initial plus four of these.
ID_CHARS = "abcdefghijkmnpqrstuvwxyz23456789"  # no 0/O/1/l confusables

N_NEW = 96  # bytes reserved for the answer; two calls take ~70
BUDGET = 384 - N_NEW  # prompt bytes that fit beside them in the context


@dataclass
class Device:
    """Device kind, caller-provided ID, and state.

    Use a kind from `KINDS`; states are on/off for light, media, and water,
    up/down for shutters, and whole degrees such as `21C` for heating.
    """

    kind: str
    id: str
    state: str


@dataclass
class House:
    """House state with an HH:MM clock, outdoor temperature in °C, and devices.

    `user_area=None` means away. Areas retain display order; temperature is
    rounded when rendered.
    """

    time: str
    outdoor_c: float
    dark: bool
    user_area: str | None
    areas: dict[str, list[Device]]


@dataclass
class Call:
    """Tool name and arguments using the house's device IDs and area names.

    `delay_s` is the delay in seconds; None means run immediately.
    """

    tool: str
    args: dict
    delay_s: float | None = None


def render_state(house: House, areas: Iterable[str] | None = None) -> str:
    """Render state for selected `areas` in house order, or all areas by default.

    19:30 out 6C dark
    user kitchen
    kitchen: light lk3m9 off, shutter sv8xr up
    """
    keep = house.areas if areas is None else set(areas)
    lines = [
        f"{house.time} out {round(house.outdoor_c)}C {'dark' if house.dark else 'bright'}",
        f"user {house.user_area or 'away'}",
    ]
    for area, devices in house.areas.items():
        if area in keep:
            lines.append(
                f"{area}: " + ", ".join(f"{d.kind} {d.id} {d.state}" for d in devices)
            )
    return "\n".join(lines) + "\n"


def select_areas(
    house: House, command: str, limit: int = 3, budget: int = BUDGET
) -> list[str]:
    """Select areas for the prompt, returning them in house order.

    Keep named areas, or the user's area and living room if none are named.
    Add other areas up to `limit`, removing those extras to meet `budget` bytes.
    """
    text = command.lower()
    named = [a for a in house.areas if re.search(rf"\b{re.escape(a.lower())}\b", text)]
    needed = named or [a for a in house.areas if a in (house.user_area, "living room")]
    others = [a for a in house.areas if a not in needed]
    keep = needed + others[: max(limit - len(needed), 0)]
    while len(keep) > len(needed) and _prompt_bytes(house, keep, command) > budget:
        keep.pop()
    return [a for a in house.areas if a in keep]


def act(
    command: str, house: House, *, model: dict | None = None
) -> tuple[list[Call], float]:
    """Return calls for `command` in `house` and the lowest answer-byte probability.

    Device IDs are aliased to the training format and mapped back in the calls.
    Return no calls if the answer fails to parse or references an unknown device
    or area. `model` is a pack from `runtime.load_pack`; default to the shipped pack.
    """
    pack = model or _packaged()
    aliased, ids = _aliased(house)
    prompt = render_state(aliased, select_areas(aliased, command)) + command + "\n"
    out, probs = runtime.generate(pack, tokenizer.encode(prompt), N_NEW)
    areas = {a.replace(" ", "_"): a for a in house.areas}
    try:
        parsed = parse_calls(tokenizer.decode(out)) or []
        calls = [_call(name, args, ids, areas) for name, args in parsed]
    except (KeyError, ValueError):
        calls = []
    return calls, min(probs)


@functools.cache
def _packaged() -> dict:
    return runtime.load_pack(runtime.MODEL)


def _prompt_bytes(house: House, areas: list[str], command: str) -> int:
    return len((render_state(house, areas) + command + "\n").encode())


def _aliased(house: House) -> tuple[House, dict[str, str]]:
    """Return a house with training-shaped device aliases and the reverse ID map.

    A fixed seed gives each house stable aliases.
    """
    rng = random.Random(0)
    ids: dict[str, str] = {}
    areas = {}
    for area, devices in house.areas.items():
        areas[area] = []
        for d in devices:
            while (alias := d.kind[0] + "".join(rng.choices(ID_CHARS, k=4))) in ids:
                pass
            ids[alias] = d.id
            areas[area].append(Device(d.kind, alias, d.state))
    return dataclasses.replace(house, areas=areas), ids


def _call(tool: str, args: dict, ids: dict[str, str], areas: dict[str, str]) -> Call:
    """Map aliases to device IDs, slugs to area names, and delays to seconds."""
    if "id" in args:
        args["id"] = ids[args["id"]]
    if "area" in args:
        args["area"] = areas[args["area"]]
    delay = args.pop("delay", None)
    if delay is None:
        return Call(tool, args)
    minutes = re.fullmatch(r"(\d+)m", str(delay))
    if not minutes:
        raise ValueError(f"unreadable delay {delay!r}")
    return Call(tool, args, int(minutes[1]) * 60.0)
