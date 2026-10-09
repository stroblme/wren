"""The library API: a house's state and a command in, device calls out.

    from wren import Device, House, act

    house = House("19:30", 6.0, dark=True, user_area="kitchen", areas={
        "kitchen": [Device("light", "light.kitchen_ceiling", "off")],
    })
    calls, confidence = act("turn on the lights in the kitchen", house)
    # [Call(tool='lights.on', args={'id': 'light.kitchen_ceiling'}, delay_s=None)]
    # and 0.96

Numpy only: nothing on this import path loads JAX. The caller describes
its house with its own device ids; the text the model reads is rendered
here, in the format the model was trained on (dataset.py renders its
training houses through `render_state` too).
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
    """One device: `kind` from KINDS, the caller's `id`, and its `state` —
    `on`/`off` (light, media, water), `up`/`down` (shutter) or whole
    degrees like `21C` (heat)."""

    kind: str
    id: str
    state: str


@dataclass
class House:
    """The state block: the clock (`HH:MM`), the outdoor temperature in °C
    (rendered rounded), whether it is dark, the area the user is in
    (None: away), and each area's devices, in display order."""

    time: str
    outdoor_c: float
    dark: bool
    user_area: str | None
    areas: dict[str, list[Device]]


@dataclass
class Call:
    """One tool call in the house's own terms, e.g. `lights.on` with
    `{"id": <a Device.id>}` or `sensors.temperature` with `{"area": <an
    area name>}`, to run `delay_s` seconds from now (None: immediately)."""

    tool: str
    args: dict
    delay_s: float | None = None


def render_state(house: House, areas: Iterable[str] | None = None) -> str:
    """The state block, with the lines of `areas` only (default: all), in
    house order:

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
    """The areas whose lines reach the context — the retrieval step that
    training simulates (dataset._keep).

    Needed are the areas the command names; failing that, the user's area
    and the living room (area-less commands resolve through the user line;
    scenes and coming home act on the living room). Other areas pad the
    block to `limit` lines, like the distractors training put beside the
    needed one, and are the first to go when the prompt would exceed
    `budget` bytes. Returned in house order.
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
    """The calls `command` asks for in `house`, and how sure the model is.

    The model reads aliases in the trained id shape instead of the
    caller's ids (it copies nothing else reliably) and its calls are
    mapped back. The confidence is the lowest probability it gave any
    byte of its answer. An answer that does not parse, or
    names a device or area not in the house, yields no calls. `model` is
    a pack from `runtime.load_pack`; default: the one the package ships.
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
    """The house with each device id replaced by a random one in the
    trained shape, and the map back. Seeded: a house always gets the same
    aliases, so the same request gets the same answer."""
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
    """A parsed call in the house's terms: aliases back to device ids,
    area slugs back to area names, `delay='20m'` to seconds."""
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
