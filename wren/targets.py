"""Domain-free targets, prompt and answer format, and `act`.

An installation lists its targets: things a command may set, and readings it
may ask about. The prompt shows the targets a command concerns with their
current values, then the command:

    centrifuge_run=false
    centrifuge_rpm=3000
    spin the centrifuge at 4000 rpm for 10 minutes

The answer is one call per line and a blank line, or only the blank line when
the command is not handled. A call sets a target or reads it, optionally after
a delay in whole hours, minutes or seconds:

    >> set(centrifuge_rpm, 4000)
    >> set(centrifuge_run, true)
    >> set(centrifuge_run, false) in 10m
"""

import functools
import re
from dataclasses import dataclass

from wren import runtime, tokenizer

N_NEW = 96  # bytes reserved for the answer
BUDGET = 384 - N_NEW  # prompt bytes beside them in the shipped context

Value = bool | int | float | str

# Words a description and a command share without naming the same target
# (written as two sets to keep the formatter from one word per line).
_STOP = {"a", "an", "and", "at", "down", "for", "in", "is", "it", "of", "off", "on"}
_STOP |= {"or", "the", "to", "up", "with"}
_LINE = re.compile(r">> (?:read\((\w+)\)|set\((\w+), (.+)\))(?: in (\d+)([hms]))?")
_UNITS = {"h": 3600, "m": 60, "s": 1}


@dataclass
class Target:
    """Something a command may set or, if `readonly`, only ask about.

    `type` is "bool", "int", "float" or "str". `values` lists the allowed
    strings and `range` the inclusive [low, high] of a number. The
    description's words help select the target; `unit` is for the caller.
    """

    name: str  # snake_case
    type: str
    value: Value
    values: list[str] | None = None
    range: list[float] | None = None
    description: str = ""
    unit: str | None = None
    readonly: bool = False


@dataclass
class Call:
    """Set `target` to `value`, or read it when `value` is None.

    `delay_s` is None to run now. `confidence` is the lowest probability of
    the bytes on the call's answer line and the answer's closing blank line.
    """

    target: str
    value: Value | None = None
    delay_s: float | None = None
    confidence: float = 1.0


def prompt(targets: list[Target], command: str) -> str:
    """Render one `name=value` line per target, then the command."""
    return "".join(_line(t) for t in targets) + command + "\n"


def select(targets: list[Target], command: str, budget: int = BUDGET) -> list[Target]:
    """Return the targets that share a word with `command`, in list order.

    Words come from a target's name (split on `_`) and description. The
    targets sharing the most words are kept while the prompt fits `budget`
    bytes; ties go to list order. No shared word gives no targets.
    """
    words = _words(command)
    scores = [len(words & _words(f"{t.name} {t.description}")) for t in targets]
    ranked = sorted((i for i, s in enumerate(scores) if s), key=lambda i: -scores[i])
    used, keep = len(prompt([], command).encode()), set()
    for i in ranked:
        used += len(_line(targets[i]).encode())
        if used > budget:
            break
        keep.add(i)
    return [t for i, t in enumerate(targets) if i in keep]


def answer(calls: list[Call]) -> str:
    """Render calls as answer lines ending in a blank line."""
    lines = []
    for c in calls:
        line = (
            f">> read({c.target})"
            if c.value is None
            else f">> set({c.target}, {_spell(c.value)})"
        )
        if c.delay_s is not None:
            line += f" in {_delay(c.delay_s)}"
        lines.append(line + "\n")
    return "".join(lines) + "\n"


def parse(out: list[int], probs: list[float], targets: list[Target]) -> list[Call]:
    """Parse generated answer bytes into calls on `targets`.

    Any line that does not parse, names an unlisted target, sets a read-only
    one or gives a value outside its type, `values` or `range` rejects the
    whole answer, as does an answer cut off before its blank line.
    """
    data = bytes(out)
    if not (data == b"\n" or data.endswith(b"\n\n")):
        return []
    by_name = {t.name: t for t in targets}
    calls, start = [], 0
    try:
        while start < len(data) - 1:  # the last byte is the blank line
            end = data.index(b"\n", start)
            m = _LINE.fullmatch(data[start:end].decode())
            if not m:
                return []
            read, set_, raw, n, unit = m.groups()
            target = by_name.get(read or set_)
            if target is None or (set_ and target.readonly):
                return []
            calls.append(
                Call(
                    target.name,
                    None if raw is None else _value(target, raw),
                    None if n is None else int(n) * _UNITS[unit],
                    min(*probs[start : end + 1], probs[-1]),
                )
            )
            start = end + 1
    except ValueError:
        return []
    return calls


def act(
    command: str, targets: list[Target], *, model: dict | None = None
) -> list[Call]:
    """Return calls for `command` on `targets`; an empty list means not handled.

    `model` is a pack from `runtime.load_pack`; default to the shipped pack.
    """
    pack = model or _packaged()
    shown = select(targets, command, pack["cfg"]["context"] - N_NEW)
    out, probs = runtime.generate(pack, tokenizer.encode(prompt(shown, command)), N_NEW)
    return parse(out, probs, targets)


@functools.cache
def _packaged() -> dict:
    return runtime.load_pack(runtime.MODEL)


def _line(t: Target) -> str:
    return f"{t.name}={_spell(t.value)}\n"


def _spell(value: Value) -> str:
    """Spell a value the way prompts and answers write it: `true`, `21.5`, `UP`."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return f"{value:g}" if isinstance(value, float) else str(value)


def _delay(seconds: float) -> str:
    """Spell a delay in the largest whole unit: 7200 -> `2h`, 90 -> `90s`."""
    n = round(seconds)
    unit = next(u for u, size in _UNITS.items() if n % size == 0)
    return f"{n // _UNITS[unit]}{unit}"


def _value(target: Target, raw: str) -> Value:
    """Read an answer's value for `target`; raise ValueError if not allowed."""
    if target.type == "bool":
        if raw not in ("true", "false"):
            raise ValueError(raw)
        return raw == "true"
    if target.type == "str":
        if target.values is not None and raw not in target.values:
            raise ValueError(raw)
        return raw
    if not re.fullmatch(r"-?\d+" if target.type == "int" else r"-?\d+(\.\d+)?", raw):
        raise ValueError(raw)
    number = int(raw) if target.type == "int" else float(raw)
    if target.range and not target.range[0] <= number <= target.range[1]:
        raise ValueError(raw)
    return number


@functools.lru_cache(maxsize=4096)  # target words repeat on every `select`
def _words(text: str) -> frozenset[str]:
    """Lowercase words without function words, a plural `s` removed."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return frozenset(
        w[:-1] if len(w) > 3 and w.endswith("s") else w for w in words if w not in _STOP
    )
