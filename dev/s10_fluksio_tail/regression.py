"""Score a packed model on the 7B panel set and the s10 probes, as fluksio runs it.

    uv run python dev/s10_fluksio_tail/regression.py [wren.bin ...]
    uv run python dev/s10_fluksio_tail/regression.py --generated [wren.bin ...]

Each command goes through `act` with a live-like house, after the node's own
steps (lower case, no sentence marks, digits, the house's aliases). An outcome
at fluksio's threshold is `ok` (applied and right), `WRONG` (applied and not
what was asked) or `-` (not applied, so the assistant takes it). A command
whose right answer is no call is `ok` when nothing is applied.

`--generated` scores held-out generated samples instead, per s10 builder and
on s8's own evaluation split (its generator as of v1.0.0, in data/s10/), through
the same packed runtime.
"""

import importlib.util
import itertools
import json
import pathlib
import random
import re
import subprocess
import sys

from wren import Device, House, act, runtime, tokenizer
from wren import dataset as s10
from wren.eval import parse_calls
from wren.home import AREAS

HERE = pathlib.Path(__file__).parent
THRESHOLD = 0.93
#: The aliases fluksio's live house sets on its wren node (Phase A).
ALIASES = {
    "outside": "in the garden",
    "page too": "patio",
    "traverse": "living room light",
}


def _house() -> House:
    """Every area lit, as on live, and the devices the panel commands name."""
    extra = {
        "living room": [("media", "off"), ("heat", "21C"), ("shutter", "up")],
        "kitchen": [("switch", "off")],
        "bathroom": [("heat", "20C")],
        "bedroom": [("heat", "19C"), ("shutter", "up")],
        "office": [("shutter", "up")],
        "garden": [("water", "off")],
        "patio": [("water", "off")],
        "balcony": [("water", "on")],
    }
    lit = {"bedroom", "office", "patio"}
    areas = {
        a: [Device("light", f"{a}:light", "on" if a in lit else "off")]
        + [Device(k, f"{a}:{k}", s) for k, s in extra.get(a, [])]
        for a in AREAS
    }
    return House("20:30", 8.0, True, None, areas)  # no presence wired: away


HOUSE = _house()


def call(tool: str, area: str = "", kind: str = "", delay: float | None = None, **args):
    if kind:
        args["id"] = f"{area}:{kind}"
    elif area:
        args["area"] = area
    return (tool, args, delay)


# (group, command as said, the calls it means). Groups follow s10's tail.
REAL = [
    ("direct", call("lights.on", "kitchen", "light")),
    ("direct", call("lights.off", "living room", "light")),
    ("direct", call("lights.off", "bedroom", "light")),
    ("direct", call("lights.on", "office", "light")),
    ("values", call("heating.set", "bathroom", "heat", temp=22)),
    ("direct", call("heating.boost", "bedroom", "heat")),
    ("values", call("heating.set", "living room", "heat", temp=19)),
    ("direct", call("shutters.down", "bedroom", "shutter")),
    ("direct", call("shutters.up", "living room", "shutter")),
    ("direct", call("shutters.down", "office", "shutter")),
    ("direct", call("media.stop", "living room", "media")),
    ("direct", call("media.play", "living room", "media")),
    ("absent", None),  # music on the patio: it has no media
    ("direct", call("watering.start", "garden", "water")),
    ("values", None),  # water for ten minutes: no duration in the grammar
    ("direct", call("watering.stop", "balcony", "water")),
    ("direct", call("lights.on", "driveway", "light")),
    ("values", call("lights.on", "garage", "light", 300.0)),
    ("values", call("lights.off", "patio", "light", 1200.0)),
    ("scope", None),  # "everything off": the node's group rule, before the model
    ("direct", call("lights.on", "kitchen", "light")),
    ("direct", call("heating.boost", "bathroom", "heat")),
    ("absent", None),  # shutters on the balcony: it has none
    ("question", call("sensors.temperature", "garden")),  # "outside", aliased
    ("question", None),  # is the garage light still on: no read call for it
    ("question", call("sensors.temperature", "bedroom")),
    ("abstract", None),
    ("abstract", None),
]

# Typed commands for what s10 adds, in the same house.
PROBES = [
    (
        "pair",
        "turn off the lights in the kitchen and the bathroom",
        [
            call("lights.off", "kitchen", "light"),
            call("lights.off", "bathroom", "light"),
        ],
    ),
    (
        "pair",
        "switch on the bedroom and the office lights",
        [call("lights.on", "bedroom", "light"), call("lights.on", "office", "light")],
    ),
    (
        "pair",
        "close the blinds in the bedroom and the office",
        [
            call("shutters.down", "bedroom", "shutter"),
            call("shutters.down", "office", "shutter"),
        ],
    ),
    (
        "arealess",
        "set the heating to 22 degrees",
        [call("heating.set", "living room", "heat", temp=22)],
    ),
    ("arealess", "turn on the light", [call("lights.on", "living room", "light")]),
    (
        "arealess",
        "it's a bit cold in here",
        [call("heating.boost", "living room", "heat")],
    ),
    ("arealess", "play some music", [call("media.play", "living room", "media")]),
    ("unknown", "turn on the hall light", []),
    ("unknown", "turn on the attic light", []),
    ("unknown", "switch on the appliances", []),
    ("unknown", "turn off the fan in the bedroom", []),
    ("unknown", "open the garage door", []),
    (
        "switch",
        "turn on the plug in the kitchen",
        [call("switches.on", "kitchen", "switch")],
    ),
    (
        "switch",
        "switch off the appliances in the kitchen",
        [call("switches.off", "kitchen", "switch")],
    ),
    (
        "direct",
        "what is the temperature in the garden",
        [call("sensors.temperature", "garden")],
    ),
    (
        "values",
        "turn on the garden light in 5 minutes",
        [call("lights.on", "garden", "light", 300.0)],
    ),
]


def node_text(text: str) -> str:
    """What the node hands the model: plain, digits (speech.plain), aliases."""
    sys.path.insert(0, str(HERE))
    from speech import plain

    text = plain(text)
    for word in sorted(ALIASES, key=len, reverse=True):
        text = re.sub(rf"\b{re.escape(word)}\b", ALIASES[word], text)
    return text


def outcome(pack: dict, text: str, gold: list) -> tuple[str, float, list]:
    calls, confidence = act(node_text(text), HOUSE, model=pack)
    got = [(c.tool, c.args, c.delay_s) for c in calls]
    applied = bool(got) and confidence >= THRESHOLD
    if not applied:
        return ("ok" if not gold else "-"), confidence, got
    return ("ok" if got == gold else "WRONG"), confidence, got


def main(packs: list[str]) -> None:
    real = json.loads(pathlib.Path("data/s10/real.json").read_text())
    rows = [
        (g, r["said"], r["heard"], [c] if c else []) for (g, c), r in zip(REAL, real)
    ]
    rows += [(g, t, t, gold) for g, t, gold in PROBES]
    for path in packs:
        pack = runtime.load_pack(path)
        counts: dict[tuple[str, str], dict[str, int]] = {}
        heard_rows: list[tuple[bool, float, bool, bool]] = []
        print(f"== {path}")
        for group, said, heard, gold in rows:
            line = [f"{group:8}"]
            for source, text in (("said", said), ("heard", heard)):
                result, conf, got = outcome(pack, text, gold)
                if source == "heard":
                    heard_rows.append((got == gold, conf, bool(got), bool(gold)))
                cell = counts.setdefault((group, source), {"ok": 0, "WRONG": 0, "-": 0})
                cell[result] += 1
                line.append(f"{result:5} {conf:.2f}")
                if source == "heard":
                    line.append(f"{node_text(text)!r} -> {got}")
            print("  ".join(line))
        print("group     said ok/wrong/abstained   heard ok/wrong/abstained")
        for group in dict.fromkeys(g for g, *_ in rows):
            s, h = counts[(group, "said")], counts[(group, "heard")]
            print(
                f"{group:8}  {s['ok']:2}/{s['WRONG']:2}/{s['-']:2}"
                f"                  {h['ok']:2}/{h['WRONG']:2}/{h['-']:2}"
            )
        print("threshold sweep, heard and probes: ok/wrong/abstained")
        for threshold in SWEEP:
            print(f"  {threshold:<5} {tally(heard_rows, threshold)}")


def _s8_generator():
    """The generator s8 was trained on: wren/dataset.py at v1.0.0."""
    path = pathlib.Path("data/s10/dataset_s8.py")
    if not path.exists():
        path.write_text(
            subprocess.check_output(
                ["git", "show", "v1.0.0:wren/dataset.py"], text=True
            )
        )
    spec = importlib.util.spec_from_file_location("dataset_s8", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evalsets(n: int = 80) -> dict[str, list]:
    """Held-out samples per builder (fresh houses, no training id), and s8's split."""
    train, mixed = s10.home(256000, 400)
    seen = set(re.findall(r"\b[lmhswp][a-z2-9]{4}\b", s10.to_text(train)))
    rng = random.Random(1)
    arealess = {p for ps in s10.AREALESS.values() for p in ps} | {"set the heating to"}
    sets: dict[str, list] = {"s10 mixed": mixed}
    for build in (s10._command, s10._reading, s10._pair, s10._unknown, s10._missing):
        for _ in range(n):
            prompt, calls = build(s10._house(rng, seen), rng)
            if len(prompt.encode()) > s10.BUDGET:
                continue
            command = re.sub(r" in \d+m$", "", prompt.splitlines()[-1])
            name = build.__name__[1:]
            if name == "command" and any(command.startswith(a) for a in arealess):
                name = "arealess"
            sets.setdefault(name, []).append((prompt, calls))
    sets["s8 split"] = _s8_generator().home(256000, 400)[1]
    return sets


#: The thresholds the sweep tries for fluksio's min_confidence.
SWEEP = (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.93)


def tally(rows: list, threshold: float) -> str:
    """ok/wrong/abstained shares of (right, confidence, called, wanted) rows."""
    ok = wrong = abstain = 0
    for right, confidence, called, wanted in rows:
        if called and confidence >= threshold:
            ok, wrong = ok + right, wrong + (not right)
        else:
            ok, abstain = ok + (not wanted), abstain + wanted
    k = len(rows)
    return f"{ok / k:4.0%}/{wrong / k:3.0%}/{abstain / k:3.0%}"


def scored(path: str, sets: dict[str, list]) -> dict[str, list]:
    """(right, confidence, called, wanted) per sample, cached beside the pack."""
    cache = pathlib.Path(path).with_suffix(".generated.json")
    if cache.exists():
        return json.loads(cache.read_text())
    pack = runtime.load_pack(path)
    out: dict[str, list] = {}
    for name, samples in sets.items():
        out[name] = []
        for prompt, ref in samples:
            o, probs = runtime.generate(pack, tokenizer.encode(prompt), 96)
            text = tokenizer.decode(o)
            got = parse_calls("\n".join(itertools.takewhile(bool, text.split("\n"))))
            out[name].append((got == ref, min(probs), bool(got), bool(ref)))
    cache.write_text(json.dumps(out))
    return out


def generated(packs: list[str]) -> None:
    sets = evalsets()
    results = {p: scored(p, sets) for p in packs}
    names = "  ".join(f"{pathlib.Path(p).stem:>22}" for p in packs)
    print(f"{'set':10} {'n':>4}  {names}")
    print(
        f"{'':10} {'':>4}  " + "  ".join(f"{'exact  ok/wrong/abst':>22}" for _ in packs)
    )
    for name, samples in sets.items():
        cells = [
            f"{sum(r[0] for r in results[p][name]) / len(samples):6.1%}  "
            f"{tally(results[p][name], THRESHOLD)}"
            for p in packs
        ]
        print(f"{name:10} {len(samples):4}  " + "  ".join(f"{c:>22}" for c in cells))
    builders = [n for n in sets if n not in ("s10 mixed", "s8 split")]
    for title, group in (("s10 builders", builders), ("s8 split", ["s8 split"])):
        print(f"\nthreshold sweep, {title}: ok/wrong/abstained")
        for threshold in SWEEP:
            cells = [
                tally([r for n in group for r in results[p][n]], threshold)
                for p in packs
            ]
            print(f"{threshold:<15}  " + "  ".join(f"{c:>22}" for c in cells))


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[:1] == ["--generated"]:
        generated(args[1:] or [str(runtime.MODEL)])
    else:
        main(args or [str(runtime.MODEL)])
