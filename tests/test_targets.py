"""Check the generic target format: prompt, selection, answer parsing and `act`."""

import json
import subprocess
import sys

import pytest

from wren import runtime
from wren.targets import N_NEW, Call, Target, act, answer, parse, prompt, select

HOME = [Target(**d) for d in json.loads(runtime.DEMO.read_text())["targets"]]
LAB = [
    Target("centrifuge_run", "bool", False),
    Target("centrifuge_rpm", "int", 3000, range=[500, 15000]),
    Target("stirrer_rpm", "int", 0, range=[0, 2000], description="magnetic stirrer"),
    Target("ph", "float", 7.04, readonly=True),
    Target("pump_mode", "str", "IDLE", values=["IDLE", "FILL", "DRAIN"]),
]


def names(targets: list[Target]) -> list[str]:
    return [t.name for t in targets]


def test_import_does_not_load_jax():
    code = "import sys, wren; from wren import act; assert 'jax' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_example_home_loads():
    assert len(HOME) == 20
    assert len(set(names(HOME))) == 20


def test_prompt_spells_values():
    shown = [HOME[1], HOME[7], HOME[12], HOME[17]]
    assert prompt(shown, "close the canopy") == (
        "kitchen_light=false\n"
        "living_room_heating=19\n"
        "canopy=UP\n"
        "outdoor_temperature=6.2\n"
        "close the canopy\n"
    )
    assert prompt([], "order a pizza") == "order a pizza\n"


def test_select_named_target():
    shown = names(select(HOME, "turn on the kitchen light"))
    assert "kitchen_light" in shown
    assert "garden_watering" not in shown
    assert shown == [n for n in names(HOME) if n in shown]  # list order


def test_select_description_and_plural():
    assert names(select(HOME, "open the awning")) == ["canopy"]
    assert names(select(HOME, "shutters down")) == [
        "kitchen_shutter",
        "bedroom_shutter",
        "user_location",
    ]
    assert names(select(LAB, "what is the ph")) == ["ph"]


def test_select_group_target():
    shown = names(select(HOME, "turn off all lights"))
    assert "all_lights" in shown
    # The group target outranks the single lights when only one line fits.
    command = "turn off all lights"
    budget = len(command) + 1 + len("all_lights=false\n")
    assert names(select(HOME, command, budget)) == ["all_lights"]


def test_select_target_outside_the_list():
    # No hallway: the lights are shown and the model has to abstain.
    shown = names(select(HOME, "turn on the hallway light"))
    assert "kitchen_light" in shown
    assert not any("hallway" in n for n in shown)
    assert select(HOME, "order a pizza") == []


def test_select_keeps_the_most_relevant_within_budget():
    command = "turn on the kitchen light"
    base = len(command) + 1
    one = base + len("kitchen_light=false\n")
    two = one + len("living_room_light=false\n")
    assert names(select(HOME, command, one)) == ["kitchen_light"]
    # Ties go to list order, and the shown lines are in list order.
    assert names(select(HOME, command, two)) == ["living_room_light", "kitchen_light"]
    assert select(HOME, command, base) == []
    for budget in (40, 100, 288):
        text = prompt(select(HOME, command, budget), command)
        assert len(text.encode()) <= budget


def canned(
    text: str, low: dict[int, float] | None = None
) -> tuple[list[int], list[float]]:
    """Answer bytes with probability 0.99, or `low[i]` at byte i."""
    out = list(text.encode())
    probs = [0.99] * len(out)
    for i, p in (low or {}).items():
        probs[i] = p
    return out, probs


def test_parse_valid_answer():
    out, probs = canned(
        ">> set(kitchen_light, true)\n"
        ">> set(canopy, DOWN) in 2h\n"
        ">> set(living_room_heating, 21.5) in 90s\n"
        ">> set(living_room_volume, 40)\n"
        ">> read(outdoor_temperature) in 15m\n\n",
        low={30: 0.4},
    )
    assert parse(out, probs, HOME) == [
        Call("kitchen_light", True, None, 0.99),
        Call("canopy", "DOWN", 7200, 0.4),
        Call("living_room_heating", 21.5, 90, 0.99),
        Call("living_room_volume", 40, None, 0.99),
        Call("outdoor_temperature", None, 900, 0.99),
    ]


def test_parse_confidence_is_per_line():
    text = ">> set(kitchen_light, true)\n>> set(all_lights, false)\n\n"
    first = len(">> set(kitchen_light, true)\n")
    # The line's own newline counts, and the closing blank line counts for
    # every call: it is the model's certainty that the answer is complete.
    out, probs = canned(text, low={first - 1: 0.7, len(text) - 1: 0.8})
    assert [c.confidence for c in parse(out, probs, HOME)] == [0.7, 0.8]


def test_parse_abstention():
    assert parse(*canned("\n"), HOME) == []


@pytest.mark.parametrize(
    "line",
    [
        ">> set(hallway_light, true)",  # unknown target
        ">> set(ph, 7.0)",  # read-only
        ">> set(centrifuge_run, on)",  # bool spelling
        ">> set(centrifuge_rpm, 99000)",  # out of range
        ">> set(centrifuge_rpm, 3000.5)",  # not an int
        ">> set(pump_mode, BOIL)",  # not an allowed value
        ">> set(centrifuge_run)",  # set without a value
        ">> read(ph, 7)",  # read with a value
        ">> set(centrifuge_run, true) in 2 hours",  # delay spelling
        "set(centrifuge_run, true)",  # no marker
    ],
)
def test_parse_rejects_the_whole_answer(line):
    out, probs = canned(f">> set(stirrer_rpm, 300)\n{line}\n\n")
    assert parse(out, probs, LAB) == []


def test_parse_rejects_a_cut_off_answer():
    out, probs = canned(">> set(stirrer_rpm, 300)\n>> set(pump_m")
    assert parse(out, probs, LAB) == []
    out, probs = canned(">> set(stirrer_rpm, 300)\n")  # no blank line yet
    assert parse(out, probs, LAB) == []


def test_answer_round_trips():
    calls = [
        Call("centrifuge_run", True),
        Call("stirrer_rpm", 300, delay_s=600),
        Call("ph"),
        Call("pump_mode", "DRAIN", delay_s=7200),
        Call("centrifuge_run", False, delay_s=45),
    ]
    text = answer(calls)
    assert text == (
        ">> set(centrifuge_run, true)\n"
        ">> set(stirrer_rpm, 300) in 10m\n"
        ">> read(ph)\n"
        ">> set(pump_mode, DRAIN) in 2h\n"
        ">> set(centrifuge_run, false) in 45s\n\n"
    )
    out = list(text.encode())
    assert parse(out, [1.0] * len(out), LAB) == calls
    assert answer([]) == "\n"


def test_act_with_a_stubbed_model(monkeypatch):
    seen = []

    def generate(pack, tokens, n_new):
        seen.append(bytes(tokens).decode())
        return canned(">> set(centrifuge_run, true)\n>> read(ph)\n\n", low={5: 0.8})

    monkeypatch.setattr("wren.runtime.generate", generate)
    calls = act("start the centrifuge", LAB, model={"cfg": {"context": 384}})
    assert seen == ["centrifuge_run=false\ncentrifuge_rpm=3000\nstart the centrifuge\n"]
    # `ph` was not shown but is listed, so its call stands.
    assert calls == [
        Call("centrifuge_run", True, None, 0.8),
        Call("ph", None, None, 0.99),
    ]


def test_act_fits_the_prompt_to_the_pack_context(monkeypatch):
    seen = []

    def generate(pack, tokens, n_new):
        seen.append(len(tokens))
        return canned("\n")

    monkeypatch.setattr("wren.runtime.generate", generate)
    assert act("turn off all lights", HOME, model={"cfg": {"context": 160}}) == []
    assert 0 < seen[0] <= 160 - N_NEW
