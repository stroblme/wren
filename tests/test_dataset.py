import random
import re
import urllib.request

import pytest

from wren import dataset
from wren.dataset import (
    AREALESS,
    _coming_home,
    _house,
    _misheard,
    _pair,
    _unknown,
    home,
    load_noisy,
    render_call,
    spoken,
    to_text,
)
from wren.eval import parse_calls


def test_home_is_deterministic():
    assert to_text(home(200, 40)[0]) == to_text(home(200, 40)[0])


def test_eval_ids_never_occur_in_training():
    train, evals = home(300, 60)
    train_text = to_text(train)
    for _, calls in evals:
        for _, args in calls:
            device = str(args.get("id", ""))
            if len(device) == 5:  # generated ids; slug fallbacks are longer
                assert device not in train_text


def test_every_argument_is_copyable_from_its_prompt():
    train, evals = home(300, 60)
    for prompt, calls in train + evals:
        for _, args in calls:
            for value in args.values():
                surface = str(value).replace("_", " ")
                assert str(value) in prompt or surface in prompt, (prompt, value)


def test_calls_parse_under_the_eval_protocol():
    train, _ = home(300, 0)
    for _, calls in train:
        rendered = "\n".join(render_call(name, args) for name, args in calls)
        assert parse_calls(rendered) == calls, rendered


def test_coming_home_rule_branches_on_state():
    house = _house(random.Random(0), set())
    living = house["areas"]["living room"]
    house["out"], house["dark"] = 3, True
    living["light"] = (living["light"][0], "off")
    _, calls = _coming_home(house, random.Random(1))
    assert [name for name, _ in calls] == ["heating.boost", "lights.on"]
    assert calls[1][1]["delay"].endswith("m")

    house["out"], house["dark"] = 25, False
    _, calls = _coming_home(house, random.Random(1))
    assert calls == []  # warm and bright: nothing to do


def test_mixture_covers_the_sample_kinds():
    train, _ = home(300, 0)
    assert any(not calls for _, calls in train)  # abstentions
    assert any(len(calls) > 1 for _, calls in train)  # scenes/implications
    assert any("user " not in prompt for prompt, _ in train)  # stateless


def test_prompts_fit_the_generation_budget():
    """Check that prompts fit context - n_new bytes without truncation."""
    train, evals = home(300, 60)
    assert max(len(p.encode()) for p, _ in train + evals) <= 288


def test_fetch_discards_an_interrupted_download(tmp_path, monkeypatch):
    monkeypatch.setattr(dataset, "DATA", tmp_path)

    def interrupted(url, path):
        path.write_text("partial")
        raise ConnectionError

    monkeypatch.setattr(urllib.request, "urlretrieve", interrupted)
    with pytest.raises(ConnectionError):
        dataset.fetch("mobile_actions")
    assert not (tmp_path / "mobile_actions" / "dataset.jsonl").exists()

    monkeypatch.setattr(
        urllib.request, "urlretrieve", lambda url, path: path.write_text("ok")
    )
    [path] = dataset.fetch("mobile_actions")
    assert path.read_text() == "ok"


def test_switches_have_a_letter_of_their_own():
    houses = [_house(random.Random(seed), set()) for seed in range(50)]
    ids = [
        d[0]
        for h in houses
        for devs in h["areas"].values()
        for k, d in devs.items()
        if k == "switch"
    ]
    assert ids and all(i.startswith("p") for i in ids)


def test_a_pair_calls_each_area_in_the_order_said():
    rng = random.Random(0)
    for _ in range(50):
        house = _house(rng, set())
        prompt, calls = _pair(house, rng)
        command = prompt.splitlines()[-1]
        assert len(calls) == 2 and calls[0][0] == calls[1][0]
        first, second = (
            next(
                a
                for a, devs in house["areas"].items()
                if i in {d[0] for d in devs.values()}
            )
            for i in (c[1]["id"] for c in calls)
        )
        assert command.index(first) < command.index(second)


def test_arealess_commands_go_to_the_living_room_when_away():
    phrases = {p for ps in AREALESS.values() for p in ps}
    train, _ = home(3000, 0)
    checked = 0
    for prompt, calls in train:
        *state, command = prompt.splitlines()
        if (
            "user away" in state
            and calls
            and re.sub(r" in \d+m$", "", command) in phrases
        ):
            living = next(line for line in state if line.startswith("living room:"))
            assert all(args["id"] in living for _, args in calls), prompt
            checked += 1
    assert checked > 20


def test_unknown_places_and_things_get_no_call():
    rng = random.Random(0)
    assert all(_unknown(_house(rng, set()), rng)[1] == [] for _ in range(50))


def test_misheard_swaps_the_command_and_keeps_state_and_calls():
    state = "12:00 out 5C bright\nuser away\nkitchen: light lab23 off\n"
    call = [("lights.on", {"id": "lab23"})]
    samples = [
        (state + "turn on x\n", call),
        (state + "turn on x in 5m\n", call),
        ("turn on x\n", []),
    ]
    noisy = {"turn on x": ["turn on ex"]}
    swapped, delayed, stateless = _misheard(samples, noisy, share=1.0, seed=0)
    assert swapped == (state + "turn on ex\n", call)
    assert delayed == (state + "turn on ex in 5 minutes\n", call)
    assert stateless == samples[2]


def test_load_noisy_drops_transcripts_that_lost_the_command(tmp_path):
    path = tmp_path / "noisy.jsonl"
    path.write_text(
        '{"command": "turn on the lights in the garage in 5m", '
        '"text": "turn on the lights in the gas in 5 minutes"}\n'
        '{"command": "water the patio", "text": "what a pity"}\n'
    )
    assert load_noisy(path) == {
        "turn on the lights in the garage in 5m": [
            "turn on the lights in the gas in 5 minutes"
        ]
    }
    assert spoken("lights on in 15m") == "lights on in 15 minutes"
