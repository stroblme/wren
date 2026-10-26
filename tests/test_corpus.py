"""Check the corpus generated from a target list."""

import json
import re

import pytest

from wren import runtime
from wren.dataset import as_case, corpus, from_log, text
from wren.targets import BUDGET, N_NEW, Call, Target, answer, parse, prompt, select

HOME_JSON = json.loads(runtime.DEMO.read_text())
HOME = [Target(**d) for d in HOME_JSON["targets"]]
EXAMPLES = HOME_JSON["examples"]
LAB = [
    Target("centrifuge_run", "bool", False),
    Target("centrifuge_rpm", "int", 3000, range=[500, 15000]),
    Target("stirrer_rpm", "int", 0, range=[0, 2000], description="magnetic stirrer"),
    Target("ph", "float", 7.04, readonly=True),
    Target("pump_mode", "str", "IDLE", values=["IDLE", "FILL", "DRAIN"]),
]


@pytest.fixture(scope="module")
def home():
    return corpus(HOME, n=3000, seed=0, examples=EXAMPLES)


@pytest.fixture(scope="module")
def lab():
    return corpus(LAB, n=1000, seed=0)


def words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def test_corpus_is_deterministic():
    assert text(corpus(HOME, n=300, seed=1, examples=EXAMPLES)) == text(
        corpus(HOME, n=300, seed=1, examples=EXAMPLES)
    )
    assert text(corpus(LAB, n=300, seed=1)) != text(corpus(LAB, n=300, seed=2))


def test_held_out_phrasings_do_not_overlap_training():
    train = corpus(LAB, n=2000, seed=0)
    held = corpus(LAB, n=500, seed=1, held_out=True)
    assert not {c for c, _, _ in train} & {c for c, _, _ in held}
    assert any(calls for _, _, calls in held)
    assert any(not calls for _, _, calls in held)


def test_corpus_excludes_reserved_commands_even_after_paraphrasing():
    reserved = {"please start the centrifuge"}
    samples = corpus(
        LAB,
        n=100,
        seed=0,
        exclude=reserved,
        paraphrase=lambda c: ["please start the centrifuge"],
    )
    assert not {c for c, _, _ in samples} & reserved


def test_text_is_prompt_and_answer(lab):
    expected = "".join(prompt(select(t, c), c) + answer(calls) for c, t, calls in lab)
    assert text(lab) == expected
    assert len(lab) == 1000


@pytest.mark.parametrize("name", ["home", "lab"])
def test_every_sample_is_learnable(name, request):
    listed = {"home": HOME, "lab": LAB}[name]
    for command, targets, calls in request.getfixturevalue(name):
        assert [t.name for t in targets] == [t.name for t in listed]
        shown = select(targets, command)
        assert {c.target for c in calls} <= {t.name for t in shown}
        assert len(prompt(shown, command).encode()) <= BUDGET
        out = list(answer(calls).encode())
        assert len(out) <= N_NEW
        assert parse(out, [1.0] * len(out), targets) == calls


@pytest.mark.parametrize("name", ["home", "lab"])
def test_every_target_and_value_is_covered(name, request):
    listed = {"home": HOME, "lab": LAB}[name]
    said = {
        (c.target, c.value)
        for _, _, calls in request.getfixturevalue(name)
        for c in calls
    }
    for t in listed:
        if t.readonly:
            assert (t.name, None) in said
        elif t.type == "bool":
            assert {(t.name, True), (t.name, False)} <= said
        elif t.type == "str":
            assert {(t.name, v) for v in t.values or []} <= said
        else:
            assert any(name == t.name for name, _ in said)


@pytest.mark.parametrize("name", ["home", "lab"])
def test_state_is_drawn_per_sample(name, request):
    listed = {"home": HOME, "lab": LAB}[name]
    seen: dict[str, set] = {t.name: set() for t in listed}
    for _, targets, _ in request.getfixturevalue(name):
        for t in targets:
            seen[t.name].add(t.value)
            if t.values:
                assert t.value in t.values
            if t.range:
                assert t.range[0] <= t.value <= t.range[1]
    for t in listed:
        if t.type == "bool":
            assert seen[t.name] == {True, False}
        elif t.type == "str" and t.values:
            assert seen[t.name] == set(t.values)
        elif t.type in ("int", "float"):
            assert len(seen[t.name]) > 10


def test_multi_target_and_delays(home):
    assert any(len(calls) == 3 for _, _, calls in home)
    assert any(len({c.value for c in calls}) == 2 for _, _, calls in home)
    delayed = [(c, calls) for c, _, calls in home if calls and calls[0].delay_s]
    assert len(delayed) > 100
    counts = []
    for command, calls in delayed:
        if command.endswith(" in half an hour"):
            assert all(c.delay_s == 1800 for c in calls)
            counts.append("half")
            continue
        said = re.search(r" in (\d+|an?) (second|minute|hour)s?$", command)
        assert said
        n, unit = said.groups()
        n = "1" if n in ("a", "an") else n
        seconds = int(n) * {"second": 1, "minute": 60, "hour": 3600}[unit]
        assert all(c.delay_s == seconds for c in calls)
        counts.append(said[1])
    # Single digits and "a minute" are said often, not one count in ten.
    assert sum(len(n) == 1 for n in counts) > len(counts) / 4
    assert {"a", "an"} <= set(counts)
    assert "half" in counts
    assert sum(n in ("1", "a", "an") for n in counts) > len(counts) / 6


def test_start_and_stop_say_on_and_off(home):
    said = {
        (c.split()[0], calls[0].value)
        for c, _, calls in home
        if calls
        and isinstance(calls[0].value, bool)
        and c.split()[0] in ("start", "stop")
    }
    assert said == {("start", True), ("stop", False)}


def test_durations(home):
    # "for N minutes" on a bool: the value now, the opposite after N.
    lasting = [
        (m.groups(), calls)
        for c, _, calls in home
        if (m := re.search(r" for (\d+|an?|half an) (second|minute|hour)s?$", c))
    ]
    assert len(lasting) > 50
    for (n, unit), calls in lasting:
        now, later = calls[: len(calls) // 2], calls[len(calls) // 2 :]
        assert all(c.delay_s is None for c in now)
        assert [(c.target, not c.value) for c in now] == [
            (c.target, c.value) for c in later
        ]
        count = 0.5 if n == "half an" else 1 if n in ("a", "an") else int(n)
        seconds = count * {"second": 1, "minute": 60, "hour": 3600}[unit]
        assert all(c.delay_s == seconds for c in later)


def test_negatives(home, lab):
    for samples in (home, lab):
        negatives = [c for c, t, calls in samples if not calls]
        assert 0.1 < len(negatives) / len(samples) < 0.3
    home_negatives = [(c, t) for c, t, calls in home if not calls]
    # Chit-chat mostly shows no target; an unlisted thing beside a listed
    # name's words ("the hallway light") does.
    assert any(not select(t, c) for c, t in home_negatives)
    assert any(select(t, c) for c, t in home_negatives)


def test_examples_are_mixed_in(home):
    by_command = {}
    for command, targets, calls in home:
        by_command.setdefault(command, []).append((targets, calls))
    for e in EXAMPLES:
        calls = [Call(*c) for c in e["calls"]]
        matches = [
            targets
            for targets, c in by_command[e["command"]]
            if c == calls
            and all(t.value == e.get("state", {}).get(t.name, t.value) for t in targets)
        ]
        assert matches


def test_an_example_that_cannot_be_learnt_raises():
    with pytest.raises(ValueError, match="too dark in here"):
        example = {
            "command": "it's too dark in here",
            "calls": [["kitchen_light", True]],
        }
        corpus(HOME, n=10, seed=0, examples=[example])
    with pytest.raises(ValueError, match="open the awning"):
        corpus(
            HOME,
            n=10,
            seed=0,
            examples=[{"command": "open the awning", "calls": [["canopy", "OPEN"]]}],
        )


def test_as_case_reads_a_record_on_the_listed_targets():
    record = {
        "state": {"user_location": "office"},
        "command": "switch on the light",
        "calls": [["office_light", True]],
    }
    command, state, calls = as_case(HOME, record)
    assert command == "switch on the light"
    assert calls == [Call("office_light", True)]
    assert [t.name for t in state] == [t.name for t in HOME]
    by_name = {t.name: t.value for t in state}
    assert by_name["user_location"] == "office"
    assert by_name["bedroom_light"] is True  # listed value kept
    # Evaluation retains a valid label even when selection cannot support it.
    assert as_case(
        HOME, {"command": "it's too dark in here", "calls": [["kitchen_light", True]]}
    )[2] == [Call("kitchen_light", True)]


def test_from_log_keeps_kept_calls_and_turns_undone_ones_into_negatives():
    log = [
        {
            "state": {"user_location": "office"},
            "command": "more light please",
            "calls": [["office_light", True]],
            "kept": True,
        },
        {
            "command": "make the garden cosy",
            "calls": [["garden_light", True]],
            "kept": False,
        },
    ]
    examples = from_log(log)
    assert examples == [
        {
            "state": {"user_location": "office"},
            "command": "more light please",
            "calls": [["office_light", True]],
        },
        {"state": {}, "command": "make the garden cosy", "calls": []},
    ]
    samples = [
        (c, calls) for c, _, calls in corpus(HOME, n=200, seed=0, examples=examples)
    ]
    assert ("more light please", [Call("office_light", True)]) in samples
    assert ("make the garden cosy", []) in samples


def test_paraphrase_widens_commands_once_per_text():
    asked: list[str] = []

    def paraphrase(command: str) -> list[str]:
        asked.append(command)
        return [f"please {command}"]

    samples = corpus(LAB, n=500, seed=0, paraphrase=paraphrase)
    assert len(asked) == len(set(asked))
    commands = [c for c, _, _ in samples]
    assert any(c.startswith("please ") for c in commands)
    assert any(not c.startswith("please ") for c in commands)
    assert any(" in " in c or " for " in c for c in asked)


def test_negated_commands_never_apply_a_call(home):
    negated = [
        (c, calls) for c, _, calls in home if c.startswith(("don't ", "do not "))
    ]
    assert len(negated) > 100
    assert all(not calls for _, calls in negated)


def test_no_home_words_in_the_generic_phrasings(lab):
    # Words a lab command uses beyond the lab's own words come from the
    # generic phrasings, so none may be a word of the demo home's names.
    own = words(" ".join(f"{t.name} {t.description} {t.values}" for t in LAB))
    used = set().union(*(words(c) for c, _, calls in lab if calls)) - own
    home = words(" ".join(t.name for t in HOME))
    assert used, "no phrasing words found"
    assert not used & home


def test_corpus_reports_selection_rejections_and_can_retain_them():
    state = [Target("pump_" + "x" * 300, "bool", False, description="pump")]
    counts = {}
    samples = corpus(state, n=100, seed=0, diagnostics=counts, filter_unlearnable=False)
    assert len(samples) == counts["accepted"] == counts["attempted"] == 100
    assert counts["missing_target"] + counts["prompt_overflow"] > 0
    assert any(calls for _, _, calls in samples)


def test_training_example_requires_its_declared_state_to_be_visible():
    state = [
        Target("pump", "bool", False),
        Target("location", "str", "lab", values=["lab", "store"], readonly=True),
    ]
    record = {
        "command": "pump on",
        "state": {"location": "lab"},
        "calls": [["pump", True]],
    }
    with pytest.raises(ValueError, match="cannot be learnt"):
        corpus(state, n=10, seed=0, examples=[record])
    assert as_case(state, record)[2] == [Call("pump", True)]


def test_paraphrase_must_preserve_visibility_of_required_state():
    state = [
        Target("pump", "bool", False),
        Target("location", "str", "lab", values=["lab", "store"], readonly=True),
    ]
    record = {
        "command": "pump on at this location",
        "state": {"location": "lab"},
        "calls": [["pump", True]],
    }
    counts = {}
    samples = corpus(
        state,
        n=500,
        seed=0,
        examples=[record],
        paraphrase=lambda c: (
            ["please power the pump"] if c == record["command"] else []
        ),
        diagnostics=counts,
    )
    assert "please power the pump" not in {c for c, _, _ in samples}
    assert counts["missing_state"] > 0
