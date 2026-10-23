"""Check the corpus generated from a target list."""

import json
import pathlib
import re

import pytest

from wren.dataset import corpus, text
from wren.targets import BUDGET, N_NEW, Call, Target, answer, parse, prompt, select

HOME_JSON = json.loads(
    (pathlib.Path(__file__).parents[1] / "examples/home.json").read_text()
)
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
    for command, calls in delayed:
        # Said in the unit `answer` writes, so the model copies number and unit.
        said = re.search(r" in (\d+) (second|minute|hour)s?$", command)
        assert said
        n, unit = said.groups()
        assert answer(calls).endswith(f" in {n}{unit[0]}\n\n")


def test_durations(home):
    # "for N minutes" on a bool: the value now, the opposite after N.
    lasting = [
        (m.groups(), calls)
        for c, _, calls in home
        if (m := re.search(r" for (\d+) (second|minute|hour)s?$", c))
    ]
    assert len(lasting) > 50
    for (n, unit), calls in lasting:
        now, later = calls[: len(calls) // 2], calls[len(calls) // 2 :]
        assert all(c.delay_s is None for c in now)
        assert [(c.target, not c.value) for c in now] == [
            (c.target, c.value) for c in later
        ]
        assert answer(calls).endswith(f" in {n}{unit[0]}\n\n")


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


def test_no_home_words_in_the_generic_phrasings(lab):
    # Words a lab command uses beyond the lab's own words come from the
    # generic phrasings, so none may be a word of the demo home's names.
    own = words(" ".join(f"{t.name} {t.description} {t.values}" for t in LAB))
    used = set().union(*(words(c) for c, _, calls in lab if calls)) - own
    home = words(" ".join(t.name for t in HOME))
    assert used, "no phrasing words found"
    assert not used & home
