import random
import urllib.request

import pytest

from wren import dataset
from wren.dataset import _coming_home, _house, home, render_call, to_text
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
