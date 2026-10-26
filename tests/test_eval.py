import math

import pytest

from wren import targets
from wren.dataset import _render_mobile_actions, render_call, to_text
from wren.eval import (
    coverage,
    evaluate,
    metrics,
    parse_calls,
    score,
    selection,
    threshold,
)
from wren.targets import Call as TCall
from wren.targets import Target

MA_ROW = {
    "metadata": "train",
    "tools": [],
    "messages": [
        {
            "role": "developer",
            "content": "Current date and time given in YYYY-MM-DDTHH:MM:SS "
            "format: 2025-06-04T15:29:23\nDay of week is Wednesday\n",
        },
        {"role": "user", "content": "Remind me of the Team Sync on Friday 2 PM."},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "create_calendar_event",
                        "arguments": {
                            "datetime": "2025-06-06T14:00:00",
                            "title": "Team Sync",
                        },
                    }
                }
            ],
        },
    ],
}


HA_ROW = {
    "conversations": [
        {"from": "system", "value": "You are 'Al' ...\nServices: ...\nDevices:\n..."},
        {"from": "user", "value": "lower the kitchen blinds"},
        {
            "from": "assistant",
            "value": "closing the blinds as you requested.\n"
            '```homeassistant\n{"service": "cover.close", '
            '"target_device": "cover.kitchen"}\n```',
        },
    ]
}


def test_render_parse_roundtrip():
    call = ("send_email", {"to": "a@b.c", "subject": "hi there"})
    assert parse_calls(render_call(*call)) == [call]


def test_parse_dotted_names():
    call = ("light.turn_on", {"brightness": 50, "target_device": "light.porch"})
    assert parse_calls(render_call(*call)) == [call]


def test_parse_rejects_malformed():
    assert parse_calls(">> send_email(to='unterminated") is None
    assert parse_calls(">> send_email('positional')") is None
    assert parse_calls(">> get()(x=1)") is None
    assert parse_calls(">> get(x={[1]: 2})") is None
    assert parse_calls("no call marker at all") == []


def test_render_home_assistant():
    from wren.dataset import _render_home_assistant

    prompt, calls = _render_home_assistant(HA_ROW)
    assert prompt == "lower the kitchen blinds\n"
    assert calls == [("cover.close", {"target_device": "cover.kitchen"})]
    assert parse_calls(to_text([(prompt, calls)])) == calls


def test_render_mobile_actions():
    prompt, calls = _render_mobile_actions(MA_ROW)
    assert prompt == (
        "2025-06-04T15:29:23 Wednesday\nRemind me of the Team Sync on Friday 2 PM.\n"
    )
    assert calls == [
        (
            "create_calendar_event",
            {"datetime": "2025-06-06T14:00:00", "title": "Team Sync"},
        )
    ]
    sample = to_text([(prompt, calls)])
    assert parse_calls(sample) == calls


def test_score_metrics():
    ref = [("lights_on", {"room": "kitchen"})]
    results = score(
        predictions=[
            ">> lights_on(room='kitchen')",  # exact
            ">> lights_on(room='garage')",  # right name, wrong argument
            ">> lights_off(room='kitchen')",  # wrong name
            ">> lights_on(room=",  # malformed
            "I would love to help!",  # parseable but no call emitted
        ],
        references=[ref, ref, ref, ref, ref],
    )
    assert results["exact_match"] == 0.2
    assert results["well_formed"] == 0.6
    assert results["name_accuracy"] == 0.4
    assert 0 < results["arg_f1"] < 1


def test_hallucination_rate():
    from wren.eval import hallucination_rate

    surface = {"lights_on": ["room"]}
    preds = [
        ">> lights_on(room='kitchen')",  # fine
        ">> lights_on(brightness=3)",  # invented argument
        ">> open_pod_bay_doors()",  # invented function
        "just chatting",  # abstained: excluded
    ]
    assert hallucination_rate(preds, surface) == 2 / 3


def test_abstain_rate():
    from wren.eval import abstain_rate

    assert abstain_rate(["", "some text", ">> call()", ">> broken("]) == 0.5


def test_calibration_error():
    from wren.eval import calibration_error

    perfect = calibration_error([1.0, 1.0, 0.0], [True, True, False], bins=4)
    assert perfect == 0.0
    overconfident = calibration_error([0.99, 0.99], [False, False], bins=4)
    assert overconfident > 0.9


def test_irrelevant_split_is_held_out():
    from wren.dataset import irrelevant

    train, evals = irrelevant(n_train=50, n_eval=10)
    train_prompts = {p.split("\n")[1] for p, _ in train}
    eval_prompts = {p.split("\n")[1] for p, _ in evals}
    assert len(evals) == 10 and not (train_prompts & eval_prompts)
    assert all(calls == [] for _, calls in train + evals)


# Held-out samples as (predicted, expected) call lists, one pair per sample.
HELD_OUT = [
    ([TCall("a", True, confidence=0.95)], [TCall("a", True)]),  # right
    ([TCall("b", 4, confidence=0.6)], [TCall("b", 3)]),  # wrong value
    ([TCall("c", True, confidence=0.4)], []),  # should abstain
    (  # right in another order, with a delay
        [TCall("b", 2, 600, confidence=0.7), TCall("a", False, confidence=0.9)],
        [TCall("a", False), TCall("b", 2, 600)],
    ),
    ([], []),  # abstains
]
PREDICTED = [p for p, _ in HELD_OUT]
EXPECTED = [e for _, e in HELD_OUT]


def test_threshold_meets_precision():
    # Applied precision: 0.6 from 0.4, 0.75 from 0.6, 1.0 from 0.7.
    assert threshold(PREDICTED, EXPECTED) == pytest.approx(0.65)
    assert threshold(PREDICTED, EXPECTED, precision=0.75) == 0.5
    assert threshold(PREDICTED, EXPECTED, precision=0.5) == 0.4


def test_threshold_counts_wrong_delay_and_extra_calls():
    expected = [[TCall("a", True, 60)], [TCall("a", True)]]
    predicted = [
        [TCall("a", True, confidence=0.9)],  # missing delay
        [TCall("a", True, confidence=0.8), TCall("b", 1, confidence=0.7)],  # extra
    ]
    assert threshold(predicted, expected, precision=0.5) == 0.75


def test_threshold_unreachable_or_empty_applies_nothing():
    wrong = [[TCall("a", True, confidence=0.9)], [TCall("b", 1, confidence=1.0)]]
    assert threshold(wrong, [[], [TCall("b", 2)]]) == math.inf
    assert threshold([], []) == math.inf
    assert threshold([[], []], [[], []]) == math.inf


def test_threshold_holds_on_each_set_at_once():
    # Apart, A reaches 0.99 from 0.1 and B from 0.9. Their higher one, 0.9,
    # leaves A one right and one wrong call: 0.99 is the first that holds.
    set_a = (
        [[TCall("a", True, confidence=0.1)]] * 99
        + [[TCall("a", False, confidence=0.95)], [TCall("a", True, confidence=0.99)]],
        [[TCall("a", True)]] * 101,
    )
    set_b = (
        [[TCall("a", False, confidence=0.8)], [TCall("a", True, confidence=0.9)]],
        [[TCall("a", True)]] * 2,
    )
    assert threshold(*set_a) == 0.1
    assert threshold(*set_b) == pytest.approx(0.85)
    assert threshold(*set_a, also=[set_b]) == 0.97
    # Tests on which the model makes no call apply nothing wrong.
    assert threshold(*set_a, also=[([[]], [[]])]) == 0.1


def test_coverage():
    assert coverage(PREDICTED, EXPECTED, 0.7) == 4 / 5  # never sets b to 3
    assert coverage(PREDICTED, EXPECTED, 0.8) == 3 / 5  # drops a needed call
    assert coverage(PREDICTED, EXPECTED, 0.3) == 3 / 5  # applies the extra call
    assert coverage(PREDICTED, EXPECTED, math.inf) == 2 / 5  # only abstentions


def test_evaluate_does_not_calibrate_on_test_answers(monkeypatch):
    replies = {
        "g1": [TCall("a", True, confidence=0.5)],
        "g2": [TCall("b", 1, confidence=0.9)],
        "t1": [TCall("a", False, confidence=0.7)],  # a confident wrong call
        "t2": [TCall("a", True, confidence=0.8)],
    }
    monkeypatch.setattr(targets, "act", lambda command, state, model: replies[command])
    state = [Target("a", "bool", False), Target("b", "int", 0)]
    generated = [("g1", state, [TCall("a", True)]), ("g2", state, [TCall("b", 1)])]
    tests = [("t1", state, [TCall("a", True)]), ("t2", state, [TCall("a", True)])]

    pack = {"cfg": {"context": 384}}
    result = evaluate(pack, generated, calibration=generated, tests=tests)
    assert result["threshold"] == 0.5  # unaffected by the wrong test call
    assert result["coverage"] == 1.0
    assert result["test_exact_match"] == result["test_coverage"] == 0.5
    assert result["tests"]["wrong_commands"] == 1
    # Installation calibration commands are checked separately from generated ones.
    result = evaluate(pack, generated, calibration=generated, calibration_tests=tests)
    assert result["threshold"] == 0.75
    assert result["coverage"] == 0.5


def test_metrics_distinguish_call_precision_and_complete_commands():
    start = TCall("pump", True, confidence=0.99)
    stop = TCall("pump", False, 60, confidence=0.6)
    result = metrics([[start, stop], [], [start]], [[start, stop], [], []], 0.9)
    assert result["n"] == 3
    assert result["coverage"] == 1 / 3  # only the correct abstention
    assert result["positive_coverage"] == 0
    assert result["call_precision"] == 0.5
    assert result["command_precision"] == 0
    assert result["partial_commands"] == 1
    assert result["wrong_commands"] == 1
    assert result["accepted_commands"] == 2
    assert result["by_kind"]["duration"]["partial_commands"] == 1
    assert result["by_kind"]["abstain"]["n"] == 2


def test_metrics_empty_denominators_are_explicit():
    result = metrics([[]], [[]], math.inf)
    assert result["positive_coverage"] is None
    assert result["call_precision"] is None
    assert result["command_precision"] is None
    assert result["coverage"] == 1
    assert metrics([], [], 0)["coverage"] is None


def test_threshold_has_room_for_small_runtime_roundoff():
    predictions = [
        [TCall("pump", True, confidence=0.973)],
        [TCall("pump", False, confidence=0.96)],
    ]
    expected = [[TCall("pump", True)]] * 2
    t = threshold(predictions, expected)
    assert 0.96 < t < 0.973
    predictions[0][0].confidence -= 2e-7
    assert metrics(predictions, expected, t)["call_precision"] == 1.0


def test_answer_calibration_counts_an_unemitted_stop_as_incorrect():
    start = TCall("pump", True, confidence=0.99)
    stop = TCall("pump", False, 60, confidence=0.8)
    assert math.isinf(threshold([[start]], [[start, stop]], policy="answer"))
    # A complete answer can pass while a more uncertain incomplete one does not.
    wrong = TCall("pump", True, confidence=0.7)
    t = threshold([[wrong], [start, stop]], [[start, stop]] * 2, policy="answer")
    assert 0.7 < t < 0.8
    result = metrics([[wrong], [start, stop]], [[start, stop]] * 2, t, policy="answer")
    assert result["coverage"] == 0.5
    assert result["command_precision"] == 1.0
    assert result["partial_commands"] == 0


def test_selection_reports_hidden_calls_state_and_length_limits():
    state = [
        Target("pump", "bool", False),
        Target("location", "str", "lab", readonly=True),
    ]
    cases = [
        ("pump on", state, [TCall("pump", True)]),
        ("too dark", state, [TCall("pump", True)]),
        ("pump " * 100, state, [TCall("pump", True)]),
    ]
    result = selection(cases, required_state=[{"location"}, set(), set()])
    assert result["target_recall"] == 1 / 3
    assert result["state_recall"] == 0
    assert result["prompt_overflow"] == 1
    assert result["failures"][0]["missing_state"] == ["location"]
    assert result["failures"][1]["missing_targets"] == ["pump"]
