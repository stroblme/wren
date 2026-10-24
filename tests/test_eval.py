import math

from wren import targets
from wren.dataset import _render_mobile_actions, render_call, to_text
from wren.eval import coverage, evaluate, parse_calls, score, threshold
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
    assert threshold(PREDICTED, EXPECTED) == 0.7
    assert threshold(PREDICTED, EXPECTED, precision=0.75) == 0.6
    assert threshold(PREDICTED, EXPECTED, precision=0.5) == 0.4


def test_threshold_counts_wrong_delay_and_extra_calls():
    expected = [[TCall("a", True, 60)], [TCall("a", True)]]
    predicted = [
        [TCall("a", True, confidence=0.9)],  # missing delay
        [TCall("a", True, confidence=0.8), TCall("b", 1, confidence=0.7)],  # extra
    ]
    assert threshold(predicted, expected, precision=0.5) == 0.8


def test_threshold_unreachable_or_empty_applies_nothing():
    wrong = [[TCall("a", True, confidence=0.9)], [TCall("b", 1, confidence=1.0)]]
    assert threshold(wrong, [[], [TCall("b", 2)]]) == math.inf
    assert threshold([], []) == math.inf
    assert threshold([[], []], [[], []]) == math.inf


def test_coverage():
    assert coverage(PREDICTED, EXPECTED, 0.7) == 4 / 5  # never sets b to 3
    assert coverage(PREDICTED, EXPECTED, 0.8) == 3 / 5  # drops a needed call
    assert coverage(PREDICTED, EXPECTED, 0.3) == 3 / 5  # applies the extra call
    assert coverage(PREDICTED, EXPECTED, math.inf) == 2 / 5  # only abstentions


def test_evaluate_takes_the_higher_threshold_of_generated_and_tests(monkeypatch):
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

    assert evaluate({}, generated) == {
        "exact_match": 1.0,
        "threshold": 0.5,
        "coverage": 1.0,
    }
    assert evaluate({}, generated, tests=tests) == {
        "exact_match": 1.0,
        "threshold": 0.8,  # the tests', above their wrong call
        "coverage": 0.5,  # g1 is no longer applied
        "test_exact_match": 0.5,
        "test_coverage": 0.5,
    }
