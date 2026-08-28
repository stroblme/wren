from wren.dataset import _render_mobile_actions, render_call, to_text
from wren.eval import parse_calls, score

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


def test_render_parse_roundtrip():
    call = ("send_email", {"to": "a@b.c", "subject": "hi there"})
    assert parse_calls(render_call(*call)) == [call]


def test_parse_rejects_malformed():
    assert parse_calls(">> send_email(to='unterminated") is None
    assert parse_calls(">> send_email('positional')") is None
    assert parse_calls("no call marker at all") == []


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
