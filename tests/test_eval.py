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
