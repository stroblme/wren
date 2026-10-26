"""Compare a retrained candidate with the shipped pack on fixed data splits."""

import dataclasses
import json
import math
import pathlib

import fluksio
from fluksio import Flow, Port, node


@node(
    requires=[Port("candidate", "str")],
    provides=[Port("report", "json"), Port("predictions", "artifact")],
    resources={"env": {"OPENBLAS_NUM_THREADS": "1"}},
)
def check(candidate: str) -> dict:
    from wren import dataset, runtime
    from wren.eval import metrics, selection, threshold
    from wren.targets import Target, act

    home = json.loads(runtime.DEMO.read_text())
    targets = [Target(**t) for t in home["targets"]]
    reserved = {r["command"] for k in ("calibration", "tests") for r in home.get(k, [])}
    counts = {}
    training = dataset.corpus(
        targets,
        n=200000,
        seed=0,
        examples=home["examples"],
        exclude=reserved,
        diagnostics=counts,
    )
    calibration = dataset.corpus(
        targets,
        n=1000,
        seed=1,
        exclude={r["command"] for r in home["tests"]},
        filter_unlearnable=False,
    )
    held = dataset.corpus(
        targets,
        n=1000,
        seed=2,
        held_out=True,
        exclude=reserved | {c for c, _, _ in training + calibration},
        filter_unlearnable=False,
    )
    # These stay fixed across packs. A pack's predictions never alter the data.
    sets = {
        "calibration": calibration,
        "held_out": held,
        "installation_calibration": [
            dataset.as_case(targets, r) for r in home["calibration"]
        ],
        "tests": [dataset.as_case(targets, r) for r in home["tests"]],
    }
    reports, details = {}, {}
    for label, path in (
        ("shipped", runtime.MODEL),
        ("candidate", pathlib.Path(candidate)),
    ):
        pack = runtime.load_pack(path)
        predictions = {
            name: [act(c, state, model=pack) for c, state, _ in cases]
            for name, cases in sets.items()
        }
        expected = {
            name: [calls for _, _, calls in cases] for name, cases in sets.items()
        }
        cutoff = threshold(
            predictions["calibration"],
            expected["calibration"],
            also=[
                (
                    predictions["installation_calibration"],
                    expected["installation_calibration"],
                )
            ],
            policy="answer",
        )
        scores: dict = {
            name: metrics(predictions[name], expected[name], cutoff, policy="answer")
            for name in sets
        }
        scores["threshold"] = cutoff if math.isfinite(cutoff) else None
        if label == "shipped":
            scores["at_saved_threshold"] = {
                name: metrics(
                    predictions[name],
                    expected[name],
                    pack["meta"]["threshold"],
                    policy=pack["meta"].get("acceptance", "call"),
                )
                for name in sets
            }
        reports[label] = scores
        details[label] = {
            name: [
                {
                    "command": c,
                    "expected": [dataclasses.asdict(call) for call in calls],
                    "predicted": [dataclasses.asdict(call) for call in predicted],
                }
                for (c, _, calls), predicted in zip(
                    cases, predictions[name], strict=True
                )
            ]
            for name, cases in sets.items()
        }
    reports["corpus"] = counts
    reports["selection"] = selection(held)
    reports["test_selection"] = selection(
        sets["tests"], required_state=[r.get("state", {}) for r in home["tests"]]
    )
    return {
        "report": reports,
        "predictions": fluksio.save_artifact(
            json.dumps(details).encode(), name="predictions.json"
        ),
    }


Flow("s13_check", nodes=[check], inputs=[Port("candidate", "str")], outputs=["report"])
