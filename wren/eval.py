"""Evaluate tool calls by ordered exact match and diagnostic metrics.

Exact match requires identical function names, call order, and argument
values. Well-formedness, name accuracy, and argument F1 measure partial
success. For calls on targets, `threshold` recommends the confidence a
host applies calls from and `coverage` gives the share it then gets right;
`evaluate` chooses a threshold on separate calibration samples and reports
complete-command results on independent evaluation and installation tests.
"""

import ast
import math
from collections.abc import Collection, Sequence

import numpy as np

from wren import targets

# One tool call: function name and keyword arguments.
Call = tuple[str, dict]


def parse_calls(text: str) -> list[Call] | None:
    """Parse `>> name(key='value', ...)` lines into (name, args) pairs.

    Names may be dotted. Return None if any marked line is not a plain call
    with literal keyword arguments; ignore unmarked lines.
    """
    calls = []
    for line in text.splitlines():
        if not line.startswith(">> "):
            continue
        try:
            expr = ast.parse(line[3:].strip(), mode="eval").body
            if not isinstance(expr, ast.Call):
                return None
            func = expr.func
            while isinstance(func, ast.Attribute):  # peel `a.b.c` to its root
                func = func.value
            if (
                not isinstance(func, ast.Name)
                or expr.args
                or any(kw.arg is None for kw in expr.keywords)
            ):
                return None
            args = {kw.arg: ast.literal_eval(kw.value) for kw in expr.keywords}
        except (SyntaxError, ValueError, TypeError):
            return None
        calls.append((ast.unparse(expr.func), args))
    return calls


def score(predictions: list[str], references: list[list[Call]]) -> dict[str, float]:
    """Score generated texts against reference call lists."""
    well_formed = exact = names = 0
    tp = fp = fn = 0
    for text, ref in zip(predictions, references, strict=True):
        pred = parse_calls(text)
        if pred is None:
            fn += sum(len(args) for _, args in ref)
            continue
        # Emitting no calls when the reference has some is a formatting
        # failure too, not a well-formed answer.
        well_formed += bool(pred) or not ref
        exact += pred == ref
        names += [n for n, _ in pred] == [n for n, _ in ref]
        # Argument micro-F1 over (call position, function, key, value) items:
        # credit for every argument placed in the right call, whether or not
        # the rest of the sample is right.
        p = {(i, n, k, repr(v)) for i, (n, a) in enumerate(pred) for k, v in a.items()}
        r = {(i, n, k, repr(v)) for i, (n, a) in enumerate(ref) for k, v in a.items()}
        tp += len(p & r)
        fp += len(p - r)
        fn += len(r - p)
    n = len(references)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "exact_match": exact / n,
        "well_formed": well_formed / n,
        "name_accuracy": names / n,
        "arg_f1": (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        ),
    }


def hallucination_rate(predictions: list[str], surface: dict[str, list[str]]) -> float:
    """Return the fraction of call predictions using unknown functions or arguments.

    Exclude malformed predictions and predictions without calls.
    """
    seen = invented = 0
    for text in predictions:
        pred = parse_calls(text)
        if not pred:
            continue
        seen += 1
        invented += any(
            name not in surface or set(args) - set(surface[name]) for name, args in pred
        )
    return invented / seen if seen else 0.0


def abstain_rate(predictions: list[str]) -> float:
    """Return the fraction emitting no calls; use on negatives for irrelevance accuracy."""
    return sum(parse_calls(t) == [] for t in predictions) / len(predictions)


def calibration_error(
    confidences: list[float], corrects: list[bool], bins: int = 10
) -> float:
    """Return expected calibration error over confidence bins.

    Weight each bin's |accuracy - mean confidence| by its sample count.
    """
    ece = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        members = [
            i for i, c in enumerate(confidences) if lo <= c < hi or (c == 1.0 == hi)
        ]
        if members:
            accuracy = sum(corrects[i] for i in members) / len(members)
            confidence = sum(confidences[i] for i in members) / len(members)
            ece += len(members) / len(confidences) * abs(accuracy - confidence)
    return ece


def threshold(
    predicted: list[list[targets.Call]],
    expected: list[list[targets.Call]],
    precision: float = 0.99,
    also: Sequence[tuple[list[list[targets.Call]], list[list[targets.Call]]]] = (),
    *,
    policy: str = "call",
) -> float:
    """Return the most permissive observed acceptance set reaching `precision`.

    With `policy="call"`, apply each call at or above the threshold. A call
    is right if its sample expects it (same target, value and delay, in
    any order), so any call on a sample that should abstain is wrong. Each
    (predicted, expected) set in `also` has to reach `precision` at the same
    threshold, checked on its own, where a set with no call left applies
    nothing wrong. Precision need not rise with the threshold, so the higher
    of two thresholds found apart can miss it on the other set.
    With `policy="answer"`, score each nonempty answer once, at its minimum
    call confidence; it is right only when every expected call is present
    and no extra call is generated.
    Place the cutoff between adjacent observed scores so tiny numerical
    changes do not reject a call sitting exactly on it. Return infinity,
    which applies nothing, when no threshold reaches `precision`.
    """
    conf, right = _marks(predicted, expected, policy)
    others = [_marks(p, e, policy) for p, e in also]
    candidates = np.unique(np.concatenate([conf, *(c for c, _ in others)]))
    for i, t in enumerate(candidates):  # ascending
        applied = conf >= t
        if (
            applied.any()
            and right[applied].mean() >= precision
            and all(
                not (sel := c >= t).any() or r[sel].mean() >= precision
                for c, r in others
            )
        ):
            lower = candidates[i - 1] if i else t
            return float(lower + (t - lower) / 2)
    return math.inf


def _marks(
    predicted: list[list[targets.Call]],
    expected: list[list[targets.Call]],
    policy: str = "call",
) -> tuple[np.ndarray, np.ndarray]:
    """Each predicted call's confidence, and whether its sample expects it."""
    conf, right = [], []
    if policy not in ("call", "answer"):
        raise ValueError(f"unknown acceptance policy: {policy}")
    for pred, exp in zip(predicted, expected, strict=True):
        want = {_key(c) for c in exp}
        if policy == "answer":
            if pred:
                conf.append(min(c.confidence for c in pred))
                right.append({_key(c) for c in pred} == want)
        else:
            conf += [c.confidence for c in pred]
            right += [_key(c) in want for c in pred]
    return np.array(conf, dtype=float), np.array(right, dtype=bool)


def coverage(
    predicted: list[list[targets.Call]],
    expected: list[list[targets.Call]],
    threshold: float,
) -> float:
    """Return the share of samples handled fully right at `threshold`.

    A sample is handled right when the calls applied at the threshold are
    exactly its expected ones: each of them and nothing else, which for a
    sample that should abstain means no call.
    """
    return sum(
        {_key(c) for c in pred if c.confidence >= threshold} == {_key(c) for c in exp}
        for pred, exp in zip(predicted, expected, strict=True)
    ) / len(expected)


def evaluate(
    pack: dict,
    cases: list[tuple[str, list[targets.Target], list[targets.Call]]],
    precision: float = 0.99,
    tests: Sequence[tuple[str, list[targets.Target], list[targets.Call]]] = (),
    *,
    calibration: Sequence[tuple[str, list[targets.Target], list[targets.Call]]],
    calibration_tests: Sequence[
        tuple[str, list[targets.Target], list[targets.Call]]
    ] = (),
    policy: str = "call",
) -> dict:
    """Run `act` with `pack` on (command, targets, expected calls) samples.

    Only `calibration` and optional installation `calibration_tests` choose
    the threshold, checked separately. `cases` and installation `tests`
    report performance at that frozen threshold; they cannot adjust it.
    Metrics include positive-command completion, incorrect and partial
    accepted answers, and counts per command kind. Empty denominators are
    None, not evidence of perfect precision.
    """
    predicted, expected = _run(pack, cases)
    tested, wanted = _run(pack, tests)
    calibrated, reference = _run(pack, calibration)
    extra, extra_reference = _run(pack, calibration_tests)
    t = threshold(
        calibrated, reference, precision, also=[(extra, extra_reference)], policy=policy
    )
    scores = metrics(predicted, expected, t, policy=policy) | {
        "threshold": t,
        "calibration": metrics(calibrated, reference, t, policy=policy),
        "selection": selection(cases, budget=pack["cfg"]["context"] - targets.N_NEW),
    }
    if calibration_tests:
        scores["calibration_tests"] = metrics(extra, extra_reference, t, policy=policy)
    if tests:
        scores["tests"] = metrics(tested, wanted, t, policy=policy)
        scores["test_exact_match"] = scores["tests"]["exact_match"]
        scores["test_coverage"] = scores["tests"]["coverage"]
    return scores


def metrics(
    predicted: list[list[targets.Call]],
    expected: list[list[targets.Call]],
    threshold: float,
    *,
    policy: str = "call",
) -> dict:
    """Score applied calls and complete commands, including abstentions.

    `partial_commands` accept a proper nonempty subset of the expected
    calls. `wrong_commands` accept any unexpected call. Command precision
    counts only complete correct answers among nonempty accepted answers.
    Duplicate calls retain the existing idempotent set semantics.
    """
    rows = []
    gate = {"meta": {"threshold": threshold, "acceptance": policy}}
    for pred, exp in zip(predicted, expected, strict=True):
        want = {_key(c) for c in exp}
        raw = {_key(c) for c in pred}
        kept = targets.accept(pred, model=gate)
        got = {_key(c) for c in kept}
        delayed = [c.delay_s is not None for c in exp]
        if not exp:
            kind = "abstain"
        elif any(delayed):
            kind = "delay" if all(delayed) else "duration"
        elif len(exp) > 1:
            kind = "multi"
        else:
            kind = "read" if exp[0].value is None else "set"
        rows.append((kind, want, raw, got, kept))

    def summary(subset):
        n = len(subset)
        positive = sum(bool(w) for _, w, _, _, _ in subset)
        accepted = sum(bool(g) for _, _, _, g, _ in subset)
        n_calls = sum(len(kept) for _, _, _, _, kept in subset)
        correct = sum(bool(g) and g == w for _, w, _, g, _ in subset)
        return {
            "n": n,
            "exact_match": sum(r == w for _, w, r, _, _ in subset) / n if n else None,
            "coverage": sum(g == w for _, w, _, g, _ in subset) / n if n else None,
            "positive_coverage": correct / positive if positive else None,
            "accepted_commands": accepted,
            "accepted_calls": n_calls,
            "command_precision": correct / accepted if accepted else None,
            "call_precision": sum(
                _key(c) in w for _, w, _, _, kept in subset for c in kept
            )
            / n_calls
            if n_calls
            else None,
            "partial_commands": sum(bool(g) and g < w for _, w, _, g, _ in subset),
            "wrong_commands": sum(bool(g - w) for _, w, _, g, _ in subset),
            "missed_commands": sum(bool(w) and not g for _, w, _, g, _ in subset),
        }

    return summary(rows) | {
        "by_kind": {
            kind: summary([r for r in rows if r[0] == kind])
            for kind in sorted({r[0] for r in rows})
        }
    }


def selection(
    cases: Sequence[tuple[str, list[targets.Target], list[targets.Call]]],
    *,
    required_state: Sequence[Collection[str]] | None = None,
    budget: int = targets.BUDGET,
) -> dict:
    """Measure selection before filtering cases, with optional state dependencies.

    `required_state[i]` names the state targets the ith label depends on,
    such as the keys of an installation record's `state` object.
    """
    wanted = shown_calls = dependencies = shown_state = 0
    prompt_overflow = answer_overflow = 0
    failures = []
    required = required_state if required_state is not None else [()] * len(cases)
    for (command, state, calls), needs in zip(cases, required, strict=True):
        shown = targets.select(state, command, budget)
        names = {t.name for t in shown}
        want, need = {c.target for c in calls}, set(needs)
        wanted += len(want)
        shown_calls += len(want & names)
        dependencies += len(need)
        shown_state += len(need & names)
        prompt_bytes = len(targets.prompt(shown, command).encode())
        answer_bytes = len(targets.answer(calls).encode())
        prompt_overflow += prompt_bytes > budget
        answer_overflow += answer_bytes > targets.N_NEW
        if (
            want - names
            or need - names
            or prompt_bytes > budget
            or answer_bytes > targets.N_NEW
        ):
            failures.append(
                {
                    "command": command,
                    "missing_targets": sorted(want - names),
                    "missing_state": sorted(need - names),
                    "prompt_bytes": prompt_bytes,
                    "answer_bytes": answer_bytes,
                }
            )
    return {
        "n": len(cases),
        "required_targets": wanted,
        "shown_targets": shown_calls,
        "target_recall": shown_calls / wanted if wanted else None,
        "required_state": dependencies,
        "shown_state": shown_state,
        "state_recall": shown_state / dependencies if dependencies else None,
        "prompt_overflow": prompt_overflow,
        "answer_overflow": answer_overflow,
        "failures": failures,
    }


def _run(
    pack: dict, cases: Sequence
) -> tuple[list[list[targets.Call]], list[list[targets.Call]]]:
    """The calls `act` returns on each sample and the calls it expects."""
    predicted = [targets.act(c, state, model=pack) for c, state, _ in cases]
    return predicted, [calls for _, _, calls in cases]


def _key(call: targets.Call) -> tuple:
    """What makes two calls the same, ignoring confidence."""
    return call.target, call.value, call.delay_s
