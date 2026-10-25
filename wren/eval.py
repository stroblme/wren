"""Evaluate tool calls by ordered exact match and diagnostic metrics.

Exact match requires identical function names, call order, and argument
values. Well-formedness, name accuracy, and argument F1 measure partial
success. For calls on targets, `threshold` recommends the confidence a
host applies calls from and `coverage` gives the share it then gets right;
`evaluate` runs a pack on generated samples and an installation's test
commands and reports both.
"""

import ast
import math
from collections.abc import Sequence

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
) -> float:
    """Return the lowest confidence at which applied calls reach `precision`.

    A host applies each call whose confidence is at least the threshold. A
    call is right if its sample expects it (same target, value and delay, in
    any order), so any call on a sample that should abstain is wrong. Each
    (predicted, expected) set in `also` has to reach `precision` at the same
    threshold, checked on its own, where a set with no call left applies
    nothing wrong. Precision need not rise with the threshold, so the higher
    of two thresholds found apart can miss it on the other set. Return
    infinity, which applies nothing, when no threshold reaches `precision`.
    """
    conf, right = _marks(predicted, expected)
    others = [_marks(p, e) for p, e in also]
    for t in np.unique(np.concatenate([conf, *(c for c, _ in others)])):  # ascending
        applied = conf >= t
        if (
            applied.any()
            and right[applied].mean() >= precision
            and all(
                not (sel := c >= t).any() or r[sel].mean() >= precision
                for c, r in others
            )
        ):
            return float(t)
    return math.inf


def _marks(
    predicted: list[list[targets.Call]], expected: list[list[targets.Call]]
) -> tuple[np.ndarray, np.ndarray]:
    """Each predicted call's confidence, and whether its sample expects it."""
    conf, right = [], []
    for pred, exp in zip(predicted, expected, strict=True):
        want = {_key(c) for c in exp}
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
) -> dict[str, float]:
    """Run `act` with `pack` on (command, targets, expected calls) samples.

    `cases` are generated samples, `tests` an installation's own commands.
    The threshold is the lowest at which applied calls reach `precision` on
    each set on its own: pooled into a thousand generated samples, a few
    tests' wrong calls would fit in the allowance.
    Return the exact match (every call right, no other, at any confidence),
    the threshold and the coverage there, and with tests their exact match
    and coverage as `test_exact_match` and `test_coverage`.
    """
    predicted, expected = _run(pack, cases)
    tested, wanted = _run(pack, tests)
    t = threshold(predicted, expected, precision, also=[(tested, wanted)])
    scores = {
        "exact_match": coverage(predicted, expected, 0.0),
        "threshold": t,
        "coverage": coverage(predicted, expected, t),
    }
    if tests:
        scores["test_exact_match"] = coverage(tested, wanted, 0.0)
        scores["test_coverage"] = coverage(tested, wanted, t)
    return scores


def _run(
    pack: dict, cases: Sequence
) -> tuple[list[list[targets.Call]], list[list[targets.Call]]]:
    """The calls `act` returns on each sample and the calls it expects."""
    predicted = [targets.act(c, state, model=pack) for c, state, _ in cases]
    return predicted, [calls for _, _, calls in cases]


def _key(call: targets.Call) -> tuple:
    """What makes two calls the same, ignoring confidence."""
    return call.target, call.value, call.delay_s
