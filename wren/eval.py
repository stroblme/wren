"""Evaluate tool calls by ordered exact match and diagnostic metrics.

Exact match requires identical function names, call order, and argument
values. Well-formedness, name accuracy, and argument F1 measure partial
success.
"""

import ast

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
        except (SyntaxError, ValueError):
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
