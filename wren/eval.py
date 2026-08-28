"""Tool-call evaluation: ordered strict exact match and its companions.

Needle's protocol, adopted as ours: a sample passes only if the function
names, the call order, and every argument value match the reference. The
companion metrics diagnose *why* exact match fails -- well-formedness (does
the output parse at all), name accuracy (right functions, wrong arguments),
and argument micro-F1 (partial credit on arguments).
"""

import ast

# One tool call: function name and keyword arguments.
Call = tuple[str, dict]


def parse_calls(text: str) -> list[Call] | None:
    """Extract `>> name(key='value', ...)` lines into (name, args) pairs.

    Returns None when any `>> ` line fails to parse as a plain call with
    literal keyword arguments -- the sample is then not well-formed.
    Lines without the marker are ignored (the model may ramble around its
    calls; only the calls are scored).
    """
    calls = []
    for line in text.splitlines():
        if not line.startswith(">> "):
            continue
        try:
            expr = ast.parse(line[3:].strip(), mode="eval").body
            if (
                not isinstance(expr, ast.Call)
                or not isinstance(expr.func, ast.Name)
                or expr.args
                or any(kw.arg is None for kw in expr.keywords)
            ):
                return None
            args = {kw.arg: ast.literal_eval(kw.value) for kw in expr.keywords}
        except (SyntaxError, ValueError):
            return None
        calls.append((expr.func.id, args))
    return calls


def score(predictions: list[str], references: list[list[Call]]) -> dict[str, float]:
    """Corpus metrics for generated texts against reference call lists."""
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
