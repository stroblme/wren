"""S7 — bottleneck autopsy (ROADMAP.md): where do exact-match points die?

Trains the reference configuration per dataset (HA-Requests under s3's
protocol, mobile-actions under s2's, stock model, muon 3e-3) and then runs
eval-side instruments on it:

- waterfall:  every positive eval sample lands in exactly one failure
  class — false abstain, ill-formed, wrong name, right-name-wrong-values
  with the missing values all copyable (their text appears in the prompt)
  or some derived (dates, arithmetic — not in the prompt), or exact.
- span bpb:   teacher-forced loss over the reference text, per byte class
  (prompt / syntax / function name / argument keys / copyable values /
  derived values) — where the residual loss lives.
- ladder:     the same eval set decoded under increasing oracles: free
  (the s2/s3 protocol), grammar (canonical call structure forced, name
  constrained to the tool surface via a byte trie, values free), gold
  name (>> name( forced per call, rest of the line free), gold structure
  (everything forced except value contents). Deltas between rungs price
  formatting, selection, and value generation separately; grammar vs
  free is the constraint-tax measurement.
- copy probe: eval samples whose value is copyable get that literal
  replaced by an unseen random one in prompt and reference; gold-structure
  decoding then measures pure copying of never-trained literals, by
  literal position and length (the induction question).
- calibration: confidence AUROC (correct vs incorrect exact match) on the
  free rung, beside the existing ECE.

The 4x-budget arm (steps 8000) excludes simple under-training. Hole
decoding ends when the model emits the template's next delimiter byte, so
a value containing its own closing quote is scored as truncated — a known
ceiling, irrelevant at these surfaces (no reference value contains one).

The training node declares `resources={"gpus": 1}`; the engine must know
the card is there or it trains on the CPU. `.fluksio/env` carries
`FLOW_GPUS=1` for this checkout; elsewhere `fluksio serve --gpus 1`.

    uv run fluksio sync dev/s7_bottleneck
    uv run fluksio sweep s7_bottleneck --param corpus=ha,mobile --param seed=0,1,2
    uv run fluksio sweep s7_bottleneck --param corpus=ha,mobile --param steps=8000 --param seed=0

Self-check (template decode, span labels, waterfall classes, trie):
`uv run python dev/s7_bottleneck/study.py`
"""

import itertools
import math
import pathlib
import random
import tempfile

import fluksio
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from fluksio import Flow, Port, node

from wren import tokenizer
from wren.dataset import render_call
from wren.eval import calibration_error, parse_calls, score
from wren.model import Config, Wren, _decode_step, generate_batch, save
from wren.train import muon, sample_batch, train_step

# The reference protocols (DECISIONS.md s2/s3 setups).
PROTOCOL = {
    "ha": {"context": 384, "seq_len": 256, "n_new": 192},
    "mobile": {"context": 768, "seq_len": 512, "n_new": 192},
}
HOLE_CAP = 64  # longest generated value/argument-list piece per hole


def wordform(value) -> str:
    """The prompt-side surface of a value: HA device slugs
    (`light.kitchen_lamp`) appear in prompts as their entity words
    (`kitchen lamp`); everything else is its own surface."""
    return str(value).split(".")[-1].replace("_", " ").strip().lower()


def copyable(value, prompt: str) -> bool:
    """A value is copyable when its text — verbatim or as its word form —
    appears in the prompt (case-insensitive): the model retrieves it,
    at most re-slugging. Anything else it must derive."""
    low = prompt.lower()
    return str(value).strip().lower() in low or wordform(value) in low


def own_lines(generated: str) -> str:
    """The model's continuation up to its first empty line -- past that
    it is hallucinating the next training sample, not answering."""
    return "\n".join(itertools.takewhile(bool, generated.split("\n")))


def call_parts(name: str, args: dict, prompt: str) -> list[tuple[str, str]]:
    """One rendered call line as (text, byte-class) pieces; concatenated
    they reproduce render_call exactly (pinned by the self-check)."""
    parts = [(">> ", "syntax"), (name, "name"), ("(", "syntax")]
    for j, (k, v) in enumerate(sorted(args.items())):
        if j:
            parts.append((", ", "syntax"))
        parts += [(k, "key"), ("=", "syntax")]
        parts.append((repr(v), "value_copy" if copyable(v, prompt) else "value_der"))
    parts.append((")", "syntax"))
    return parts


# ---------------------------------------------------------------- waterfall


def waterfall(predictions: list[str], samples: list) -> dict:
    """Sample counts per failure class, plus argument-item counts."""
    counts = dict.fromkeys(
        [
            "exact",
            "values_copyable",
            "values_derived",
            "wrong_name",
            "abstain",
            "ill_formed",
        ],
        0,
    )
    items = {"copy_total": 0, "copy_missed": 0, "der_total": 0, "der_missed": 0}
    for text, (prompt, ref) in zip(predictions, samples, strict=True):
        pred = parse_calls(text)
        ref = [(n, dict(a)) for n, a in ref]
        p = (
            {(i, n, k, repr(v)) for i, (n, a) in enumerate(pred) for k, v in a.items()}
            if pred
            else set()
        )
        r = {(i, n, k, repr(v)) for i, (n, a) in enumerate(ref) for k, v in a.items()}
        for i, n, k, rv in r:
            copy = copyable(ref[i][1][k], prompt)
            items["copy_total" if copy else "der_total"] += 1
            if (i, n, k, rv) not in p:
                items["copy_missed" if copy else "der_missed"] += 1
        if pred is None:
            counts["ill_formed"] += 1
        elif not pred:
            counts["abstain"] += 1
        elif [n for n, _ in pred] != [n for n, _ in ref]:
            counts["wrong_name"] += 1
        elif pred == ref:
            counts["exact"] += 1
        else:
            missed = [ref[i][1][k] for i, n, k, rv in r - p]
            derived = any(not copyable(v, prompt) for v in missed)
            counts["values_derived" if derived else "values_copyable"] += 1
    return {"samples": counts, "items": items}


# ----------------------------------------------------------------- span bpb


@nnx.jit
def _byte_nats(model: Wren, tokens: jax.Array) -> jax.Array:
    """(batch, seq-1) per-byte cross-entropy: nats of tokens[:, 1:]."""
    logits = model(tokens)[:, :-1]
    return optax.softmax_cross_entropy_with_integer_labels(logits, tokens[:, 1:])


def span_bpb(model: Wren, samples: list, batch_size: int = 16) -> dict:
    """Mean bits/byte per byte class over the reference texts."""
    texts, labels = [], []
    for prompt, calls in samples:
        parts = [(prompt, "prompt")]
        for name, args in calls:
            parts += call_parts(name, dict(args), prompt) + [("\n", "syntax")]
        text = "".join(t for t, _ in parts)
        lab = [c for t, c in parts for _ in t.encode()]
        toks = tokenizer.encode(text)[-model.cfg.context :]
        texts.append(toks)
        labels.append(lab[-len(toks) :])
    nats = dict.fromkeys(
        ["prompt", "syntax", "name", "key", "value_copy", "value_der"], 0.0
    )
    n = dict.fromkeys(nats, 0)
    order = np.argsort([len(t) for t in texts])  # bucket similar lengths
    for i in range(0, len(order), batch_size):
        idx = order[i : i + batch_size]
        width = -(-max(len(texts[j]) for j in idx) // 64) * 64  # pad: few retraces
        buf = np.zeros((len(idx), width), dtype=np.int32)
        for row, j in enumerate(idx):
            buf[row, : len(texts[j])] = texts[j]
        loss = np.asarray(_byte_nats(model, jnp.asarray(buf)))
        for row, j in enumerate(idx):
            for pos in range(1, len(texts[j])):  # byte 0 has no prediction
                nats[labels[j][pos]] += float(loss[row, pos - 1])
                n[labels[j][pos]] += 1
    return {c: nats[c] / n[c] / math.log(2) if n[c] else None for c in nats}


# ----------------------------------------------------- template decoding


def _trie_allowed(names: list[bytes], prefix: bytes, delim: int) -> list[int]:
    """Bytes that extend `prefix` toward some name; the delimiter once the
    prefix is a complete name."""
    allowed = {
        n[len(prefix)] for n in names if len(n) > len(prefix) and n.startswith(prefix)
    }
    if any(n == prefix for n in names):
        allowed.add(delim)
    return sorted(allowed)


def template_decode(
    model: Wren,
    templates: list[list],
    trie_names: list[str] | None = None,
    gen_batch: int = 64,
) -> list[list[list[int]]]:
    """Decode each template — segments alternating forced byte lists and
    holes ({"cap": int, "trie": bool}) — and return the per-sample hole
    outputs. Forced bytes are teacher-forced through the KV cache; in a
    hole the model's argmax continues until it emits the next segment's
    first byte (the delimiter) or the cap is reached. Templates start and
    end with forced segments; a first segment longer than the context
    budget loses its head, as in generate_batch.
    """
    names = [n.encode() for n in trie_names or []]

    def size(segs: list) -> int:
        return sum(len(s) if isinstance(s, list) else s["cap"] + 1 for s in segs)

    results = []
    for start in range(0, len(templates), gen_batch):
        seqs = []
        for segs in templates[start : start + gen_batch]:
            over = size(segs) - model.cfg.context
            if over > 0:  # trim the prompt's head, as generate_batch does
                cut = min(over, len(segs[0]) - 1)
                segs = [segs[0][cut:]] + segs[1:]
            over = size(segs) - model.cfg.context
            if over > 0:  # still oversize: shrink the hole caps evenly
                n_holes = sum(not isinstance(s, list) for s in segs)
                shrink = -(-over // n_holes)
                segs = [
                    s if isinstance(s, list) else s | {"cap": max(8, s["cap"] - shrink)}
                    for s in segs
                ]
            seqs.append(segs)
        batch, width = len(seqs), max(size(segs) for segs in seqs)
        buf = np.zeros((batch, width), dtype=np.int32)
        si = np.zeros(batch, dtype=int)  # current segment
        off = np.zeros(batch, dtype=int)  # offset within it
        done = np.zeros(batch, dtype=bool)
        holes: list[list[list[int]]] = [[] for _ in range(batch)]
        for i, segs in enumerate(seqs):
            buf[i, 0] = segs[0][0]  # every template starts forced
            off[i] = 1
        caches = model.init_cache(batch)
        for t in range(width - 1):
            if done.all():
                break
            logits, caches = _decode_step(
                model, jnp.asarray(buf[:, t : t + 1]), caches, jnp.int32(t)
            )
            probs = np.asarray(logits)
            for i in range(batch):
                if done[i]:
                    continue
                segs = seqs[i]
                # Step out of exhausted forced segments before choosing the
                # byte — a hole may follow a one-byte delimiter.
                while isinstance(segs[si[i]], list) and off[i] >= len(segs[si[i]]):
                    si[i] += 1
                    off[i] = 0
                    if si[i] == len(segs):
                        done[i] = True
                        break
                    if not isinstance(segs[si[i]], list):
                        holes[i].append([])
                if done[i]:
                    continue
                seg = segs[si[i]]
                if isinstance(seg, list):  # forced byte, model's logits unused
                    buf[i, t + 1] = seg[off[i]]
                    off[i] += 1
                    continue
                delim = segs[si[i] + 1][0]  # holes are followed by forced
                if seg.get("trie"):
                    prefix = bytes(holes[i][-1])
                    allowed = _trie_allowed(names, prefix, delim)
                    nxt = allowed[int(np.argmax(probs[i, allowed]))]
                else:
                    nxt = int(np.argmax(probs[i]))
                if nxt == delim or len(holes[i][-1]) >= seg["cap"]:
                    buf[i, t + 1] = delim  # the hole closed; delimiter forced
                    si[i] += 1
                    off[i] = 1
                else:
                    buf[i, t + 1] = nxt
                    holes[i][-1].append(nxt)
        results += holes
    return results


def rung_templates(samples: list, rung: str) -> tuple[list[list], list[list[str]]]:
    """(templates, per-hole labels) for one oracle rung.

    gold_structure: holes are value contents (string values between their
    forced quotes; other literals whole). grammar: additionally the name
    is a trie hole. gold_name: one hole per call covering everything after
    `>> name(` to the end of the line.
    """
    templates, labels = [], []
    for prompt, calls in samples:
        segs: list = [tokenizer.encode(prompt)]
        labs = []

        def force(text: str, segs: list = segs) -> None:
            if isinstance(segs[-1], list):
                segs[-1] += tokenizer.encode(text)
            else:
                segs.append(tokenizer.encode(text))

        for name, args in calls:
            if rung == "gold_name":
                force(f">> {name}(")
                segs.append({"cap": 3 * HOLE_CAP, "trie": False})
                labs.append("line")
                force("\n")
                continue
            force(">> ")
            if rung == "grammar":
                segs.append({"cap": HOLE_CAP, "trie": True})
                labs.append("g_name")
            else:
                force(name)
            force("(")
            for j, (k, v) in enumerate(sorted(dict(args).items())):
                if j:
                    force(", ")
                force(f"{k}=")
                rep = repr(v)
                cls = "value_copy" if copyable(v, prompt) else "value_der"
                if isinstance(v, str):
                    force(rep[0])
                    segs.append({"cap": HOLE_CAP, "trie": False})
                    labs.append(cls)
                    force(rep[-1])
                else:
                    segs.append({"cap": HOLE_CAP, "trie": False})
                    labs.append(cls)
            force(")\n")
        force("\n")
        templates.append(segs)
        labels.append(labs)
    return templates, labels


def render_from_holes(template: list, holes: list[list[int]]) -> str:
    """The decoded text (prompt included) with hole outputs spliced in."""
    out, h = [], iter(holes)
    for seg in template:
        out += seg if isinstance(seg, list) else next(h, [])
    return tokenizer.decode(out)


def hole_expected(samples: list, rung: str) -> list[list[str]]:
    """Per sample, the reference content of each hole, aligned with
    rung_templates' hole order (gold_structure/grammar value holes only)."""
    expected = []
    for prompt, calls in samples:
        exp = []
        for name, args in calls:
            if rung == "grammar":
                exp.append(name)
            for _, v in sorted(dict(args).items()):
                rep = repr(v)
                exp.append(rep[1:-1] if isinstance(v, str) else rep)
        expected.append(exp)
    return expected


def auroc(scores: list[float], labels: list[bool]) -> float | None:
    """Rank-based AUROC of scores for the positive labels."""
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    order = np.argsort(np.array(scores, dtype=float))
    rank = np.empty(len(scores))
    rank[order] = np.arange(len(scores))
    r_pos = sum(rank[i] for i, y in enumerate(labels) if y)
    return float((r_pos - len(pos) * (len(pos) - 1) / 2) / (len(pos) * len(neg)))


# ------------------------------------------------------------------ nodes


@node(
    requires=[Port("corpus", "str")],
    provides=[
        Port("train_text", "str"),
        Port("eval_samples", "json"),
        Port("surface_names", "json"),
    ],
)
def dataset(corpus: str) -> dict:
    from wren.dataset import home_assistant, irrelevant, mobile_actions, to_text

    if corpus == "ha":
        train, test = home_assistant()
        random.Random(0).shuffle(train)
        mixed = train
    else:  # s2's stream: positives plus ~2% synthetic negatives
        train, test = mobile_actions()
        neg_train, _ = irrelevant()
        mixed = train + neg_train
        random.Random(0).shuffle(mixed)
        test = [s for s in test if s[1]]  # the autopsy scores positives
    names = sorted({n for _, calls in mixed for n, _ in calls})
    return {
        "train_text": to_text(mixed),
        "eval_samples": [[p, [[n, a] for n, a in calls]] for p, calls in test],
        "surface_names": names,
    }


@node(
    requires=[
        Port("train_text", "str"),
        Port("eval_samples", "json"),
        Port("surface_names", "json"),
        Port("corpus", "str"),
        Port("lr", "float"),
        Port("seed", "int"),
        Port("steps", "int"),
        Port("n_eval", "int"),
    ],
    provides=[
        Port("loss", "float", stream=True),
        Port("final_loss", "float"),
        Port("exact_match", "float"),
        Port("arg_f1", "float"),
        Port("report", "json"),
        Port("weights", "artifact"),
    ],
    # preallocate=false: jax otherwise takes 75% of the card at import, so a
    # second engine (a `--local` run beside a running `serve`) collides with
    # this one instead of fitting beside it.
    resources={"gpus": 1, "env": {"XLA_PYTHON_CLIENT_PREALLOCATE": "false"}},
)
def study(
    train_text: str,
    eval_samples: list,
    surface_names: list,
    corpus: str,
    lr: float,
    seed: int,
    steps: int = 2000,
    n_eval: int = 0,
    batch_size: int = 16,
    gen_batch: int = 64,
    log_every: int = 50,
    n_probe: int = 400,
) -> dict:
    import os

    if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        raise RuntimeError("no card declared: `fluksio serve --gpus 1`")
    if jax.default_backend() == "cpu":
        raise RuntimeError("jax sees no card: `uv sync --extra gpu`, restart engine")

    proto = PROTOCOL[corpus]
    model = Wren(Config(context=proto["context"]), rngs=nnx.Rngs(seed))
    optimizer = nnx.Optimizer(model, muon(lr), wrt=nnx.Param)
    data = jnp.array(tokenizer.encode(train_text), dtype=jnp.int32)
    key = jax.random.PRNGKey(seed)
    loss = float("nan")
    for step in range(steps):
        key, sub = jax.random.split(key)
        x, y = sample_batch(data, batch_size, proto["seq_len"], sub)
        loss = float(train_step(model, optimizer, x, y))
        if step % log_every == 0 or step == steps - 1:
            fluksio.emit(loss=loss)

    samples = [(p, [(n, a) for n, a in calls]) for p, calls in eval_samples]
    if n_eval:
        samples = samples[:n_eval]
    n_new = proto["n_new"]

    # Rung 1 — free decode, the s2/s3 protocol, plus calibration raw material.
    predictions, confidences = [], []
    for i in range(0, len(samples), gen_batch):
        prompts = [tokenizer.encode(p) for p, _ in samples[i : i + gen_batch]]
        outs, confs = generate_batch(model, prompts, n_new, stop=(10, 10))
        predictions += [own_lines(tokenizer.decode(o)) for o in outs]
        confidences += confs
    references = [calls for _, calls in samples]
    free = score(predictions, references)
    corrects = [parse_calls(p) == r for p, r in zip(predictions, references)]
    ladder: dict[str, dict] = {
        "free": free
        | {
            "ece": float(calibration_error(confidences, corrects)),
            "auroc": auroc(confidences, corrects),
        }
    }

    # Rungs 2–4 — template decodes.
    for rung in ("grammar", "gold_name", "gold_structure"):
        templates, labels = rung_templates(samples, rung)
        holes = template_decode(
            model, templates, trie_names=surface_names, gen_batch=gen_batch
        )
        texts = [
            own_lines(render_from_holes(t, h)[len(p.encode()) :])
            for t, h, (p, _) in zip(templates, holes, samples)
        ]
        ladder[rung] = score(texts, references)
        if rung in ("gold_structure", "grammar"):
            expected = hole_expected(samples, rung)
            hit = dict.fromkeys(["value_copy", "value_der", "g_name"], 0)
            tot = dict.fromkeys(hit, 0)
            for h, exp, labs in zip(holes, expected, labels):
                for got, want, lab in zip(h, exp, labs):
                    tot[lab] += 1
                    hit[lab] += tokenizer.decode(got) == want
            ladder[rung]["holes"] = {
                k: hit[k] / tot[k] if tot[k] else None for k in hit
            }

    # Copy probe — unseen literals substituted into copyable value slots.
    rng = random.Random(seed)
    probes = []
    for prompt, calls in samples:
        for ci, (name, args) in enumerate(calls):
            for k, v in args.items():
                if not (isinstance(v, str) and len(v) >= 3):
                    continue
                # The literal to replace in the prompt: the value itself
                # (mobile) or its word form (HA slugs).
                surface = next(
                    (s for s in (v, wordform(v)) if s.lower() in prompt.lower()), None
                )
                if surface is None:
                    continue
                lit = "".join(
                    ch if not ch.isalnum() else rng.choice("abcdefghijklmnopqrstuvwxyz")
                    for ch in surface
                )
                start = prompt.lower().index(surface.lower())
                new_prompt = prompt[:start] + lit + prompt[start + len(surface) :]
                # The value gets the same substitution in its own format:
                # verbatim, or re-slugged under the entity prefix.
                new_v = (
                    lit if surface == v else v[: -len(surface)] + lit.replace(" ", "_")
                )
                new_calls = [
                    (n, {**a, k: new_v} if i == ci else dict(a))
                    for i, (n, a) in enumerate(calls)
                ]
                probes.append(
                    (new_prompt, new_calls, new_v, start / max(len(prompt), 1))
                )
                break
            else:
                continue
            break
    probes = probes[:n_probe]
    copy_report = {"n": len(probes)}
    if probes:
        p_samples = [(p, c) for p, c, _, _ in probes]
        templates, _ = rung_templates(p_samples, "gold_structure")
        holes = template_decode(model, templates, gen_batch=gen_batch)
        results = []
        for (prompt, calls, lit, pos), h in zip(probes, holes):
            exp = hole_expected([(prompt, calls)], "gold_structure")[0]
            # The substituted literal is plain lowercase, so its repr adds
            # no escapes: the hole is right iff it reproduces lit exactly.
            ok = any(
                want == lit and tokenizer.decode(got) == lit
                for got, want in zip(h, exp)
            )
            results.append((ok, pos, len(lit)))
        copy_report["accuracy"] = sum(ok for ok, _, _ in results) / len(results)
        for name_, lo, hi in (
            ("early", 0.0, 1 / 3),
            ("mid", 1 / 3, 2 / 3),
            ("late", 2 / 3, 1.01),
        ):
            grp = [ok for ok, pos, _ in results if lo <= pos < hi]
            copy_report[f"acc_{name_}"] = sum(grp) / len(grp) if grp else None
        short = [ok for ok, _, ln in results if ln <= 12]
        long_ = [ok for ok, _, ln in results if ln > 12]
        copy_report["acc_short"] = sum(short) / len(short) if short else None
        copy_report["acc_long"] = sum(long_) / len(long_) if long_ else None

    report = {
        "waterfall": waterfall(predictions, samples),
        "span_bpb": span_bpb(model, samples, batch_size),
        "ladder": ladder,
        "copy_probe": copy_report,
        "n_eval": len(samples),
    }

    path = pathlib.Path(tempfile.mkdtemp()) / "weights.npz"
    save(model, path)
    ref = fluksio.save_artifact(path.read_bytes(), name="weights.npz")
    return {
        "final_loss": loss,
        "exact_match": free["exact_match"],
        "arg_f1": free["arg_f1"],
        "report": report,
        "weights": ref,
    }


Flow(
    "s7_bottleneck",
    title="S7 — bottleneck autopsy",
    nodes=[dataset, study],
    inputs=[
        Port("corpus", "str", initial="ha"),
        Port("lr", "float", initial=3e-3),
        Port("seed", "int", initial=0),
        Port("steps", "int", initial=2000),
        Port("n_eval", "int", initial=0),
    ],
    outputs=[
        "final_loss",
        "exact_match",
        "arg_f1",
        # json blob, declared as an output so `export runs` carries it --
        # neither export surface reaches plain node ports.
        "report",
    ],
)


if __name__ == "__main__":
    # Self-checks, here rather than in a pytest file beside this one: a test
    # module importing `study` gives the sync walk a second copy of it, and
    # two Flow declarations of the same name.
    _prompt = "turn on the kitchen light\n"
    _calls = [("light.turn_on", {"name": "kitchen light", "brightness": 3})]

    # call_parts reproduces render_call byte for byte, labels tile it.
    _parts = call_parts(*_calls[0], _prompt)
    _text = "".join(t for t, _ in _parts)
    assert _text == render_call(*_calls[0]), _text
    assert sum(len(t.encode()) for t, _ in _parts) == len(_text.encode())
    # 'kitchen light' is in the prompt, 3 is not.
    _labels = [c for t, c in _parts for _ in t.encode()]
    assert "value_copy" in _labels and "value_der" in _labels

    # waterfall: one crafted prediction per class.
    _samples = [(_prompt, _calls)] * 5
    _preds = [
        ">> light.turn_on(brightness=3, name='kitchen light')",  # exact
        ">> light.turn_on(brightness=3, name='bedroom light')",  # copyable miss
        ">> light.turn_on(brightness=7, name='kitchen light')",  # derived miss
        ">> media.play(name='kitchen light')",  # wrong name
        ">> light.turn_on(",  # ill-formed
    ]
    _w = waterfall(_preds, _samples)["samples"]
    for _cls in (
        "exact",
        "values_copyable",
        "values_derived",
        "wrong_name",
        "ill_formed",
    ):
        assert _w[_cls] == 1, (_cls, _w)

    # template decode on a tiny random model: forced-only templates come
    # back verbatim; a trie hole emits a valid surface name.
    _cfg = Config(dim=64, n_layers=2, context=256)
    _model = Wren(_cfg, rngs=nnx.Rngs(0))
    _tmpl, _labs = rung_templates(_samples[:1], "gold_structure")
    _holes = template_decode(_model, _tmpl)
    assert len(_holes[0]) == len(_labs[0]) == 2  # two value holes
    _full = render_from_holes(_tmpl[0], _holes[0])
    assert _full.startswith(_prompt) and ">> light.turn_on(" in _full

    _tmplg, _labsg = rung_templates(_samples[:1], "grammar")
    _names = ["light.turn_on", "light.turn_off", "media.play"]
    _holesg = template_decode(_model, _tmplg, trie_names=_names)
    assert tokenizer.decode(_holesg[0][0]) in _names  # trie forces a valid name

    _tmpln, _ = rung_templates(_samples[:1], "gold_name")
    _holesn = template_decode(_model, _tmpln)
    assert len(_holesn[0]) == 1  # one free hole per call

    # hole_expected aligns with the template's hole order.
    assert hole_expected(_samples[:1], "gold_structure")[0] == ["3", "kitchen light"]
    assert hole_expected(_samples[:1], "grammar")[0] == [
        "light.turn_on",
        "3",
        "kitchen light",
    ]

    # auroc: separable scores give 1.0, swapped give 0.0.
    assert auroc([0.9, 0.8, 0.2, 0.1], [True, True, False, False]) == 1.0
    assert auroc([0.1, 0.2, 0.8, 0.9], [True, True, False, False]) == 0.0

    print("s7 self-checks ok")
