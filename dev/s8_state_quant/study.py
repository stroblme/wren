"""Compare fp32, int8, and ternary training on home and mobile-actions samples.

Quantized forward passes use fp32 master weights and straight-through
gradients. Embeddings and norms remain fp32. See README.md for the
protocol and reproduction commands.
"""

import itertools
import pathlib
import re
import tempfile

import fluksio
import jax
import jax.numpy as jnp
from flax import nnx
from fluksio import Flow, Port, node

from wren import tokenizer
from wren.eval import abstain_rate, parse_calls, score
from wren.model import Config, Wren, generate_batch, save
from wren.train import muon, sample_batch, train_step

# Per-corpus protocol: home windows span a full sample (pruned state block
# included); mobile keeps s2's sizing so its numbers stay comparable.
PROTOCOL = {
    "home": {"context": 384, "seq_len": 384, "n_new": 96},
    "mobile": {"context": 768, "seq_len": 512, "n_new": 192},
}

# A generated device id: kind letter + 4 chars of the id alphabet. Matches
# no slug, state word, temperature or delay value (test_dataset pins the
# alphabet), so it separates copy-requiring references from the rest.
_ID = re.compile(r"^[lmhsw][a-z2-9]{4}$")


def _ste(w: jax.Array, quant) -> jax.Array:
    """Use quantized weights in the forward pass and pass gradients through unchanged."""
    return w + jax.lax.stop_gradient(quant(w) - w)


def _int8(w: jax.Array) -> jax.Array:
    scale = jnp.max(jnp.abs(w)) / 127 + 1e-12
    return jnp.round(w / scale) * scale


def _ternary(w: jax.Array) -> jax.Array:
    """Quantize weights to {-1, 0, +1} times their mean absolute value."""
    scale = jnp.mean(jnp.abs(w)) + 1e-12
    return jnp.clip(jnp.round(w / scale), -1, 1) * scale


class Int8Linear(nnx.Linear):
    def __call__(self, inputs: jax.Array, out_sharding=None) -> jax.Array:
        return inputs @ _ste(self.kernel[...], _int8)


class TernaryLinear(nnx.Linear):
    def __call__(self, inputs: jax.Array, out_sharding=None) -> jax.Array:
        return inputs @ _ste(self.kernel[...], _ternary)


QUANT = {"fp32": None, "int8": Int8Linear, "ternary": TernaryLinear}
BITS = {"fp32": 32, "int8": 8, "ternary": 2}


def quantize_(model: Wren, mode: str) -> Wren:
    """Use quantized attention projections while retaining fp32 master parameters."""
    if (cls := QUANT[mode]) is not None:
        for block in model.blocks:
            block.wqkv.__class__ = cls
            block.wo.__class__ = cls
    return model


def packed_mb(model: Wren, mode: str) -> float:
    """Estimate packed megabytes using the arm's attention-weight bit width."""
    quantized = sum(b.wqkv.kernel.size + b.wo.kernel.size for b in model.blocks)
    rest = model.n_params() - quantized
    return (quantized * BITS[mode] + rest * 32) / 8 / 1e6


def own_lines(generated: str) -> str:
    """Keep the generated continuation through its first blank line."""
    return "\n".join(itertools.takewhile(bool, generated.split("\n")))


@node(
    requires=[Port("corpus", "str"), Port("n_train", "int")],
    provides=[Port("train_text", "str"), Port("eval_samples", "json")],
)
def dataset(corpus: str, n_train: int) -> dict:
    """Build home samples or the mobile-actions stream with synthetic negatives.

    `n_train` controls the number of unique home training samples; mobile-actions
    uses its own split and ignores this parameter.
    """
    import random

    from wren.dataset import home, irrelevant, mobile_actions, to_text

    if corpus == "mobile":
        train, evals = mobile_actions()
        neg_train, neg_eval = irrelevant()
        mixed = train + neg_train
        random.Random(0).shuffle(mixed)
        train, evals = mixed, evals + neg_eval
    else:
        train, evals = home(n_train=n_train)
    return {
        "train_text": to_text(train),
        "eval_samples": [[p, [[n, a] for n, a in calls]] for p, calls in evals],
    }


@node(
    requires=[
        Port("train_text", "str"),
        Port("eval_samples", "json"),
        Port("corpus", "str"),
        Port("quant", "str"),
        Port("lr", "float"),
        Port("seed", "int"),
        Port("steps", "int"),
    ],
    provides=[
        Port("loss", "float", stream=True),
        Port("final_loss", "float"),
        Port("exact_match", "float"),
        Port("well_formed", "float"),
        Port("name_accuracy", "float"),
        Port("arg_f1", "float"),
        Port("abstain_acc", "float"),
        Port("copy_exact", "float"),
        Port("nocopy_exact", "float"),
        Port("id_copy_acc", "float"),
        Port("n_params", "float"),
        Port("packed_mb", "float"),
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
    corpus: str,
    quant: str,
    lr: float,
    seed: int,
    steps: int = 8000,
    batch_size: int = 32,
    gen_batch: int = 64,
    log_every: int = 50,
) -> dict:
    import os

    if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        raise RuntimeError("no card declared: `fluksio serve --gpus 1`")
    if jax.default_backend() == "cpu":
        raise RuntimeError("jax sees no card: `uv sync --group cuda12`, restart engine")

    proto = PROTOCOL[corpus]
    model = quantize_(
        Wren(Config(context=proto["context"]), rngs=nnx.Rngs(seed)), quant
    )
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

    samples = [(p, [(n, dict(a)) for n, a in calls]) for p, calls in eval_samples]
    predictions = []
    for i in range(0, len(samples), gen_batch):
        prompts = [tokenizer.encode(p) for p, _ in samples[i : i + gen_batch]]
        outs, _ = generate_batch(model, prompts, proto["n_new"], stop=(10, 10))
        predictions += [own_lines(tokenizer.decode(o)) for o in outs]

    # Positives carry the copy question (unseen ids); negatives the abstention.
    positives = [i for i, (_, calls) in enumerate(samples) if calls]
    negatives = [i for i, (_, calls) in enumerate(samples) if not calls]
    metrics = score(
        [predictions[i] for i in positives], [samples[i][1] for i in positives]
    )

    # The copy split: exact match over samples whose reference needs an
    # unseen device id vs the rest, and per-value id reproduction. -1 when
    # a class is empty (the mobile corpus has no generated ids).
    parsed = {i: parse_calls(predictions[i]) or [] for i in positives}
    needs = {
        i: any(_ID.match(str(v)) for _, a in samples[i][1] for v in a.values())
        for i in positives
    }
    copy_idx = [i for i in positives if needs[i]]
    nocopy_idx = [i for i in positives if not needs[i]]
    id_hit = id_total = 0
    for i in copy_idx:
        values = {
            (ci, k): v for ci, (_, a) in enumerate(parsed[i]) for k, v in a.items()
        }
        for ci, (_, a) in enumerate(samples[i][1]):
            for k, v in a.items():
                if _ID.match(str(v)):
                    id_total += 1
                    id_hit += values.get((ci, k)) == v

    def exact_over(idx: list[int]) -> float:
        return sum(parsed[i] == samples[i][1] for i in idx) / len(idx) if idx else -1.0

    path = pathlib.Path(tempfile.mkdtemp()) / "weights.npz"
    save(model, path)
    return {
        "final_loss": loss,
        "copy_exact": exact_over(copy_idx),
        "nocopy_exact": exact_over(nocopy_idx),
        "id_copy_acc": id_hit / id_total if id_total else -1.0,
        "abstain_acc": abstain_rate([predictions[i] for i in negatives]),
        "n_params": float(model.n_params()),
        "packed_mb": packed_mb(model, quant),
        "weights": fluksio.save_artifact(path.read_bytes(), name="weights.npz"),
    } | metrics


Flow(
    "s8_state_quant",
    title="S8 — state corpus × quantization",
    nodes=[dataset, study],
    inputs=[
        Port("corpus", "str", initial="home"),
        Port("quant", "str", initial="fp32"),
        Port("lr", "float", initial=3e-3),
        Port("seed", "int", initial=0),
        Port("steps", "int", initial=8000),
        Port("n_train", "int", initial=8000),
    ],
    outputs=[
        "final_loss",
        "exact_match",
        "well_formed",
        "name_accuracy",
        "arg_f1",
        "abstain_acc",
        "copy_exact",
        "nocopy_exact",
        "id_copy_acc",
        "n_params",
        "packed_mb",
    ],
)


if __name__ == "__main__":
    # Quantizer values: ternary lands on {-s, 0, +s}, int8 on a 255-point grid.
    _w = jax.random.normal(jax.random.PRNGKey(0), (64, 64))
    _tq = _ternary(_w)
    assert len(jnp.unique(_tq)) <= 3, jnp.unique(_tq)
    assert jnp.allclose(jnp.unique(jnp.abs(_tq)), jnp.unique(jnp.abs(_tq))[0:2])
    assert len(jnp.unique(_int8(_w))) <= 255

    # Straight-through: the gradient equals the unquantized gradient.
    _x = jax.random.normal(jax.random.PRNGKey(1), (4, 64))
    _g = jax.grad(lambda w: jnp.sum(_x @ _ste(w, _ternary)))(_w)
    _g_plain = jax.grad(lambda w: jnp.sum(_x @ w))(_w)
    assert jnp.allclose(_g, _g_plain)

    # Arm parity: fp32 mode leaves the stock forward byte-identical, and a
    # quantized arm changes the forward without touching the parameters.
    _cfg = Config(dim=64, n_layers=2, context=64)
    _stock = Wren(_cfg, rngs=nnx.Rngs(0))
    _q = quantize_(Wren(_cfg, rngs=nnx.Rngs(0)), "ternary")
    _tok = jnp.zeros((1, 8), dtype=jnp.int32)
    _ref = Wren(_cfg, rngs=nnx.Rngs(0))(_tok)
    assert (quantize_(Wren(_cfg, rngs=nnx.Rngs(0)), "fp32")(_tok) == _ref).all()
    assert not (_q(_tok) == _ref).all()
    assert (_q.blocks[0].wqkv.kernel[...] == _stock.blocks[0].wqkv.kernel[...]).all()

    # A ternary model trains: loss falls on a repetitive stream.
    _opt = nnx.Optimizer(_q, muon(3e-3), wrt=nnx.Param)
    _data = jnp.array(tokenizer.encode("ab\n" * 200), dtype=jnp.int32)
    _first = _last = None
    for _step in range(30):
        _xb, _yb = sample_batch(_data, 8, 16, jax.random.PRNGKey(_step))
        _last = float(train_step(_q, _opt, _xb, _yb))
        _first = _last if _first is None else _first
    assert _last < _first, (_first, _last)

    # packed_mb at the default shape: ternary ships an order of magnitude
    # smaller (the tiny self-check shape is embedding-dominated).
    _full = Wren(Config(), rngs=nnx.Rngs(0))
    assert packed_mb(_full, "fp32") / packed_mb(_full, "ternary") > 8

    print("s8 self-checks ok")
