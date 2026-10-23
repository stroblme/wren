"""Train s8's configuration on the s10 corpus, with and without noise.

s8's shipped model is its pruned ternary run (256k samples, 16000 steps). This
study keeps that configuration and changes only the corpus: switches, pairs of
areas, area-less commands, unknown places and things, and a share of commands
replaced by what a recogniser heard (`speech.py noisy`). See README.md.
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

N_NEW = 96  # the home protocol's answer budget (s8)
# A generated device id: kind letter (switches are `p`) + 4 chars.
_ID = re.compile(r"^[lmhswp][a-z2-9]{4}$")


def _ternary(w: jax.Array) -> jax.Array:
    """Quantize weights to {-1, 0, +1} times their mean absolute value."""
    scale = jnp.mean(jnp.abs(w)) + 1e-12
    return jnp.clip(jnp.round(w / scale), -1, 1) * scale


def _ste(w: jax.Array) -> jax.Array:
    """The ternary weights forward, the unquantized gradient back."""
    return w + jax.lax.stop_gradient(_ternary(w) - w)


class TernaryLinear(nnx.Linear):
    def __call__(self, inputs: jax.Array, out_sharding=None) -> jax.Array:
        return inputs @ _ste(self.kernel[...])


def ternary_(model: Wren) -> Wren:
    """s8's ternary arm: quantized attention projections, fp32 master weights."""
    for block in model.blocks:
        block.wqkv.__class__ = TernaryLinear
        block.wo.__class__ = TernaryLinear
    return model


@node(
    requires=[Port("n_train", "int"), Port("noisy", "str"), Port("share", "float")],
    provides=[Port("train_text", "str"), Port("eval_samples", "json")],
)
def dataset(n_train: int, noisy: str, share: float) -> dict:
    """The s10 home corpus; `noisy` is the path of speech.py's table, or empty."""
    from wren.dataset import home, load_noisy, to_text

    table = load_noisy(noisy) if noisy and share > 0 else None
    train, evals = home(n_train=n_train, noisy=table, share=share)
    return {
        "train_text": to_text(train),
        "eval_samples": [[p, [[n, a] for n, a in calls]] for p, calls in evals],
    }


@node(
    requires=[
        Port("train_text", "str"),
        Port("eval_samples", "json"),
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
        Port("id_copy_acc", "float"),
        Port("weights", "artifact"),
    ],
    # The card is shared with the live speech service and its LLM: grow as
    # needed (s8), and never past a fifth of it.
    resources={
        "gpus": 1,
        "env": {
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.2",
        },
    },
)
def study(
    train_text: str,
    eval_samples: list,
    lr: float,
    seed: int,
    steps: int = 16000,
    batch_size: int = 32,
    gen_batch: int = 64,
    log_every: int = 50,
) -> dict:
    if jax.default_backend() == "cpu":
        raise RuntimeError("jax sees no card: `uv sync --group cuda12`, restart engine")

    cfg = Config()
    model = ternary_(Wren(cfg, rngs=nnx.Rngs(seed)))
    optimizer = nnx.Optimizer(model, muon(lr), wrt=nnx.Param)
    data = jnp.array(tokenizer.encode(train_text), dtype=jnp.int32)
    key = jax.random.PRNGKey(seed)
    loss = float("nan")
    for step in range(steps):
        key, sub = jax.random.split(key)
        x, y = sample_batch(data, batch_size, cfg.context, sub)
        loss = float(train_step(model, optimizer, x, y))
        if step % log_every == 0 or step == steps - 1:
            fluksio.emit(loss=loss)

    samples = [(p, [(n, dict(a)) for n, a in calls]) for p, calls in eval_samples]
    predictions = []
    for i in range(0, len(samples), gen_batch):
        prompts = [tokenizer.encode(p) for p, _ in samples[i : i + gen_batch]]
        outs, _ = generate_batch(model, prompts, N_NEW, stop=(10, 10))
        # Through the first blank line, as s8 reads an answer.
        predictions += [
            "\n".join(itertools.takewhile(bool, tokenizer.decode(o).split("\n")))
            for o in outs
        ]

    positives = [i for i, (_, calls) in enumerate(samples) if calls]
    negatives = [i for i, (_, calls) in enumerate(samples) if not calls]
    metrics = score(
        [predictions[i] for i in positives], [samples[i][1] for i in positives]
    )
    hit = total = 0
    for i in positives:
        got = parse_calls(predictions[i]) or []
        values = {(c, k): v for c, (_, a) in enumerate(got) for k, v in a.items()}
        for c, (_, a) in enumerate(samples[i][1]):
            for k, v in a.items():
                if _ID.match(str(v)):
                    total += 1
                    hit += values.get((c, k)) == v

    path = pathlib.Path(tempfile.mkdtemp()) / "weights.npz"
    save(model, path)
    return {
        "final_loss": loss,
        "abstain_acc": abstain_rate([predictions[i] for i in negatives]),
        "id_copy_acc": hit / total if total else -1.0,
        "weights": fluksio.save_artifact(path.read_bytes(), name="weights.npz"),
    } | metrics


Flow(
    "s10_fluksio_tail",
    title="S10 — the fluksio tail",
    nodes=[dataset, study],
    inputs=[
        Port("n_train", "int", initial=256000),
        Port("noisy", "str", initial=""),
        Port("share", "float", initial=0.0),
        Port("lr", "float", initial=3e-3),
        Port("seed", "int", initial=0),
        Port("steps", "int", initial=16000),
    ],
    outputs=[
        "final_loss",
        "exact_match",
        "well_formed",
        "name_accuracy",
        "arg_f1",
        "abstain_acc",
        "id_copy_acc",
    ],
)


if __name__ == "__main__":
    # The quantizer lands on three values and passes gradients straight through.
    _w = jax.random.normal(jax.random.PRNGKey(0), (64, 64))
    assert len(jnp.unique(_ternary(_w))) <= 3
    _x = jax.random.normal(jax.random.PRNGKey(1), (4, 64))
    _g = jax.grad(lambda w: jnp.sum(_x @ _ste(w)))(_w)
    assert jnp.allclose(_g, jax.grad(lambda w: jnp.sum(_x @ w))(_w))
    # A switch id counts as copied; a slug does not.
    assert _ID.match("pk3m9") and not _ID.match("garage")
    print("s10 self-checks ok")
