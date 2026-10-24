"""Train byte decoders to predict the next byte at each position.

`fit` trains an installation's model from its target list and writes the
ternary pack that `wren.targets.act` runs.
"""

import dataclasses
import datetime
import inspect
import math
import pathlib
import tempfile
from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from wren import runtime, tokenizer
from wren.dataset import as_case, corpus, text
from wren.eval import evaluate
from wren.export import export
from wren.model import Config, Wren, save
from wren.targets import Target


def muon(lr: optax.ScalarOrSchedule) -> optax.GradientTransformation:
    """Apply Muon to attention matrices and its inner AdamW to embeddings and norms."""

    def hidden_matrices(params: dict) -> dict:
        return jax.tree_util.tree_map_with_path(
            lambda path, p: (
                optax.contrib.MuonDimensionNumbers()
                if p.ndim == 2 and "embed" not in jax.tree_util.keystr(path)
                else None
            ),
            params,
        )

    # ty: muon's annotation omits the callable form its source handles
    return optax.contrib.muon(
        lr,
        muon_weight_dimension_numbers=hidden_matrices,  # ty: ignore[invalid-argument-type]
    )


def sample_batch(
    data: jax.Array, batch_size: int, seq_len: int, key: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Sample corpus windows with targets shifted forward by one byte."""
    starts = jax.random.randint(key, (batch_size,), 0, len(data) - seq_len - 1)
    idx = starts[:, None] + jnp.arange(seq_len + 1)[None, :]
    chunk = data[idx]
    return chunk[:, :-1], chunk[:, 1:]


@nnx.jit
def train_step(
    model: Wren, optimizer: nnx.Optimizer, x: jax.Array, y: jax.Array
) -> jax.Array:
    def loss_fn(model: Wren) -> jax.Array:
        logits = model(x)
        return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss


def train(
    model: Wren,
    text: str,
    steps: int = 4000,
    batch_size: int = 32,
    seq_len: int = 384,
    lr: float = 3e-3,
    on_log: Callable[[int, float], None] | None = None,
) -> list[float]:
    """Train the model in place and return the loss at each step.

    The learning rate decays from `lr` to zero along a cosine, so the run
    ends settled rather than wherever the last noisy step left it. Call
    `on_log(step, loss)` after every step if provided.
    """
    data = jnp.array(tokenizer.encode(text), dtype=jnp.int32)
    schedule = optax.cosine_decay_schedule(lr, steps)
    optimizer = nnx.Optimizer(model, muon(schedule), wrt=nnx.Param)
    key = jax.random.PRNGKey(0)
    losses = []
    for step in range(steps):
        key, sub = jax.random.split(key)
        x, y = sample_batch(data, batch_size, seq_len, sub)
        loss = float(train_step(model, optimizer, x, y))
        losses.append(loss)
        if on_log is not None:
            on_log(step, loss)
    return losses


def fit(
    targets: list[Target],
    *,
    out: str | pathlib.Path,
    examples: Sequence[dict] = (),
    tests: Sequence[dict] = (),
    paraphrase: Callable[[str], list[str]] | None = None,
    dim: int = 256,
    n_layers: int = 4,
    n_heads: int = 4,
    steps: int = 4000,
    batch_size: int = 16,
    lr: float = 3e-3,
    n_train: int = 200_000,
    n_eval: int = 1000,
    precision: float = 0.99,
    seed: int = 0,
    on_log: Callable[[int, float], None] | None = None,
) -> dict:
    """Train a model for an installation's `targets` and write its pack to `out`.

    Trains from scratch, quantization-aware, on `n_train` samples of
    `wren.dataset.corpus` (with the installation's `examples`, widened by
    `paraphrase` if given; the held-out samples are not), then runs
    `act` with the pack on `n_eval` held-out samples drawn with seed + 1 and
    on the installation's `tests` (shaped like `examples`, never trained on).
    The threshold is the higher of those `wren.eval.threshold` recommends at
    `precision` on each (`wren.eval.evaluate`). The pack's metadata holds the
    targets without their values, the threshold, the tests' exact match, the
    steps and the date. Raise RuntimeError and write nothing if no threshold
    reaches `precision`.

    Return the held-out exact match, threshold and coverage, the tests'
    exact match and coverage if given, the final loss, the parameter count
    and the wall time in seconds.
    """
    start = datetime.datetime.now(datetime.UTC)
    cases = corpus(
        targets, n=n_train, seed=seed, examples=examples, paraphrase=paraphrase
    )
    held_out = corpus(targets, n=n_eval, seed=seed + 1, examples=examples)
    tested = [as_case(targets, t) for t in tests]
    cfg = Config(dim=dim, n_layers=n_layers, n_heads=n_heads, ternary=True)
    model = Wren(cfg, rngs=nnx.Rngs(seed))
    losses = train(
        model,
        text(cases),
        steps=steps,
        batch_size=batch_size,
        seq_len=cfg.context,
        lr=lr,
        on_log=on_log,
    )
    with tempfile.TemporaryDirectory() as tmp:
        weights, pack = pathlib.Path(tmp, "weights.npz"), pathlib.Path(tmp, "wren.bin")
        save(model, weights)
        export(str(weights), str(pack))
        scores = evaluate(runtime.load_pack(pack), held_out, precision, tested)
        if math.isinf(scores["threshold"]):
            raise RuntimeError(
                f"no threshold reaches precision {precision} on {n_eval} held-out "
                f"samples and {len(tested)} tests (exact match "
                f"{scores['exact_match']:.1%}); not writing {out}"
            )
        meta = {
            "targets": [
                {k: v for k, v in dataclasses.asdict(t).items() if k != "value"}
                for t in targets
            ],
            "threshold": scores["threshold"],
            "precision": precision,
            "steps": steps,
            "date": start.date().isoformat(),
        }
        if tested:
            meta["test_exact_match"] = scores["test_exact_match"]
        export(str(weights), str(out), meta)
    seconds = (datetime.datetime.now(datetime.UTC) - start).total_seconds()
    return scores | {
        "final_loss": losses[-1],
        "n_params": model.n_params(),
        "seconds": seconds,
    }


#: Every knob `wren --train` exposes, and the one place its default lives:
#: `fit`'s numeric keywords, read off its signature rather than listed again,
#: so editing it is the whole change -- the CLI turns each into a `--flag`,
#: the flow into an input (wren/flow.py), and the run records what it was given.
DEFAULTS: dict[str, int | float] = {
    name: parameter.default
    for name, parameter in inspect.signature(fit).parameters.items()
    if isinstance(parameter.default, int | float)
}
