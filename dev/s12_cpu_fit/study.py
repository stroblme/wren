"""Train the demo home's model on a CPU: step budget, shape and prompt selection.

The steps of `wren.train.fit`, with the held-out evaluation repeated every
`every` steps. Training has no schedule, so the model at step k is the one
a k-step run ends with, and one run gives every smaller budget. `select`
picks how many targets a prompt shows: "all" shares a word with the command
(`wren.targets.select`), "top" only the most words. See README.md.
"""

import json
import math
import os
import pathlib
import statistics
import tempfile
import time

import fluksio
import jax
import jax.numpy as jnp
from flax import nnx
from fluksio import Flow, Port, node

from wren import dataset, eval, runtime, targets, tokenizer
from wren.export import export
from wren.model import Config, Wren, save
from wren.train import muon, sample_batch, train_step

HOME = runtime.DEMO
SELECT = targets.select


def top(
    shown: list[targets.Target], command: str, budget: int = targets.BUDGET
) -> list[targets.Target]:
    """`select`, keeping only the targets that share the most words."""

    def score(t: targets.Target) -> int:
        return len(
            targets._words(command) & targets._words(f"{t.name} {t.description}")
        )

    shown = SELECT(shown, command, budget)
    best = max(map(score, shown), default=0)
    return [t for t in shown if score(t) == best]


def use(variant: str) -> None:
    """Build prompts with `variant` in both the corpus and `act`."""
    # ty: a module's function is typed as that function, so any swap is flagged
    fn = {"all": SELECT, "top": top}[variant]
    dataset.select = targets.select = fn  # ty: ignore[invalid-assignment]


@node(
    requires=[
        Port("dim", "int"),
        Port("n_layers", "int"),
        Port("n_heads", "int"),
        Port("steps", "int"),
        Port("every", "int"),
        Port("batch_size", "int"),
        Port("lr", "float"),
        Port("n_train", "int"),
        Port("n_eval", "int"),
        Port("select", "str"),
        Port("seed", "int"),
    ],
    provides=[
        Port("loss", "float", stream=True),
        Port("curve", "json"),
        Port("exact_match", "float"),
        Port("threshold", "float"),
        Port("coverage", "float"),
        Port("unseen_exact", "float"),
        Port("seen_share", "float"),
        Port("corpus_s", "float"),
        Port("train_s", "float"),
        Port("latency_ms", "float"),
        Port("n_params", "float"),
        Port("pack", "artifact"),
    ],
    # The CPU, one BLAS thread for `act` as a host runs it.
    resources={"env": {"JAX_PLATFORMS": "cpu", "OPENBLAS_NUM_THREADS": "1"}},
)
def study(
    dim: int,
    n_layers: int,
    n_heads: int,
    steps: int,
    every: int,
    batch_size: int,
    lr: float,
    n_train: int,
    n_eval: int,
    select: str,
    seed: int,
) -> dict:
    if jax.default_backend() != "cpu":
        raise RuntimeError(f"jax runs on {jax.default_backend()}, not the CPU")
    home = json.loads(HOME.read_text())
    ts = [targets.Target(**t) for t in home["targets"]]

    start = time.perf_counter()
    # Held out under the shipped selection, so both arms answer the same commands.
    use("all")
    held_out = dataset.corpus(ts, n=n_eval, seed=seed + 1, examples=home["examples"])
    use(select)
    cases = dataset.corpus(ts, n=n_train, seed=seed, examples=home["examples"])
    data = jnp.array(tokenizer.encode(dataset.text(cases)), dtype=jnp.int32)
    corpus_s = time.perf_counter() - start
    seen = {c for c, _, _ in cases}
    unseen = [i for i, (c, _, _) in enumerate(held_out) if c not in seen]

    cfg = Config(dim=dim, n_layers=n_layers, n_heads=n_heads, ternary=True)
    model = Wren(cfg, rngs=nnx.Rngs(seed))
    optimizer = nnx.Optimizer(model, muon(lr), wrt=nnx.Param)
    key = jax.random.PRNGKey(0)  # the batches `wren.train.train` draws
    tmp = pathlib.Path(tempfile.mkdtemp())
    curve, losses, train_s = [], [], 0.0
    for step in range(1, steps + 1):
        tic = time.perf_counter()
        key, sub = jax.random.split(key)
        x, y = sample_batch(data, batch_size, cfg.context, sub)
        losses.append(float(train_step(model, optimizer, x, y)))
        train_s += time.perf_counter() - tic
        if step % 50 == 0:
            fluksio.emit(loss=statistics.fmean(losses[-50:]))
        if step % every and step != steps:
            continue
        save(model, tmp / "weights.npz")
        export(str(tmp / "weights.npz"), str(tmp / "wren.bin"))
        pack = runtime.load_pack(tmp / "wren.bin")
        predicted = [targets.act(c, state, model=pack) for c, state, _ in held_out]
        expected = [calls for _, _, calls in held_out]
        t = eval.threshold(predicted, expected)
        curve.append(
            {
                "step": step,
                "train_s": train_s,
                "loss": statistics.fmean(losses[-50:]),
                "exact_match": eval.coverage(predicted, expected, 0.0),
                "threshold": t if math.isfinite(t) else -1.0,  # none reaches 0.99
                "coverage": eval.coverage(predicted, expected, t),
                "unseen_exact": eval.coverage(
                    [predicted[i] for i in unseen], [expected[i] for i in unseen], 0.0
                ),
                "load": os.getloadavg()[1],  # other work on the machine
            }
        )
        fluksio.logger.info(json.dumps(curve[-1]))

    times = []
    for c, state, _ in held_out[:48]:
        tic = time.perf_counter()
        targets.act(c, state, model=pack)
        times.append(time.perf_counter() - tic)
    use("all")
    final = ("exact_match", "threshold", "coverage", "unseen_exact", "train_s")
    return {k: curve[-1][k] for k in final} | {
        "curve": curve,
        "seen_share": 1 - len(unseen) / len(held_out),
        "corpus_s": corpus_s,
        "latency_ms": 1000 * statistics.median(times),
        "n_params": float(model.n_params()),
        "pack": fluksio.save_artifact((tmp / "wren.bin").read_bytes(), name="wren.bin"),
    }


Flow(
    "s12_cpu_fit",
    title="S12 — training an installation on a CPU",
    nodes=[study],
    inputs=[
        Port("dim", "int", initial=256),
        Port("n_layers", "int", initial=4),
        Port("n_heads", "int", initial=4),
        Port("steps", "int", initial=4000),
        Port("every", "int", initial=500),
        Port("batch_size", "int", initial=32),
        Port("lr", "float", initial=3e-3),
        Port("n_train", "int", initial=200000),
        Port("n_eval", "int", initial=1000),
        Port("select", "str", initial="all"),
        Port("seed", "int", initial=0),
    ],
    outputs=[
        "exact_match",
        "threshold",
        "coverage",
        "unseen_exact",
        "seen_share",
        "corpus_s",
        "train_s",
        "latency_ms",
        "n_params",
        "curve",
    ],
)


if __name__ == "__main__":
    # "top" keeps the best-matching target and every target tied with it.
    home = json.loads(HOME.read_text())
    ts = [targets.Target(**t) for t in home["targets"]]
    assert [t.name for t in top(ts, "turn on the kitchen light")] == ["kitchen_light"]
    assert [t.name for t in top(ts, "turn off all lights")] == ["all_lights"]
    assert len(top(ts, "lights on please")) == len(SELECT(ts, "lights on please")) == 8
    use("top")
    assert dataset.select is targets.select is top
    use("all")
    print("s12 self-checks ok")
