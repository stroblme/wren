"""Compare stream windows, intact examples and prompt loss weights on a CPU."""

import json
import math
import pathlib
import tempfile
import time

import fluksio
import numpy as np
from fluksio import Flow, Port, node


def prepare(cases: list, context: int, prompt_weight: float) -> tuple:
    """Right-pad intact examples; weight next-byte targets, never padding."""
    from wren.targets import answer, prompt, select

    x = np.zeros((len(cases), context), dtype=np.int32)
    y = np.zeros_like(x)
    weights = np.zeros(x.shape, dtype=np.float32)
    for i, (command, state, calls) in enumerate(cases):
        prefix = prompt(select(state, command), command).encode()
        data = np.frombuffer(prefix + answer(calls).encode(), dtype=np.uint8)
        if len(data) > context:
            raise ValueError("example exceeds the training context")
        n = len(data) - 1
        x[i, :n], y[i, :n] = data[:-1], data[1:]
        weights[i, : len(prefix) - 1] = prompt_weight
        weights[i, len(prefix) - 1 : n] = 1.0
    return x, y, weights


INPUTS = [
    Port("mode", "str", initial="stream"),
    Port("seed", "int", initial=0),
    Port("steps", "int", initial=4000),
    Port("n_train", "int", initial=200000),
    Port("n_eval", "int", initial=1000),
    Port("dim", "int", initial=256),
    Port("n_layers", "int", initial=4),
    Port("batch_size", "int", initial=16),
    Port("lr", "float", initial=0.003),
]


@node(
    requires=[Port(p.name, p.dtype) for p in INPUTS],
    provides=[
        Port("report", "json"),
        Port("pack", "artifact"),
        Port("weights", "artifact"),
        Port("loss", "float", stream=True),
    ],
    resources={"env": {"JAX_PLATFORMS": "cpu", "OPENBLAS_NUM_THREADS": "1"}},
)
def study(
    mode: str,
    seed: int,
    steps: int,
    n_train: int,
    n_eval: int,
    dim: int,
    n_layers: int,
    batch_size: int,
    lr: float,
) -> dict:
    import jax
    import jax.numpy as jnp
    import optax
    from flax import nnx

    from wren import dataset, runtime, tokenizer
    from wren.eval import evaluate
    from wren.export import export
    from wren.model import Config, Wren, save
    from wren.targets import Target
    from wren.train import muon, sample_batch, train_step

    if mode not in ("stream", "aligned", "weighted"):
        raise ValueError("mode must be stream, aligned or weighted")
    if jax.default_backend() != "cpu":
        raise RuntimeError("this study measures CPU training")
    home = json.loads(runtime.DEMO.read_text())
    targets = [Target(**t) for t in home["targets"]]
    reserved = {r["command"] for k in ("calibration", "tests") for r in home.get(k, [])}
    # Fixed data and sampling seeds isolate initialization variance across arms.
    cases = dataset.corpus(
        targets, n=n_train, seed=0, examples=home["examples"], exclude=reserved
    )
    calibration = dataset.corpus(targets, n=n_eval, seed=1, exclude=reserved)
    held = dataset.corpus(
        targets,
        n=n_eval,
        seed=2,
        held_out=True,
        exclude=reserved | {c for c, _, _ in cases + calibration},
    )
    tests = [dataset.as_case(targets, t) for t in home["tests"]]
    checked = [dataset.as_case(targets, t) for t in home.get("calibration", [])]
    cfg = Config(dim=dim, n_layers=n_layers, n_heads=4, ternary=True)
    model = Wren(cfg, rngs=nnx.Rngs(seed))
    optimizer = nnx.Optimizer(
        model, muon(optax.cosine_decay_schedule(lr, steps)), wrt=nnx.Param
    )
    stream = jnp.array(tokenizer.encode(dataset.text(cases)), dtype=jnp.int32)
    aligned = (
        tuple(
            jnp.asarray(a)
            for a in prepare(cases, cfg.context, 0.1 if mode == "weighted" else 1.0)
        )
        if mode != "stream"
        else None
    )
    key = jax.random.PRNGKey(0)
    trained_bytes = 0
    started = time.perf_counter()
    for step in range(steps):
        key, sub = jax.random.split(key)
        if aligned is None:
            x, y = sample_batch(stream, batch_size, cfg.context, sub)
            loss = float(train_step(model, optimizer, x, y))
            trained_bytes += batch_size * cfg.context
        else:
            idx = jax.random.randint(sub, (batch_size,), 0, len(cases))
            x, y, weights = (a[idx] for a in aligned)
            loss = float(train_step(model, optimizer, x, y, weights))
            trained_bytes += int(jnp.count_nonzero(weights))
        if step % 50 == 0 or step == steps - 1:
            fluksio.emit(loss=loss)
    seconds = time.perf_counter() - started
    with tempfile.TemporaryDirectory() as tmp:
        weights_path, pack_path = (
            pathlib.Path(tmp, "weights.npz"),
            pathlib.Path(tmp, "candidate.bin"),
        )
        save(model, weights_path)
        export(str(weights_path), str(pack_path))
        report = evaluate(
            runtime.load_pack(pack_path),
            held,
            tests=tests,
            calibration=calibration,
            calibration_tests=checked,
            policy="call",
        )
        report |= {
            "mode": mode,
            "seed": seed,
            "steps": steps,
            "train_seconds": seconds,
            "trained_bytes": trained_bytes,
            "final_loss": loss,
        }
        if not math.isfinite(report["threshold"]):
            report["threshold"] = None
        return {
            "report": report,
            "pack": fluksio.save_artifact(pack_path.read_bytes(), name="candidate.bin"),
            "weights": fluksio.save_artifact(
                weights_path.read_bytes(), name="weights.npz"
            ),
        }


Flow("s13_accuracy", nodes=[study], inputs=INPUTS, outputs=["report"])
