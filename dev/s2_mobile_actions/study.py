"""Train and evaluate the baseline on google/mobile-actions.

Prompts contain timestamps and requests; tool schemas are learned from
training. See README.md for the protocol and reproduction commands.
"""

import math
import pathlib
import tempfile

import fluksio
from fluksio import Flow, Port, node


@node(
    provides=[
        Port("train_text", "str"),
        Port("eval_samples", "json"),
        Port("neg_prompts", "json"),
        Port("surface", "json"),
    ]
)
def dataset() -> dict:
    from wren.dataset import irrelevant, mobile_actions, mobile_actions_surface, to_text

    train, evals = mobile_actions()
    neg_train, neg_eval = irrelevant()
    # Negatives (~2% of samples) teach abstention: an off-domain prompt is
    # followed by nothing before the blank-line boundary.
    import random

    mixed = train + neg_train
    random.Random(0).shuffle(mixed)
    return {
        "train_text": to_text(mixed),
        "eval_samples": [[p, [[n, a] for n, a in calls]] for p, calls in evals],
        "neg_prompts": [p for p, _ in neg_eval],
        "surface": mobile_actions_surface(),
    }


@node(
    requires=[
        Port("train_text", "str"),
        Port("eval_samples", "json"),
        Port("neg_prompts", "json"),
        Port("surface", "json"),
        Port("seed", "int"),
        Port("steps", "int"),
        Port("lr", "float"),
    ],
    provides=[
        Port("loss", "float", stream=True),
        Port("final_loss", "float"),
        Port("bits_per_byte", "float"),
        Port("exact_match", "float"),
        Port("well_formed", "float"),
        Port("name_accuracy", "float"),
        Port("arg_f1", "float"),
        Port("hallucination", "float"),
        Port("irrelevance_acc", "float"),
        Port("ece", "float"),
        Port("gen_bytes_per_s", "float"),
        Port("peak_ram_mb", "float"),
        Port("n_params", "float"),
        Port("weights", "artifact"),
    ],
    # preallocate=false: jax otherwise takes 75% of the card at import, so a
    # second engine (a `--local` run beside a running `serve`) collides with
    # this one instead of fitting beside it.
    resources={"gpus": 1, "env": {"XLA_PYTHON_CLIENT_PREALLOCATE": "false"}},
)
def benchmark(
    train_text: str,
    eval_samples: list,
    neg_prompts: list,
    surface: dict,
    seed: int,
    steps: int = 2000,
    lr: float = 3e-3,
    batch_size: int = 16,
    seq_len: int = 512,
    log_every: int = 50,
    gen_batch: int = 64,
    n_new: int = 192,
) -> dict:
    import itertools
    import os
    import resource
    import time

    import jax
    import jax.numpy as jnp
    import optax
    from flax import nnx

    from wren import tokenizer
    from wren.dataset import to_text
    from wren.eval import (
        abstain_rate,
        calibration_error,
        hallucination_rate,
        parse_calls,
        score,
    )
    from wren.model import Config, Wren, generate_batch, save
    from wren.train import sample_batch, train_step

    # An unsatisfiable `gpus: 1` arrives as CUDA_VISIBLE_DEVICES="" and would
    # train on the CPU without saying so; fail instead.
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        raise RuntimeError("no card declared: `fluksio serve --gpus 1`")

    def own_lines(generated: str) -> str:
        """Keep the generated continuation through its first blank line."""
        return "\n".join(itertools.takewhile(bool, generated.split("\n")))

    model = Wren(Config(context=768), rngs=nnx.Rngs(seed))
    data = jnp.array(tokenizer.encode(train_text), dtype=jnp.int32)
    optimizer = nnx.Optimizer(model, optax.adamw(lr), wrt=nnx.Param)
    key = jax.random.PRNGKey(seed)
    loss = float("nan")
    for step in range(steps):
        key, sub = jax.random.split(key)
        x, y = sample_batch(data, batch_size, seq_len, sub)
        loss = float(train_step(model, optimizer, x, y))
        if step % log_every == 0 or step == steps - 1:
            fluksio.emit(loss=loss)

    samples = [(p, [(n, a) for n, a in calls]) for p, calls in eval_samples]

    # Bits per byte on the held-out split: the LM-level health metric.
    held = jnp.array(tokenizer.encode(to_text(samples)), dtype=jnp.int32)
    windows = (len(held) - 1) // seq_len
    xs = held[: windows * seq_len].reshape(windows, seq_len)
    ys = held[1 : windows * seq_len + 1].reshape(windows, seq_len)
    nats = [
        float(
            optax.softmax_cross_entropy_with_integer_labels(
                model(xs[i : i + batch_size]), ys[i : i + batch_size]
            ).mean()
        )
        for i in range(0, windows, batch_size)
    ]
    bpb = sum(nats) / len(nats) / math.log(2)

    # Greedy generation, KV-cached, a batch of prompts at a time; the
    # blank line the training stream uses as sample boundary is the stop.
    predictions, confidences = [], []
    started = time.perf_counter()
    for i in range(0, len(samples), gen_batch):
        prompts = [tokenizer.encode(p) for p, _ in samples[i : i + gen_batch]]
        outs, confs = generate_batch(model, prompts, n_new, stop=(10, 10))
        predictions += [own_lines(tokenizer.decode(o)) for o in outs]
        confidences += confs
        print(f"generated {len(predictions)}/{len(samples)}")
    references = [calls for _, calls in samples]
    generated_bytes = sum(len(p.encode()) for p in predictions)
    metrics = score(predictions, references)

    # Abstention on held-out off-domain prompts (prompt types never trained).
    neg_out, _ = generate_batch(
        model, [tokenizer.encode(p) for p in neg_prompts], n_new, stop=(10, 10)
    )
    neg_predictions = [own_lines(tokenizer.decode(o)) for o in neg_out]

    path = pathlib.Path(tempfile.mkdtemp()) / "weights.npz"
    save(model, path)
    ref = fluksio.save_artifact(path.read_bytes(), name="weights.npz")
    return {
        "final_loss": loss,
        "bits_per_byte": bpb,
        "hallucination": hallucination_rate(predictions, surface),
        "irrelevance_acc": abstain_rate(neg_predictions),
        "ece": calibration_error(
            confidences, [parse_calls(p) == r for p, r in zip(predictions, references)]
        ),
        "gen_bytes_per_s": generated_bytes / (time.perf_counter() - started),
        "peak_ram_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "n_params": float(model.n_params()),
        "weights": ref,
    } | metrics


Flow(
    "s2_mobile_actions",
    title="S2 — mobile-actions benchmark",
    nodes=[dataset, benchmark],
    inputs=[
        Port("seed", "int", initial=0),
        Port("steps", "int", initial=2000),
        Port("lr", "float", initial=3e-3),
    ],
    outputs=[
        "exact_match",
        "well_formed",
        "name_accuracy",
        "arg_f1",
        "bits_per_byte",
        "hallucination",
        "irrelevance_acc",
        "ece",
        "gen_bytes_per_s",
        "peak_ram_mb",
    ],
)
