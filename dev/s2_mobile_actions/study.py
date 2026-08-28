"""S2 — external benchmark baseline on google/mobile-actions (ROADMAP.md).

Implicit-tool mode: the 7-function surface lives in the weights; the prompt
is the timestamp line plus the user request. Scored with ordered strict
exact match (needle's protocol) plus its diagnostic companions, and
bits-per-byte as the language-model-level metric. Reference points on this
dataset: needle 2 (45M params) 63.7%, FunctionGemma 270M (fine-tuned).
"""

import math
import pathlib
import tempfile

import fluksio
from fluksio import Flow, Port, node


@node(provides=[Port("train_text", "str"), Port("eval_samples", "json")])
def dataset() -> dict:
    from wren.dataset import mobile_actions, to_text

    train, evals = mobile_actions()
    return {
        "train_text": to_text(train),
        "eval_samples": [[p, [[n, a] for n, a in calls]] for p, calls in evals],
    }


@node(
    requires=[
        Port("train_text", "str"),
        Port("eval_samples", "json"),
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
        Port("weights", "artifact"),
    ],
)
def benchmark(
    train_text: str,
    eval_samples: list,
    seed: int,
    steps: int = 2000,
    lr: float = 3e-3,
    batch_size: int = 16,
    seq_len: int = 512,
    log_every: int = 50,
    gen_batch: int = 64,
    n_new: int = 192,
) -> dict:
    import jax
    import jax.numpy as jnp
    import optax
    from flax import nnx

    from wren import tokenizer
    from wren.dataset import to_text
    from wren.eval import score
    from wren.model import Config, Wren, generate_batch, save
    from wren.train import sample_batch, train_step

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
    predictions = []
    for i in range(0, len(samples), gen_batch):
        prompts = [tokenizer.encode(p) for p, _ in samples[i : i + gen_batch]]
        outs = generate_batch(model, prompts, n_new, stop=(10, 10))
        predictions += [tokenizer.decode(o).split("\n\n")[0] for o in outs]
        print(f"generated {len(predictions)}/{len(samples)}")
    metrics = score(predictions, [calls for _, calls in samples])

    path = pathlib.Path(tempfile.mkdtemp()) / "weights.npz"
    save(model, path)
    ref = fluksio.save_artifact(path.read_bytes(), name="weights.npz")
    return {"final_loss": loss, "bits_per_byte": bpb, "weights": ref} | metrics


Flow(
    "s2_mobile_actions",
    title="S2 — mobile-actions benchmark",
    nodes=[dataset, benchmark],
    inputs=[
        Port("seed", "int", initial=0),
        Port("steps", "int", initial=2000),
        Port("lr", "float", initial=3e-3),
    ],
    outputs=["exact_match", "well_formed", "name_accuracy", "arg_f1", "bits_per_byte"],
)
