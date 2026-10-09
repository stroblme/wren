"""S3 — optimizer bake-off: AdamW vs Muon vs K-FAC.

Fixed step budget, matched tuning effort: one LR sweep per optimizer (the
grids differ because their step scales differ), then >=3 seeds at each
winner, then depth 16 at the winners for the depth question.

The training node declares `resources={"gpus": 1}`, which is what serializes
the arms: concurrent jax workers each preallocate ~75% of VRAM and kill each
other's cublas init. The engine must know the card is there, or the
declaration is clamped to zero and the worker is handed
CUDA_VISIBLE_DEVICES="" -- training then runs on the CPU. `.fluksio/env`
carries `FLOW_GPUS=1` for this checkout; elsewhere start it with
`fluksio serve --gpus 1`.

    uv run fluksio sweep s3_optimizers --param optimizer=adamw --param lr=3e-4,1e-3,3e-3,1e-2 --param seed=0
    uv run fluksio sweep s3_optimizers --param optimizer=muon --param lr=3e-3,1e-2,3e-2,1e-1 --param seed=0
    uv run fluksio sweep s3_optimizers --param optimizer=kfac --param lr=3e-4,1e-3,3e-3,1e-2 --param seed=0
    uv run fluksio sweep s3_optimizers --param optimizer=... --param lr=<winner> --param seed=1,2
    uv run fluksio sweep s3_optimizers --param optimizer=... --param lr=<winner> --param seed=0 --param depth=16

All three precondition the same matrices, which is what makes the comparison
clean: Muon orthogonalizes the attention-weight updates and leaves the tied
embedding to its inner AdamW (standard practice); K-FAC's auto-registration
puts Kronecker factors on exactly those attention matrices and diagonal
approximations on the embedding and the norm scales; AdamW is the
diagonal-only (empirical Fisher) baseline. K-FAC runs with fixed momentum
0.9 and fixed damping 1e-3; its LR sweep is the matched tuning knob.

Data: Home-Assistant-Requests train split (pristine, no synthetic
negatives — abstention is out of scope here), benchmark metrics on its
test split. Primary metric: loss at the fixed step budget; bits-per-byte
and the needle-protocol metrics on the test split are the secondary check
that a loss win is real.
"""

import itertools
import math
import pathlib
import tempfile

import fluksio
from fluksio import Flow, Port, node


@node(provides=[Port("train_text", "str"), Port("eval_samples", "json")])
def dataset() -> dict:
    import random

    from wren.dataset import home_assistant, to_text

    train, test = home_assistant()
    # The raw file is grouped by service type; shuffle so a training window
    # is not locally homogeneous.
    random.Random(0).shuffle(train)
    return {
        "train_text": to_text(train),
        "eval_samples": [[p, [[n, a] for n, a in calls]] for p, calls in test],
    }


@node(
    requires=[
        Port("train_text", "str"),
        Port("eval_samples", "json"),
        Port("optimizer", "str"),
        Port("lr", "float"),
        Port("seed", "int"),
        Port("steps", "int"),
        Port("depth", "int"),
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
    # preallocate=false: jax otherwise takes 75% of the card at import, so a
    # second engine (a `--local` run beside a running `serve`) collides with
    # this one instead of fitting beside it.
    resources={"gpus": 1, "env": {"XLA_PYTHON_CLIENT_PREALLOCATE": "false"}},
)
def bakeoff(
    train_text: str,
    eval_samples: list,
    optimizer: str,
    lr: float,
    seed: int,
    steps: int = 2000,
    depth: int = 8,
    batch_size: int = 16,
    seq_len: int = 256,
    log_every: int = 50,
    gen_batch: int = 64,
    n_new: int = 192,
) -> dict:
    import os

    import jax
    import jax.numpy as jnp
    import optax
    from flax import nnx

    from wren import tokenizer
    from wren.dataset import to_text
    from wren.eval import score
    from wren.model import Config, Wren, generate_batch, save
    from wren.train import sample_batch, train_step

    # An unsatisfiable `gpus: 1` arrives as CUDA_VISIBLE_DEVICES="" and would
    # train on the CPU without saying so; fail instead.
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        raise RuntimeError("no card declared: `fluksio serve --gpus 1`")

    # context 384: longest prompt (151 B) + n_new fit; longest sample 330 B.
    model = Wren(Config(context=384, n_layers=depth), rngs=nnx.Rngs(seed))
    data = jnp.array(tokenizer.encode(train_text), dtype=jnp.int32)
    key = jax.random.PRNGKey(seed)
    loss = float("nan")

    if optimizer.startswith("kfac"):
        import kfac_jax

        # "kfac-adaptive" is the library's designed operating mode: lr,
        # momentum and damping self-tune from its quadratic model (the lr
        # input is ignored -- run it with lr=0). Plain "kfac" fixes
        # momentum 0.9 / damping 1e-3 and sweeps lr like the others.
        adaptive = optimizer == "kfac-adaptive"
        graphdef, params = nnx.split(model, nnx.Param)

        def loss_fn(params: nnx.State, batch: tuple) -> jax.Array:
            x, y = batch
            logits = nnx.merge(graphdef, params)(x)
            flat, tgt = logits.reshape(-1, logits.shape[-1]), y.reshape(-1)
            kfac_jax.register_softmax_cross_entropy_loss(flat, tgt)
            return optax.softmax_cross_entropy_with_integer_labels(flat, tgt).mean()

        opt = kfac_jax.Optimizer(
            value_and_grad_func=jax.value_and_grad(loss_fn),
            l2_reg=0.0,
            value_func_has_aux=False,
            value_func_has_state=False,
            value_func_has_rng=False,
            use_adaptive_learning_rate=adaptive,
            use_adaptive_momentum=adaptive,
            use_adaptive_damping=adaptive,
            initial_damping=1.0 if adaptive else None,
            num_burnin_steps=0,
            multi_device=False,
        )
        key, sub = jax.random.split(key)
        opt_state = opt.init(params, key, sample_batch(data, batch_size, seq_len, sub))
        knobs = (
            {}
            if adaptive
            else {
                "learning_rate": jnp.array(lr),
                "momentum": jnp.array(0.9),
                "damping": jnp.array(1e-3),
            }
        )
        for step in range(steps):
            key, sub, rng = jax.random.split(key, 3)
            # ty: kfac's step returns 3 values when value_func_has_state=False,
            # and its signature lets **knobs match the data iterator
            params, opt_state, stats = opt.step(  # ty: ignore[invalid-assignment]
                params,
                opt_state,
                rng,
                batch=sample_batch(data, batch_size, seq_len, sub),
                global_step_int=step,
                **knobs,  # ty: ignore[invalid-argument-type]
            )
            loss = float(stats["loss"])
            if step % log_every == 0 or step == steps - 1:
                fluksio.emit(loss=loss)
        nnx.update(model, params)
    else:

        def hidden_matrices(params: dict) -> dict:
            # Muon preconditions the attention matrices only; the tied
            # embedding (also 2D) falls through to the inner AdamW.
            return jax.tree_util.tree_map_with_path(
                lambda path, p: (
                    optax.contrib.MuonDimensionNumbers()
                    if p.ndim == 2 and "embed" not in jax.tree_util.keystr(path)
                    else None
                ),
                params,
            )

        tx = {
            "adamw": lambda: optax.adamw(lr),
            # ty: muon's annotation omits the callable form its source handles
            "muon": lambda: optax.contrib.muon(
                lr,
                muon_weight_dimension_numbers=hidden_matrices,  # ty: ignore[invalid-argument-type]
            ),
        }[optimizer]()
        state = nnx.Optimizer(model, tx, wrt=nnx.Param)
        for step in range(steps):
            key, sub = jax.random.split(key)
            x, y = sample_batch(data, batch_size, seq_len, sub)
            loss = float(train_step(model, state, x, y))
            if step % log_every == 0 or step == steps - 1:
                fluksio.emit(loss=loss)

    samples = [(p, [(n, a) for n, a in calls]) for p, calls in eval_samples]

    # Bits per byte on the held-out split, windowed like training.
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

    def own_lines(generated: str) -> str:
        """The model's continuation up to its first empty line -- past that
        it is hallucinating the next training sample, not answering."""
        return "\n".join(itertools.takewhile(bool, generated.split("\n")))

    predictions = []
    for i in range(0, len(samples), gen_batch):
        prompts = [tokenizer.encode(p) for p, _ in samples[i : i + gen_batch]]
        outs, _ = generate_batch(model, prompts, n_new, stop=(10, 10))
        predictions += [own_lines(tokenizer.decode(o)) for o in outs]
        print(f"generated {len(predictions)}/{len(samples)}")
    metrics = score(predictions, [calls for _, calls in samples])

    path = pathlib.Path(tempfile.mkdtemp()) / "weights.npz"
    save(model, path)
    ref = fluksio.save_artifact(path.read_bytes(), name="weights.npz")
    return {"final_loss": loss, "bits_per_byte": bpb, "weights": ref} | metrics


Flow(
    "s3_optimizers",
    title="S3 — optimizer bake-off",
    nodes=[dataset, bakeoff],
    inputs=[
        Port("optimizer", "str", initial="adamw"),
        Port("lr", "float", initial=1e-3),
        Port("seed", "int", initial=0),
        Port("steps", "int", initial=2000),
        Port("depth", "int", initial=8),
    ],
    outputs=[
        "final_loss",
        "bits_per_byte",
        "exact_match",
        "well_formed",
        "name_accuracy",
        "arg_f1",
    ],
)
