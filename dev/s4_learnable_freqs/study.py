"""Compare fixed, learned, and FoPE-style RoPE spectra.

Evaluate held-out bits/byte at multiple window lengths and probe one
frequency after training. Saved weights include `log_freqs`; reload with
SpectralWren, since `wren.model.load` restores only the stock model. See
README.md for the protocol and reproduction commands.
"""

import math
import pathlib
import tempfile

import fluksio
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from fluksio import Flow, Port, node

from wren.model import AttentionBlock, Config, Wren, save
from wren.train import muon, sample_batch, train_step

LOG_ZERO = -30.0  # exp(-30) ~ 1e-13: DC at any context, and log-representable


def spectrum(head_dim: int) -> jax.Array:
    """Return the stock RoPE spectrum used by `wren.model.rope`."""
    half = head_dim // 2
    return 1.0 / (10000.0 ** (jnp.arange(half) / half))


def rope_with(x: jax.Array, freqs: jax.Array) -> jax.Array:
    """Apply RoPE with an explicit spectrum to a full window, starting at position 0."""
    _, seq, _, _ = x.shape
    half = freqs.shape[0]
    angles = jnp.arange(seq)[:, None] * freqs[None, :]  # (seq, half)
    cos = jnp.cos(angles)[None, :, None, :]
    sin = jnp.sin(angles)[None, :, None, :]
    x1, x2 = x[..., :half], x[..., half:]
    return jnp.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)


class SpectralBlock(AttentionBlock):
    """Attention block using an explicit RoPE spectrum in `forward`.

    Inherited `__call__` and `decode` methods use the stock spectrum.
    """

    def forward(self, x: jax.Array, freqs: jax.Array) -> jax.Array:
        batch, seq, dim = x.shape
        h = self.norm(x)
        q, k, v = jnp.split(self.wqkv(h), 3, axis=-1)
        q = q.reshape(batch, seq, self.n_heads, self.head_dim)
        k = k.reshape(batch, seq, self.n_heads, self.head_dim)
        v = v.reshape(batch, seq, self.n_heads, self.head_dim)
        q = rope_with(self.q_norm(q), freqs)
        k = rope_with(self.k_norm(k), freqs)

        scores = jnp.einsum("bqhd,bkhd->bhqk", q, k) / jnp.sqrt(self.head_dim)
        causal = jnp.tril(jnp.ones((seq, seq), dtype=bool))
        scores = jnp.where(causal, scores, -jnp.inf)
        attn = jax.nn.softmax(scores, axis=-1)

        out = jnp.einsum("bhqk,bkhd->bqhd", attn, v).reshape(batch, seq, dim)
        return x + self.wo(out)


class SpectralWren(Wren):
    """Wren with a shared log-parameterized RoPE spectrum.

    Stock frequencies and the same seed reproduce stock Wren parameters.
    `freq_routed` controls whether the spectrum is trained.
    """

    def __init__(self, cfg: Config, rngs: nnx.Rngs, init_freqs: jax.Array):
        self.cfg = cfg
        self.embed = nnx.Embed(cfg.vocab_size, cfg.dim, rngs=rngs)
        self.blocks: nnx.List[SpectralBlock] = nnx.List(
            SpectralBlock(cfg, rngs) for _ in range(cfg.n_layers)
        )
        self.norm = nnx.RMSNorm(cfg.dim, rngs=rngs)
        floor = math.exp(LOG_ZERO)
        self.log_freqs = nnx.Param(jnp.log(jnp.maximum(init_freqs, floor)))

    def __call__(self, tokens: jax.Array) -> jax.Array:
        x = self.embed(tokens)
        freqs = jnp.exp(self.log_freqs[...])
        for block in self.blocks:
            x = block.forward(x, freqs)
        return self.embed.attend(self.norm(x))


def freq_routed(
    lr: float, freq_lr: float, trainable: bool
) -> optax.GradientTransformation:
    """Use Muon for model weights and separate Adam at `freq_lr` for `log_freqs`.

    Freeze spectrum updates in the fixed and FoPE arms.
    """

    def labels(params: dict) -> dict:
        return jax.tree_util.tree_map_with_path(
            lambda path, p: (
                "freqs" if "log_freqs" in jax.tree_util.keystr(path) else "rest"
            ),
            params,
        )

    freq_tx = optax.adam(freq_lr) if trainable else optax.set_to_zero()
    return optax.multi_transform({"rest": muon(lr), "freqs": freq_tx}, labels)


@nnx.jit
def _ce(model: SpectralWren, x: jax.Array, y: jax.Array) -> jax.Array:
    return optax.softmax_cross_entropy_with_integer_labels(model(x), y).mean()


def held_windows(held: jax.Array, seq_len: int) -> tuple[jax.Array, jax.Array]:
    """Split the held-out stream into non-overlapping (input, target) windows."""
    windows = (len(held) - 1) // seq_len
    xs = held[: windows * seq_len].reshape(windows, seq_len)
    ys = held[1 : windows * seq_len + 1].reshape(windows, seq_len)
    return xs, ys


def eval_bpb(
    model: SpectralWren, held: jax.Array, seq_len: int, batch_size: int
) -> float:
    """Return held-out cross-entropy in bits/byte at the given window length."""
    xs, ys = held_windows(held, seq_len)
    nats = [
        float(_ce(model, xs[i : i + batch_size], ys[i : i + batch_size]))
        for i in range(0, len(xs), batch_size)
    ]
    return sum(nats) / len(nats) / math.log(2)


def probe_curve(
    model: SpectralWren,
    held: jax.Array,
    seq_len: int,
    idx: int,
    n_points: int = 256,
    n_windows: int = 64,
) -> tuple[list[float], list[float]]:
    """Measure loss over one period of frequency `idx`, holding other weights fixed.

    Restore the original spectrum afterward.
    """
    xs, ys = held_windows(held, seq_len)
    x, y = xs[:n_windows], ys[:n_windows]
    saved = model.log_freqs[...]
    omegas = np.linspace(0.0, 2.0 * math.pi, n_points, endpoint=False)[1:]
    losses = []
    for w in omegas:
        model.log_freqs[...] = saved.at[idx].set(math.log(float(w)))
        losses.append(float(_ce(model, x, y)))
    model.log_freqs[...] = saved
    return omegas.tolist(), losses


@node(provides=[Port("train_text", "str"), Port("eval_text", "str")])
def dataset() -> dict:
    import random

    from wren.dataset import home_assistant, to_text

    train, test = home_assistant()
    # The raw file is grouped by service type; shuffle so a training window
    # is not locally homogeneous.
    random.Random(0).shuffle(train)
    return {"train_text": to_text(train), "eval_text": to_text(test)}


@node(
    requires=[
        Port("train_text", "str"),
        Port("eval_text", "str"),
        Port("mode", "str"),
        Port("lr", "float"),
        Port("freq_lr", "float"),
        Port("seed", "int"),
        Port("steps", "int"),
        Port("seq_len", "int"),
    ],
    provides=[
        Port("loss", "float", stream=True),
        Port("final_loss", "float"),
        Port("bpb_128", "float"),
        Port("bpb_192", "float"),
        Port("bpb_256", "float"),
        Port("freq_history", "json"),
        Port("probe_omegas", "json"),
        Port("probe_losses", "json"),
        Port("weights", "artifact"),
    ],
    # preallocate=false: jax otherwise takes 75% of the card at import, so a
    # second engine (a `--local` run beside a running `serve`) collides with
    # this one instead of fitting beside it.
    resources={"gpus": 1, "env": {"XLA_PYTHON_CLIENT_PREALLOCATE": "false"}},
)
def study(
    train_text: str,
    eval_text: str,
    mode: str,
    lr: float,
    freq_lr: float,
    seed: int,
    steps: int = 2000,
    seq_len: int = 128,
    batch_size: int = 32,  # seq 128 at batch 32 = s3's tokens per step
    log_every: int = 50,
    probe_idx: int = 8,  # omega = 0.1, period ~63 bytes — mid-band at seq 128
) -> dict:
    import os

    from wren import tokenizer

    # An unsatisfiable `gpus: 1` arrives as CUDA_VISIBLE_DEVICES="" and would
    # train on the CPU without saying so; fail instead.
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        raise RuntimeError("no card declared: `fluksio serve --gpus 1`")

    cfg = Config(context=256)  # must hold the longest eval window
    init = spectrum(cfg.dim // cfg.n_heads)
    if mode == "fope":
        init = jnp.where(init >= 2.0 * math.pi / seq_len, init, 0.0)
    model = SpectralWren(cfg, nnx.Rngs(seed), init)
    optimizer = nnx.Optimizer(
        model, freq_routed(lr, freq_lr, mode == "learned"), wrt=nnx.Param
    )

    data = jnp.array(tokenizer.encode(train_text), dtype=jnp.int32)
    key = jax.random.PRNGKey(seed)
    loss = float("nan")
    freq_history = []
    for step in range(steps):
        key, sub = jax.random.split(key)
        x, y = sample_batch(data, batch_size, seq_len, sub)
        loss = float(train_step(model, optimizer, x, y))
        if step % log_every == 0 or step == steps - 1:
            fluksio.emit(loss=loss)
            freq_history.append(np.exp(np.asarray(model.log_freqs[...])).tolist())

    held = jnp.array(tokenizer.encode(eval_text), dtype=jnp.int32)
    bpb = {f"bpb_{n}": eval_bpb(model, held, n, batch_size) for n in (128, 192, 256)}

    path = pathlib.Path(tempfile.mkdtemp()) / "weights.npz"
    save(model, path)
    ref = fluksio.save_artifact(path.read_bytes(), name="weights.npz")

    omegas, probe = probe_curve(model, held, seq_len, probe_idx)
    return {
        "final_loss": loss,
        "freq_history": freq_history,
        "probe_omegas": omegas,
        "probe_losses": probe,
        "weights": ref,
    } | bpb


Flow(
    "s4_learnable_freqs",
    title="S4 — learnable RoPE frequencies",
    nodes=[dataset, study],
    inputs=[
        Port("mode", "str", initial="fixed"),
        Port("lr", "float", initial=3e-3),
        Port("freq_lr", "float", initial=3e-3),
        Port("seed", "int", initial=0),
        Port("steps", "int", initial=2000),
        Port("seq_len", "int", initial=128),
    ],
    outputs=[
        "final_loss",
        "bpb_128",
        "bpb_192",
        "bpb_256",
        # json blobs, declared as outputs so `export runs` carries them --
        # neither export surface reaches plain node ports.
        "freq_history",
        "probe_omegas",
        "probe_losses",
    ],
)


if __name__ == "__main__":
    # Self-checks, here rather than in a pytest file beside this one: a test
    # module importing `study` gives the sync walk a second copy of it, and
    # two Flow declarations of the same name.
    _cfg = Config(dim=64, n_layers=2, context=64)
    _stock = spectrum(_cfg.dim // _cfg.n_heads)
    _tokens = jax.random.randint(jax.random.PRNGKey(1), (2, 32), 0, 256)

    # Same draw order as Wren, so the fixed arm is the stock model exactly.
    _base = Wren(_cfg, rngs=nnx.Rngs(0))
    _spec = SpectralWren(_cfg, nnx.Rngs(0), _stock)
    assert jnp.allclose(_base(_tokens), _spec(_tokens), atol=1e-5)

    # The spectrum moves in the learned arm and only there; everything else
    # trains in all three.
    for _mode, _moves in (("learned", True), ("fixed", False), ("fope", False)):
        _m = SpectralWren(_cfg, nnx.Rngs(0), _stock)
        _opt = nnx.Optimizer(
            _m, freq_routed(3e-3, 3e-3, _mode == "learned"), wrt=nnx.Param
        )
        _freqs_before = _m.log_freqs[...].copy()
        _wqkv_before = _m.blocks[0].wqkv.kernel[...].copy()
        train_step(_m, _opt, _tokens, _tokens)
        assert bool(jnp.allclose(_freqs_before, _m.log_freqs[...])) != _moves, _mode
        assert not jnp.allclose(_wqkv_before, _m.blocks[0].wqkv.kernel[...]), _mode

    print("s4 self-checks ok")
