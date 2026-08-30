"""S6 — spectral-aware attention heads (ROADMAP.md, go per the s5 verdict).

Three arms over the s4 protocol (Home-Assistant-Requests, seq 128, 2000
steps, muon 3e-3), differing only in how the rope spectrum is laid out
across the 4 heads:

- shared:         the stock spectrum 10000^(-2i/d) in every head, frozen —
                  the control, byte-identical to wren.model.Wren.
- banded:         each head one contiguous frequency band, frozen. Edges at
                  periods {2pi, 32, 128, 1024, 2pi*10000^(31/32)} bytes —
                  byte/short-word, word/line (up to the train window),
                  sample (s5's high-power 149–265 B ramp band), long/DC.
                  32 log-spaced frequencies per head; the union spans
                  exactly the stock range, so only the allocation differs.
- banded_learned: banded init, frequencies trainable — except slots with
                  omega < 2pi/seq_len, frozen at the prior. s4 showed
                  omega-drift in that band is pure extrapolation damage and
                  s5 that its *coefficients* still do the logit work, so the
                  prior keeps those representatives and data adjusts the
                  spectrum only where the training window can rank it.

Parameter counts match across arms (log_freqs exists everywhere; the fixed
arms just never step it — s4's routing pattern). Length generalization is
held-out bits/byte at windows {128, 192, 256}. Per-head instruments, taken
on held-out windows at 128 and 256: normalized attention entropy (s5's
estimator, kept per head) and the attention-distance profile — attention
mass per log2 distance bin, which shows directly whether banded heads
specialize to their scale.

The training node declares `resources={"gpus": 1}`; the engine must know
the card is there or it trains on the CPU. `.fluksio/env` carries
`FLOW_GPUS=1` for this checkout; elsewhere `fluksio serve --gpus 1`.

    uv run fluksio sync dev/s6_banded_heads
    uv run fluksio sweep s6_banded_heads --param mode=shared,banded,banded_learned --param seed=0,1,2

Self-check (parity with the stock model, band tiling, instruments,
optimizer routing): `uv run python dev/s6_banded_heads/study.py`
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

# Interior band edges as periods in bytes; the outer edges come from the
# stock spectrum's own range. Chosen from the byte-level scales s5 measured
# (logit power at 6–8 B and 149–265 B, ~1%/slot in the deep tail).
BAND_PERIODS = (32.0, 128.0, 1024.0)
NBINS = 9  # log2 distance bins: 0, 1, 2–3, ..., 128–255 (window <= 256)


def stock_spectrum(n_heads: int, half: int) -> jax.Array:
    """The stock rope spectrum, one copy per head: shape (heads, half)."""
    freqs = 1.0 / (10000.0 ** (jnp.arange(half) / half))
    return jnp.tile(freqs[None, :], (n_heads, 1))


def band_spectrum(n_heads: int, half: int) -> jax.Array:
    """Per-head banded spectrum, shape (heads, half), high frequency first.

    Head h gets `half` log-spaced frequencies over [edge_h, edge_{h+1}),
    half-open so heads tile without duplicates; the last head includes its
    low edge so the union endpoints equal the stock spectrum's.
    """
    assert n_heads == len(BAND_PERIODS) + 1
    # Descending omega: stock maximum, the interior scale edges, stock minimum.
    edges = [
        1.0,
        *(2 * math.pi / p for p in BAND_PERIODS),
        10000 ** -((half - 1) / half),
    ]
    rows = [
        np.geomspace(edges[h], edges[h + 1], half + 1)[:half]
        if h < n_heads - 1
        else np.geomspace(edges[h], edges[h + 1], half)
        for h in range(n_heads)
    ]
    return jnp.array(np.stack(rows))


def rope_heads(x: jax.Array, freqs: jax.Array) -> jax.Array:
    """wren.model.rope with a per-head spectrum, freqs (heads, half).

    No offset: s6 measures loss on full windows, there is no decode path.
    """
    _, seq, _, _ = x.shape
    half = freqs.shape[1]
    angles = jnp.arange(seq)[:, None, None] * freqs[None, :, :]  # (seq, heads, half)
    cos = jnp.cos(angles)[None, :, :, :]
    sin = jnp.sin(angles)[None, :, :, :]
    x1, x2 = x[..., :half], x[..., half:]
    return jnp.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)


class BandedBlock(AttentionBlock):
    """AttentionBlock with a per-head rope spectrum injected per call.

    `forward(x, freqs)` is the stock `__call__` with rope_heads; the
    inherited __call__/decode are unused (no generation in s6).
    """

    def forward(self, x: jax.Array, freqs: jax.Array) -> jax.Array:
        batch, seq, dim = x.shape
        h = self.norm(x)
        q, k, v = jnp.split(self.wqkv(h), 3, axis=-1)
        q = q.reshape(batch, seq, self.n_heads, self.head_dim)
        k = k.reshape(batch, seq, self.n_heads, self.head_dim)
        v = v.reshape(batch, seq, self.n_heads, self.head_dim)
        q = rope_heads(self.q_norm(q), freqs)
        k = rope_heads(self.k_norm(k), freqs)

        scores = jnp.einsum("bqhd,bkhd->bhqk", q, k) / jnp.sqrt(self.head_dim)
        causal = jnp.tril(jnp.ones((seq, seq), dtype=bool))
        scores = jnp.where(causal, scores, -jnp.inf)
        attn = jax.nn.softmax(scores, axis=-1)

        out = jnp.einsum("bhqk,bkhd->bqhd", attn, v).reshape(batch, seq, dim)
        return x + self.wo(out)

    def head_stats(
        self, x: jax.Array, freqs: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """forward with per-head instruments.

        Returns (block output, per-head normalized attention entropy
        (heads,), per-head attention-distance profile (heads, NBINS) —
        attention mass per log2 distance bin, rows summing to 1).
        """
        batch, seq, dim = x.shape
        h = self.norm(x)
        q, k, v = jnp.split(self.wqkv(h), 3, axis=-1)
        q = q.reshape(batch, seq, self.n_heads, self.head_dim)
        k = k.reshape(batch, seq, self.n_heads, self.head_dim)
        v = v.reshape(batch, seq, self.n_heads, self.head_dim)
        q = rope_heads(self.q_norm(q), freqs)
        k = rope_heads(self.k_norm(k), freqs)

        scores = jnp.einsum("bqhd,bkhd->bhqk", q, k) / jnp.sqrt(self.head_dim)
        causal = jnp.tril(jnp.ones((seq, seq), dtype=bool))
        scores = jnp.where(causal, scores, -jnp.inf)
        attn = jax.nn.softmax(scores, axis=-1)

        # Entropy normalized per query by its maximum log(pos+1), position 0
        # excluded — s5's estimator, kept per head instead of averaged.
        plogp = jnp.where(attn > 0, attn * jnp.log(attn), 0.0)
        max_ent = jnp.log(jnp.arange(seq) + 1.0)
        entropy = (-plogp.sum(-1)[..., 1:] / max_ent[1:]).mean(axis=(0, 2))

        # Distance q-k -> bin {0, 1, 2–3, 4–7, ...}; attn is exactly zero
        # beyond the causal mask, so negative distances never carry mass.
        dist = jnp.maximum(jnp.arange(seq)[:, None] - jnp.arange(seq)[None, :], 0)
        bins = jnp.where(dist == 0, 0, jnp.floor(jnp.log2(jnp.maximum(dist, 1))) + 1)
        onehot = jax.nn.one_hot(bins.astype(jnp.int32), NBINS)  # (seq, seq, NBINS)
        profile = jnp.einsum("bhqk,qkn->hn", attn, onehot) / (batch * seq)

        out = jnp.einsum("bhqk,bkhd->bqhd", attn, v).reshape(batch, seq, dim)
        return x + self.wo(out), entropy, profile


class BandedWren(Wren):
    """Wren with an explicit per-head rope spectrum, shape (heads, half).

    Draws parameters in the same order as Wren.__init__, so with the stock
    spectrum and the same seed it is byte-identical to the stock model
    (pinned by the self-check below). Whether — and which slots of —
    log_freqs train is the optimizer's business (freq_routed), not the
    model's.
    """

    def __init__(self, cfg: Config, rngs: nnx.Rngs, init_freqs: jax.Array):
        self.cfg = cfg
        self.embed = nnx.Embed(cfg.vocab_size, cfg.dim, rngs=rngs)
        self.blocks: nnx.List[BandedBlock] = nnx.List(
            BandedBlock(cfg, rngs) for _ in range(cfg.n_layers)
        )
        self.norm = nnx.RMSNorm(cfg.dim, rngs=rngs)
        self.log_freqs = nnx.Param(jnp.log(init_freqs))

    def __call__(self, tokens: jax.Array) -> jax.Array:
        x = self.embed(tokens)
        freqs = jnp.exp(self.log_freqs.value)
        for block in self.blocks:
            x = block.forward(x, freqs)
        return self.embed.attend(self.norm(x))

    def profile(self, tokens: jax.Array) -> tuple[jax.Array, jax.Array]:
        """(per-layer per-head entropy (L, H), distance profile (L, H, NBINS))."""
        x = self.embed(tokens)
        freqs = jnp.exp(self.log_freqs.value)
        ents, profiles = [], []
        for block in self.blocks:
            x, ent, prof = block.head_stats(x, freqs)
            ents.append(ent)
            profiles.append(prof)
        return jnp.stack(ents), jnp.stack(profiles)


def freq_routed(
    lr: float, freq_lr: float, mask: jax.Array | None
) -> optax.GradientTransformation:
    """wren.train.muon on everything but log_freqs; the spectrum gets its own
    Adam at freq_lr on the slots `mask` marks trainable (gradients on frozen
    slots are zeroed before Adam, so their moments never accumulate), or
    frozen updates entirely in the fixed arms (mask None) — s4's routing."""

    def labels(params: dict) -> dict:
        return jax.tree_util.tree_map_with_path(
            lambda path, p: (
                "freqs" if "log_freqs" in jax.tree_util.keystr(path) else "rest"
            ),
            params,
        )

    if mask is None:
        freq_tx = optax.set_to_zero()
    else:
        keep = jnp.asarray(mask, dtype=jnp.float32)
        freeze = optax.stateless(lambda u, p: jax.tree.map(lambda g: g * keep, u))
        freq_tx = optax.chain(freeze, optax.adam(freq_lr))
    return optax.multi_transform({"rest": muon(lr), "freqs": freq_tx}, labels)


@nnx.jit
def _ce(model: BandedWren, x: jax.Array, y: jax.Array) -> jax.Array:
    return optax.softmax_cross_entropy_with_integer_labels(model(x), y).mean()


@nnx.jit
def _profile(model: BandedWren, x: jax.Array) -> tuple[jax.Array, jax.Array]:
    return model.profile(x)


def held_windows(held: jax.Array, seq_len: int) -> tuple[jax.Array, jax.Array]:
    """Non-overlapping (inputs, targets) windows over the held-out stream."""
    windows = (len(held) - 1) // seq_len
    xs = held[: windows * seq_len].reshape(windows, seq_len)
    ys = held[1 : windows * seq_len + 1].reshape(windows, seq_len)
    return xs, ys


def eval_bpb(
    model: BandedWren, held: jax.Array, seq_len: int, batch_size: int
) -> float:
    """Held-out cross-entropy in bits/byte at the given window length."""
    xs, ys = held_windows(held, seq_len)
    nats = [
        float(_ce(model, xs[i : i + batch_size], ys[i : i + batch_size]))
        for i in range(0, len(xs), batch_size)
    ]
    return sum(nats) / len(nats) / math.log(2)


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
        Port("head_profiles", "json"),
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
    profile_windows: int = 64,
) -> dict:
    import os

    from wren import tokenizer

    # An unsatisfiable `gpus: 1` arrives as CUDA_VISIBLE_DEVICES="" and would
    # train on the CPU without saying so; fail instead.
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        raise RuntimeError("no card declared: `fluksio serve --gpus 1`")
    # A bare `uv sync` prunes the gpu extra and jax falls back to the CPU
    # with the card declared and granted; a 2-minute run becomes 31.
    if jax.default_backend() == "cpu":
        raise RuntimeError("jax sees no card: `uv sync --extra gpu`, restart engine")

    cfg = Config(context=256)  # must hold the longest eval window
    half = cfg.dim // cfg.n_heads // 2
    init = (
        stock_spectrum(cfg.n_heads, half)
        if mode == "shared"
        else band_spectrum(cfg.n_heads, half)
    )
    # Trainable where the training window can rank the frequency (completes
    # a cycle in-window); below that the prior keeps the representative.
    mask = init >= 2 * math.pi / seq_len if mode == "banded_learned" else None
    model = BandedWren(cfg, nnx.Rngs(seed), init)
    optimizer = nnx.Optimizer(model, freq_routed(lr, freq_lr, mask), wrt=nnx.Param)

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
            freq_history.append(np.exp(np.asarray(model.log_freqs.value)).tolist())

    held = jnp.array(tokenizer.encode(eval_text), dtype=jnp.int32)
    bpb = {f"bpb_{n}": eval_bpb(model, held, n, batch_size) for n in (128, 192, 256)}

    profiles = {}
    for n in (128, 256):
        xs, _ = held_windows(held, n)
        ent, prof = _profile(model, xs[:profile_windows])
        profiles[str(n)] = {
            "entropy": np.asarray(ent).tolist(),
            "dist": np.asarray(prof).tolist(),
        }

    path = pathlib.Path(tempfile.mkdtemp()) / "weights.npz"
    save(model, path)
    ref = fluksio.save_artifact(path.read_bytes(), name="weights.npz")

    return {
        "final_loss": loss,
        "freq_history": freq_history,
        "head_profiles": profiles,
        "weights": ref,
    } | bpb


Flow(
    "s6_banded_heads",
    title="S6 — spectral-aware attention heads",
    nodes=[dataset, study],
    inputs=[
        Port("mode", "str", initial="shared"),
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
        "head_profiles",
    ],
)


if __name__ == "__main__":
    # Self-checks, here rather than in a pytest file beside this one: a test
    # module importing `study` gives the sync walk a second copy of it, and
    # two Flow declarations of the same name.
    _cfg = Config(dim=64, n_layers=2, context=64)
    _half = _cfg.dim // _cfg.n_heads // 2
    _tokens = jax.random.randint(jax.random.PRNGKey(1), (2, 32), 0, 256)

    # Same draw order as Wren, so the shared arm is the stock model exactly.
    _base = Wren(_cfg, rngs=nnx.Rngs(0))
    _shared = BandedWren(_cfg, nnx.Rngs(0), stock_spectrum(_cfg.n_heads, _half))
    assert jnp.allclose(_base(_tokens), _shared(_tokens), atol=1e-5)

    # Bands tile the stock range: descending within each head, contiguous
    # across heads, union endpoints equal to the stock spectrum's.
    _bands = np.asarray(band_spectrum(_cfg.n_heads, _half))
    assert (np.diff(_bands, axis=1) < 0).all()
    assert np.isclose(_bands[0, 0], 1.0)
    assert np.isclose(_bands[-1, -1], 10000 ** -((_half - 1) / _half))
    for _h in range(_cfg.n_heads - 1):
        assert _bands[_h, -1] > _bands[_h + 1, 0]

    # Instruments: distance profile rows sum to 1, entropy lands in [0, 1].
    _ent, _prof = _shared.profile(_tokens)
    assert _ent.shape == (2, 4) and _prof.shape == (2, 4, NBINS)
    assert jnp.allclose(_prof.sum(-1), 1.0, atol=1e-5)
    assert bool((_ent >= 0.0).all()) and bool((_ent <= 1.0).all())

    # The spectrum moves only in the learned arm and only on unmasked slots;
    # everything else trains in all three arms.
    _init = band_spectrum(_cfg.n_heads, _half)
    _mask = np.asarray(_init >= 2 * math.pi / 32)  # seq 32: mixed per head
    assert _mask.any() and not _mask.all()
    for _mode, _m in (("banded_learned", jnp.asarray(_mask)), ("fixed", None)):
        _model = BandedWren(_cfg, nnx.Rngs(0), _init)
        _opt = nnx.Optimizer(_model, freq_routed(3e-3, 3e-3, _m), wrt=nnx.Param)
        _before = _model.log_freqs.value.copy()
        _wqkv_before = _model.blocks[0].wqkv.kernel.value.copy()
        train_step(_model, _opt, _tokens, _tokens)
        _moved = ~np.isclose(np.asarray(_before), np.asarray(_model.log_freqs.value))
        assert (_moved == _mask).all() if _m is not None else not _moved.any(), _mode
        assert not jnp.allclose(_wqkv_before, _model.blocks[0].wqkv.kernel.value), _mode

    print("s6 self-checks ok")
