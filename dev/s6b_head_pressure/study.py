"""S6b — head-dim pressure: does banding pay when heads get narrow?

Follow-up to s6, which found banding loss-neutral at 4 heads x 32 frequency
pairs: each shared head is a spectral superset, so the prior had nothing to
add. This study shrinks head_dim by raising n_heads at fixed dim 256 —
heads {4, 8, 16, 32}, head_dim {64, 32, 16, 8} — so the shared arm's
per-head spectrum falls from 32 to 4 distinct frequencies while the banded
union always holds n_heads x half = 128 distinct frequencies tiling the
same range (s6's four scale bands, each subdivided log-uniformly into
n_heads/4 contiguous sub-bands, one head per sub-band). Prediction from
the s6 verdict: the shared arm degrades before the banded arm as per-head
resolution collapses; the arm *gap* at each head count identifies the
spectral-resolution effect, while absolute degradation with head count is
confounded (value-head width, per-head softmax) and shared by both arms.

Two arms only — the learnable-spectrum thread is closed (s6c). Parameter
counts match between arms at every head count. Protocol and instruments
are s6's: Home-Assistant-Requests, dim 256, seq 128, batch 32, 2000 steps,
muon 3e-3; held-out bits/byte at windows {128, 192, 256}; per-head
normalized attention entropy and attention-distance profiles at windows
{128, 256}.

The training node declares `resources={"gpus": 1}`; the engine must know
the card is there or it trains on the CPU. `.fluksio/env` carries
`FLOW_GPUS=1` for this checkout; elsewhere `fluksio serve --gpus 1`.

    uv run fluksio sync dev/s6b_head_pressure
    uv run fluksio sweep s6b_head_pressure --param mode=shared,banded --param n_heads=4,8,16,32 --param seed=0,1,2

Self-check (parity with the stock model at 4 and 8 heads, band tiling,
instruments, frozen spectrum): `uv run python dev/s6b_head_pressure/study.py`
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

# Interior scale-band edges as periods in bytes (s6's); outer edges come
# from the stock spectrum's own range at the given head_dim.
BAND_PERIODS = (32.0, 128.0, 1024.0)
NBINS = 9  # log2 distance bins: 0, 1, 2–3, ..., 128–255 (window <= 256)


def stock_spectrum(n_heads: int, half: int) -> jax.Array:
    """The stock rope spectrum, one copy per head: shape (heads, half)."""
    freqs = 1.0 / (10000.0 ** (jnp.arange(half) / half))
    return jnp.tile(freqs[None, :], (n_heads, 1))


def band_spectrum(n_heads: int, half: int) -> jax.Array:
    """Per-head banded spectrum, shape (heads, half), high frequency first.

    The four scale bands are each split log-uniformly into n_heads/4
    contiguous sub-bands; head h gets `half` log-spaced frequencies over
    its sub-band, half-open so heads tile without duplicates (the last
    head includes its low edge). The union is always n_heads * half
    distinct frequencies over the stock range — constant total resolution
    while the shared arm's per-head resolution falls with head_dim.
    At n_heads=4 this reduces to s6's band_spectrum exactly.
    """
    assert n_heads % 4 == 0
    edges = [
        1.0,
        *(2 * math.pi / p for p in BAND_PERIODS),
        10000 ** -((half - 1) / half),
    ]
    k = n_heads // 4
    sub = []
    for b in range(4):
        sub.extend(np.geomspace(edges[b], edges[b + 1], k + 1)[:-1])
    sub.append(edges[-1])
    rows = [
        np.geomspace(sub[h], sub[h + 1], half + 1)[:half]
        if h < n_heads - 1
        else np.geomspace(sub[h], sub[h + 1], half)
        for h in range(n_heads)
    ]
    return jnp.array(np.stack(rows))


def rope_heads(x: jax.Array, freqs: jax.Array) -> jax.Array:
    """wren.model.rope with a per-head spectrum, freqs (heads, half).

    No offset: s6b measures loss on full windows, there is no decode path.
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
    inherited __call__/decode are unused (no generation in s6b).
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
    spectrum and the same seed it is byte-identical to the stock model at
    any head count (pinned by the self-check below). The spectrum never
    trains in s6b; freq_routed keeps it frozen.
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


def freq_routed(lr: float) -> optax.GradientTransformation:
    """wren.train.muon on everything but log_freqs, which stays frozen —
    without the route muon would orthogonalize the (heads, half) spectrum
    as if it were a weight matrix."""

    def labels(params: dict) -> dict:
        return jax.tree_util.tree_map_with_path(
            lambda path, p: (
                "freqs" if "log_freqs" in jax.tree_util.keystr(path) else "rest"
            ),
            params,
        )

    return optax.multi_transform(
        {"rest": muon(lr), "freqs": optax.set_to_zero()}, labels
    )


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
        Port("n_heads", "int"),
        Port("lr", "float"),
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
    n_heads: int,
    lr: float,
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

    cfg = Config(context=256, n_heads=n_heads)  # context must hold window 256
    half = cfg.dim // cfg.n_heads // 2
    init = (
        stock_spectrum(cfg.n_heads, half)
        if mode == "shared"
        else band_spectrum(cfg.n_heads, half)
    )
    model = BandedWren(cfg, nnx.Rngs(seed), init)
    optimizer = nnx.Optimizer(model, freq_routed(lr), wrt=nnx.Param)

    data = jnp.array(tokenizer.encode(train_text), dtype=jnp.int32)
    key = jax.random.PRNGKey(seed)
    loss = float("nan")
    for step in range(steps):
        key, sub = jax.random.split(key)
        x, y = sample_batch(data, batch_size, seq_len, sub)
        loss = float(train_step(model, optimizer, x, y))
        if step % log_every == 0 or step == steps - 1:
            fluksio.emit(loss=loss)

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

    return {"final_loss": loss, "head_profiles": profiles, "weights": ref} | bpb


Flow(
    "s6b_head_pressure",
    title="S6b — head-dim pressure",
    nodes=[dataset, study],
    inputs=[
        Port("mode", "str", initial="shared"),
        Port("n_heads", "int", initial=4),
        Port("lr", "float", initial=3e-3),
        Port("seed", "int", initial=0),
        Port("steps", "int", initial=2000),
        Port("seq_len", "int", initial=128),
    ],
    outputs=[
        "final_loss",
        "bpb_128",
        "bpb_192",
        "bpb_256",
        # json blob, declared as an output so `export runs` carries it --
        # neither export surface reaches plain node ports.
        "head_profiles",
    ],
)


if __name__ == "__main__":
    # Self-checks, here rather than in a pytest file beside this one: a test
    # module importing `study` gives the sync walk a second copy of it, and
    # two Flow declarations of the same name.
    _tokens = jax.random.randint(jax.random.PRNGKey(1), (2, 32), 0, 256)

    for _nh in (4, 8):
        _cfg = Config(dim=64, n_layers=2, n_heads=_nh, context=64)
        _half = _cfg.dim // _cfg.n_heads // 2

        # Same draw order as Wren, so the shared arm is the stock model
        # exactly, at any head count.
        _base = Wren(_cfg, rngs=nnx.Rngs(0))
        _shared = BandedWren(_cfg, nnx.Rngs(0), stock_spectrum(_nh, _half))
        assert jnp.allclose(_base(_tokens), _shared(_tokens), atol=1e-5), _nh

        # Bands tile the stock range: descending within each head,
        # contiguous across heads, union endpoints equal to the stock
        # spectrum's, n_heads * half slots in total.
        _bands = np.asarray(band_spectrum(_nh, _half))
        assert _bands.shape == (_nh, _half)
        assert (np.diff(_bands, axis=1) < 0).all()
        assert np.isclose(_bands[0, 0], 1.0)
        assert np.isclose(_bands[-1, -1], 10000 ** -((_half - 1) / _half))
        for _h in range(_nh - 1):
            assert _bands[_h, -1] > _bands[_h + 1, 0]

        # Instruments: distance profile rows sum to 1, entropy in [0, 1].
        _banded = BandedWren(_cfg, nnx.Rngs(0), band_spectrum(_nh, _half))
        _ent, _prof = _banded.profile(_tokens)
        assert _ent.shape == (2, _nh) and _prof.shape == (2, _nh, NBINS)
        assert jnp.allclose(_prof.sum(-1), 1.0, atol=1e-5)
        assert bool((_ent >= 0.0).all()) and bool((_ent <= 1.0).all())

        # The spectrum never moves; everything else trains.
        _opt = nnx.Optimizer(_banded, freq_routed(3e-3), wrt=nnx.Param)
        _before = _banded.log_freqs.value.copy()
        _wqkv_before = _banded.blocks[0].wqkv.kernel.value.copy()
        train_step(_banded, _opt, _tokens, _tokens)
        assert jnp.allclose(_before, _banded.log_freqs.value), _nh
        assert not jnp.allclose(_wqkv_before, _banded.blocks[0].wqkv.kernel.value), _nh

    print("s6b self-checks ok")
