"""S5 — QK-norm as a barren-plateau diagnostic.

QML's trainability instrument — gradient variance as a function of depth
(McClean et al., arXiv:1803.11173) — applied to the SAN paper's claim that
QK-norm is what keeps deep attention-only stacks trainable
(arXiv:2607.18363). Two arms identical except for the per-head
RMS-normalization of queries and keys, at depths {4, 8, 16, 32}, on the s4
protocol (Home-Assistant-Requests, seq 128, batch 32, muon lr 3e-3 fixed —
the tuned default; per-depth LR sweeps are out of scope).

Instruments, all on K fixed probe batches shared across every run:

- Per-layer gradient statistics over the block matrices (wqkv, wo), at init
  and every probe_every steps. The per-parameter gradient over the K batches
  splits into *signal* (squared mean gradient) and *noise* (variance across
  batches, at the training batch size); their sum is the BP-style second
  moment E[g^2] (over random inits the mean gradient is ~0, so E[g^2] is
  the variance QML plots), their ratio the SNR the optimizer actually sees.
- Attention entropy per layer, each query position normalized by its
  maximum log(position+1), on one probe batch — softmax saturation is the
  suspected classical mechanism where training fails.
- Band-resolved (s4 addendum): the pre-softmax logit decomposes exactly
  over RoPE's d/2 2D frequency pairs, so we record each slot's share of
  logit power and, from the q/k column blocks of wqkv, per-slot gradient
  signal/noise — s4 predicts the under-trained band (period > train
  window) carries mostly batch noise. This is the s6 go/no-go data.

The training node declares `resources={"gpus": 1}`; the engine must know
the card is there or it trains on the CPU. `.fluksio/env` carries
`FLOW_GPUS=1` for this checkout; elsewhere `fluksio serve --gpus 1`.

    uv run fluksio sync dev/s5_qknorm
    uv run fluksio sweep s5_qknorm --param depth=4,8,16,32 \
        --param qk_norm=on,off --param seed=0,1,2

Self-check (parity with the stock model, decomposition exactness, probe
algebra): `uv run python dev/s5_qknorm/study.py`
"""

import math
import pathlib
import re
import tempfile

import fluksio
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from fluksio import Flow, Port, node

from wren.model import AttentionBlock, Config, Wren, rope, save
from wren.train import muon, sample_batch, train_step


class ProbeBlock(AttentionBlock):
    """AttentionBlock with QK-norm switchable off.

    `use_qk_norm` is a static flag; the norm modules are created either way
    so the parameter draw order matches the stock block exactly (pinned by
    the parity self-check), they are just bypassed in the off arm.
    """

    def __init__(self, cfg: Config, rngs: nnx.Rngs, qk_norm: bool):
        super().__init__(cfg, rngs)
        self.use_qk_norm = qk_norm

    def __call__(self, x: jax.Array) -> jax.Array:
        batch, seq, dim = x.shape
        h = self.norm(x)
        q, k, v = jnp.split(self.wqkv(h), 3, axis=-1)
        q = q.reshape(batch, seq, self.n_heads, self.head_dim)
        k = k.reshape(batch, seq, self.n_heads, self.head_dim)
        v = v.reshape(batch, seq, self.n_heads, self.head_dim)
        if self.use_qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        q, k = rope(q), rope(k)

        scores = jnp.einsum("bqhd,bkhd->bhqk", q, k) / jnp.sqrt(self.head_dim)
        causal = jnp.tril(jnp.ones((seq, seq), dtype=bool))
        scores = jnp.where(causal, scores, -jnp.inf)
        attn = jax.nn.softmax(scores, axis=-1)

        out = jnp.einsum("bhqk,bkhd->bqhd", attn, v).reshape(batch, seq, dim)
        return x + self.wo(out)

    def forward_stats(self, x: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        """__call__ with the logit decomposed over RoPE's frequency pairs.

        Returns (block output, normalized attention entropy, per-slot logit
        power). The rotated q.k product for pair i is exactly a_i cos(w_i
        (n-m)) + b_i sin(w_i (n-m)), so summing the
        per-pair contributions reproduces the attention scores — pinned by
        the decomposition self-check.
        """
        batch, seq, dim = x.shape
        half = self.head_dim // 2
        h = self.norm(x)
        q, k, v = jnp.split(self.wqkv(h), 3, axis=-1)
        q = q.reshape(batch, seq, self.n_heads, self.head_dim)
        k = k.reshape(batch, seq, self.n_heads, self.head_dim)
        v = v.reshape(batch, seq, self.n_heads, self.head_dim)
        if self.use_qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        q, k = rope(q), rope(k)
        qp = jnp.stack([q[..., :half], q[..., half:]], axis=-1)
        kp = jnp.stack([k[..., :half], k[..., half:]], axis=-1)
        # (batch, heads, q, k, slot): pair i's contribution to the logit.
        contrib = jnp.einsum("bqhfc,bkhfc->bhqkf", qp, kp) / jnp.sqrt(self.head_dim)

        causal = jnp.tril(jnp.ones((seq, seq), dtype=bool))
        scores = jnp.where(causal, contrib.sum(-1), -jnp.inf)
        attn = jax.nn.softmax(scores, axis=-1)

        plogp = jnp.where(attn > 0, attn * jnp.log(attn), 0.0)
        max_ent = jnp.log(jnp.arange(seq) + 1.0)  # position m attends m+1 keys
        entropy = (-plogp.sum(-1)[..., 1:] / max_ent[1:]).mean()

        masked = jnp.where(causal[None, None, :, :, None], contrib, 0.0)
        n_valid = seq * (seq + 1) / 2
        power = (masked * masked).sum((0, 1, 2, 3)) / (batch * self.n_heads * n_valid)

        out = jnp.einsum("bhqk,bkhd->bqhd", attn, v).reshape(batch, seq, dim)
        return x + self.wo(out), entropy, power


class ProbeWren(Wren):
    """Wren at configurable depth with switchable QK-norm.

    Draws parameters in the same order as Wren.__init__, so with qk_norm on
    and the same seed it is byte-identical to the stock model (pinned by the
    self-check below).
    """

    def __init__(self, cfg: Config, rngs: nnx.Rngs, qk_norm: bool):
        self.cfg = cfg
        self.embed = nnx.Embed(cfg.vocab_size, cfg.dim, rngs=rngs)
        self.blocks: nnx.List[ProbeBlock] = nnx.List(
            ProbeBlock(cfg, rngs, qk_norm) for _ in range(cfg.n_layers)
        )
        self.norm = nnx.RMSNorm(cfg.dim, rngs=rngs)

    def probe_stats(self, tokens: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        """(logits, per-layer entropy (L,), per-layer slot power (L, d/2))."""
        x = self.embed(tokens)
        ents, powers = [], []
        for block in self.blocks:
            x, ent, power = block.forward_stats(x)
            ents.append(ent)
            powers.append(power)
        logits = self.embed.attend(self.norm(x))
        return logits, jnp.stack(ents), jnp.stack(powers)


@nnx.jit
def _grads(model: ProbeWren, x: jax.Array, y: jax.Array):
    def loss_fn(model: ProbeWren) -> jax.Array:
        logits = model(x)
        return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()

    return nnx.value_and_grad(loss_fn)(model)


@nnx.jit
def _probe_stats(model: ProbeWren, tokens: jax.Array):
    return model.probe_stats(tokens)


def kernel_grads(grads) -> dict[tuple[int, str], np.ndarray]:
    """(layer, 'wqkv'|'wo') -> gradient array, from an nnx grad state."""
    out = {}
    for path, leaf in jax.tree_util.tree_flatten_with_path(grads)[0]:
        m = re.search(
            r"blocks.*?(\d+).*?(wqkv|wo).*?kernel", jax.tree_util.keystr(path)
        )
        if m:
            out[int(m.group(1)), m.group(2)] = np.asarray(leaf)
    return out


def snapshot(
    model: ProbeWren,
    batches: list[tuple[jax.Array, jax.Array]],
    probe_x: jax.Array,
    step: int,
) -> dict:
    """One probe point: per-layer gradient signal/noise, attention entropy,
    and the band-resolved stats, all as plain (json-able) lists."""
    cfg = model.cfg
    n = len(batches)
    sums: dict[tuple[int, str], list[np.ndarray]] = {}
    losses = []
    for x, y in batches:
        loss, grads = _grads(model, x, y)
        losses.append(float(loss))
        for key, g in kernel_grads(grads).items():
            # float64: squared BP-regime gradients can underflow float32.
            g = g.astype(np.float64)
            s = sums.setdefault(key, [np.zeros_like(g), np.zeros_like(g)])
            s[0] += g
            s[1] += g * g

    dim, heads, layers = cfg.dim, cfg.n_heads, cfg.n_layers
    half = dim // heads // 2
    sig_sum, var_sum, count = np.zeros(layers), np.zeros(layers), np.zeros(layers)
    bands = {
        t: np.zeros((layers, half)) for t in ("sig_q", "noise_q", "sig_k", "noise_k")
    }
    for (layer, name), (s0, s1) in sums.items():
        mean = s0 / n
        var = s1 / n - mean**2  # biased (K batches); comparisons only
        sig_sum[layer] += (mean**2).sum()
        var_sum[layer] += var.sum()
        count[layer] += mean.size
        if name == "wqkv":
            for j, tag in ((0, "q"), (1, "k")):
                m = mean[:, j * dim : (j + 1) * dim].reshape(dim, heads, 2 * half)
                v = var[:, j * dim : (j + 1) * dim].reshape(dim, heads, 2 * half)
                pair_m = np.stack([m[..., :half], m[..., half:]])
                pair_v = np.stack([v[..., :half], v[..., half:]])
                bands[f"sig_{tag}"][layer] = (pair_m**2).mean((0, 1, 2))
                bands[f"noise_{tag}"][layer] = pair_v.mean((0, 1, 2))

    _, entropy, power = _probe_stats(model, probe_x)
    return {
        "step": step,
        "probe_loss": sum(losses) / n,
        "sig": (sig_sum / count).tolist(),
        "noise": (var_sum / count).tolist(),
        "entropy": np.asarray(entropy).tolist(),
        "power": np.asarray(power).tolist(),
        **{f"band_{t}": v.tolist() for t, v in bands.items()},
    }


def clean(x):
    """json-safe: non-finite -> None (a diverged run must still export),
    floats rounded to 6 significant digits (the blobs are large)."""
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, list):
        return [clean(v) for v in x]
    if isinstance(x, float):
        return float(f"{x:.6g}") if math.isfinite(x) else None
    return x


@nnx.jit
def _ce(model: ProbeWren, x: jax.Array, y: jax.Array) -> jax.Array:
    return optax.softmax_cross_entropy_with_integer_labels(model(x), y).mean()


def eval_bpb(model: ProbeWren, held: jax.Array, seq_len: int, batch_size: int) -> float:
    """Held-out cross-entropy in bits/byte at the given window length."""
    windows = (len(held) - 1) // seq_len
    xs = held[: windows * seq_len].reshape(windows, seq_len)
    ys = held[1 : windows * seq_len + 1].reshape(windows, seq_len)
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
        Port("depth", "int"),
        Port("qk_norm", "str"),
        Port("lr", "float"),
        Port("seed", "int"),
        Port("steps", "int"),
        Port("seq_len", "int"),
    ],
    provides=[
        Port("loss", "float", stream=True),
        Port("final_loss", "float"),
        Port("bpb_128", "float"),
        Port("probes", "json"),
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
    depth: int,
    qk_norm: str,
    lr: float,
    seed: int,
    steps: int = 2000,
    seq_len: int = 128,
    batch_size: int = 32,  # the probe measures noise at the training batch size
    log_every: int = 50,
    probe_every: int = 100,
    n_probe_batches: int = 8,
) -> dict:
    import os

    from wren import tokenizer

    # An unsatisfiable `gpus: 1` arrives as CUDA_VISIBLE_DEVICES="" and would
    # train on the CPU without saying so; fail instead.
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        raise RuntimeError("no card declared: `fluksio serve --gpus 1`")
    if qk_norm not in ("on", "off"):
        raise ValueError(f"qk_norm must be 'on' or 'off', got {qk_norm!r}")

    cfg = Config(n_layers=depth, context=seq_len)
    model = ProbeWren(cfg, nnx.Rngs(seed), qk_norm == "on")
    optimizer = nnx.Optimizer(model, muon(lr), wrt=nnx.Param)

    data = jnp.array(tokenizer.encode(train_text), dtype=jnp.int32)
    # Probe batches are fixed across runs and probe points: seed-independent
    # key, so every arm sees the same data and differences are model-side.
    batches = [
        sample_batch(data, batch_size, seq_len, k)
        for k in jax.random.split(jax.random.PRNGKey(5150), n_probe_batches)
    ]
    probe_x = batches[0][0][:8]  # entropy/power need one small batch only

    probes = [snapshot(model, batches, probe_x, 0)]  # at init, before any update
    key = jax.random.PRNGKey(seed)
    loss = float("nan")
    for step in range(1, steps + 1):
        key, sub = jax.random.split(key)
        x, y = sample_batch(data, batch_size, seq_len, sub)
        loss = float(train_step(model, optimizer, x, y))
        if step % log_every == 0 or step == steps:
            fluksio.emit(loss=loss)
        if step % probe_every == 0:
            probes.append(snapshot(model, batches, probe_x, step))

    held = jnp.array(tokenizer.encode(eval_text), dtype=jnp.int32)
    bpb = eval_bpb(model, held, seq_len, batch_size)

    path = pathlib.Path(tempfile.mkdtemp()) / "weights.npz"
    save(model, path)
    ref = fluksio.save_artifact(path.read_bytes(), name="weights.npz")

    return {
        "final_loss": loss if math.isfinite(loss) else None,
        "bpb_128": bpb if math.isfinite(bpb) else None,
        "probes": clean(probes),
        "weights": ref,
    }


Flow(
    "s5_qknorm",
    title="S5 — QK-norm as a barren-plateau diagnostic",
    nodes=[dataset, study],
    inputs=[
        Port("depth", "int", initial=8),
        Port("qk_norm", "str", initial="on"),
        Port("lr", "float", initial=3e-3),
        Port("seed", "int", initial=0),
        Port("steps", "int", initial=2000),
        Port("seq_len", "int", initial=128),
    ],
    outputs=[
        "final_loss",
        "bpb_128",
        # json blob, declared as an output so `export runs` carries it.
        "probes",
    ],
)


if __name__ == "__main__":
    # Self-checks, here rather than in a pytest file beside this one: a test
    # module importing `study` gives the sync walk a second copy of it, and
    # two Flow declarations of the same name.
    _cfg = Config(dim=64, n_layers=2, context=64)
    _tokens = jax.random.randint(jax.random.PRNGKey(1), (2, 32), 0, 256)

    # Same draw order as Wren, so the on arm is the stock model exactly.
    _base = Wren(_cfg, rngs=nnx.Rngs(0))
    _on = ProbeWren(_cfg, nnx.Rngs(0), qk_norm=True)
    _off = ProbeWren(_cfg, nnx.Rngs(0), qk_norm=False)
    assert jnp.allclose(_base(_tokens), _on(_tokens), atol=1e-5)
    assert not jnp.allclose(_on(_tokens), _off(_tokens), atol=1e-5)

    # The decomposed forward is the same computation: per-pair contributions
    # sum exactly to the scores, so the logits agree; entropy is in (0, 1].
    for _m in (_on, _off):
        _logits, _ent, _power = _m.probe_stats(_tokens)
        assert jnp.allclose(_logits, _m(_tokens), atol=1e-5)
        assert _ent.shape == (2,) and _power.shape == (2, 8)
        assert bool(jnp.all((_ent > 0) & (_ent <= 1.0 + 1e-6)))

    # Off arm: the bypassed norm scales get zero gradient, wqkv still trains.
    _, _g = _grads(_off, _tokens, _tokens)
    for _path, _leaf in jax.tree_util.tree_flatten_with_path(_g)[0]:
        _key = jax.tree_util.keystr(_path)
        if "q_norm" in _key or "k_norm" in _key:
            assert not jnp.any(jnp.asarray(_leaf)), _key
    assert (0, "wqkv") in kernel_grads(_g) and (1, "wo") in kernel_grads(_g)

    # Probe algebra: identical batches mean zero cross-batch noise, and the
    # signal is then the full second moment.
    _x, _y = _tokens, _tokens
    _snap = snapshot(_on, [(_x, _y)] * 3, _tokens, step=0)
    assert all(abs(v) < 1e-12 * s for v, s in zip(_snap["noise"], _snap["sig"]))
    assert np.all(np.abs(_snap["band_noise_q"]) < 1e-12 * np.max(_snap["band_sig_q"]))
    assert len(_snap["sig"]) == 2 and np.shape(_snap["band_sig_q"]) == (2, 8)

    # Divergence must still export: non-finite values become None.
    assert clean([float("nan"), {"a": float("inf")}, 1.5]) == [None, {"a": None}, 1.5]

    print("s5 self-checks ok")
