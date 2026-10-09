"""Wren: an attention-only transformer (a "Simple Attention Network").

A standard transformer block is attention followed by a feed-forward MLP,
and the MLP holds ~2/3 of the parameters. Following
https://arxiv.org/abs/2607.18363 we drop the MLPs entirely and spend the
saved parameters on *more attention layers* instead. The paper shows this
costs almost nothing in loss (0.006 nats at matched parameter count),
provided queries and keys are normalized (QK-norm) -- that normalization,
not the MLPs, is what keeps deep attention-only stacks trainable.

The known weakness of attention-only models is recalling facts stored in
the weights; their strength is answers grounded in the prompt. Wren gets
its knowledge via RAG/context, so we lean into that strength.
"""

import pathlib
from dataclasses import asdict, dataclass

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from wren.tokenizer import VOCAB_SIZE


@dataclass
class Config:
    vocab_size: int = VOCAB_SIZE
    dim: int = 256  # width of the residual stream
    n_layers: int = 8  # depth: the CPU latency axis -- keep shallow (README)
    n_heads: int = 4
    context: int = 384  # maximum sequence length; holds a pruned state block + calls


def rope(x: jax.Array, offset: jax.Array | int = 0) -> jax.Array:
    """Rotary position embedding (RoPE), applied to queries and keys.

    Rotates each (i, i+half) feature pair by an angle proportional to the
    token's position, at a different frequency per pair. A dot product of
    two rotated vectors then depends only on their *relative* distance, so
    the model needs no learned position table and positions generalize.
    `offset` shifts the positions -- decoding one token at position p passes
    a length-1 sequence with offset=p.
    """
    _, seq, _, head_dim = x.shape  # (batch, seq, heads, head_dim)
    half = head_dim // 2
    freqs = 1.0 / (10000.0 ** (jnp.arange(half) / half))
    angles = (jnp.arange(seq) + offset)[:, None] * freqs[None, :]  # (seq, half)
    cos = jnp.cos(angles)[None, :, None, :]
    sin = jnp.sin(angles)[None, :, None, :]
    x1, x2 = x[..., :half], x[..., half:]
    return jnp.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)


class AttentionBlock(nnx.Module):
    """Pre-norm causal self-attention with QK-norm. That's the whole block:
    there is deliberately no feed-forward network after it."""

    def __init__(self, cfg: Config, rngs: nnx.Rngs):
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.dim // cfg.n_heads
        self.norm = nnx.RMSNorm(cfg.dim, rngs=rngs)
        self.wqkv = nnx.Linear(cfg.dim, 3 * cfg.dim, use_bias=False, rngs=rngs)
        self.wo = nnx.Linear(cfg.dim, cfg.dim, use_bias=False, rngs=rngs)
        # QK-norm: RMS-normalize queries and keys per head before their dot
        # product, bounding attention logits. Per the paper, this is the one
        # ingredient that keeps deep attention-only stacks trainable.
        self.q_norm = nnx.RMSNorm(self.head_dim, rngs=rngs)
        self.k_norm = nnx.RMSNorm(self.head_dim, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        batch, seq, dim = x.shape
        h = self.norm(x)
        q, k, v = jnp.split(self.wqkv(h), 3, axis=-1)
        q = q.reshape(batch, seq, self.n_heads, self.head_dim)
        k = k.reshape(batch, seq, self.n_heads, self.head_dim)
        v = v.reshape(batch, seq, self.n_heads, self.head_dim)
        q = rope(self.q_norm(q))
        k = rope(self.k_norm(k))

        scores = jnp.einsum("bqhd,bkhd->bhqk", q, k) / jnp.sqrt(self.head_dim)
        # Causal mask: token q may only attend to tokens k <= q, so the
        # model can't peek at the future it is being trained to predict.
        causal = jnp.tril(jnp.ones((seq, seq), dtype=bool))
        scores = jnp.where(causal, scores, -jnp.inf)
        attn = jax.nn.softmax(scores, axis=-1)

        out = jnp.einsum("bhqk,bkhd->bqhd", attn, v).reshape(batch, seq, dim)
        return x + self.wo(out)

    def prefill(
        self, x: jax.Array, k_cache: jax.Array, v_cache: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """__call__ plus cache filling: the same attention over the whole
        prompt in one pass, with the roped keys/values written to slots
        0..seq-1 — the batched equivalent of `decode` over those positions."""
        batch, seq, dim = x.shape
        h = self.norm(x)
        q, k, v = jnp.split(self.wqkv(h), 3, axis=-1)
        q = q.reshape(batch, seq, self.n_heads, self.head_dim)
        k = k.reshape(batch, seq, self.n_heads, self.head_dim)
        v = v.reshape(batch, seq, self.n_heads, self.head_dim)
        q = rope(self.q_norm(q))
        k = rope(self.k_norm(k))

        scores = jnp.einsum("bqhd,bkhd->bhqk", q, k) / jnp.sqrt(self.head_dim)
        causal = jnp.tril(jnp.ones((seq, seq), dtype=bool))
        scores = jnp.where(causal, scores, -jnp.inf)
        attn = jax.nn.softmax(scores, axis=-1)

        out = jnp.einsum("bhqk,bkhd->bqhd", attn, v).reshape(batch, seq, dim)
        k_cache = jax.lax.dynamic_update_slice_in_dim(k_cache, k, 0, axis=1)
        v_cache = jax.lax.dynamic_update_slice_in_dim(v_cache, v, 0, axis=1)
        return x + self.wo(out), k_cache, v_cache

    def decode(
        self, x: jax.Array, k_cache: jax.Array, v_cache: jax.Array, pos: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """One token at position `pos`, attending over the cached keys/values.

        The full __call__ recomputes every pairwise score (O(seq^2)); during
        generation only the new token's row changes, so caching k/v makes a
        step O(seq). x is (batch, 1, dim); caches are (batch, context,
        heads, head_dim) and come back updated at `pos`.
        """
        batch, _, dim = x.shape
        h = self.norm(x)
        q, k, v = jnp.split(self.wqkv(h), 3, axis=-1)
        q = q.reshape(batch, 1, self.n_heads, self.head_dim)
        k = k.reshape(batch, 1, self.n_heads, self.head_dim)
        v = v.reshape(batch, 1, self.n_heads, self.head_dim)
        q = rope(self.q_norm(q), offset=pos)
        k = rope(self.k_norm(k), offset=pos)
        k_cache = jax.lax.dynamic_update_slice_in_dim(k_cache, k, pos, axis=1)
        v_cache = jax.lax.dynamic_update_slice_in_dim(v_cache, v, pos, axis=1)

        scores = jnp.einsum("bqhd,bkhd->bhqk", q, k_cache) / jnp.sqrt(self.head_dim)
        mask = jnp.arange(k_cache.shape[1]) <= pos  # causal: only filled slots
        scores = jnp.where(mask[None, None, None, :], scores, -jnp.inf)
        attn = jax.nn.softmax(scores, axis=-1)

        out = jnp.einsum("bhqk,bkhd->bqhd", attn, v_cache).reshape(batch, 1, dim)
        return x + self.wo(out), k_cache, v_cache


class Wren(nnx.Module):
    """Byte-in, next-byte-logits-out decoder."""

    def __init__(self, cfg: Config, rngs: nnx.Rngs):
        self.cfg = cfg
        self.embed = nnx.Embed(cfg.vocab_size, cfg.dim, rngs=rngs)
        self.blocks = nnx.List(AttentionBlock(cfg, rngs) for _ in range(cfg.n_layers))
        self.norm = nnx.RMSNorm(cfg.dim, rngs=rngs)

    def __call__(self, tokens: jax.Array) -> jax.Array:
        """(batch, seq) int32 byte ids -> (batch, seq, vocab) logits."""
        x = self.embed(tokens)
        for block in self.blocks:
            x = block(x)
        # Tied output head: attend() reuses the embedding matrix transposed
        # as the unembedding -- standard for small models, saves vocab*dim.
        return self.embed.attend(self.norm(x))

    def n_params(self) -> int:
        return sum(p.size for p in jax.tree.leaves(nnx.state(self, nnx.Param)))

    def init_cache(self, batch: int) -> list[tuple[jax.Array, jax.Array]]:
        """Empty per-layer (k, v) caches for `decode`."""
        shape = (
            batch,
            self.cfg.context,
            self.cfg.n_heads,
            self.cfg.dim // self.cfg.n_heads,
        )
        return [(jnp.zeros(shape), jnp.zeros(shape)) for _ in self.blocks]

    def decode(
        self,
        tokens: jax.Array,
        caches: list[tuple[jax.Array, jax.Array]],
        pos: jax.Array,
    ) -> tuple[jax.Array, list[tuple[jax.Array, jax.Array]]]:
        """(batch, 1) tokens at position `pos` -> (batch, vocab) logits,
        with the caches advanced."""
        x = self.embed(tokens)
        updated = []
        for block, (k, v) in zip(self.blocks, caches):
            x, k, v = block.decode(x, k, v, pos)
            updated.append((k, v))
        return self.embed.attend(self.norm(x))[:, 0], updated

    def prefill(
        self, tokens: jax.Array, caches: list[tuple[jax.Array, jax.Array]]
    ) -> list[tuple[jax.Array, jax.Array]]:
        """Fill the caches for (batch, seq) prompt tokens in one batched
        pass instead of seq sequential `decode` steps."""
        x = self.embed(tokens)
        updated = []
        for block, (k, v) in zip(self.blocks, caches):
            x, k, v = block.prefill(x, k, v)
            updated.append((k, v))
        return updated


def generate(
    model: Wren,
    prompt: list[int],
    n_new: int,
    temperature: float = 1.0,
    seed: int = 0,
) -> list[int]:
    """Sample n_new bytes autoregressively after the prompt, through the
    same prefill + compiled-loop path as generate_batch. A prompt longer
    than context - n_new loses its head."""
    assert prompt, "empty prompt"
    outs, _ = generate_batch(model, [prompt], n_new, temperature=temperature, seed=seed)
    return list(prompt) + outs[0]


@nnx.jit
def _decode_step(
    model: Wren,
    tokens: jax.Array,
    caches: list[tuple[jax.Array, jax.Array]],
    pos: jax.Array,
) -> tuple[jax.Array, list[tuple[jax.Array, jax.Array]]]:
    return model.decode(tokens, caches, pos)


@nnx.jit
def _prefill(
    model: Wren, tokens: jax.Array, caches: list[tuple[jax.Array, jax.Array]]
) -> list[tuple[jax.Array, jax.Array]]:
    return model.prefill(tokens, caches)


@nnx.jit
def _decode_loop(
    model: Wren,
    buf: jax.Array,
    caches: list[tuple[jax.Array, jax.Array]],
    lens: jax.Array,
    ends: jax.Array,
    start: jax.Array,
    stop_arr: jax.Array,
    temperature: jax.Array,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """The whole generation loop as one compiled program: no per-byte
    dispatch, caches never leave the device. Temperature 0 decodes
    greedily; above 0 it samples. Returns the filled buffer, per-row end
    positions, summed log-probabilities and written byte counts."""
    batch, width = buf.shape
    s = stop_arr.shape[0]

    def cond(carry):
        t, _buf, _caches, done, ends, _logprob, _written, _ring = carry
        return (t < width - 1) & jnp.any(~done & (t + 1 < ends))

    def body(carry):
        t, buf, caches, done, ends, logprob, written, ring = carry
        tokens = jax.lax.dynamic_slice_in_dim(buf, t, 1, axis=1)
        logits, caches = model.decode(tokens, caches, t)
        sampled = jax.random.categorical(
            jax.random.fold_in(key, t), logits / jnp.maximum(temperature, 1e-6)
        )
        greedy = jnp.argmax(logits, axis=-1)
        nxt = jnp.where(temperature > 0, sampled, greedy).astype(jnp.int32)
        write = ~done & (t + 1 < ends) & (t + 1 >= lens)
        cur = jax.lax.dynamic_slice_in_dim(buf, t + 1, 1, axis=1)[:, 0]
        new = jnp.where(write, nxt, cur)
        buf = jax.lax.dynamic_update_slice_in_dim(buf, new[:, None], t + 1, axis=1)
        picked = jnp.take_along_axis(jax.nn.log_softmax(logits), nxt[:, None], 1)[:, 0]
        logprob += jnp.where(write, picked, 0.0)
        written += write
        if s:  # static: traced once per stop length
            ring = jnp.concatenate([ring[:, 1:], new[:, None]], axis=1)
            # Only a window of purely generated bytes may match the stop.
            matched = write & (written >= s) & jnp.all(ring == stop_arr, axis=1)
            done |= matched
            ends = jnp.where(matched, t + 2, ends)
        return t + 1, buf, caches, done, ends, logprob, written, ring

    carry = (
        start,
        buf,
        caches,
        jnp.zeros(batch, dtype=bool),
        ends,
        jnp.zeros(batch),
        jnp.zeros(batch, dtype=jnp.int32),
        jnp.zeros((batch, max(s, 1)), dtype=jnp.int32),
    )
    _, buf, _, _, ends, logprob, written, _ = jax.lax.while_loop(cond, body, carry)
    return buf, ends, logprob, written


def generate_batch(
    model: Wren,
    prompts: list[list[int]],
    n_new: int,
    stop: tuple[int, ...] = (),
    temperature: float = 0.0,
    seed: int = 0,
) -> tuple[list[list[int]], list[float]]:
    """Decode continuations for many prompts in one pass — greedily by
    default, sampled when temperature is above 0.

    The prompts are prefilled in one batched forward (garbage cache slots
    past a short row's prompt are rewritten by the loop before they are
    ever attended), then a single compiled loop decodes from the first
    position any row generates at. Prompts longer than context - n_new
    lose their head. Returns the generated tokens per prompt, cut after
    `stop` appears.

    Also returns each sample's per-byte confidence: the geometric mean of
    the probabilities of its generated bytes -- the raw material for
    calibration analysis.
    """
    cfg = model.cfg
    prompts = [p[-(cfg.context - n_new) :] for p in prompts]
    batch = len(prompts)
    lens = np.array([len(p) for p in prompts])
    # Rounding the width up bounds recompilation to a handful of shapes.
    width = min(-(-(int(lens.max()) + n_new) // 64) * 64, cfg.context)
    buf = np.zeros((batch, width), dtype=np.int32)
    for i, p in enumerate(prompts):
        buf[i, : len(p)] = p
    caches = model.init_cache(batch)
    if width > 2:  # one batched pass instead of width sequential steps
        caches = _prefill(model, jnp.asarray(buf[:, : width - 1]), caches)
    buf, ends, logprob, written = _decode_loop(
        model,
        jnp.asarray(buf),
        caches,
        jnp.asarray(lens, dtype=jnp.int32),
        jnp.asarray(lens + n_new, dtype=jnp.int32),
        jnp.int32(max(int(lens.min()) - 1, 0)),
        jnp.array(stop, dtype=jnp.int32),
        jnp.float32(temperature),
        jax.random.PRNGKey(seed),
    )
    buf, ends = np.asarray(buf), np.asarray(ends)
    outs = [buf[i, lens[i] : ends[i]].tolist() for i in range(batch)]
    conf = np.exp(np.asarray(logprob) / np.maximum(np.asarray(written), 1))
    return outs, [float(c) for c in conf]


WEIGHTS = pathlib.Path(__file__).parent / "weights.npz"


def save(model: Wren, path: pathlib.Path = WEIGHTS) -> None:
    """Write parameters plus the config needed to rebuild the model."""
    leaves, _ = jax.tree_util.tree_flatten_with_path(nnx.state(model, nnx.Param))
    arrays = {jax.tree_util.keystr(k): np.asarray(v) for k, v in leaves}
    arrays |= {f"cfg:{k}": v for k, v in asdict(model.cfg).items()}
    np.savez(path, **arrays)


def load(path: pathlib.Path = WEIGHTS) -> Wren:
    """Rebuild the model saved at `path`."""
    npz = np.load(path)
    cfg = Config(**{k.removeprefix("cfg:"): int(npz[k]) for k in npz if "cfg:" in k})
    model = Wren(cfg, rngs=nnx.Rngs(0))
    leaves, treedef = jax.tree_util.tree_flatten_with_path(nnx.state(model, nnx.Param))
    restored = [jnp.asarray(npz[jax.tree_util.keystr(k)]) for k, _ in leaves]
    nnx.update(model, jax.tree_util.tree_unflatten(treedef, restored))
    return model
