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
    context: int = 512  # maximum sequence length; holds a state block + calls


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


@nnx.jit
def _forward(model: Wren, tokens: jax.Array) -> jax.Array:
    return model(tokens)


def generate(
    model: Wren,
    prompt: list[int],
    n_new: int,
    temperature: float = 1.0,
    seed: int = 0,
) -> list[int]:
    """Sample n_new bytes autoregressively after the prompt.

    The token buffer has fixed shape (1, context) so jax compiles the
    forward pass exactly once; the causal mask makes the zero-padding after
    position `pos` invisible. When the window fills we slide it left --
    RoPE only encodes relative distance, so sliding is safe.

    ponytail: recomputes the full window every step; fine for the CLI's
    single prompt. Bulk evaluation uses generate_batch, which decodes
    through the KV cache instead.
    """
    cfg = model.cfg
    assert 0 < len(prompt) <= cfg.context
    key = jax.random.PRNGKey(seed)
    buf = jnp.zeros((1, cfg.context), dtype=jnp.int32)
    buf = buf.at[0, : len(prompt)].set(jnp.array(prompt, dtype=jnp.int32))
    pos = len(prompt)
    out = list(prompt)
    for _ in range(n_new):
        logits = _forward(model, buf)[0, pos - 1] / max(temperature, 1e-6)
        key, sub = jax.random.split(key)
        nxt = int(jax.random.categorical(sub, logits))
        out.append(nxt)
        if pos < cfg.context:
            buf = buf.at[0, pos].set(nxt)
            pos += 1
        else:
            buf = jnp.concatenate(
                [buf[:, 1:], jnp.array([[nxt]], dtype=jnp.int32)], axis=1
            )
    return out


@nnx.jit
def _decode_step(
    model: Wren,
    tokens: jax.Array,
    caches: list[tuple[jax.Array, jax.Array]],
    pos: jax.Array,
) -> tuple[jax.Array, list[tuple[jax.Array, jax.Array]]]:
    return model.decode(tokens, caches, pos)


def generate_batch(
    model: Wren,
    prompts: list[list[int]],
    n_new: int,
    stop: tuple[int, ...] = (),
) -> tuple[list[list[int]], list[float]]:
    """Greedy-decode continuations for many prompts in one pass.

    One KV-cached decode loop serves the whole batch: prompt tokens are fed
    in teacher-forced (their logits discarded), then each sample continues
    with its own argmax tokens, so ragged prompt lengths need no separate
    prefill pass. Prompts longer than context - n_new lose their head.
    Returns the generated tokens per prompt, cut after `stop` appears.

    Also returns each sample's per-byte confidence: the geometric mean of
    the probabilities of its generated bytes -- the raw material for
    calibration analysis.
    """
    cfg = model.cfg
    prompts = [p[-(cfg.context - n_new) :] for p in prompts]
    batch, stop_arr = len(prompts), np.array(stop, dtype=np.int32)
    lens = np.array([len(p) for p in prompts])
    width = lens.max() + n_new
    buf = np.zeros((batch, width), dtype=np.int32)
    for i, p in enumerate(prompts):
        buf[i, : len(p)] = p
    caches = model.init_cache(batch)
    ends = lens + n_new
    done = np.zeros(batch, dtype=bool)
    logprob = np.zeros(batch)
    written = np.zeros(batch, dtype=np.int64)
    for t in range(width - 1):
        active = ~done & (t + 1 < ends)
        if not active.any():
            break
        logits, caches = _decode_step(
            model, jnp.asarray(buf[:, t : t + 1]), caches, jnp.int32(t)
        )
        nxt = np.asarray(jnp.argmax(logits, axis=-1), dtype=np.int32)
        write = active & (t + 1 >= lens)
        buf[write, t + 1] = nxt[write]
        if write.any():
            probs = np.asarray(jax.nn.log_softmax(logits, axis=-1))
            logprob[write] += probs[write, nxt[write]]
            written[write] += 1
        for i in np.nonzero(write)[0] if stop else ():
            if (
                t + 2 - lens[i] >= len(stop_arr)
                and (buf[i, t + 2 - len(stop_arr) : t + 2] == stop_arr).all()
            ):
                done[i], ends[i] = True, t + 2
    outs = [buf[i, lens[i] : ends[i]].tolist() for i in range(batch)]
    return outs, list(np.exp(logprob / np.maximum(written, 1)))


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
