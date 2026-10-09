"""Numpy inference for the packed ternary format (`wren.bin`).

The format, as runtime/export.py writes it: a header, the fp32 embedding
and final norm, then per layer the fp32 norm scales and the two attention
kernels as {-1, 0, +1} times one per-tensor scale, packed 4 weights per
byte and stored (out, in) row-major.

This module is the format's specification and the runtime the library
ships — numpy only, no JAX. It holds what runtime/wren.c holds (float32,
kernels unpacked with their scale folded in) and is pinned against the
JAX ternary forward, and wren.c against it byte for byte
(tests/test_runtime.py).
"""

import pathlib
import struct

import numpy as np

MAGIC, VERSION = b"WREN", 1
EPS = 1e-6  # flax RMSNorm epsilon
NEWLINE = 10
MODEL = pathlib.Path(__file__).parent / "wren.bin"  # the packaged model


def _unpack2(raw: bytes, n: int) -> np.ndarray:
    """2 bits each (value+1), 4 per byte -> {-1,0,1} int8."""
    b = np.frombuffer(raw, dtype=np.uint8)
    q = np.stack([b & 3, (b >> 2) & 3, (b >> 4) & 3, (b >> 6) & 3], 1).reshape(-1)
    return q[:n].astype(np.int8) - 1


def load_pack(path: str | pathlib.Path) -> dict:
    """Read a pack into float32 arrays, kernels (out, in) with their scale
    folded in."""
    with open(path, "rb") as f:
        assert f.read(4) == MAGIC, f"{path}: not a wren pack"
        version, vocab, dim, n_layers, n_heads = struct.unpack("<5i", f.read(20))
        (context,) = struct.unpack("<i", f.read(4))
        assert version == VERSION
        head_dim = dim // n_heads

        def f32(n):
            return np.frombuffer(f.read(4 * n), dtype="<f4").astype(np.float32)

        def kernel(rows, cols):
            (scale,) = struct.unpack("<f", f.read(4))
            raw = f.read((rows * cols + 3) // 4)
            return _unpack2(raw, rows * cols).reshape(rows, cols) * np.float32(scale)

        pack = {
            "cfg": {
                "vocab": vocab,
                "dim": dim,
                "n_layers": n_layers,
                "n_heads": n_heads,
                "context": context,
            },
            "embed": f32(vocab * dim).reshape(vocab, dim),
            "final_norm": f32(dim),
            "layers": [],
        }
        for _ in range(n_layers):
            layer = {"norm": f32(dim), "q_norm": f32(head_dim), "k_norm": f32(head_dim)}
            layer["wqkv"] = kernel(3 * dim, dim)
            layer["wo"] = kernel(dim, dim)
            pack["layers"].append(layer)
        return pack


def _rms(x: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return x / np.sqrt(np.mean(x * x, -1, keepdims=True) + EPS) * scale


def _rope(x: np.ndarray, pos: int) -> np.ndarray:
    """Rotary embedding for (seq, heads, head_dim) at positions pos.."""
    seq, _, head_dim = x.shape
    half = head_dim // 2
    angles = np.arange(pos, pos + seq)[:, None, None] / 10000.0 ** (
        np.arange(half) / half
    )
    c, s = np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([x1 * c - x2 * s, x1 * s + x2 * c], -1)


def _caches(pack: dict) -> tuple[np.ndarray, np.ndarray]:
    """Empty per-layer key/value caches, (layers, context, heads, head_dim)."""
    cfg = pack["cfg"]
    heads = cfg["n_heads"]
    shape = (cfg["n_layers"], cfg["context"], heads, cfg["dim"] // heads)
    return np.zeros(shape, np.float32), np.zeros(shape, np.float32)


def _forward(pack: dict, caches: tuple, tokens: list[int], pos: int) -> np.ndarray:
    """Logits for `tokens` at positions pos, pos+1, ... — a whole prompt in
    one pass (prefill) or one byte per call (decode). Their keys and values
    are written into the caches, and each attends to every slot up to its
    own position."""
    heads = pack["cfg"]["n_heads"]
    head_dim = pack["cfg"]["dim"] // heads
    n, end = len(tokens), pos + len(tokens)
    ks, vs = caches
    causal = np.arange(end)[None, :] <= np.arange(pos, end)[:, None]  # (n, end)
    x = pack["embed"][tokens]
    for li, layer in enumerate(pack["layers"]):
        qkv = _rms(x, layer["norm"]) @ layer["wqkv"].T
        q, k, v = qkv.reshape(n, 3, heads, head_dim).transpose(1, 0, 2, 3)
        q = _rope(_rms(q, layer["q_norm"]), pos)
        ks[li, pos:end] = _rope(_rms(k, layer["k_norm"]), pos)
        vs[li, pos:end] = v
        # Per head: (n, head_dim) queries against (end, head_dim) keys.
        scores = q.swapaxes(0, 1) @ ks[li, :end].transpose(1, 2, 0) / head_dim**0.5
        scores = np.where(causal, scores, -np.inf)
        attn = np.exp(scores - scores.max(-1, keepdims=True))
        attn /= attn.sum(-1, keepdims=True)
        o = (attn @ vs[li, :end].swapaxes(0, 1)).swapaxes(0, 1).reshape(n, -1)
        x = x + o @ layer["wo"].T
    return _rms(x, pack["final_norm"]) @ pack["embed"].T


def generate(
    pack: dict, prompt: list[int], n_new: int
) -> tuple[list[int], list[float]]:
    """Greedy continuation of `prompt`, stopping at a blank line — the spec
    wren.c is pinned to. Only the tail of a prompt longer than
    context - n_new is kept, as in wren.c.

    Returns the generated bytes and the probability the model gave each.
    """
    prompt = prompt[-(pack["cfg"]["context"] - n_new) :]
    caches = _caches(pack)
    logits = _forward(pack, caches, prompt, 0)[-1]
    out, probs = [], []
    while True:
        nxt = int(np.argmax(logits))
        out.append(nxt)
        # softmax at the argmax, whose shifted logit is exp(0) = 1
        probs.append(float(1 / np.exp(logits - logits[nxt]).sum()))
        if (prompt[-1:] + out)[-2:] == [NEWLINE, NEWLINE] or len(out) == n_new:
            return out, probs
        logits = _forward(pack, caches, [nxt], len(prompt) + len(out) - 1)[-1]
