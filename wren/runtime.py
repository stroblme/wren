"""NumPy inference for packed ternary weights (`wren.bin`).

The format has a header, fp32 embedding and final norm, then each layer's
fp32 norm scales and attention kernels. Kernels store {-1, 0, +1} values
at four weights per byte with a scale per tensor, in (out, in) row order.
An optional trailer holds metadata as a little-endian u32 byte count and
that much UTF-8 JSON, such as the `targets` a model was trained on (each a
`Target` without its value) and the recommended `threshold`.
`wren/export.py` writes this format; `runtime/wren.c` also reads it and
ignores the trailer.
"""

import json
import pathlib
import struct

import numpy as np

MAGIC, VERSION = b"WREN", 1
EPS = 1e-6  # flax RMSNorm epsilon
NEWLINE = 10
MODEL = pathlib.Path(__file__).parent / "wren.bin"  # the packaged model
DEMO = pathlib.Path(__file__).parent / "home.json"  # the installation it learnt


def _unpack2(raw: bytes, n: int) -> np.ndarray:
    """Unpack four 2-bit (value+1) weights per byte into {-1, 0, 1} int8 values."""
    b = np.frombuffer(raw, dtype=np.uint8)
    q = np.stack([b & 3, (b >> 2) & 3, (b >> 4) & 3, (b >> 6) & 3], 1).reshape(-1)
    return q[:n].astype(np.int8) - 1


def load_pack(path: str | pathlib.Path) -> dict:
    """Load float32 arrays, scaling attention kernels in (out, in) order.

    `pack["meta"]` holds the trailer's metadata, `{}` if there is none.
    """
    with open(path, "rb") as f:
        if f.read(4) != MAGIC:
            raise ValueError(f"{path}: not a wren pack")
        version, vocab, dim, n_layers, n_heads = struct.unpack("<5i", f.read(20))
        (context,) = struct.unpack("<i", f.read(4))
        if version != VERSION:
            raise ValueError(f"{path}: unsupported version {version}")
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
        pack["meta"] = {}
        if size := f.read(4):
            pack["meta"] = json.loads(f.read(*struct.unpack("<I", size)))
        return pack


def _rms(x: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return x / np.sqrt(np.mean(x * x, -1, keepdims=True) + EPS) * scale


def _rope(x: np.ndarray, pos: int) -> np.ndarray:
    """Apply rotary positions to (seq, heads, head_dim), starting at `pos`."""
    seq, _, head_dim = x.shape
    half = head_dim // 2
    angles = np.arange(pos, pos + seq)[:, None, None] / 10000.0 ** (
        np.arange(half) / half
    )
    c, s = np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([x1 * c - x2 * s, x1 * s + x2 * c], -1)


def _caches(pack: dict) -> tuple[np.ndarray, np.ndarray]:
    """Create key/value caches with shape (layers, context, heads, head_dim)."""
    cfg = pack["cfg"]
    heads = cfg["n_heads"]
    shape = (cfg["n_layers"], cfg["context"], heads, cfg["dim"] // heads)
    return np.zeros(shape, np.float32), np.zeros(shape, np.float32)


def _forward(pack: dict, caches: tuple, tokens: list[int], pos: int) -> np.ndarray:
    """Compute logits for tokens starting at `pos` and update their cache slots.

    Each token attends to all preceding slots and its own slot. Use a whole
    prompt for prefill or one token for decoding.
    """
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
    """Generate bytes greedily from `prompt`, stopping at a blank line.

    Keep only the last context - n_new prompt tokens. Return generated bytes
    and their probabilities.
    """
    context = pack["cfg"]["context"]
    if not 0 < n_new < context:
        raise ValueError(f"n_new must be in 1..{context - 1}, got {n_new}")
    prompt = prompt[-(context - n_new) :]
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
