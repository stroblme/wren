"""Export Wren weights to the packed ternary format `wren.c` loads.

The attention kernels are ternarized exactly as the s8 QAT arms train
them (BitNet b1.58 absmean, arXiv:2402.17764): {-1, 0, +1} times one
per-tensor scale, packed 4 weights per byte on disk. Embedding and norm
scales stay fp32 (~3% of parameters). Kernels are stored transposed
(out, in) row-major so the C dot products walk memory linearly.

    uv run python runtime/export.py wren/weights.npz wren.bin

`reference_generate` is the format's specification: a numpy
implementation of the packed model that the C binary is pinned against
(tests/test_runtime.py).
"""

import struct
import sys

import numpy as np

MAGIC, VERSION = b"WREN", 1
EPS = 1e-6  # flax RMSNorm epsilon


def ternary(w: np.ndarray) -> tuple[np.ndarray, float]:
    """(int8 values in {-1,0,1}, scale) — the b1.58 absmean quantizer."""
    scale = float(np.mean(np.abs(w))) + 1e-12
    return np.clip(np.round(w / scale), -1, 1).astype(np.int8), scale


def _pack2(q: np.ndarray) -> bytes:
    """{-1,0,1} int8 -> 2 bits each (value+1), 4 per byte."""
    flat = (q.reshape(-1) + 1).astype(np.uint8)
    flat = np.pad(flat, (0, -len(flat) % 4))
    return (
        flat[0::4] | (flat[1::4] << 2) | (flat[2::4] << 4) | (flat[3::4] << 6)
    ).tobytes()


def _unpack2(raw: bytes, n: int) -> np.ndarray:
    b = np.frombuffer(raw, dtype=np.uint8)
    q = np.stack([b & 3, (b >> 2) & 3, (b >> 4) & 3, (b >> 6) & 3], 1).reshape(-1)
    return q[:n].astype(np.int8) - 1


def export(npz_path: str, out_path: str) -> None:
    npz = np.load(npz_path)
    cfg = {k.removeprefix("cfg:"): int(npz[k]) for k in npz.files if "cfg:" in k}
    n_layers = cfg["n_layers"]
    with open(out_path, "wb") as f:
        f.write(MAGIC)
        f.write(
            struct.pack(
                "<5i",
                VERSION,
                cfg["vocab_size"],
                cfg["dim"],
                cfg["n_layers"],
                cfg["n_heads"],
            )
        )
        f.write(struct.pack("<i", cfg["context"]))
        f.write(npz["['embed']['embedding'].value"].astype("<f4").tobytes())
        f.write(npz["['norm']['scale'].value"].astype("<f4").tobytes())
        for li in range(n_layers):
            b = f"['blocks'][{li}]"
            f.write(npz[f"{b}['norm']['scale'].value"].astype("<f4").tobytes())
            f.write(npz[f"{b}['q_norm']['scale'].value"].astype("<f4").tobytes())
            f.write(npz[f"{b}['k_norm']['scale'].value"].astype("<f4").tobytes())
            for name in ("wqkv", "wo"):
                q, scale = ternary(npz[f"{b}['{name}']['kernel'].value"])
                f.write(struct.pack("<f", scale))
                f.write(_pack2(q.T.copy()))  # (out, in) row-major


# ------------------------------------------------- reference implementation


def load_pack(path: str) -> dict:
    with open(path, "rb") as f:
        assert f.read(4) == MAGIC
        version, vocab, dim, n_layers, n_heads = struct.unpack("<5i", f.read(20))
        (context,) = struct.unpack("<i", f.read(4))
        assert version == VERSION
        head_dim = dim // n_heads

        def f32(n):
            return np.frombuffer(f.read(4 * n), dtype="<f4").astype(np.float64)

        def kernel(rows, cols):
            (scale,) = struct.unpack("<f", f.read(4))
            raw = f.read((rows * cols + 3) // 4)
            return _unpack2(raw, rows * cols).reshape(rows, cols), scale

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
            layer = {
                "norm": f32(dim),
                "q_norm": f32(head_dim),
                "k_norm": f32(head_dim),
            }
            layer["wqkv"], layer["wqkv_scale"] = kernel(3 * dim, dim)
            layer["wo"], layer["wo_scale"] = kernel(dim, dim)
            pack["layers"].append(layer)
        return pack


def _rms(x: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return x / np.sqrt(np.mean(x * x, -1, keepdims=True) + EPS) * scale


def _rope(x: np.ndarray, pos: int) -> np.ndarray:
    half = x.shape[-1] // 2
    angles = pos / 10000.0 ** (np.arange(half) / half)
    c, s = np.cos(angles), np.sin(angles)
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([x1 * c - x2 * s, x1 * s + x2 * c], -1)


def _step(pack: dict, caches: tuple, token: int, pos: int) -> np.ndarray:
    """One decode step: logits for position pos, caches updated in place."""
    heads = pack["cfg"]["n_heads"]
    head_dim = pack["cfg"]["dim"] // heads
    ks, vs = caches
    x = pack["embed"][token].copy()
    for li, layer in enumerate(pack["layers"]):
        h = _rms(x, layer["norm"])
        qkv = layer["wqkv"] @ h * layer["wqkv_scale"]
        q, k, v = qkv.reshape(3, heads, head_dim)
        q = _rope(_rms(q, layer["q_norm"]), pos)
        ks[li, pos] = _rope(_rms(k, layer["k_norm"]), pos)
        vs[li, pos] = v
        o = np.zeros((heads, head_dim))
        for hd in range(heads):
            scores = ks[li, : pos + 1, hd] @ q[hd] / np.sqrt(head_dim)
            attn = np.exp(scores - scores.max())
            o[hd] = attn @ vs[li, : pos + 1, hd] / attn.sum()
        x = x + layer["wo"] @ o.reshape(-1) * layer["wo_scale"]
    return pack["embed"] @ _rms(x, pack["final_norm"])


def reference_generate(pack: dict, prompt: list[int], n_new: int) -> list[int]:
    """Greedy continuation, stopping at a blank line — the spec for wren.c."""
    cfg = pack["cfg"]
    heads, head_dim = cfg["n_heads"], cfg["dim"] // cfg["n_heads"]
    caches = (
        np.zeros((cfg["n_layers"], cfg["context"], heads, head_dim)),
        np.zeros((cfg["n_layers"], cfg["context"], heads, head_dim)),
    )
    tokens = list(prompt)
    limit = min(len(prompt) + n_new, cfg["context"])
    pos = 0
    while pos < len(tokens):
        logits = _step(pack, caches, tokens[pos], pos)
        if pos == len(tokens) - 1 and len(tokens) < limit:
            tokens.append(int(np.argmax(logits)))
            if tokens[-2:] == [10, 10]:
                break
        pos += 1
    return tokens[len(prompt) :]


if __name__ == "__main__":
    export(sys.argv[1], sys.argv[2])
    import os

    print(f"{sys.argv[2]}: {os.path.getsize(sys.argv[2]) / 1e6:.2f} MB")
