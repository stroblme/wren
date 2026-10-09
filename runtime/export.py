"""Export Wren weights to the packed ternary format `wren.c` loads.

The attention kernels are ternarized exactly as the s8 QAT arms train
them (BitNet b1.58 absmean, arXiv:2402.17764): {-1, 0, +1} times one
per-tensor scale, packed 4 weights per byte on disk. Embedding and norm
scales stay fp32 (~3% of parameters). Kernels are stored transposed
(out, in) row-major so the C dot products walk memory linearly.

    uv run python runtime/export.py runtime/weights-qat.npz wren/wren.bin

The reader lives in the package: wren/runtime.py is the format's
specification and the numpy runtime the library ships.
"""

import struct
import sys

import numpy as np

from wren.runtime import MAGIC, VERSION


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


if __name__ == "__main__":
    export(sys.argv[1], sys.argv[2])
    import os

    print(f"{sys.argv[2]}: {os.path.getsize(sys.argv[2]) / 1e6:.2f} MB")
