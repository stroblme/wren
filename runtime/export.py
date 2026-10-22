"""Export weights to the ternary format read by Wren's NumPy and C runtimes.

Attention kernels use {-1, 0, +1} times a scale per tensor, packed four
weights per byte in (out, in) row order. Embeddings and norms remain fp32.

    uv run python runtime/export.py runtime/weights-qat.npz wren/wren.bin
"""

import os
import struct
import sys

import numpy as np

from wren.runtime import MAGIC, VERSION


def ternary(w: np.ndarray) -> tuple[np.ndarray, float]:
    """Return int8 ternary values and their mean-absolute-weight scale."""
    scale = float(np.mean(np.abs(w))) + 1e-12
    return np.clip(np.round(w / scale), -1, 1).astype(np.int8), scale


def _pack2(q: np.ndarray) -> bytes:
    """Pack {-1, 0, 1} as 2-bit (value+1) codes, four weights per byte."""
    flat = (q.reshape(-1) + 1).astype(np.uint8)
    flat = np.pad(flat, (0, -len(flat) % 4))
    return (
        flat[0::4] | (flat[1::4] << 2) | (flat[2::4] << 4) | (flat[3::4] << 6)
    ).tobytes()


def export(npz_path: str, out_path: str) -> None:
    npz = np.load(npz_path)
    cfg = {k.removeprefix("cfg:"): int(npz[k]) for k in npz.files if "cfg:" in k}
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
        for li in range(cfg["n_layers"]):
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
    print(f"{sys.argv[2]}: {os.path.getsize(sys.argv[2]) / 1e6:.2f} MB")
