"""The packed runtime's two parity pins: the numpy reference (the format
spec) matches the JAX model with ternarized kernels, and the C binary
matches the reference byte for byte."""

import pathlib
import shutil
import subprocess
import sys

import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from wren.model import Config, Wren, save

RUNTIME = pathlib.Path(__file__).parent.parent / "runtime"
sys.path.insert(0, str(RUNTIME))
import export


@pytest.fixture(scope="module")
def packed(tmp_path_factory):
    """A tiny random model, saved, exported, and loaded back."""
    tmp = tmp_path_factory.mktemp("runtime")
    cfg = Config(dim=64, n_layers=2, n_heads=2, context=96)
    model = Wren(cfg, rngs=nnx.Rngs(0))
    save(model, tmp / "weights.npz")
    export.export(str(tmp / "weights.npz"), str(tmp / "model.bin"))
    return model, export.load_pack(str(tmp / "model.bin")), tmp / "model.bin"


def test_reference_matches_jax_ternary_forward(packed):
    model, pack, _ = packed
    # Ternarize the JAX model's kernels in place, exactly as export does.
    for block in model.blocks:
        for lin in (block.wqkv, block.wo):
            q, scale = export.ternary(np.asarray(lin.kernel.value))
            lin.kernel.value = jnp.asarray(q.astype(np.float32) * scale)
    prompt = list(b"turn on the lights\n")
    logits = np.asarray(model(jnp.array([prompt]))[0])

    caches = (
        np.zeros((2, pack["cfg"]["context"], 2, 32)),
        np.zeros((2, pack["cfg"]["context"], 2, 32)),
    )
    ref = np.stack(
        [export._step(pack, caches, tok, pos) for pos, tok in enumerate(prompt)]
    )
    assert np.allclose(logits, ref, atol=1e-3), np.abs(logits - ref).max()


@pytest.mark.skipif(shutil.which("cc") is None, reason="no C compiler")
def test_c_binary_matches_reference(packed, tmp_path):
    _, pack, bin_path = packed
    exe = tmp_path / "wren-rt"
    subprocess.run(
        ["cc", "-O3", "-o", str(exe), str(RUNTIME / "wren.c"), "-lm"], check=True
    )
    for prompt in ("turn on the lights in the kitchen", "what is the temperature"):
        want = export.reference_generate(pack, list(f"{prompt}\n".encode()), 48)
        got = subprocess.run(
            [str(exe), str(bin_path), "-n", "48", prompt],
            capture_output=True,
            check=True,
        ).stdout
        assert list(got) == want, (bytes(want), got)
