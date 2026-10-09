"""The packed runtime's parity pins: the numpy runtime (the format spec,
wren/runtime.py) matches the JAX model with ternarized kernels, the C
binary matches the numpy runtime byte for byte, and the shipped model
still answers what the original per-byte reference answered."""

import pathlib
import shutil
import subprocess
import sys

import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from wren import runtime
from wren.model import Config, Wren, save

RUNTIME = pathlib.Path(__file__).parent.parent / "runtime"
sys.path.insert(0, str(RUNTIME))
import export  # ty: ignore[unresolved-import]


@pytest.fixture(scope="module")
def packed(tmp_path_factory):
    """A tiny random model, saved, exported, and loaded back."""
    tmp = tmp_path_factory.mktemp("runtime")
    cfg = Config(dim=64, n_layers=2, n_heads=2, context=96)
    model = Wren(cfg, rngs=nnx.Rngs(0))
    save(model, tmp / "weights.npz")
    export.export(str(tmp / "weights.npz"), str(tmp / "model.bin"))
    return model, runtime.load_pack(tmp / "model.bin"), tmp / "model.bin"


def test_numpy_matches_jax_ternary_forward(packed):
    model, pack, _ = packed
    # Ternarize the JAX model's kernels in place, exactly as export does.
    for block in model.blocks:
        for lin in (block.wqkv, block.wo):
            q, scale = export.ternary(np.asarray(lin.kernel.value))
            lin.kernel.value = jnp.asarray(q.astype(np.float32) * scale)
    prompt = list(b"turn on the lights\n")
    logits = np.asarray(model(jnp.array([prompt]))[0])

    # The whole prompt in one pass (prefill) and one byte at a time (decode).
    prefill = runtime._forward(pack, runtime._caches(pack), prompt, 0)
    caches = runtime._caches(pack)
    decode = np.concatenate(
        [runtime._forward(pack, caches, [tok], pos) for pos, tok in enumerate(prompt)]
    )
    assert np.allclose(logits, prefill, atol=1e-3), np.abs(logits - prefill).max()
    assert np.allclose(logits, decode, atol=1e-3), np.abs(logits - decode).max()


@pytest.mark.skipif(shutil.which("cc") is None, reason="no C compiler")
def test_c_binary_matches_numpy(packed, tmp_path):
    _, pack, bin_path = packed
    exe = tmp_path / "wren-rt"
    subprocess.run(
        ["cc", "-O3", "-o", str(exe), str(RUNTIME / "wren.c"), "-lm"], check=True
    )
    for prompt in ("turn on the lights in the kitchen", "what is the temperature"):
        want, _ = runtime.generate(pack, list(f"{prompt}\n".encode()), 48)
        got = subprocess.run(
            [str(exe), str(bin_path), "-n", "48", prompt],
            capture_output=True,
            check=True,
        ).stdout
        assert list(got) == want, (bytes(want), got)


# Held-out prompts (dataset.home() eval split) and what the original
# per-byte float64 reference (runtime/export.py before the move) generated.
REFERENCE = [
    (
        (
            "11:35 out 27C bright\nuser living room\nbathroom: light lzi6n off\n"
            "bedroom: light lxhnv off, shutter sv8xr down\n"
            "it's too bright in the bathroom\n"
        ),
        ">> lights.off(id='lzi6n')\n\n",
    ),
    (
        (
            "23:17 out 8C dark\nuser away\n"
            "living room: light ldf4m off, media mc5nc on, heat h3j2h 20C\n"
            "kitchen: light leaem off, shutter skscq up\n"
            "bathroom: light ln59c on, media mb6ca off, heat h4thd 20C, "
            "shutter sd3hn down\ncoming home in 20m\n"
        ),
        ">> heating.boost(id='h3j2h')\n>> lights.on(delay='20m', id='ldf4m')\n\n",
    ),
    (
        (
            "21:22 out 18C dark\nuser kitchen\n"
            "living room: light l789e off, media m9q7d off, heat h6ww5 23C\n"
            "office: light lngrm on, heat hhvji 18C, shutter swbuj up\n"
            "turn that off in the office\n"
        ),
        "\n",
    ),
]


@pytest.mark.parametrize(("prompt", "want"), REFERENCE)
def test_packaged_model_reproduces_reference(prompt, want):
    pack = runtime.load_pack(runtime.MODEL)
    out, probs = runtime.generate(pack, list(prompt.encode()), 96)
    assert bytes(out).decode() == want
    assert len(probs) == len(out) and all(0 < p <= 1 for p in probs)
