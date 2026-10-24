"""Check NumPy/JAX parity, C output, pack metadata and shipped-model responses."""

import json
import pathlib
import shutil
import struct
import subprocess
from dataclasses import asdict

import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from wren import export, runtime
from wren.model import Config, Wren, save
from wren.targets import Target

RUNTIME = pathlib.Path(__file__).parent.parent / "runtime"
# What a pack trained on two targets carries: the targets without their
# current values, and the recommended threshold.
TRAINED_ON = [
    Target("kitchen_light", "bool", False, description="ceiling light"),
    Target("ph", "float", 7.0, range=[0, 14], unit="pH", readonly=True),
]
META = {
    "targets": [
        {k: v for k, v in asdict(t).items() if k != "value"} for t in TRAINED_ON
    ],
    "threshold": 0.8,
}


@pytest.fixture(scope="module")
def packed(tmp_path_factory):
    """Create a small random model, export it, and load its packed weights."""
    tmp = tmp_path_factory.mktemp("runtime")
    cfg = Config(dim=64, n_layers=2, n_heads=2, context=96)
    model = Wren(cfg, rngs=nnx.Rngs(0))
    save(model, tmp / "weights.npz")
    export.export(str(tmp / "weights.npz"), str(tmp / "model.bin"))
    return model, runtime.load_pack(tmp / "model.bin"), tmp / "model.bin"


@pytest.fixture(scope="module")
def with_meta(packed):
    """Export the same weights with a metadata trailer."""
    path = packed[2].with_name("meta.bin")
    export.export(str(packed[2].with_name("weights.npz")), str(path), META)
    return path


def test_export_appends_meta_trailer(packed, with_meta):
    plain, data = packed[2].read_bytes(), with_meta.read_bytes()
    assert data.startswith(plain)
    (n,) = struct.unpack("<I", data[len(plain) : len(plain) + 4])
    assert len(data) == len(plain) + 4 + n


def test_load_pack_reads_meta(with_meta):
    assert runtime.load_pack(with_meta)["meta"] == META


def test_load_pack_without_trailer_has_empty_meta(packed):
    assert packed[1]["meta"] == {}


def test_packaged_model_carries_its_installation():
    meta = runtime.load_pack(runtime.MODEL)["meta"]
    demo = json.loads(runtime.DEMO.read_text())["targets"]
    assert [t["name"] for t in meta["targets"]] == [t["name"] for t in demo]
    assert 0 < meta["threshold"] < 1
    assert 0 < meta["test_exact_match"] <= 1


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


@pytest.fixture(scope="module")
def exe(tmp_path_factory):
    """Compile runtime/wren.c once per module."""
    if shutil.which("cc") is None:
        pytest.skip("no C compiler")
    path = tmp_path_factory.mktemp("c") / "wren-rt"
    subprocess.run(
        ["cc", "-O3", "-o", str(path), str(RUNTIME / "wren.c"), "-lm"], check=True
    )
    return path


def test_c_binary_matches_numpy(packed, exe):
    _, pack, bin_path = packed
    for prompt in ("turn on the lights in the kitchen", "what is the temperature"):
        want, _ = runtime.generate(pack, list(f"{prompt}\n".encode()), 48)
        got = subprocess.run(
            [str(exe), str(bin_path), "-n", "48", prompt],
            capture_output=True,
            check=True,
        ).stdout
        assert list(got) == want, (bytes(want), got)


def test_c_binary_ignores_meta_trailer(packed, exe, with_meta):
    _, pack, _ = packed
    want, _ = runtime.generate(pack, list(b"turn on the lights in the kitchen\n"), 48)
    got = subprocess.run(
        [str(exe), str(with_meta), "-n", "48", "turn on the lights in the kitchen"],
        capture_output=True,
        check=True,
    ).stdout
    assert list(got) == want, (bytes(want), got)


def test_c_binary_keeps_tail_of_long_prompt(packed, exe):
    _, pack, bin_path = packed
    prompt = b"turn on the lights in the kitchen\n" * 150  # ~5000 bytes
    want, _ = runtime.generate(pack, list(prompt), 48)
    got = subprocess.run(
        [str(exe), str(bin_path), "-n", "48"],
        input=prompt,
        capture_output=True,
        check=True,
    ).stdout
    assert list(got) == want, (bytes(want), got)


@pytest.mark.parametrize("n_new", [-200, 0, 96])
def test_c_binary_rejects_bad_n_new(packed, exe, n_new):
    run = subprocess.run(
        [str(exe), str(packed[2]), "-n", str(n_new), "hi"],
        capture_output=True,
        check=False,
    )
    assert run.returncode == 2 and b"-n must be" in run.stderr


# (vocab, dim, n_layers, n_heads, context)
BAD_HEADERS = [(256, 64, 1, 0, 96), (256, 64, 1, 3, 96), (256, 1024, 1, 2, 96)]
BAD_HEADERS += [(256, 64, 1, 2, 9000), (128, 64, 1, 2, 96)]


@pytest.mark.parametrize("hdr", BAD_HEADERS)
def test_c_binary_rejects_bad_header(exe, tmp_path, hdr):
    path = tmp_path / "bad.bin"
    path.write_bytes(runtime.MAGIC + struct.pack("<6i", runtime.VERSION, *hdr))
    run = subprocess.run([str(exe), str(path), "hi"], capture_output=True, check=False)
    assert run.returncode != 0 and b"bad header" in run.stderr


def test_c_binary_reports_truncated_file(packed, exe, tmp_path):
    path = tmp_path / "short.bin"
    path.write_bytes(packed[2].read_bytes()[:-100])
    run = subprocess.run([str(exe), str(path), "hi"], capture_output=True, check=False)
    assert run.returncode != 0 and b"truncated file" in run.stderr


@pytest.mark.parametrize("n_new", [0, 96, 200])
def test_generate_rejects_bad_n_new(packed, n_new):
    with pytest.raises(ValueError, match="n_new"):
        runtime.generate(packed[1], list(b"hi\n"), n_new)


def test_load_pack_rejects_bad_magic(tmp_path):
    path = tmp_path / "bad.bin"
    path.write_bytes(b"NOPE" + bytes(24))
    with pytest.raises(ValueError):
        runtime.load_pack(path)


# Prompts as `act` builds them on the demo home, and the shipped model's
# answers: a delayed set, a number among several heating targets, a reading
# and a command it must pass on.
REFERENCE = [
    ("canopy=UP\nclose the awning in 10 minutes\n", ">> set(canopy, DOWN) in 10m\n\n"),
    (
        (
            "bedroom_light=true\nliving_room_heating=19\nbedroom_heating=17.5\n"
            "bathroom_heating=21\nbedroom_shutter=DOWN\nuser_location=kitchen\n"
            "set the bedroom heating to 21\n"
        ),
        ">> set(bedroom_heating, 21)\n\n",
    ),
    (
        "outdoor_temperature=6.2\nwhat is the outdoor temperature\n",
        ">> read(outdoor_temperature)\n\n",
    ),
    ("what is the capital of France\n", "\n"),
]


@pytest.mark.parametrize(("prompt", "want"), REFERENCE)
def test_packaged_model_reproduces_reference(prompt, want):
    pack = runtime.load_pack(runtime.MODEL)
    out, probs = runtime.generate(pack, list(prompt.encode()), 96)
    assert bytes(out).decode() == want
    assert len(probs) == len(out) and all(0 < p <= 1 for p in probs)


@pytest.mark.parametrize(("prompt", "want"), REFERENCE)
def test_c_binary_reproduces_reference(exe, prompt, want):
    got = subprocess.run(
        [str(exe), str(runtime.MODEL)],
        input=prompt.encode(),
        capture_output=True,
        check=True,
    ).stdout
    assert got.decode() == want
