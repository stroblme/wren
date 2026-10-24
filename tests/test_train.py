"""Check quantization-aware training and `fit` on a tiny installation."""

import functools
import math

import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from wren import runtime, tokenizer
from wren.export import export
from wren.model import Config, Wren, save
from wren.targets import Target, act
from wren.train import fit, train

LAB = [
    Target("pump", "bool", False),
    Target("ph", "float", 7.0, readonly=True),
]
# Small enough to train in seconds, large enough to learn the two targets.
tiny_fit = functools.partial(fit, dim=64, n_layers=2, n_heads=2, batch_size=8, lr=1e-2)


def test_ternary_kernels_survive_export(tmp_path):
    cfg = Config(dim=32, n_layers=2, n_heads=2, context=48, ternary=True)
    model = Wren(cfg, rngs=nnx.Rngs(0))
    train(model, "ab\n\n" * 200, steps=20, batch_size=8, seq_len=16)
    save(model, tmp_path / "weights.npz")
    export(str(tmp_path / "weights.npz"), str(tmp_path / "wren.bin"))
    pack = runtime.load_pack(tmp_path / "wren.bin")

    for layer in pack["layers"]:  # {-s, 0, +s} per kernel
        assert len(np.unique(layer["wqkv"])) <= 3 and len(np.unique(layer["wo"])) <= 3
    tokens = tokenizer.encode("ab\nab\n")
    want = np.asarray(model(jnp.array([tokens]))[0])
    got = runtime._forward(pack, runtime._caches(pack), tokens, 0)
    np.testing.assert_allclose(got, want, atol=1e-4)


def test_fit_writes_a_pack_that_act_runs(tmp_path):
    out = tmp_path / "lab.bin"
    result = tiny_fit(LAB, out=out, steps=200, n_train=3000, n_eval=60)

    pack = runtime.load_pack(out)
    assert [t["name"] for t in pack["meta"]["targets"]] == ["pump", "ph"]
    assert all("value" not in t for t in pack["meta"]["targets"])
    assert pack["meta"]["threshold"] == result["threshold"]
    assert math.isfinite(result["threshold"])
    assert 0 < result["coverage"] <= result["exact_match"] <= 1
    calls = act("turn on the pump", LAB, model=pack)
    assert all(c.target in ("pump", "ph") for c in calls)


def test_fit_refuses_a_model_below_the_precision(tmp_path):
    out = tmp_path / "lab.bin"
    # One step: the model writes no answer that parses, so no call is right.
    with pytest.raises(RuntimeError, match="precision 0.99"):
        tiny_fit(LAB, out=out, steps=1, n_train=300, n_eval=20)
    assert not out.exists()
