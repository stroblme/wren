"""Response weighting keeps full contexts and excludes padding from loss."""

import jax.numpy as jnp
import numpy as np
from flax import nnx

from dev.s13_accuracy.study import prepare
from wren.model import Config, Wren
from wren.targets import Call, Target, answer, prompt
from wren.train import muon, train_step


def test_aligned_examples_mask_padding_and_weight_answer_targets():
    state = [Target("pump", "bool", False)]
    cases = [("pump on", state, [Call("pump", True)]), ("hello", state, [])]
    x, y, weight = prepare(cases, context=96, prompt_weight=0.1)
    first = (prompt(state, "pump on") + answer(cases[0][2])).encode()
    n_prompt = len(prompt(state, "pump on").encode())
    assert bytes(x[0, : len(first) - 1].tolist()) == first[:-1]
    assert bytes(y[0, : len(first) - 1].tolist()) == first[1:]
    np.testing.assert_allclose(weight[0, : n_prompt - 1], 0.1)
    assert (weight[0, n_prompt - 1 : len(first) - 1] == 1).all()
    assert (weight[0, len(first) - 1 :] == 0).all()
    # The blank line teaching abstention is an answer token, not padding.
    assert weight[1, len("hello\n") - 1] == 1


def test_zero_weight_padding_does_not_change_loss_or_update():
    cfg = Config(dim=16, n_layers=1, n_heads=2, context=16)
    a = Wren(cfg, rngs=nnx.Rngs(0))
    b = Wren(cfg, rngs=nnx.Rngs(0))
    opt_a = nnx.Optimizer(a, muon(1e-3), wrt=nnx.Param)
    opt_b = nnx.Optimizer(b, muon(1e-3), wrt=nnx.Param)
    x, y = jnp.array([[10, 20, 30]]), jnp.array([[20, 30, 40]])
    loss_a = train_step(a, opt_a, x, y)
    loss_b = train_step(
        b,
        opt_b,
        jnp.pad(x, ((0, 0), (0, 3))),
        jnp.pad(y, ((0, 0), (0, 3))),
        jnp.array([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0]]),
    )
    np.testing.assert_allclose(loss_a, loss_b, rtol=1e-5)
    np.testing.assert_allclose(a(x), b(x), atol=1e-5)
