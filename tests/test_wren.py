import subprocess
import sys

import jax.numpy as jnp
import pytest
from flax import nnx

from wren import cli, tokenizer
from wren.model import Config, Wren, generate, load, save
from wren.targets import Call, Target
from wren.train import train


def test_tokenizer_roundtrip():
    text = "Setup a scene for dinner — 21.5°C"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_forward_shape():
    cfg = Config(dim=32, n_layers=2, n_heads=2, context=32)
    model = Wren(cfg, rngs=nnx.Rngs(0))
    logits = model(jnp.zeros((2, 16), dtype=jnp.int32))
    assert logits.shape == (2, 16, cfg.vocab_size)


def test_overfits_tiny_corpus():
    cfg = Config(dim=64, n_layers=2, n_heads=2, context=64)
    model = Wren(cfg, rngs=nnx.Rngs(0))
    text = "the lights are on. " * 60
    losses = train(model, text, steps=150, batch_size=8, seq_len=32)
    assert losses[-1] < losses[0] * 0.2, (
        f"loss barely moved: {losses[0]} -> {losses[-1]}"
    )
    out = tokenizer.decode(
        generate(model, tokenizer.encode("the li"), 20, temperature=0.1)
    )
    assert "lights are on" in out, f"overfit model failed to reproduce corpus: {out!r}"


def test_save_load_roundtrip(tmp_path):
    cfg = Config(dim=32, n_layers=2, n_heads=2, context=32)
    model = Wren(cfg, rngs=nnx.Rngs(1))
    path = tmp_path / "weights.npz"
    save(model, path)
    restored = load(path)
    x = jnp.zeros((1, 8), dtype=jnp.int32)
    assert restored.cfg == cfg
    assert jnp.allclose(model(x), restored(x))


def test_cli_prints_each_call_with_its_confidence(tmp_path, monkeypatch, capsys):
    installation = tmp_path / "lab.json"
    installation.write_text(
        '{"targets": [{"name": "pump", "type": "bool", "value": false}]}'
    )
    seen = {}

    def fake_act(command, targets, *, model):
        seen.update(command=command, targets=targets, model=model)
        return [Call("pump", True, confidence=0.97), Call("pump", False, 600, 0.9)]

    monkeypatch.setattr(cli, "act", fake_act)
    monkeypatch.setattr(cli.runtime, "load_pack", lambda path: f"pack {path.name}")
    cli.main(["--targets", str(installation), "--model", "lab.bin", "run", "the pump"])
    assert seen == {
        "command": "run the pump",
        "targets": [Target("pump", "bool", False)],
        "model": "pack lab.bin",
    }
    assert capsys.readouterr().out == (
        ">> set(pump, true)  0.97\n>> set(pump, false) in 10m  0.90\n"
    )


def test_cli_says_when_a_command_is_not_handled(monkeypatch, capsys):
    monkeypatch.setattr(cli, "act", lambda command, targets, *, model: [])
    cli.main(["order a pizza"])
    assert capsys.readouterr().out == "not handled\n"


def test_cli_runs_the_demo_model_without_jax():
    code = (
        "import sys; sys.modules['jax'] = None; from wren import cli; "
        "cli.main(['turn on the kitchen light'])"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.startswith(">> set(kitchen_light, true)  ")


def test_cli_train_without_jax_names_the_extra():
    code = (
        "import sys; sys.modules['jax'] = None; from wren import cli; "
        "cli.main(['--train', 'home.json'])"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 1
    assert "pip install 'wren-lm[train]'" in result.stderr


def test_generate_batch_matches_full_forward():
    import jax.numpy as jnp

    from wren.model import generate_batch

    cfg = Config(dim=32, n_layers=2, n_heads=2, context=48)
    model = Wren(cfg, rngs=nnx.Rngs(3))
    prompts = [tokenizer.encode("hello world"), tokenizer.encode("hi")]
    got, _ = generate_batch(model, prompts, n_new=8)
    for prompt, out in zip(prompts, got):
        toks = list(prompt)
        for _ in range(8):  # reference: argmax through the uncached forward
            logits = model(jnp.array([toks]))[0, -1]
            toks.append(int(jnp.argmax(logits)))
        assert out == toks[len(prompt) :]


def test_generate_batch_stop_sequence():
    from wren.model import generate_batch

    cfg = Config(dim=64, n_layers=2, n_heads=2, context=64)
    model = Wren(cfg, rngs=nnx.Rngs(0))
    train(model, "ab\n\n" * 200, steps=60, batch_size=8, seq_len=16)
    [out], _ = generate_batch(model, [tokenizer.encode("ab")], n_new=20, stop=(10, 10))
    assert tokenizer.decode(out).endswith("\n\n") and len(out) < 20


def test_generate_batch_confidence():
    from wren.model import generate_batch

    cfg = Config(dim=32, n_layers=2, n_heads=2, context=48)
    model = Wren(cfg, rngs=nnx.Rngs(3))
    outs, confs = generate_batch(model, [tokenizer.encode("ab")], n_new=6)
    assert len(confs) == 1 and 0.0 < confs[0] <= 1.0
    assert outs == generate_batch(model, [tokenizer.encode("ab")], n_new=6)[0]


def test_generate_batch_rejects_n_new_outside_context():
    from wren.model import generate_batch

    cfg = Config(dim=32, n_layers=1, n_heads=2, context=32)
    model = Wren(cfg, rngs=nnx.Rngs(0))
    for n_new in (0, cfg.context, cfg.context + 8):
        with pytest.raises(ValueError, match="n_new"):
            generate_batch(model, [tokenizer.encode("ab")], n_new=n_new)


def test_flow_takes_every_knob():
    """Check that flow inputs expose every training parameter."""
    import inspect

    from wren import flow
    from wren.train import DEFAULTS

    assert set(inspect.signature(flow.fit).parameters) == {"installation", *DEFAULTS}
    assert {port.name: port.initial for port in flow.PORTS} == DEFAULTS
