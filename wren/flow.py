"""The training run, declared as a fluksio flow.

`wren --train` submits this flow to the project-local engine
(`uv run fluksio serve`), which records what it ran with, the loss curve it
reported, and the weights it produced. The node is a thin wrapper around
`wren.train.train`, so the library itself stays fluksio-free and a plain
`train(model, text)` keeps working.

Every knob is a flow *input* rather than a node setting, because
`fluksio sweep --param` varies inputs (DECISIONS.md), and its default comes
from `wren.train.DEFAULTS` -- the model shape included, so a run records the
model it trained instead of whatever `Config` happened to say on the worker.
"""

import pathlib
import tempfile

import fluksio
from fluksio import Flow, Port, node

from wren.train import DEFAULTS

NAME = "wren_train"

#: One input per knob; the CLI spells the same names `--dim`, `--lr`, ...
PORTS = [
    Port(name, "float" if isinstance(default, float) else "int", initial=default)
    for name, default in DEFAULTS.items()
]


@node(
    requires=[Port("corpus", "str"), *(Port(p.name, p.dtype) for p in PORTS)],
    provides=[
        Port("loss", "float", stream=True),
        Port("final_loss", "float"),
        Port("n_params", "float"),
        Port("weights", "artifact"),
    ],
    # No `gpus` declaration: this is the shipped training command, so a machine
    # without a card should train on the CPU rather than refuse the run.
    # preallocate=false leaves room beside a study holding the same card --
    # jax otherwise takes 75% of it at import (NOTEPAD.md).
    resources={"env": {"XLA_PYTHON_CLIENT_PREALLOCATE": "false"}},
)
def fit(
    corpus: str,
    dim: int,
    n_layers: int,
    n_heads: int,
    context: int,
    steps: int,
    batch_size: int,
    seq_len: int,
    lr: float,
) -> dict:
    """Train from scratch on `corpus` (a path; empty means the demo corpus)."""
    from flax import nnx

    from wren.dataset import home_corpus
    from wren.model import Config, Wren, save
    from wren.train import train

    text = pathlib.Path(corpus).read_text() if corpus else home_corpus()
    cfg = Config(dim=dim, n_layers=n_layers, n_heads=n_heads, context=context)
    model = Wren(cfg, rngs=nnx.Rngs(0))
    losses = train(
        model,
        text,
        steps=steps,
        batch_size=batch_size,
        seq_len=seq_len,
        lr=lr,
        on_log=lambda step, loss: fluksio.emit(loss=loss),
    )

    path = pathlib.Path(tempfile.mkdtemp()) / "weights.npz"
    save(model, path)
    return {
        "final_loss": losses[-1],
        "n_params": model.n_params(),
        "weights": fluksio.save_artifact(path.read_bytes(), name="weights.npz"),
    }


FLOW = Flow(
    NAME,
    title="wren --train",
    nodes=[fit],
    # The corpus travels as a path: it is half a megabyte of text, which is a
    # file for the worker to read, not a parameter to record on every run.
    inputs=[Port("corpus", "str", initial=""), *PORTS],
    outputs=["final_loss", "n_params"],
)
