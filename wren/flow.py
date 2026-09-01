"""The training run, declared as a fluksio flow.

`wren --train` submits this flow to the project-local engine
(`uv run fluksio serve`), which records what it ran with, the loss curve it
reported, and the weights it produced. The node is a thin wrapper around
`wren.train.train`, so the library itself stays fluksio-free and a plain
`train(model, text)` keeps working.

Flow *inputs* rather than node settings, because `fluksio sweep --param`
varies inputs (DECISIONS.md); their initials mirror `train()`'s defaults,
which a test pins.
"""

import pathlib
import tempfile

import fluksio
from fluksio import Flow, Port, node

NAME = "wren_train"

#: The flow's knobs, in `train()`'s order. The CLI turns each into `--name`.
PARAMS = [
    Port("steps", "int", initial=1000),
    Port("batch_size", "int", initial=64),
    Port("seq_len", "int", initial=128),
    Port("lr", "float", initial=1e-3),
    Port("log_every", "int", initial=25),
]


@node(
    requires=[Port("corpus", "str"), *(Port(p.name, p.dtype) for p in PARAMS)],
    provides=[
        Port("loss", "float", stream=True),
        Port("final_loss", "float"),
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
    steps: int,
    batch_size: int,
    seq_len: int,
    lr: float,
    log_every: int,
) -> dict:
    """Train from scratch on `corpus` (a path; empty means the demo corpus)."""
    from flax import nnx

    from wren.dataset import home_corpus
    from wren.model import Config, Wren, save
    from wren.train import train

    text = pathlib.Path(corpus).read_text() if corpus else home_corpus()
    model = Wren(Config(), rngs=nnx.Rngs(0))
    losses = train(
        model,
        text,
        steps=steps,
        batch_size=batch_size,
        seq_len=seq_len,
        lr=lr,
        log_every=log_every,
        on_log=lambda step, loss: fluksio.emit(loss=loss),
    )

    path = pathlib.Path(tempfile.mkdtemp()) / "weights.npz"
    save(model, path)
    return {
        "final_loss": losses[-1],
        "weights": fluksio.save_artifact(path.read_bytes(), name="weights.npz"),
    }


FLOW = Flow(
    NAME,
    title="wren --train",
    nodes=[fit],
    # The corpus travels as a path: it is half a megabyte of text, which is a
    # file for the worker to read, not a parameter to record on every run.
    inputs=[Port("corpus", "str", initial=""), *PARAMS],
    outputs=["final_loss"],
)
