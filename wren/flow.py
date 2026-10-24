"""Fluksio training flow used by `wren --train`.

Flow inputs use `wren.train.DEFAULTS` and support parameter sweeps. The
engine records parameters, loss, held-out scores and the pack; training
uses `wren.train.fit`.
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
    requires=[Port("installation", "str"), *(Port(p.name, p.dtype) for p in PORTS)],
    provides=[
        Port("loss", "float", stream=True),
        Port("final_loss", "float"),
        Port("n_params", "float"),
        Port("exact_match", "float"),
        Port("threshold", "float"),
        Port("coverage", "float"),
        Port("pack", "artifact"),
    ],
    # No `gpus` declaration: this is the shipped training command, so a machine
    # without a card should train on the CPU rather than refuse the run.
    # preallocate=false leaves room beside a study holding the same card --
    # jax otherwise takes 75% of it at import.
    resources={"env": {"XLA_PYTHON_CLIENT_PREALLOCATE": "false"}},
)
def fit(
    installation: str,
    dim: int,
    n_layers: int,
    n_heads: int,
    steps: int,
    batch_size: int,
    lr: float,
    n_train: int,
    n_eval: int,
    precision: float,
    seed: int,
) -> dict:
    """Train on an installation's JSON file (`targets`, `examples`) with `wren.train.fit`."""
    import json

    from wren import train
    from wren.targets import Target

    data = json.loads(pathlib.Path(installation).read_text())
    path = pathlib.Path(tempfile.mkdtemp()) / "wren.bin"
    result = train.fit(
        [Target(**t) for t in data["targets"]],
        out=path,
        examples=data.get("examples", ()),
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        steps=steps,
        batch_size=batch_size,
        lr=lr,
        n_train=n_train,
        n_eval=n_eval,
        precision=precision,
        seed=seed,
        on_log=lambda step, loss: fluksio.emit(loss=loss),
    )
    keys = ("final_loss", "n_params", "exact_match", "threshold", "coverage")
    return {k: result[k] for k in keys} | {
        "pack": fluksio.save_artifact(path.read_bytes(), name="wren.bin"),
    }


FLOW = Flow(
    NAME,
    title="wren --train",
    nodes=[fit],
    # The installation travels as a path, for the worker to read.
    inputs=[Port("installation", "str"), *PORTS],
    outputs=["final_loss", "n_params", "exact_match", "threshold", "coverage"],
)
