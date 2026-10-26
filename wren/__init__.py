"""Wren turns short commands on an installation's targets into calls.

`act(command, targets, model=load_pack(path))` runs a trained pack in NumPy;
without `model` it runs the shipped demo model. Training (`wren.train.fit`)
requires the `train` extra.
"""

from wren.runtime import load_pack
from wren.targets import Call, Target, accept, act

__all__ = ["Call", "Target", "accept", "act", "load_pack"]
