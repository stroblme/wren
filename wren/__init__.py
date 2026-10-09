"""Wren: a tiny attention-only byte-level model for grounded tool calls.

The package root is the inference library (numpy only, see wren.home);
training lives in wren.model and wren.train and needs the `train` extra.
"""

from wren.home import AREAS, KINDS, Call, Device, House, act, render_state, select_areas

__all__ = [
    "AREAS",
    "KINDS",
    "Call",
    "Device",
    "House",
    "act",
    "render_state",
    "select_areas",
]
