"""Probe frequency redundancy in saved fixed and learned s4 models.

Sweep slots 8 and 18 over [1e-3, pi), measuring held-out loss at windows
128 and 256. Write `figures/s4-redundancy-probe.svg` and print loss spans.
"""

import csv
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

SLOTS = (8, 18)
WINDOWS = (128, 256)
N_POINTS = 64
N_BATCH = 64  # held windows per loss evaluation
COLORS = {"fixed": "#2a78d6", "learned": "#eb6834"}
INK, MUTED = "#0b0b0b", "#52514e"


def fetch_weights(directory: Path) -> dict[str, Path]:
    """Fetch weights from the fixed and learned@3e-3 seed-0 runs."""
    runs = directory / "runs.csv"
    subprocess.run(
        [
            "fluksio",
            "export",
            "runs",
            "--local",
            "--flow",
            "s4_learnable_freqs",
            "--status",
            "ok",
            "-o",
            str(runs),
        ],
        check=True,
    )
    wanted = {}
    for row in sorted(
        csv.DictReader(runs.read_text().splitlines()), key=lambda row: row.get("id", "")
    ):
        if (row.get("param.steps") or "2000") != "2000":
            continue
        if (row.get("param.seed") or "0") != "0":
            continue
        mode = row.get("param.mode") or "fixed"
        if mode == "fixed" or (
            mode == "learned" and float(row.get("param.freq_lr") or 0) == 3e-3
        ):
            wanted[mode] = row["id"]
    paths = {}
    for mode, run_id in wanted.items():
        paths[mode] = directory / f"{mode}.npz"
        subprocess.run(
            [
                "fluksio",
                "artifacts",
                "--local",
                run_id,
                "s4_learnable_freqs.weights",  # the port-qualified name
                "-o",
                str(paths[mode]),
            ],
            check=True,
        )
    return paths


def main() -> None:
    # Before jax loads: share the card politely.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    sys.path.insert(0, str(Path(__file__).parent))

    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np
    from flax import nnx
    from study import SpectralWren, _ce, held_windows, spectrum

    from wren import tokenizer
    from wren.dataset import home_assistant, to_text
    from wren.model import Config

    def load_spectral(path: Path) -> SpectralWren:
        """Load SpectralWren weights, including the extra `log_freqs` parameter."""
        npz = np.load(path)
        cfg = Config(
            **{k.removeprefix("cfg:"): int(npz[k]) for k in npz if "cfg:" in k}
        )
        model = SpectralWren(cfg, nnx.Rngs(0), spectrum(cfg.dim // cfg.n_heads))
        leaves, treedef = jax.tree_util.tree_flatten_with_path(
            nnx.state(model, nnx.Param)
        )
        restored = [jnp.asarray(npz[jax.tree_util.keystr(k)]) for k, _ in leaves]
        nnx.update(model, jax.tree_util.tree_unflatten(treedef, restored))
        return model

    with tempfile.TemporaryDirectory() as directory:
        models = {
            m: load_spectral(p) for m, p in fetch_weights(Path(directory)).items()
        }

    _, test = home_assistant()
    held = jnp.array(tokenizer.encode(to_text(test)), dtype=jnp.int32)
    batches = {n: tuple(w[:N_BATCH] for w in held_windows(held, n)) for n in WINDOWS}

    omegas = np.geomspace(1e-3, math.pi, N_POINTS)
    curves: dict[tuple[str, int, int], list[float]] = {}
    for mode, model in models.items():
        saved = model.log_freqs[...]
        for slot in SLOTS:
            for n in WINDOWS:
                x, y = batches[n]
                losses = []
                for w in omegas:
                    model.log_freqs[...] = saved.at[slot].set(math.log(float(w)))
                    losses.append(float(_ce(model, x, y)))
                curves[mode, slot, n] = losses
        model.log_freqs[...] = saved

    # Span table: near-DC region (omega < 2*pi/128, redundant at the train
    # window by the aliasing argument) and the full sweep, in nats.
    dc = omegas < 2 * math.pi / 128
    print(
        f"{'model':8s} {'slot':>4s} {'window':>6s} {'span near-DC':>12s} {'span full':>10s}"
    )
    for (mode, slot, n), losses in sorted(curves.items()):
        arr = np.array(losses)
        print(
            f"{mode:8s} {slot:4d} {n:6d} {arr[dc].max() - arr[dc].min():12.4f} "
            f"{arr.max() - arr.min():10.4f}"
        )

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True, dpi=120)
    fig.patch.set_facecolor("#fcfcfb")
    for ax, slot in zip(axes, SLOTS):
        ax.set_facecolor("#fcfcfb")
        ax.set_xscale("log")
        ax.grid(axis="y", color="#e6e5e1", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=8)
        # The aliasing thresholds: everything left of 2*pi/L is near-DC at
        # window L. Growing the window moves the wall left.
        for n, style in zip(WINDOWS, (":", "--")):
            ax.axvline(2 * math.pi / n, color=MUTED, linewidth=0.8, linestyle=style)
        ax.text(
            2 * math.pi / 128,
            0.98,
            " 2π/128",
            color=MUTED,
            fontsize=7,
            transform=ax.get_xaxis_transform(),
            va="top",
        )
        ax.text(
            2 * math.pi / 256,
            0.98,
            "2π/256 ",
            color=MUTED,
            fontsize=7,
            transform=ax.get_xaxis_transform(),
            va="top",
            ha="right",
        )
        for mode, color in COLORS.items():
            if mode not in models:
                continue
            for n, width in zip(WINDOWS, (2.0, 1.2)):
                ax.plot(
                    omegas,
                    curves[mode, slot, n],
                    color=color,
                    linewidth=width,
                    alpha=1.0 if n == 128 else 0.65,
                    label=f"{mode}, window {n}",
                )
            cur = float(np.exp(np.asarray(models[mode].log_freqs[...]))[slot])
            ax.plot(
                [cur],
                [np.interp(cur, omegas, curves[mode, slot, 128])],
                "o",
                color=color,
                markersize=5,
            )
        band = "in-band" if slot == SLOTS[0] else "under-trained"
        ax.set_title(f"slot {slot} ({band})", color=INK, fontsize=10, loc="left")
        ax.set_xlabel("omega", color=INK, fontsize=9)
    axes[0].set_ylabel("held-out loss (nats)", color=INK, fontsize=9)
    axes[0].legend(fontsize=7, frameon=False, labelcolor=MUTED)
    fig.suptitle(
        "s4 addendum — redundancy probe · loss along one spectral coordinate, "
        "markers at trained values",
        color=INK,
        fontsize=11,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = Path(__file__).parent / "figures" / "s4-redundancy-probe.svg"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, facecolor=fig.get_facecolor())


if __name__ == "__main__":
    main()
