"""S6b deliverable figure: the shared−banded gap as heads get narrow.

Reads the engine's run table and writes figures/s6b-head-pressure.svg.
Panels 1–3: held-out bits/byte at windows 128 and 256 and train loss at
budget, each vs head count, shared vs banded with seed spread — the gap
between the arms is the result. Panel 4: per-head entropy at 32 heads,
where banding should show a scale ladder if specialization survives
narrow heads.

Data comes from `fluksio export runs`; `--local` boots the engine in this
process, so plotting needs no server.
"""

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

csv.field_size_limit(sys.maxsize)  # head_profiles exceeds the 128KB default

# Categorical slots 1-2 of the validated palette; color follows the arm.
COLORS = {"shared": "#2a78d6", "banded": "#eb6834"}
INK, MUTED = "#0b0b0b", "#52514e"
HEAD_COUNTS = (4, 8, 16, 32)

METRICS = ("head_profiles", "final_loss", "bpb_128", "bpb_192", "bpb_256")


def rows() -> list[dict]:
    """One dict per finished non-smoke run, deduplicated by combination
    (newest wins), straight from `export runs`."""
    with tempfile.TemporaryDirectory() as directory:
        out = Path(directory) / "runs.csv"
        subprocess.run(
            [
                "fluksio",
                "export",
                "runs",
                "--local",
                "--flow",
                "s6b_head_pressure",
                "--status",
                "ok",
                "--metrics",
                ",".join(METRICS),
                "-o",
                str(out),
            ],
            check=True,
        )
        exported = list(csv.DictReader(out.read_text().splitlines()))

    kept: dict[tuple[str, int, int], dict] = {}
    for row in sorted(exported, key=lambda row: row.get("id", "")):
        if (row.get("param.steps") or "2000") != "2000":  # smoke runs
            continue
        if not row.get("metric.bpb_128"):
            continue
        key = (
            row.get("param.mode") or "shared",
            int(row.get("param.n_heads") or 4),
            int(row.get("param.seed") or 0),
        )
        kept[key] = {
            "mode": key[0],
            "n_heads": key[1],
            "seed": key[2],
            "loss": float(row["metric.final_loss"]),
            "bpb": {n: float(row[f"metric.bpb_{n}"]) for n in (128, 192, 256)},
            "profiles": json.loads(row["metric.head_profiles"]),
        }
    return list(kept.values())


def main() -> None:
    runs = rows()
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), dpi=120)
    fig.patch.set_facecolor("#fcfcfb")

    for ax in axes:
        ax.set_facecolor("#fcfcfb")
        ax.grid(axis="y", color="#e6e5e1", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=8)

    # Panels 1–3: metric vs head count per arm, seed dots and mean lines.
    panels = [
        (axes[0], lambda r: r["bpb"][128], "held-out bits/byte, window 128"),
        (axes[1], lambda r: r["bpb"][256], "held-out bits/byte, window 256"),
        (axes[2], lambda r: r["loss"], "train loss at budget"),
    ]
    for ax, value, title in panels:
        for mode, color in COLORS.items():
            means = []
            for nh in HEAD_COUNTS:
                grp = [r for r in runs if r["mode"] == mode and r["n_heads"] == nh]
                vals = [value(r) for r in grp]
                if vals:
                    ax.plot(
                        [nh] * len(vals),
                        vals,
                        "o",
                        color=color,
                        markersize=3.5,
                        alpha=0.6,
                    )
                    means.append((nh, float(np.mean(vals))))
            if means:
                ax.plot(*zip(*means), "-", color=color, linewidth=1.5, label=mode)
        ax.set_xscale("log", base=2)
        ax.set_xticks(HEAD_COUNTS, [str(n) for n in HEAD_COUNTS])
        ax.set_xlabel("heads (head_dim = 256/heads)", color=INK, fontsize=9)
        ax.set_title(title, color=INK, fontsize=10, loc="left")
        ax.legend(fontsize=7, frameon=False, labelcolor=MUTED)

    # Panel 4: per-head attention mass at distances 128–255 B — beyond the
    # train window — at window 256 (layer- and seed-mean). Sharing puts the
    # wrap-zone slot in every head (flat, high = correlated failure);
    # banding isolates it in dedicated heads (spikes = contained failure).
    ax = axes[3]
    for nh, style in ((16, "-"), (32, ":")):
        for mode, color in COLORS.items():
            grp = [r for r in runs if r["mode"] == mode and r["n_heads"] == nh]
            if not grp:
                continue
            mass = np.mean(
                [np.array(r["profiles"]["256"]["dist"])[:, :, 8] for r in grp],
                axis=(0, 1),
            )
            ax.plot(
                np.arange(nh) / nh,
                mass,
                style,
                color=color,
                linewidth=1.3,
                label=f"{mode} h{nh}",
            )
    ax.set_xlabel("head position (high → low ω band)", color=INK, fontsize=9)
    ax.set_ylabel("attention mass at 128–255 B", color=INK, fontsize=9)
    ax.set_title("extrapolated-range mass, w256", color=INK, fontsize=10, loc="left")
    ax.legend(fontsize=7, frameon=False, labelcolor=MUTED)

    fig.suptitle(
        "s6b — head-dim pressure · Home-Assistant-Requests, dim 256, seq 128, "
        "muon 3e-3, 2000 steps, 3 seeds",
        color=INK,
        fontsize=11,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = Path(__file__).parent / "figures" / "s6b-head-pressure.svg"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, facecolor=fig.get_facecolor())


if __name__ == "__main__":
    main()
