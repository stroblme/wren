"""Plot spectrum changes, length generalization, and the frequency probe.

Read Fluksio runs and write `figures/s4-learnable-freqs.svg`. Export with
`--local` to read runs without a running server.
"""

import csv
import json
import math
import subprocess
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt

# Categorical slots 1-3 of the validated palette; color follows the mode.
COLORS = {"fixed": "#2a78d6", "learned": "#eb6834", "fope": "#1baf7a"}
INK, MUTED = "#0b0b0b", "#52514e"
LENGTHS = (128, 192, 256)
PROBE_OMEGA = 0.1  # the probed frequency's stock value, omega_8


# The lists and the per-window numbers the panels read. Only the numeric
# leaves of a run's outputs are columns by default, so the lists are named.
METRICS = ("freq_history", "probe_omegas", "probe_losses") + tuple(
    f"bpb_{n}" for n in LENGTHS
)


def rows() -> list[dict]:
    """Return completed runs, excluding smoke runs.

    Export through a file because engine startup also writes to stdout. Fill
    missing parameters in older runs from the flow's original defaults.
    """
    with tempfile.TemporaryDirectory() as directory:
        out = Path(directory) / "runs.csv"
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
                "--metrics",
                ",".join(METRICS),
                "-o",
                str(out),
            ],
            check=True,
        )
        exported = list(csv.DictReader(out.read_text().splitlines()))

    kept = []
    # Oldest first. The export hands back newest first, and the panels that
    # pick one run per mode would otherwise draw a different arm than before.
    for row in sorted(exported, key=lambda row: row.get("id", "")):
        if (row.get("param.steps") or "2000") != "2000":  # smoke runs
            continue
        if not row.get("metric.bpb_128"):
            continue
        kept.append(
            {
                "param.mode": row.get("param.mode") or "fixed",
                "param.freq_lr": row.get("param.freq_lr") or "",
                "param.seed": row.get("param.seed") or "",
                **{f"metric.{name}": row.get(f"metric.{name}", "") for name in METRICS},
            }
        )
    return kept


def main() -> None:
    runs = rows()
    fig, (spec_ax, len_ax, probe_ax) = plt.subplots(1, 3, figsize=(12, 4), dpi=120)
    fig.patch.set_facecolor("#fcfcfb")

    for ax in (spec_ax, len_ax, probe_ax):
        ax.set_facecolor("#fcfcfb")
        ax.grid(axis="y", color="#e6e5e1", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=8)

    # Left: spectrum before and after training, one line per learned freq_lr.
    stock = [10000.0 ** (-i / 32) for i in range(32)]
    spec_ax.set_yscale("log")
    spec_ax.axhspan(2 * math.pi / 64, 2 * math.pi / 2, color="#eb6834", alpha=0.06)
    spec_ax.text(
        31,
        2 * math.pi / 16,
        "byte-scale\nperiods 2–64",
        color=MUTED,
        fontsize=7,
        ha="right",
        va="center",
    )
    spec_ax.plot(stock, "--", color=MUTED, linewidth=1.2, label="stock / init")
    learned = [
        r
        for r in runs
        if r["param.mode"] == "learned"
        and (r["param.seed"] or "0") == "0"
        and r["metric.freq_history"]
    ]
    for i, r in enumerate(
        sorted(learned, key=lambda r: float(r["param.freq_lr"] or 0))
    ):
        history = json.loads(r["metric.freq_history"])
        alpha = 1.0 - 0.55 * (len(learned) - 1 - i) / max(len(learned) - 1, 1)
        spec_ax.plot(
            history[-1],
            color=COLORS["learned"],
            linewidth=2,
            alpha=alpha,
            label=f"learned, freq_lr {float(r['param.freq_lr'] or 3e-3):g}",
        )
    spec_ax.legend(fontsize=7, frameon=False, labelcolor=MUTED)
    spec_ax.set_xlabel("frequency index", color=INK, fontsize=9)
    spec_ax.set_ylabel("omega", color=INK, fontsize=9)
    spec_ax.set_title("spectrum: init vs learned", color=INK, fontsize=10, loc="left")

    # Middle: length generalization, trained at seq 128.
    for i, (mode, color) in enumerate(COLORS.items()):
        pts = [r for r in runs if r["param.mode"] == mode]
        for r in pts:
            xs = [x + (i - 1) * 6 for x in LENGTHS]
            ys = [float(r[f"metric.bpb_{n}"]) for n in LENGTHS]
            len_ax.plot(xs, ys, "o-", color=color, markersize=4, linewidth=1, alpha=0.8)
        if pts:
            y = float(pts[0]["metric.bpb_256"])
            len_ax.annotate(
                mode,
                (256 + (i - 1) * 6, y),
                xytext=(8, 0),
                textcoords="offset points",
                color=color,
                fontsize=9,
                fontweight="bold",
                va="center",
            )
    len_ax.set_xticks(LENGTHS)
    len_ax.set_xlim(112, 300)
    len_ax.set_xlabel("eval window (bytes)", color=INK, fontsize=9)
    len_ax.set_ylabel("bits/byte", color=INK, fontsize=9)
    len_ax.set_title(
        "length generalization (train seq 128)", color=INK, fontsize=10, loc="left"
    )

    # Right: loss over one full period of frequency 8, everything else frozen.
    for mode, color in COLORS.items():
        r = next(
            (
                r
                for r in runs
                if r["param.mode"] == mode
                and (r["param.seed"] or "0") == "0"
                and r["metric.probe_losses"]
            ),
            None,
        )
        if r:
            probe_ax.plot(
                json.loads(r["metric.probe_omegas"]),
                json.loads(r["metric.probe_losses"]),
                color=color,
                linewidth=1.5,
                label=mode,
            )
    probe_ax.axvline(PROBE_OMEGA, color=MUTED, linewidth=0.8, linestyle=":")
    probe_ax.text(
        PROBE_OMEGA,
        0.98,
        " omega_8 = 0.1",
        color=MUTED,
        fontsize=7,
        transform=probe_ax.get_xaxis_transform(),
        va="top",
    )
    probe_ax.legend(fontsize=7, frameon=False, labelcolor=MUTED)
    probe_ax.set_xlabel("omega (one period)", color=INK, fontsize=9)
    probe_ax.set_ylabel("held-out loss", color=INK, fontsize=9)
    probe_ax.set_title(
        "Rotosolve probe, frequency 8", color=INK, fontsize=10, loc="left"
    )

    fig.suptitle(
        "s4 — learnable RoPE frequencies · Home-Assistant-Requests, dim 256, "
        "2000 steps",
        color=INK,
        fontsize=11,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = Path(__file__).parent / "figures" / "s4-learnable-freqs.svg"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, facecolor=fig.get_facecolor())


if __name__ == "__main__":
    main()
