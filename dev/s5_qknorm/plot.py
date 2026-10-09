"""Plot gradient moments, attention entropy, and frequency-pair statistics.

Read Fluksio runs and write `figures/s5-qknorm.svg`. Export with `--local`
to read runs without a running server.
"""

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

csv.field_size_limit(sys.maxsize)  # the probes blob exceeds the 128KB default

# Categorical slots 1-2 of the validated palette; color follows the arm.
COLORS = {"on": "#2a78d6", "off": "#eb6834"}
INK, MUTED = "#0b0b0b", "#52514e"
DEPTHS = (4, 8, 16, 32)
WALL = 10.5  # slots >= 11 have periods > the train window (omega < 2pi/128)


def rows() -> list[dict]:
    """Return completed 2000-step runs from the Fluksio export."""
    with tempfile.TemporaryDirectory() as directory:
        out = Path(directory) / "runs.csv"
        subprocess.run(
            [
                "fluksio",
                "export",
                "runs",
                "--local",
                "--flow",
                "s5_qknorm",
                "--status",
                "ok",
                "--metrics",
                "probes,final_loss,bpb_128",
                "-o",
                str(out),
            ],
            check=True,
        )
        exported = list(csv.DictReader(out.read_text().splitlines()))

    kept = []
    for row in sorted(exported, key=lambda row: row.get("id", "")):
        if (row.get("param.steps") or "2000") != "2000":  # smoke runs
            continue
        kept.append(
            {
                "depth": int(row["param.depth"] or 8),
                "qk": row["param.qk_norm"] or "on",
                "seed": int(row["param.seed"] or 0),
                "probes": json.loads(row["metric.probes"]),
            }
        )
    return kept


def arr(snap: dict, key: str) -> np.ndarray:
    """Convert a snapshot field to an array, replacing JSON null with NaN."""

    def conv(v):
        if isinstance(v, list):
            return [conv(u) for u in v]
        return float("nan") if v is None else float(v)

    return np.array(conv(snap[key]), dtype=float)


def g2(run: dict, idx: int) -> np.ndarray:
    """Return each layer's gradient second moment: signal + noise."""
    return arr(run["probes"][idx], "sig") + arr(run["probes"][idx], "noise")


def main() -> None:
    runs = rows()
    fig, (bp_ax, layer_ax, ent_ax, band_ax) = plt.subplots(
        1, 4, figsize=(15, 3.8), dpi=120
    )
    fig.patch.set_facecolor("#fcfcfb")

    for ax in (bp_ax, layer_ax, ent_ax, band_ax):
        ax.set_facecolor("#fcfcfb")
        ax.grid(axis="y", color="#e6e5e1", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=8)

    # Panel 1: the BP curve — E[g^2] at init (mean over layers) vs depth.
    for qk, color in COLORS.items():
        means = []
        for depth in DEPTHS:
            vals = [
                np.nanmean(g2(r, 0))
                for r in runs
                if r["depth"] == depth and r["qk"] == qk
            ]
            if vals:
                bp_ax.plot([depth] * len(vals), vals, "o", color=color, markersize=4)
                means.append((depth, float(np.mean(vals))))
        if means:
            bp_ax.plot(*zip(*means), "-", color=color, linewidth=1.5, label=f"qk {qk}")
    anchor = [np.nanmean(g2(r, 0)) for r in runs if r["depth"] == 4 and r["qk"] == "on"]
    if anchor:
        ds = np.array(DEPTHS, dtype=float)
        bp_ax.plot(
            ds,
            np.mean(anchor) * 4.0 / ds,
            ":",
            color=MUTED,
            linewidth=1,
            label="1/depth",
        )
    bp_ax.set_xscale("log", base=2)
    bp_ax.set_yscale("log")
    bp_ax.set_xticks(DEPTHS, [str(d) for d in DEPTHS])
    bp_ax.legend(fontsize=7, frameon=False, labelcolor=MUTED)
    bp_ax.set_xlabel("depth (layers)", color=INK, fontsize=9)
    bp_ax.set_ylabel("E[g²] at init", color=INK, fontsize=9)
    bp_ax.set_title("gradient variance vs depth", color=INK, fontsize=10, loc="left")

    # Panels 2 and 3: per-layer resolution at depths 8 and 32 (seed means).
    for depth, alpha in ((8, 0.4), (32, 1.0)):
        for qk, color in COLORS.items():
            grp = [r for r in runs if r["depth"] == depth and r["qk"] == qk]
            if not grp:
                continue
            layer_ax.plot(
                np.mean([g2(r, 0) for r in grp], axis=0),
                color=color,
                alpha=alpha,
                linewidth=1.5,
                label=f"d{depth} {qk}",
            )
            ent_ax.plot(
                np.mean([arr(r["probes"][-1], "entropy") for r in grp], axis=0),
                color=color,
                alpha=alpha,
                linewidth=1.5,
            )
            ent_ax.plot(
                np.mean([arr(r["probes"][0], "entropy") for r in grp], axis=0),
                color=color,
                alpha=0.35 * alpha,
                linewidth=1,
                linestyle=":",
            )
    layer_ax.set_yscale("log")
    layer_ax.legend(fontsize=7, frameon=False, labelcolor=MUTED, ncols=2)
    layer_ax.set_xlabel("layer", color=INK, fontsize=9)
    layer_ax.set_ylabel("E[g²] at init", color=INK, fontsize=9)
    layer_ax.set_title("per-layer profile at init", color=INK, fontsize=10, loc="left")

    ent_ax.set_ylim(0, 1.05)
    ent_ax.set_xlabel("layer", color=INK, fontsize=9)
    ent_ax.set_ylabel("attention entropy (normalized)", color=INK, fontsize=9)
    ent_ax.set_title(
        "entropy per layer (dotted: init)", color=INK, fontsize=10, loc="left"
    )

    # Panel 4: band instruments at depth 8, averaged over layers, seeds, and
    # the training probes (init excluded — there is no signal yet).
    d8 = {qk: [r for r in runs if r["depth"] == 8 and r["qk"] == qk] for qk in COLORS}
    if d8["on"]:
        power = np.mean(
            [[arr(p, "power") for p in r["probes"][1:]] for r in d8["on"]],
            axis=(0, 1, 2),
        )
        share = power / power.sum()
        band_ax.bar(range(len(share)), share, color=MUTED, alpha=0.35, width=1.0)
        band_ax.set_ylabel("logit power share (qk on)", color=INK, fontsize=9)
        snr_ax = band_ax.twinx()
        for qk, color in COLORS.items():
            if not d8[qk]:
                continue
            sig = np.mean(
                [[arr(p, "band_sig_q") for p in r["probes"][1:]] for r in d8[qk]],
                axis=(0, 1, 2),
            )
            noise = np.mean(
                [[arr(p, "band_noise_q") for p in r["probes"][1:]] for r in d8[qk]],
                axis=(0, 1, 2),
            )
            snr_ax.plot(sig / noise, color=color, linewidth=1.5, label=f"qk {qk}")
        snr_ax.set_ylabel("W_q gradient SNR", color=INK, fontsize=9)
        snr_ax.set_ylim(bottom=0)
        snr_ax.tick_params(colors=MUTED, labelsize=8)
        for side in ("top",):
            snr_ax.spines[side].set_visible(False)
        snr_ax.legend(fontsize=7, frameon=False, labelcolor=MUTED, loc="upper right")
    band_ax.axvline(WALL, color=MUTED, linewidth=0.8, linestyle=":")
    band_ax.text(
        WALL,
        0.98,
        "aliasing wall (period > 128) ",
        color=MUTED,
        fontsize=7,
        transform=band_ax.get_xaxis_transform(),
        va="top",
        ha="right",
    )
    band_ax.set_xlabel("frequency slot (high → low ω)", color=INK, fontsize=9)
    band_ax.set_title("band instruments, depth 8", color=INK, fontsize=10, loc="left")

    fig.suptitle(
        "s5 — QK-norm as a barren-plateau diagnostic · Home-Assistant-Requests, "
        "dim 256, seq 128, muon 3e-3, 2000 steps, 3 seeds",
        color=INK,
        fontsize=11,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = Path(__file__).parent / "figures" / "s5-qknorm.svg"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, facecolor=fig.get_facecolor())


if __name__ == "__main__":
    main()
