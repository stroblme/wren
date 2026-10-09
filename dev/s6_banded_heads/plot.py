"""Plot length generalization, head statistics, and spectrum changes.

Read Fluksio runs and write `figures/s6-banded-heads.svg`. Export with
`--local` to read runs without a running server.
"""

import csv
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

csv.field_size_limit(sys.maxsize)  # freq_history exceeds the 128KB default

# Categorical slots 1-3 of the validated palette; color follows the arm.
COLORS = {"shared": "#2a78d6", "banded": "#eb6834", "banded_learned": "#1baf7a"}
HEAD_COLORS = ("#eb6834", "#b6522a", "#7a3d24", "#3d2a1e")  # byte -> long
INK, MUTED = "#0b0b0b", "#52514e"
LENGTHS = (128, 192, 256)
BINS = ("0", "1", "2–3", "4–7", "8–15", "16–31", "32–63", "64–127", "128–255")
HEADS = ("byte 6–32 B", "word 32–128 B", "sample 128–1024 B", "long ≥1024 B")

METRICS = ("freq_history", "head_profiles") + tuple(f"bpb_{n}" for n in LENGTHS)


def rows() -> list[dict]:
    """Return completed runs from the Fluksio export, excluding smoke runs."""
    with tempfile.TemporaryDirectory() as directory:
        out = Path(directory) / "runs.csv"
        subprocess.run(
            [
                "fluksio",
                "export",
                "runs",
                "--local",
                "--flow",
                "s6_banded_heads",
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

    # Deduplicate by (mode, seed), newest complete run wins: the deadlocked
    # first sweep left runs marked ok with no metrics, and re-runs of a
    # finished combination come back as cache hits of the original.
    kept: dict[tuple[str, int], dict] = {}
    for row in sorted(exported, key=lambda row: row.get("id", "")):
        if (row.get("param.steps") or "2000") != "2000":  # smoke runs
            continue
        if not row.get("metric.bpb_128"):
            continue
        mode = row.get("param.mode") or "shared"
        seed = int(row.get("param.seed") or 0)
        kept[(mode, seed)] = {
            "mode": mode,
            "seed": seed,
            "bpb": [float(row[f"metric.bpb_{n}"]) for n in LENGTHS],
            "profiles": json.loads(row["metric.head_profiles"]),
            "freqs": json.loads(row["metric.freq_history"]),
        }
    return list(kept.values())


def main() -> None:
    runs = rows()
    by_mode = {m: [r for r in runs if r["mode"] == m] for m in COLORS}
    fig, (bpb_ax, dist_ax, ent_ax, freq_ax) = plt.subplots(
        1, 4, figsize=(15, 3.8), dpi=120
    )
    fig.patch.set_facecolor("#fcfcfb")

    for ax in (bpb_ax, dist_ax, ent_ax, freq_ax):
        ax.set_facecolor("#fcfcfb")
        ax.grid(axis="y", color="#e6e5e1", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=8)

    # Panel 1: bits/byte per eval window, seed points and mean line per arm.
    for mode, color in COLORS.items():
        grp = by_mode[mode]
        if not grp:
            continue
        for r in grp:
            bpb_ax.plot(LENGTHS, r["bpb"], "o", color=color, markersize=3, alpha=0.6)
        bpb_ax.plot(
            LENGTHS,
            np.mean([r["bpb"] for r in grp], axis=0),
            "-",
            color=color,
            linewidth=1.5,
            label=mode,
        )
    bpb_ax.set_xticks(LENGTHS, [str(n) for n in LENGTHS])
    bpb_ax.legend(fontsize=7, frameon=False, labelcolor=MUTED)
    bpb_ax.set_xlabel("eval window (bytes)", color=INK, fontsize=9)
    bpb_ax.set_ylabel("held-out bits/byte", color=INK, fontsize=9)
    bpb_ax.set_title("length generalization", color=INK, fontsize=10, loc="left")

    # Panel 2: attention-distance profiles at window 256, layer- and
    # seed-averaged. Banded heads as lines; the shared arm's heads are
    # unordered, so they appear as a min-max spread band.
    if by_mode["shared"]:
        sh = np.mean(
            [np.array(r["profiles"]["256"]["dist"]) for r in by_mode["shared"]], axis=0
        ).mean(axis=0)  # (heads, bins)
        dist_ax.fill_between(
            range(len(BINS)),
            sh.min(axis=0),
            sh.max(axis=0),
            color=COLORS["shared"],
            alpha=0.25,
            label="shared (head spread)",
        )
    if by_mode["banded"]:
        bd = np.mean(
            [np.array(r["profiles"]["256"]["dist"]) for r in by_mode["banded"]], axis=0
        ).mean(axis=0)
        for h, (color, name) in enumerate(zip(HEAD_COLORS, HEADS)):
            dist_ax.plot(bd[h], color=color, linewidth=1.5, label=name)
    dist_ax.set_xticks(range(len(BINS)), BINS, rotation=45)
    dist_ax.legend(fontsize=6.5, frameon=False, labelcolor=MUTED)
    dist_ax.set_xlabel("attention distance (bytes)", color=INK, fontsize=9)
    dist_ax.set_ylabel("attention mass, window 256", color=INK, fontsize=9)
    dist_ax.set_title("per-head distance profile", color=INK, fontsize=10, loc="left")

    # Panel 3: per-head entropy at window 256, layer-averaged, seed points.
    for i, (mode, color) in enumerate(COLORS.items()):
        for r in by_mode[mode]:
            ent = np.array(r["profiles"]["256"]["entropy"]).mean(axis=0)  # (heads,)
            x = np.arange(len(ent)) + (i - 1) * 0.18
            ent_ax.plot(x, ent, "o", color=color, markersize=3.5, alpha=0.7)
    ent_ax.set_xticks(range(4), [f"head {h}" for h in range(4)])
    ent_ax.set_ylim(0, 1.05)
    ent_ax.set_ylabel("attention entropy (normalized)", color=INK, fontsize=9)
    ent_ax.set_title("per-head entropy, window 256", color=INK, fontsize=10, loc="left")

    # Panel 4: banded_learned spectrum, init vs final over the 128 global
    # slots, frozen slots (omega < 2pi/128 at init) shaded.
    if by_mode["banded_learned"]:
        r0 = by_mode["banded_learned"][0]
        init = np.array(r0["freqs"][0]).reshape(-1)
        wall = 2 * math.pi / 128
        for r in by_mode["banded_learned"]:
            final = np.array(r["freqs"][-1]).reshape(-1)
            freq_ax.plot(final, color=COLORS["banded_learned"], linewidth=1, alpha=0.6)
        freq_ax.plot(init, color=MUTED, linewidth=1, linestyle=":", label="init")
        freq_ax.fill_between(
            range(len(init)),
            1e-5,
            1e1,
            where=init < wall,
            color=MUTED,
            alpha=0.12,
            label="frozen (period > 128)",
        )
        for edge in (32, 64, 96):
            freq_ax.axvline(edge - 0.5, color="#e6e5e1", linewidth=0.8)
        freq_ax.set_yscale("log")
        freq_ax.set_ylim(init.min() * 0.5, 3.0)
        freq_ax.legend(fontsize=7, frameon=False, labelcolor=MUTED)
    freq_ax.set_xlabel("slot (head 0 … head 3, high → low ω)", color=INK, fontsize=9)
    freq_ax.set_ylabel("ω", color=INK, fontsize=9)
    freq_ax.set_title("learned banded spectrum", color=INK, fontsize=10, loc="left")

    fig.suptitle(
        "s6 — spectral-aware attention heads · Home-Assistant-Requests, dim 256, "
        "seq 128, muon 3e-3, 2000 steps, 3 seeds",
        color=INK,
        fontsize=11,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = Path(__file__).parent / "figures" / "s6-banded-heads.svg"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, facecolor=fig.get_facecolor())


if __name__ == "__main__":
    main()
