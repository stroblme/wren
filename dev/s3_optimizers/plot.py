"""S3 deliverable figure: loss at the fixed step budget per optimizer.

Reads the engine's run table (`fluksio export runs`) and writes
docs/s3-optimizer-bakeoff.svg. Left: the LR sweep (seed 0, depth 8).
Right: seed spread at each winner, with depth 16 as open markers.
"""

import csv
import io
import subprocess

import matplotlib.pyplot as plt
from matplotlib import ticker

# Categorical slots 1-3 of the validated palette; color follows the
# optimizer across both panels.
COLORS = {"adamw": "#2a78d6", "muon": "#eb6834", "kfac": "#1baf7a"}
WINNERS = {"adamw": 1e-3, "muon": 3e-3, "kfac": 3e-3}
INK, MUTED = "#0b0b0b", "#52514e"


def rows() -> list[dict]:
    out = subprocess.run(
        ["fluksio", "export", "runs", "--flow", "s3_optimizers"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [
        r
        for r in csv.DictReader(io.StringIO(out))
        if r["status"] == "ok" and r["param.optimizer"]
    ]


def main() -> None:
    runs = rows()
    fig, (lr_ax, seed_ax) = plt.subplots(
        1, 2, figsize=(9, 4), width_ratios=[3, 2], sharey=True, dpi=120
    )
    fig.patch.set_facecolor("#fcfcfb")

    for ax in (lr_ax, seed_ax):
        ax.set_facecolor("#fcfcfb")
        ax.set_yscale("log")
        ax.set_ylim(0.11, 3.2)
        ax.yaxis.set_major_locator(ticker.FixedLocator([0.2, 0.3, 0.5, 1, 2, 3]))
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.yaxis.set_minor_locator(ticker.NullLocator())
        ax.grid(axis="y", color="#e6e5e1", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=8)

    for opt, color in COLORS.items():
        pts = sorted(
            (float(r["param.lr"]), float(r["metric.final_loss"]))
            for r in runs
            if r["param.optimizer"] == opt
            and not r["param.depth"]
            and (r["param.seed"] or "0") == "0"
        )
        lrs, losses = zip(*pts)
        lr_ax.plot(lrs, losses, "o-", color=color, linewidth=2, markersize=5)
        lr_ax.annotate(
            opt,
            (lrs[-1], losses[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            color=color,
            fontsize=9,
            fontweight="bold",
            va="center",
        )
    lr_ax.set_xscale("log")
    lr_ax.set_xlabel("learning rate", color=INK, fontsize=9)
    lr_ax.set_ylabel("loss at 2000 steps", color=INK, fontsize=9)
    lr_ax.set_title("LR sweep (seed 0, depth 8)", color=INK, fontsize=10, loc="left")

    for i, (opt, color) in enumerate(COLORS.items()):
        seeds = [
            float(r["metric.final_loss"])
            for r in runs
            if r["param.optimizer"] == opt
            and not r["param.depth"]
            and float(r["param.lr"]) == WINNERS[opt]
        ]
        deep = [
            float(r["metric.final_loss"])
            for r in runs
            if r["param.optimizer"] == opt and r["param.depth"] == "16"
        ]
        x = [i - 0.14, i - 0.06, i + 0.02][: len(seeds)]
        seed_ax.plot(x, seeds, "o", color=color, markersize=7)
        seed_ax.plot(
            [i + 0.18] * len(deep),
            deep,
            "D",
            color=color,
            markersize=7,
            markerfacecolor="#fcfcfb",
        )
        seed_ax.annotate(
            f"{min(seeds):.3f}",
            (i - 0.06, min(seeds)),
            xytext=(0, -18),
            textcoords="offset points",
            color=MUTED,
            fontsize=8,
            ha="center",
        )
    seed_ax.set_xticks(range(3), [f"{o}\nlr {WINNERS[o]:g}" for o in COLORS])
    seed_ax.set_xlim(-0.5, 2.5)
    seed_ax.set_title(
        "winner: seeds 0–2 (●), depth 16 (◇)", color=INK, fontsize=10, loc="left"
    )

    fig.suptitle(
        "s3 — optimizer bake-off · Home-Assistant-Requests, dim 256, 2000 steps",
        color=INK,
        fontsize=11,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig("docs/s3-optimizer-bakeoff.svg", facecolor=fig.get_facecolor())


if __name__ == "__main__":
    main()
