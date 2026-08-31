"""S7 deliverable figure: the failure waterfall and where the oracle points.

Reads the engine's run table and writes docs/s7-bottleneck.svg. Panel 1:
the waterfall — eval samples per failure class, per corpus. Panel 2: the
oracle ladder — exact match under free, grammar-constrained, gold-name,
and gold-structure decoding. Panel 3: value-hole accuracy under gold
structure for trained values vs the copy probe's unseen literals — the
memorization-vs-copying contrast. Panel 4: span-resolved bits/byte.

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

csv.field_size_limit(sys.maxsize)

INK, MUTED = "#0b0b0b", "#52514e"
CORPORA = {"ha": "#2a78d6", "mobile": "#eb6834"}
CLASSES = [  # waterfall order, best to worst
    ("exact", "#1baf7a"),
    ("values_copyable", "#2a78d6"),
    ("values_derived", "#7a5cc4"),
    ("wrong_name", "#eb6834"),
    ("ill_formed", "#c94040"),
    ("abstain", "#52514e"),
]
RUNGS = ["free", "grammar", "gold_name", "gold_structure"]
SPANS = ["prompt", "syntax", "name", "key", "value_copy", "value_der"]


def rows() -> list[dict]:
    """One dict per finished run, deduplicated by (corpus, seed, steps)."""
    with tempfile.TemporaryDirectory() as directory:
        out = Path(directory) / "runs.csv"
        subprocess.run(
            [
                "fluksio",
                "export",
                "runs",
                "--local",
                "--flow",
                "s7_bottleneck",
                "--status",
                "ok",
                "--metrics",
                "report",
                "-o",
                str(out),
            ],
            check=True,
        )
        exported = list(csv.DictReader(out.read_text().splitlines()))

    kept: dict[tuple, dict] = {}
    for row in sorted(exported, key=lambda row: row.get("id", "")):
        if not row.get("metric.report"):
            continue
        report = json.loads(row["metric.report"])
        if report["n_eval"] < 500:  # smoke runs
            continue
        key = (
            row.get("param.corpus") or "ha",
            int(row.get("param.seed") or 0),
            int(row.get("param.steps") or 2000),
        )
        kept[key] = {
            "corpus": key[0],
            "seed": key[1],
            "steps": key[2],
            "report": report,
        }
    return list(kept.values())


def main() -> None:
    runs = rows()
    base = {
        c: [r for r in runs if r["corpus"] == c and r["steps"] == 2000] for c in CORPORA
    }
    big = {
        c: [r for r in runs if r["corpus"] == c and r["steps"] == 8000] for c in CORPORA
    }
    fig, (wf_ax, lad_ax, copy_ax, span_ax) = plt.subplots(
        1, 4, figsize=(15, 3.8), dpi=120
    )
    fig.patch.set_facecolor("#fcfcfb")
    for ax in (wf_ax, lad_ax, copy_ax, span_ax):
        ax.set_facecolor("#fcfcfb")
        ax.grid(axis="y", color="#e6e5e1", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=8)

    # Panel 1: the waterfall, seed-mean stacked bars per corpus (+4x arm).
    bars = [(c, grp) for c, grp in base.items() if grp] + [
        (f"{c} 4x", grp) for c, grp in big.items() if grp
    ]
    for x, (label, grp) in enumerate(bars):
        n = grp[0]["report"]["n_eval"]
        bottom = 0.0
        for cls, color in CLASSES:
            v = np.mean([r["report"]["waterfall"]["samples"][cls] / n for r in grp])
            wf_ax.bar(x, v, bottom=bottom, color=color, width=0.62)
            if v > 0.04:
                wf_ax.text(
                    x,
                    bottom + v / 2,
                    cls.replace("values_", "val "),
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="white",
                )
            bottom += v
    wf_ax.set_xticks(range(len(bars)), [b[0] for b in bars])
    wf_ax.set_ylim(0, 1)
    wf_ax.set_ylabel("fraction of eval samples", color=INK, fontsize=9)
    wf_ax.set_title("failure waterfall", color=INK, fontsize=10, loc="left")

    # Panel 2: exact match per oracle rung.
    for c, color in CORPORA.items():
        for grp, style in ((base[c], "-"), (big[c], ":")):
            if not grp:
                continue
            for r in grp:
                ys = [r["report"]["ladder"][rg]["exact_match"] for rg in RUNGS]
                lad_ax.plot(
                    range(len(RUNGS)), ys, "o", color=color, markersize=3, alpha=0.5
                )
            mean = np.mean(
                [
                    [r["report"]["ladder"][rg]["exact_match"] for rg in RUNGS]
                    for r in grp
                ],
                axis=0,
            )
            label = c if style == "-" else f"{c} 4x"
            lad_ax.plot(mean, style, color=color, linewidth=1.5, label=label)
    lad_ax.set_xticks(
        range(len(RUNGS)), ["free", "grammar", "gold\nname", "gold\nstructure"]
    )
    lad_ax.set_ylabel("exact match", color=INK, fontsize=9)
    lad_ax.legend(fontsize=7, frameon=False, labelcolor=MUTED)
    lad_ax.set_title("oracle ladder", color=INK, fontsize=10, loc="left")

    # Panel 3: value generation under gold structure — trained values vs
    # the copy probe's unseen literals.
    groups = ["trained copyable", "trained derived", "unseen (copy probe)"]
    for gi, c in enumerate(CORPORA):
        grp = base[c]
        if not grp:
            continue
        holes = [r["report"]["ladder"]["gold_structure"]["holes"] for r in grp]
        vals = [
            np.mean([h["value_copy"] or 0 for h in holes]),
            np.mean([h["value_der"] or 0 for h in holes]),
            np.mean([r["report"]["copy_probe"].get("accuracy", 0) for r in grp]),
        ]
        x = np.arange(len(groups)) + (gi - 0.5) * 0.34
        copy_ax.bar(x, vals, width=0.32, color=CORPORA[c], label=c)
        for xi, v in zip(x, vals):
            copy_ax.text(
                xi, v + 0.02, f"{v:.0%}", ha="center", fontsize=6.5, color=MUTED
            )
    copy_ax.set_xticks(range(len(groups)), ["copyable", "derived", "unseen"])
    copy_ax.set_ylim(0, 1)
    copy_ax.set_ylabel("value accuracy (gold structure)", color=INK, fontsize=9)
    copy_ax.legend(fontsize=7, frameon=False, labelcolor=MUTED)
    copy_ax.set_title("memorization, not copying", color=INK, fontsize=10, loc="left")

    # Panel 4: span-resolved bits/byte.
    for gi, c in enumerate(CORPORA):
        grp = base[c]
        if not grp:
            continue
        vals = np.mean(
            [[r["report"]["span_bpb"][s] or 0 for s in SPANS] for r in grp], axis=0
        )
        x = np.arange(len(SPANS)) + (gi - 0.5) * 0.34
        span_ax.bar(x, vals, width=0.32, color=CORPORA[c], label=c)
    span_ax.set_xticks(
        range(len(SPANS)), ["prompt", "syntax", "name", "key", "val\ncopy", "val\nder"]
    )
    span_ax.set_ylabel("bits/byte (teacher-forced)", color=INK, fontsize=9)
    span_ax.legend(fontsize=7, frameon=False, labelcolor=MUTED)
    span_ax.set_title("span-resolved loss", color=INK, fontsize=10, loc="left")

    fig.suptitle(
        "s7 — bottleneck autopsy · HA-Requests + mobile-actions, dim 256, "
        "muon 3e-3, 2000 steps (4x arm dotted), 3 seeds",
        color=INK,
        fontsize=11,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig("docs/s7-bottleneck.svg", facecolor=fig.get_facecolor())


if __name__ == "__main__":
    main()
