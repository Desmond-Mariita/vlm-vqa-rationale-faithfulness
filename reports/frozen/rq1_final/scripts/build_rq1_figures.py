#!/usr/bin/env python3
"""Build the two minimal frozen RQ1 figures and their source-data tables."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("reports/frozen/rq1_final")
FIG = ROOT / "figures"
ARMS = ["plain", "point", "plain_desc", "point_desc"]
CONDITIONS = ["source", "grey", "mismatch", "mask", "noise", "mirror"]
PERTURBATIONS = CONDITIONS[1:]
COLORS = {
    "plain": "#0072B2",
    "point": "#D55E00",
    "plain_desc": "#009E73",
    "point_desc": "#CC79A7",
}
MARKERS = {"plain": "o", "point": "s", "plain_desc": "^", "point_desc": "D"}


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def save(fig, stem: str) -> None:
    fig.savefig(FIG / f"{stem}.png", dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    cells = read_tsv(ROOT / "data/RQ1_CELL_RESULTS.tsv")
    gaps = read_tsv(ROOT / "data/RQ1_GAP_RESULTS.tsv")
    write_tsv(FIG / "RQ1_FIGURE_1_ACCURACY_SOURCE_DATA.tsv", cells)
    write_tsv(FIG / "RQ1_FIGURE_2_GAP_SOURCE_DATA.tsv", gaps)

    cell = {(r["arm"], r["condition"]): r for r in cells}
    x = np.arange(len(CONDITIONS))
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    for arm in ARMS:
        y = np.asarray([float(cell[(arm, c)]["accuracy"]) for c in CONDITIONS])
        lo = np.asarray([float(cell[(arm, c)]["lower_95%"]) for c in CONDITIONS])
        hi = np.asarray([float(cell[(arm, c)]["upper_95%"]) for c in CONDITIONS])
        ax.errorbar(x, y, yerr=np.vstack([y-lo, hi-y]), label=arm, color=COLORS[arm], marker=MARKERS[arm],
                    linewidth=1.8, markersize=5.5, capsize=2.5)
    ax.axhline(0.25, color="#5f6368", linestyle="--", linewidth=1.2, label="chance (0.25)")
    ax.set_xticks(x, ["Source", "Grey", "Mismatch", "Mask", "Noise", "Mirror"])
    ax.set_ylabel("Stage-1 accuracy")
    ax.set_xlabel("Image condition")
    ax.set_ylim(0.2, 0.87)
    ax.set_title("Final Stage-1 accuracy across image conditions")
    ax.grid(axis="y", color="#d9dde3", linewidth=0.7, alpha=0.8)
    ax.legend(ncol=3, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.28))
    fig.subplots_adjust(bottom=0.24)
    save(fig, "RQ1_FIGURE_1_ACCURACY_BY_CONDITION")

    gap = {(r["arm"], r["condition"]): r for r in gaps}
    x = np.arange(len(PERTURBATIONS))
    offsets = np.linspace(-0.27, 0.27, len(ARMS))
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    for offset, arm in zip(offsets, ARMS):
        y = np.asarray([float(gap[(arm, c)]["source_minus_condition_gap"]) for c in PERTURBATIONS])
        lo = np.asarray([float(gap[(arm, c)]["lower_95%"]) for c in PERTURBATIONS])
        hi = np.asarray([float(gap[(arm, c)]["upper_95%"]) for c in PERTURBATIONS])
        ax.errorbar(x + offset, y, yerr=np.vstack([y-lo, hi-y]), label=arm, color=COLORS[arm], marker=MARKERS[arm],
                    linestyle="none", markersize=6, capsize=3, elinewidth=1.6)
    ax.axhline(0, color="#5f6368", linewidth=1)
    ax.set_xticks(x, ["Grey", "Mismatch", "Mask", "Noise", "Mirror"])
    ax.set_ylabel("Source − condition accuracy")
    ax.set_xlabel("Image perturbation")
    ax.set_ylim(-0.03, 0.42)
    ax.set_title("Source-relative Stage-1 accuracy gaps")
    ax.grid(axis="y", color="#d9dde3", linewidth=0.7, alpha=0.8)
    ax.legend(ncol=4, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.26))
    fig.subplots_adjust(bottom=0.23)
    save(fig, "RQ1_FIGURE_2_SOURCE_RELATIVE_DROP")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
