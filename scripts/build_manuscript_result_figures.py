#!/usr/bin/env python3
"""Render manuscript result figures from frozen publication source data.

This script changes presentation labels only. It verifies the frozen TSV hashes,
performs no inference, and computes no new comparison.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import string
import zlib
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.backends.backend_pdf as _backend_pdf
import matplotlib.pyplot as plt
import numpy as np


def _stable_subset_prefix(charset) -> str:
    """A reproducible six-letter PDF font subset tag.

    Matplotlib builds this tag from ``hash(charset)``. On the Type 1 path the
    argument is a dictionary view, which has no value-based hash, so Python
    falls back to the object's identity and the tag changes with the memory
    address. The figure is then byte-different on every run while not one mark
    on the page moves, and the builder can never be shown to reproduce its own
    output. Deriving the tag from the character set itself fixes that, and the
    tag is the only thing it changes.
    """
    digest = zlib.crc32(repr(sorted(map(repr, charset))).encode("utf-8"))
    letters = ""
    for _ in range(6):
        letters = string.ascii_uppercase[digest % 26] + letters
        digest //= 26
    return letters + "+"


_backend_pdf.PdfFile._get_subset_prefix = staticmethod(_stable_subset_prefix)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "thesis/Figures/final"
RQ1 = Path(
    "reports/frozen/rq1_final/figures/"
    "RQ1_FIGURE_1_ACCURACY_SOURCE_DATA.tsv"
)
RQ2_DRIFT = Path(
    "reports/frozen/rq2_final/figures/"
    "RQ2_FIGURE_1_SOURCE_DATA.tsv"
)
RQ2_ANCHOR = Path(
    "reports/frozen/rq2_final/figures/"
    "RQ2_FIGURE_2_SOURCE_DATA.tsv"
)
EXPECTED = {
    RQ1: "afc9fdf6b793651f4f1e37b76fe5a6e60d37adbf7b54ede940afc0c3e6111b27",
    RQ2_DRIFT: "5713c7f928b1c8e93e48477b31d5d634f70057f492da37f72f7ce475001135ff",
    RQ2_ANCHOR: "be1e12d949f3f28bd7ae0b1813880022ead87b5febe1f4cc53dc4babadf1ae06",
}
ARMS = ("plain", "point", "plain_desc", "point_desc")
# Legend and dodge order is the arm order used by every table in the manuscript.
ARM_LABELS = {
    "plain": r"\texttt{plain}",
    "point": r"\texttt{point}",
    "plain_desc": r"\texttt{plain\_desc}",
    "point_desc": r"\texttt{point\_desc}",
}
COLORS = {
    "plain": "#0072B2",
    "point": "#D55E00",
    "plain_desc": "#009E73",
    "point_desc": "#CC79A7",
}
# Filled and open markers alternate so the four series stay separable in
# grayscale, where the four hues converge.
MARKERS = {"plain": "o", "point": "s", "plain_desc": "^", "point_desc": "D"}
FILLED = {"plain": True, "point": False, "plain_desc": True, "point_desc": False}
DODGE = 0.17

TEXTWIDTH_PT = 404.02908
SMALL_PT = 10.95
FOOTNOTE_PT = 8.85


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_verified(path: Path) -> list[dict[str, str]]:
    actual = sha256(path)
    if actual != EXPECTED[path]:
        raise RuntimeError(f"Frozen source hash mismatch: {path}: {actual}")
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def save_pdf(figure: plt.Figure, name: str) -> None:
    figure.savefig(
        OUT / name,
        facecolor="white",
        metadata={"Creator": "Frozen manuscript figure renderer", "CreationDate": None},
    )
    plt.close(figure)


def configure() -> None:
    mpl.rcParams.update(
        {
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage{mathpazo}",
            "font.family": "serif",
            "font.serif": ["Palatino"],
            "font.size": FOOTNOTE_PT,
            "axes.titlesize": SMALL_PT,
            "axes.labelsize": SMALL_PT,
            "xtick.labelsize": FOOTNOTE_PT,
            "ytick.labelsize": FOOTNOTE_PT,
            "legend.fontsize": FOOTNOTE_PT,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def draw_series(axis, x, arm, values, low, high, index):
    """Draw one arm as dodged points with error bars and no connecting line."""
    offset = (index - (len(ARMS) - 1) / 2) * DODGE
    axis.errorbar(
        np.asarray(x) + offset,
        values,
        yerr=np.vstack((values - low, high - values)),
        linestyle="none",
        marker=MARKERS[arm],
        color=COLORS[arm],
        markerfacecolor=COLORS[arm] if FILLED[arm] else "white",
        markeredgecolor=COLORS[arm],
        markeredgewidth=1.2,
        markersize=5.0,
        elinewidth=1.2,
        capsize=2.5,
        label=ARM_LABELS[arm],
    )


def rq1_figure(rows: list[dict[str, str]]) -> None:
    conditions = ("source", "grey", "mismatch", "mask", "noise", "mirror")
    cells = {(row["arm"], row["condition"]): row for row in rows}
    x = np.arange(len(conditions))
    figure, axis = plt.subplots(figsize=(0.88 * TEXTWIDTH_PT / 72.27, 3.65))
    for index, arm in enumerate(ARMS):
        values = np.asarray([float(cells[(arm, c)]["accuracy"]) for c in conditions])
        low = np.asarray([float(cells[(arm, c)]["lower_95%"]) for c in conditions])
        high = np.asarray([float(cells[(arm, c)]["upper_95%"]) for c in conditions])
        draw_series(axis, x, arm, values, low, high, index)
    axis.axhline(0.25, color="#5f6368", linestyle="--", linewidth=1.1)
    axis.annotate(
        "Chance (0.25)",
        xy=(len(conditions) - 0.55, 0.25),
        xytext=(0, 4),
        textcoords="offset points",
        ha="right",
        va="bottom",
        color="#5f6368",
        fontsize=FOOTNOTE_PT,
    )
    axis.set_xticks(x, [c.title() for c in conditions])
    axis.set_xlim(-0.5, len(conditions) - 0.5)
    axis.set_ylabel("Stage 1 accuracy")
    axis.set_xlabel("Image condition")
    axis.set_ylim(0.2, 0.87)
    axis.grid(axis="y", color="#d9dde3", linewidth=0.7, alpha=0.8)
    axis.set_axisbelow(True)
    axis.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.24),
                handletextpad=0.4, columnspacing=1.4)
    figure.subplots_adjust(left=0.15, right=0.97, top=0.96, bottom=0.30)
    save_pdf(figure, "rq1_accuracy_by_condition.pdf")


def rq2_drift_figure(rows: list[dict[str, str]]) -> None:
    conditions = ("grey", "mismatch", "mask", "noise", "mirror")
    cells = {(row["arm"], row["condition"]): row for row in rows}
    x = np.arange(len(conditions))
    figure, axis = plt.subplots(figsize=(0.88 * TEXTWIDTH_PT / 72.27, 3.45))
    for index, arm in enumerate(ARMS):
        values = np.asarray([float(cells[(arm, c)]["mean_raw_drift"]) for c in conditions])
        low = np.asarray([float(cells[(arm, c)]["ci_lower_95"]) for c in conditions])
        high = np.asarray([float(cells[(arm, c)]["ci_upper_95"]) for c in conditions])
        draw_series(axis, x, arm, values, low, high, index)
    axis.set_xticks(x, [c.title() for c in conditions])
    axis.set_xlim(-0.5, len(conditions) - 0.5)
    axis.set_ylabel("Raw BERTScore Drift")
    axis.set_xlabel("Image intervention")
    axis.set_ylim(0.04, 0.098)
    axis.grid(axis="y", color="#d9dde3", linewidth=0.7, alpha=0.8)
    axis.set_axisbelow(True)
    axis.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.25),
                handletextpad=0.4, columnspacing=1.4)
    figure.subplots_adjust(left=0.16, right=0.97, top=0.96, bottom=0.31)
    save_pdf(figure, "rq2_drift_by_perturbation.pdf")


def rq2_anchor_figure(rows: list[dict[str, str]]) -> None:
    names = (
        "mirror",
        "grey",
        "mismatch",
        "deliberate_answer_control",
        "unrelated_rationale_reference",
    )
    labels = (
        "Mirror",
        "Grey",
        "Mismatch",
        "Wrong\nanswer",
        "Unrelated\nrationale",
    )
    scopes = (
        "$n\\approx$2,653",
        "$n\\approx$2,653",
        "$n\\approx$2,653",
        "$n$=200/arm",
        "$n\\approx$2,653",
    )
    cells = {(row["arm"], row["anchor"]): row for row in rows}
    x = np.arange(len(names))
    figure, axis = plt.subplots(figsize=(0.92 * TEXTWIDTH_PT / 72.27, 3.95))
    for index, arm in enumerate(ARMS):
        values = np.asarray([float(cells[(arm, n)]["mean_raw_drift"]) for n in names])
        low = np.asarray([float(cells[(arm, n)]["ci_lower_95"]) for n in names])
        high = np.asarray([float(cells[(arm, n)]["ci_upper_95"]) for n in names])
        draw_series(axis, x, arm, values, low, high, index)

    axis.set_ylim(-0.006, 0.158)
    axis.axhline(0.0, color="#5f6368", linestyle=":", linewidth=1.1)
    axis.annotate(
        "Identity (Drift 0)",
        xy=(-0.42, 0.0),
        xytext=(0, 3),
        textcoords="offset points",
        ha="left",
        va="bottom",
        color="#5f6368",
        fontsize=FOOTNOTE_PT,
    )
    # Separate the three kinds of comparison point: they are not one continuum.
    for boundary in (2.5, 3.5):
        axis.axvline(boundary, color="#b9bec6", linewidth=0.8, linestyle="-")
    for centre, text in ((1.0, "Image interventions"), (3.0, "Answer"), (4.0, "Reference")):
        axis.annotate(
            text,
            xy=(centre, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(0, -11),
            textcoords="offset points",
            ha="center",
            va="top",
            color="#5f6368",
            fontsize=FOOTNOTE_PT,
        )

    axis.set_xticks(x, [f"{label}\n{scope}" for label, scope in zip(labels, scopes)])
    axis.set_xlim(-0.5, len(names) - 0.5)
    axis.set_ylabel("Raw BERTScore Drift")
    axis.set_xlabel("Empirical comparison point")
    axis.grid(axis="y", color="#d9dde3", linewidth=0.7, alpha=0.8)
    axis.set_axisbelow(True)
    axis.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.30),
                handletextpad=0.4, columnspacing=1.4)
    figure.subplots_adjust(left=0.16, right=0.97, top=0.90, bottom=0.32)
    save_pdf(figure, "rq2_anchor_ladder.pdf")


def main() -> None:
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    OUT = args.output_dir.resolve()
    if OUT == (ROOT / "thesis/Figures/final").resolve():
        raise SystemExit("Choose a separate generated output directory")
    OUT.mkdir(parents=True, exist_ok=True)
    configure()
    rq1_figure(read_verified(RQ1))
    rq2_drift_figure(read_verified(RQ2_DRIFT))
    rq2_anchor_figure(read_verified(RQ2_ANCHOR))
    print("rendered three manuscript figures from frozen publication source data")


if __name__ == "__main__":
    main()
