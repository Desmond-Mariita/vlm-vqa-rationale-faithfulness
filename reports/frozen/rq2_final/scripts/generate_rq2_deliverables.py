#!/usr/bin/env python3
"""Create the frozen RQ2 publication tables, secondary records, and minimal figures."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ARMS = ("plain", "point", "plain_desc", "point_desc")
CONDITIONS = ("grey", "mismatch", "mask", "noise", "mirror")
ARM_LABELS = {
    "plain": "Plain",
    "point": "Point",
    "plain_desc": "Plain + description",
    "point_desc": "Point + description",
}


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def tex_escape(value: object) -> str:
    text = str(value)
    for source, target in (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
    ):
        text = text.replace(source, target)
    return text


def write_tex(path: Path, rows: list[dict], caption: str, label: str) -> None:
    fields = list(rows[0])
    alignment = "l" + "r" * (len(fields) - 1)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        rf"\caption{{{tex_escape(caption)}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{alignment}}}",
        r"\toprule",
        " & ".join(tex_escape(field.replace("_", " ")) for field in fields) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(tex_escape(row[field]) for field in fields) + r" \\")
    lines.extend((r"\bottomrule", r"\end{tabular}", r"\end{table}", ""))
    path.write_text("\n".join(lines), encoding="utf-8")


def write_table_family(base: Path, stem: str, rows: list[dict], caption: str, label: str) -> None:
    write_tsv(base / f"{stem}.tsv", rows)
    write_csv(base / f"{stem}.csv", rows)
    write_tex(base / f"{stem}.tex", rows, caption, label)


def f4(value: str | float) -> str:
    return f"{float(value):.4f}"


def interval(mean: str | float, low: str | float, high: str | float, n: str | int) -> str:
    return f"{f4(mean)} [{f4(low)}, {f4(high)}]; n={n}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--pred-provenance", required=True, type=Path)
    args = parser.parse_args()
    root = args.root
    tables = root / "tables"
    figures = root / "figures"
    secondary = root / "secondary"
    figures.mkdir(parents=True, exist_ok=True)

    coverage = read_tsv(root / "audits/RQ2_PARSE_COVERAGE.tsv")
    primary = read_tsv(tables / "RQ2_PRIMARY_DRIFT_RESULTS.tsv")
    contrasts = read_tsv(tables / "RQ2_PLANNED_CONTRASTS.tsv")
    known = read_tsv(tables / "RQ2_KNOWN_CHANGE_RESULTS.tsv")
    anchors = read_tsv(tables / "RQ2_ANCHOR_SCALE_SUMMARY.tsv")
    description = read_tsv(tables / "RQ2_DESCRIPTION_AVAILABILITY_RESULTS.tsv")
    bootstrap = read_tsv(root / "audits/RQ2_BOOTSTRAP_RESULTS.tsv")

    primary_map = {(row["arm"], row["condition"]): row for row in primary}
    bootstrap_map = {row["estimate_id"]: row for row in bootstrap}

    # Table A: primary parse and coverage audit.
    table_a = []
    for row in coverage:
        if row["family"] != "stage2_primary":
            continue
        table_a.append({
            "arm": row["arm"],
            "condition": row["condition"],
            "raw_n": row["raw_generation_n"],
            "usable_final_n": row["tolerant_final_success_n"],
            "usable_final_percent": f"{100 * float(row['tolerant_final_success_rate']):.2f}",
            "pairable_source_relative_n": row["source_relative_pairable_final_n"] or "not applicable",
        })
    condition_order = ("source",) + CONDITIONS
    table_a.sort(key=lambda row: (ARMS.index(row["arm"]), condition_order.index(row["condition"])))
    write_table_family(
        tables,
        "RQ2_TABLE_A_PARSE_COVERAGE",
        table_a,
        "Final Stage-2 parse coverage by arm and condition.",
        "tab:rq2-parse-coverage",
    )

    # Table B: requested wide primary battery.
    table_b = []
    for arm in ARMS:
        row = {"arm": arm}
        for condition in CONDITIONS:
            result = primary_map[(arm, condition)]
            row[condition] = interval(result["mean_raw_drift"], result["ci_lower_95"], result["ci_upper_95"], result["valid_pair_n"])
        table_b.append(row)
    write_table_family(
        tables,
        "RQ2_TABLE_B_PRIMARY_DRIFT_BATTERY",
        table_b,
        "Raw final-span BERTScore Drift (mean, source-frame-cluster 95 percent CI, paired n).",
        "tab:rq2-primary-drift",
    )

    # Table C: frozen contrasts.
    table_c = []
    for row in contrasts:
        table_c.append({
            "contrast": row["contrast_id"],
            "status": row["status"],
            "left_mean": f4(row["left_mean_drift"]),
            "right_mean": f4(row["right_mean_drift"]),
            "paired_n": row["paired_n"],
            "mean_difference": f4(row["mean_difference"]),
            "ci_95": f"[{f4(row['ci_lower_95'])}, {f4(row['ci_upper_95'])}]",
        })
    write_table_family(
        tables,
        "RQ2_TABLE_C_CONDITION_CONTRASTS",
        table_c,
        "Frozen paired RQ2 condition and description-arm contrasts.",
        "tab:rq2-contrasts",
    )

    # Table D: known-change anchor.
    table_d = []
    for row in known:
        table_d.append({
            "arm": row["arm"],
            "n": row["valid_paired_n"],
            "matched_grey_drift": f4(row["image_removal_mean_drift"]),
            "deliberate_answer_drift": f4(row["alternate_answer_mean_drift"]),
            "answer_minus_grey": f4(row["answer_minus_image_mean_difference"]),
            "difference_ci_95": f"[{f4(row['difference_ci_lower_95'])}, {f4(row['difference_ci_upper_95'])}]",
        })
    write_table_family(
        tables,
        "RQ2_TABLE_D_KNOWN_CHANGE_ANCHOR",
        table_d,
        "Same-lineage deliberate-answer known-change control on the fixed matched sample.",
        "tab:rq2-known-change",
    )

    # Table E: anchor ladder with explicit sample scope.
    table_e = []
    for row in anchors:
        table_e.append({
            "arm": row["arm"],
            "anchor_or_condition": row["anchor_or_condition"],
            "n": row["n"],
            "mean_raw_drift": f4(row["mean_raw_drift"]),
            "ci_95": "definition" if not row["ci_lower_95"] else f"[{f4(row['ci_lower_95'])}, {f4(row['ci_upper_95'])}]",
            "scope": row["sample_scope"],
        })
    write_table_family(
        tables,
        "RQ2_TABLE_E_ANCHOR_SCALE_SUMMARY",
        table_e,
        "Empirical BERTScore Drift anchors; differing sample sizes and scopes are explicit.",
        "tab:rq2-anchor-scale",
    )

    # Table F: bounded description availability diagnostic.
    table_f = []
    for row in description:
        table_f.append({
            "arm": row["arm"],
            "n": row["valid_paired_n"],
            "D_on_mean": f4(row["D_on_mean"]),
            "D_off_mean": f4(row["D_off_mean"]),
            "Delta_off_minus_on": f4(row["Delta_description_D_off_minus_D_on"]),
            "delta_ci_95": f"[{f4(row['delta_ci_lower_95'])}, {f4(row['delta_ci_upper_95'])}]",
        })
    write_table_family(
        tables,
        "RQ2_TABLE_F_DESCRIPTION_AVAILABILITY",
        table_f,
        "Description-availability ablation in description-trained Stage-2 models (fixed n=200 per arm).",
        "tab:rq2-description-ablation",
    )

    # Secondary PRED-A table from the frozen source-only provenance record.
    pred = json.loads(args.pred_provenance.read_text(encoding="utf-8"))
    pred_rows = []
    for arm in ARMS:
        row = pred["arms"][arm]
        split = row["correct_wrong_cmc_split"]
        pred_rows.append({
            "arm": arm,
            "total_source_ids": row["records"],
            "usable_final_n": row["hardened_usable_final_n"],
            "usable_final_rate": f"{row['hardened_coverage']:.4f}",
            "stage1_correct_n": row["stage1_correct_n"],
            "stage1_wrong_n": row["stage1_wrong_n"],
            "conditional_cmc": f"{row['conditional_cmc']:.4f}",
            "pipeline_cmc": f"{row['pipeline_cmc']:.4f}",
            "correct_conditional_pipeline": f"{split['correct']['conditional']:.3f} / {split['correct']['pipeline']:.3f}",
            "wrong_conditional_pipeline": f"{split['wrong']['conditional']:.3f} / {split['wrong']['pipeline']:.3f}",
            "lineage": row["source_generation_hardware_runtime"],
        })
    write_table_family(
        tables,
        "PRED_A_FINAL_SECONDARY_SUMMARY",
        pred_rows,
        "Historical predicted-conditioned source-only pipeline summary; not merged with primary RQ2.",
        "tab:pred-a-secondary",
    )
    write_tsv(secondary / "PRED_A_FINAL_SECONDARY_SUMMARY.tsv", pred_rows)

    # Figure 1: primary perturbation battery.
    fig1_rows = []
    x = np.arange(len(CONDITIONS))
    fig, ax = plt.subplots(figsize=(8.4, 5.2), constrained_layout=True)
    markers = ("o", "s", "^", "D")
    for index, arm in enumerate(ARMS):
        values = [float(primary_map[(arm, condition)]["mean_raw_drift"]) for condition in CONDITIONS]
        low = [float(primary_map[(arm, condition)]["ci_lower_95"]) for condition in CONDITIONS]
        high = [float(primary_map[(arm, condition)]["ci_upper_95"]) for condition in CONDITIONS]
        errors = np.vstack((np.asarray(values) - low, np.asarray(high) - values))
        ax.errorbar(x, values, yerr=errors, marker=markers[index], linewidth=1.8, capsize=3, label=ARM_LABELS[arm])
        for condition, value, lower, upper in zip(CONDITIONS, values, low, high):
            fig1_rows.append({"arm": arm, "condition": condition, "mean_raw_drift": value, "ci_lower_95": lower, "ci_upper_95": upper, "n": primary_map[(arm, condition)]["valid_pair_n"]})
    ax.set_xticks(x, [condition.title() for condition in CONDITIONS])
    ax.set_ylabel("Raw final-span BERTScore Drift")
    ax.set_xlabel("Image intervention")
    ax.set_ylim(0.04, 0.095)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.savefig(figures / "RQ2_FIGURE_1_DRIFT_BY_PERTURBATION.png", dpi=240)
    fig.savefig(figures / "RQ2_FIGURE_1_DRIFT_BY_PERTURBATION.pdf")
    plt.close(fig)
    write_tsv(figures / "RQ2_FIGURE_1_SOURCE_DATA.tsv", fig1_rows)

    # Figure 2: empirical anchor ladder.
    anchor_names = ("mirror", "grey", "mismatch", "deliberate_answer_control", "unrelated_rationale_reference")
    anchor_labels = ("Mirror", "Grey", "Mismatch", "Deliberate answer\n(n=200/arm)", "Unrelated rationale")
    anchor_map = {(row["arm"], row["anchor_or_condition"]): row for row in anchors}
    fig2_rows = []
    x = np.arange(len(anchor_names))
    fig, ax = plt.subplots(figsize=(9.0, 5.3), constrained_layout=True)
    for index, arm in enumerate(ARMS):
        values = [float(anchor_map[(arm, name)]["mean_raw_drift"]) for name in anchor_names]
        low = [float(anchor_map[(arm, name)]["ci_lower_95"]) for name in anchor_names]
        high = [float(anchor_map[(arm, name)]["ci_upper_95"]) for name in anchor_names]
        errors = np.vstack((np.asarray(values) - low, np.asarray(high) - values))
        ax.errorbar(x, values, yerr=errors, marker=markers[index], linewidth=1.8, capsize=3, label=ARM_LABELS[arm])
        for name, value, lower, upper in zip(anchor_names, values, low, high):
            fig2_rows.append({"arm": arm, "anchor": name, "mean_raw_drift": value, "ci_lower_95": lower, "ci_upper_95": upper, "n": anchor_map[(arm, name)]["n"], "sample_scope": anchor_map[(arm, name)]["sample_scope"]})
    ax.set_xticks(x, anchor_labels)
    ax.set_ylabel("Raw final-span BERTScore Drift")
    ax.set_xlabel("Empirical comparison point")
    ax.set_ylim(0.04, 0.135)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.savefig(figures / "RQ2_FIGURE_2_ANCHOR_LADDER.png", dpi=240)
    fig.savefig(figures / "RQ2_FIGURE_2_ANCHOR_LADDER.pdf")
    plt.close(fig)
    write_tsv(figures / "RQ2_FIGURE_2_SOURCE_DATA.tsv", fig2_rows)

    # Figure 3: the bounded diagnostic is interpretable and strongly arm-dependent.
    fig3_rows = []
    fig, ax = plt.subplots(figsize=(7.2, 4.9), constrained_layout=True)
    xpos = np.arange(2)
    width = 0.34
    for offset_index, (state, prefix, color) in enumerate((("Description ON", "description_d_on", "#4c78a8"), ("Description OFF", "description_d_off", "#f58518"))):
        values = []
        lows = []
        highs = []
        for arm in ("plain_desc", "point_desc"):
            b = bootstrap_map[f"{prefix}__{arm}"]
            values.append(float(b["point_estimate"]))
            lows.append(float(b["lower_95"]))
            highs.append(float(b["upper_95"]))
            fig3_rows.append({"arm": arm, "description_state": state, "mean_image_removal_drift": b["point_estimate"], "ci_lower_95": b["lower_95"], "ci_upper_95": b["upper_95"], "n": b["record_n"]})
        position = xpos + (offset_index - 0.5) * width
        errors = np.vstack((np.asarray(values) - lows, np.asarray(highs) - values))
        ax.bar(position, values, width=width, color=color, alpha=0.9, label=state)
        ax.errorbar(position, values, yerr=errors, fmt="none", color="black", capsize=3, linewidth=1.2)
    ax.set_xticks(xpos, ["Plain + description", "Point + description"])
    ax.set_ylabel("Source-to-grey raw BERTScore Drift")
    ax.set_ylim(0, 0.175)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(figures / "RQ2_FIGURE_3_DESCRIPTION_ABLATION.png", dpi=240)
    fig.savefig(figures / "RQ2_FIGURE_3_DESCRIPTION_ABLATION.pdf")
    plt.close(fig)
    write_tsv(figures / "RQ2_FIGURE_3_SOURCE_DATA.tsv", fig3_rows)

    print(json.dumps({"status": "PASS", "publication_table_families": 7, "figures": 3}, indent=2))


if __name__ == "__main__":
    main()
