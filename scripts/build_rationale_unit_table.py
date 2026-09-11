#!/usr/bin/env python3
"""Emit the appendix table supporting the rationale-unit sensitivity claim.

Every value is copied from the frozen text-unit sensitivity record. Nothing is
rescored and no metric configuration changes.
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = ROOT / "provenance" / "selected_final_audits"
OUTPUT = ROOT / "thesis/Tables/final/rq2_text_unit_sensitivity.tex"
RQ2 = Path("reports/frozen/rq2_final")
UNITS = RQ2 / "secondary/RQ2_TEXT_UNIT_SENSITIVITY.tsv"
ECHO = RQ2 / "secondary/RQ2_KNOWN_CHANGE_ECHO_SENSITIVITY.tsv"

ARMS = ("plain", "point", "plain_desc", "point_desc")
CONDITIONS = ("grey", "mismatch", "mask", "noise", "mirror")
UNIT_KEYS = ("tolerant_final_primary", "tolerant_reasoning_sensitivity", "raw_full_generation_sensitivity")
REPRESENTATIONS = (
    ("final_span", "final span"),
    ("reasoning_only", "reasoning span"),
    ("answer_choice_masked_final", "final span, answer choices masked"),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def arm_label(arm: str) -> str:
    return r"\texttt{" + arm.replace("_", r"\_") + "}"


def main() -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    units = {(r["arm"], r["text_unit"], r["condition"]): r
             for r in csv.DictReader(UNITS.open(encoding="utf-8"), delimiter="\t")}
    echo = {(r["arm"], r["representation"]): r
            for r in csv.DictReader(ECHO.open(encoding="utf-8"), delimiter="\t")}

    out = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption[Drift by rationale text unit]{Mean raw BERTScore Drift for the same "
        r"pairs measured on three text units: the final span used for the primary analysis, "
        r"the reasoning span, and the complete generation. Pairable $n$ is the count for "
        r"the final span; the sensitivity units use the same records.}",
        r"\label{tab:app-rq2-text-unit}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Arm & Image condition & $n$ & Final span & Reasoning span & Full generation \\",
        r"\midrule",
    ]
    rows = []
    for arm in ARMS:
        for position, condition in enumerate(CONDITIONS):
            primary = units[(arm, "tolerant_final_primary", condition)]
            values = [float(units[(arm, unit, condition)]["mean_raw_drift"]) for unit in UNIT_KEYS]
            out.append(" & ".join([
                arm_label(arm) if position == 0 else "",
                condition,
                f"{int(primary['valid_pair_n']):,}",
                *[f"{v:.4f}" for v in values],
            ]) + r" \\")
            rows.append({"arm": arm, "condition": condition,
                         "n": int(primary["valid_pair_n"]),
                         "final": round(values[0], 4), "reasoning": round(values[1], 4),
                         "full": round(values[2], 4)})
    out.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])

    out.extend([
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption[Wrong-answer comparison by text unit]{The wrong-answer comparison on "
        r"the fixed 200-record sample, measured on three text representations. The difference is "
        r"Drift from the wrong answer minus matched grey Drift, with a 95\% confidence "
        r"interval from a bootstrap clustered by source frame.}",
        r"\label{tab:app-rq2-echo-unit}",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Arm & Representation & Grey Drift & Wrong-answer Drift & Difference [95\% CI] \\",
        r"\midrule",
    ])
    echo_rows = []
    for arm in ARMS:
        for position, (key, label) in enumerate(REPRESENTATIONS):
            r = echo[(arm, key)]
            diff = float(r["answer_minus_image_mean_difference"])
            lo, hi = float(r["difference_ci_lower_95"]), float(r["difference_ci_upper_95"])
            out.append(" & ".join([
                arm_label(arm) if position == 0 else "",
                label,
                f"{float(r['image_removal_mean_drift']):.4f}",
                f"{float(r['alternate_answer_mean_drift']):.4f}",
                f"{diff:+.4f} [{lo:.4f}, {hi:.4f}]",
            ]) + r" \\")
            echo_rows.append({"arm": arm, "representation": key, "difference": round(diff, 4),
                              "ci": [round(lo, 4), round(hi, 4)]})
    out.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    OUTPUT.write_text("\n".join(out), encoding="utf-8")

    checks = [
        {"check": "twenty text-unit rows", "status": "PASS" if len(rows) == 20 else "FAIL", "evidence": len(rows)},
        {"check": "twelve wrong-answer rows", "status": "PASS" if len(echo_rows) == 12 else "FAIL", "evidence": len(echo_rows)},
        {"check": "final-span means match the primary Drift table",
         "status": "PASS" if all(
             f"{r['final']:.4f}" in (ROOT / "thesis/Tables/final/rq2_drift.tex").read_text(encoding="utf-8")
             for r in rows) else "FAIL", "evidence": None},
        {"check": "no local type reduction", "status": "PASS" if not any(
            t in "\n".join(out) for t in (r"\scriptsize", r"\tiny", r"\resizebox")) else "FAIL", "evidence": None},
    ]
    audit = {
        "schema": "v11-rationale-unit-table-audit-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "sources": [{"path": str(UNITS), "sha256": sha256(UNITS)}, {"path": str(ECHO), "sha256": sha256(ECHO)}],
        "output": {"path": str(OUTPUT.relative_to(ROOT)), "sha256": sha256(OUTPUT)},
        "rows": rows, "deliberate_answer_rows": echo_rows, "checks": checks,
    }
    audit["status"] = "PASS" if all(c["status"] == "PASS" for c in checks) else "FAIL"
    (AUDIT_DIR / "RATIONALE_UNIT_TABLE_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": audit["status"], "failed": [c["check"] for c in checks if c["status"] != "PASS"]}, indent=2))
    if audit["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
