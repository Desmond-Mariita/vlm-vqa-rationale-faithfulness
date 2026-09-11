#!/usr/bin/env python3
"""Independent consistency and deterministic-rerun audit for the RQ1 package."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("reports/frozen/rq1_final")
RUN = Path("data/external/execution/v11-final-2653-20260830-f8349fa0")
ARMS = ["plain", "point", "plain_desc", "point_desc"]
CONDITIONS = ["source", "grey", "mismatch", "mask", "noise", "mirror"]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rows(path: Path, delimiter="\t"):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter=delimiter))


def main() -> int:
    deterministic = sorted((ROOT / "data").glob("*.tsv")) + sorted((ROOT / "tables").glob("*")) + [
        ROOT / "audits/RQ1_ALIGNMENT_AUDIT.json",
        ROOT / "audits/RQ1_TRANSITION_IDENTITY_AUDIT.json",
        ROOT / "audits/RQ1_BOOTSTRAP_REPRODUCIBILITY_AUDIT.json",
        ROOT / "audits/RQ1_POINT_DESC_MASK_NOISE_ANOMALY.json",
        ROOT / "provenance/RQ1_INPUT_PROVENANCE.tsv",
    ]
    before = {str(p.relative_to(ROOT)): sha(p) for p in deterministic}
    subprocess.run(["python", str(ROOT / "scripts/rq1_analysis.py")], check=True, stdout=subprocess.DEVNULL)
    after = {str(p.relative_to(ROOT)): sha(p) for p in deterministic}

    cell = rows(ROOT / "data/RQ1_CELL_RESULTS.tsv")
    gap = rows(ROOT / "data/RQ1_GAP_RESULTS.tsv")
    trans = rows(ROOT / "data/RQ1_TRANSITION_RESULTS.tsv")
    contrast = rows(ROOT / "data/RQ1_PLANNED_CONTRASTS.tsv")
    records = rows(ROOT / "data/RQ1_FINAL_RECORD_RESULTS.tsv")
    provenance = rows(ROOT / "provenance/RQ1_INPUT_PROVENANCE.tsv")
    by_cell = {(r["arm"], r["condition"]): r for r in cell}
    by_gap = {(r["arm"], r["condition"]): r for r in gap}

    checks = {}
    checks["every_arm_condition_N_2653"] = len(cell) == 24 and all(int(r["N"]) == 2653 for r in cell)
    checks["every_arm_condition_frame_N_2400"] = all(int(r["source_frames"]) == 2400 for r in cell)
    checks["correct_over_N_equals_accuracy"] = all(
        abs(int(r["correct_count"]) / int(r["N"]) - float(r["accuracy"])) < 5e-13 and
        int(r["correct_count"]) + int(r["wrong_count"]) == int(r["N"])
        for r in cell
    )
    checks["transition_cells_sum_to_N"] = len(trans) == 20 and all(
        sum(int(r[k]) for k in ["C_to_C", "C_to_W", "W_to_C", "W_to_W"]) == int(r["N"])
        for r in trans
    )
    checks["transition_identity_20_of_20"] = all(
        int(r["source_correct"]) - round(float(r["condition_accuracy"]) * int(r["N"]))
        == int(r["C_to_W"]) - int(r["W_to_C"]) and r["identity_pass"] == "True"
        for r in trans
    )
    checks["source_gap_reconciles"] = all(
        abs(float(r["source_accuracy"]) - float(r["condition_accuracy"]) - float(r["source_minus_condition_gap"])) < 2e-12
        for r in gap
    )
    checks["sufficiency_gap_is_source_minus_grey"] = all(
        abs(
            float(next(r for r in contrast if r["contrast_family"] == "description_vs_no_description_sufficiency_gap" and r["description_arm"] == arm)["description_sufficiency_gap"])
            - (float(by_cell[(arm, "source")]["accuracy"]) - float(by_cell[(arm, "grey")]["accuracy"]))
        ) < 2e-12
        for arm in ["plain_desc", "point_desc"]
    ) and all(
        abs(float(by_gap[(arm, "grey")]["source_minus_condition_gap"]) - (float(by_cell[(arm, "source")]["accuracy"]) - float(by_cell[(arm, "grey")]["accuracy"]))) < 2e-12
        for arm in ARMS
    )
    checks["bootstrap_seed_42_exact_rerun"] = before == after and json.loads((ROOT / "audits/RQ1_BOOTSTRAP_REPRODUCIBILITY_AUDIT.json").read_text())["exact_second_run_array_equality"]

    expected_mismatch = {}
    for arm in ARMS:
        p = RUN / f"shards/stage1_wrong/{arm}/worker_0/records.jsonl"
        expected_mismatch[arm] = {json.loads(line)["record_id"]: json.loads(line)["prediction"] for line in p.read_text().splitlines() if line.strip()}
    checks["final_six_condition_table_uses_Option_B_only"] = all(
        int(r["pred_mismatch"]) == int(expected_mismatch[r["arm"]][r["record_id"]]) for r in records
    ) and all(
        r["status"] == ("FINAL_OPTION_B" if r["condition"] == "mismatch" else "FINAL_RETAINED_HISTORICAL") for r in provenance
    )
    checks["historical_mismatch_only_in_supersession_material"] = "pred_wrong" not in records[0] and len(rows(ROOT / "data/RQ1_MISMATCH_SUPERSESSION_COMPARISON.tsv")) == 4

    freeze = json.loads((ROOT / "qualitative/RQ1_W2C_SAMPLE_FREEZE.json").read_text())
    freeze_time = datetime.fromisoformat(freeze["freeze_utc"]).timestamp()
    material_mtime = (ROOT / "qualitative/RQ1_W2C_SELECTED_MATERIAL.tsv").stat().st_mtime
    checks["qualitative_mismatch_sample_hashed_before_content_inspection"] = (
        freeze["status"] == "FROZEN_BEFORE_SUBJECTIVE_INSPECTION"
        and freeze["final_sample_sha256"] == sha(ROOT / "qualitative/RQ1_W2C_FINAL_SAMPLE.tsv")
        and freeze["eligible_frame_sha256"] == sha(ROOT / "qualitative/RQ1_W2C_MISMATCH_ELIGIBLE_FRAME.tsv")
        and freeze_time < material_mtime
        and not freeze["subjective_content_inspected_before_freeze"]
    )

    table_a = rows(ROOT / "tables/RQ1_TABLE_A_FINAL_ACCURACY_BATTERY.tsv")
    a_ok = True
    for r in table_a:
        for condition in CONDITIONS:
            c = by_cell[(r["arm"], condition)]
            a_ok &= all(r[f"{condition}_{field}"] == c[source] for field, source in [
                ("accuracy", "accuracy"), ("N", "N"), ("lower_95%", "lower_95%"), ("upper_95%", "upper_95%")
            ])
    b_tsv = rows(ROOT / "tables/RQ1_TABLE_B_SOURCE_RELATIVE_GAPS.tsv")
    c_tsv = rows(ROOT / "tables/RQ1_TABLE_C_TRANSITION_DECOMPOSITION.tsv")
    e_tsv = rows(ROOT / "tables/RQ1_TABLE_E_MISMATCH_SUPERSESSION.tsv")
    d_tsv = rows(ROOT / "tables/RQ1_TABLE_D_SUFFICIENCY_GAP_CONTRASTS.tsv")
    sg = [r for r in contrast if r["contrast_family"] == "description_vs_no_description_sufficiency_gap"]
    checks["all_table_values_reconcile_to_machine_results"] = (
        a_ok and b_tsv == gap and c_tsv == trans and d_tsv == sg
        and e_tsv == rows(ROOT / "data/RQ1_MISMATCH_SUPERSESSION_COMPARISON.tsv")
        and rows(ROOT / "tables/RQ1_TABLE_A_FINAL_ACCURACY_BATTERY.csv", ",") == table_a
        and rows(ROOT / "tables/RQ1_TABLE_B_SOURCE_RELATIVE_GAPS.csv", ",") == b_tsv
        and rows(ROOT / "tables/RQ1_TABLE_C_TRANSITION_DECOMPOSITION.csv", ",") == c_tsv
        and rows(ROOT / "tables/RQ1_TABLE_D_SUFFICIENCY_GAP_CONTRASTS.csv", ",") == d_tsv
        and rows(ROOT / "tables/RQ1_TABLE_E_MISMATCH_SUPERSESSION.csv", ",") == e_tsv
    )
    checks["point_desc_mask_noise_anomaly_resolved"] = (
        json.loads((ROOT / "audits/RQ1_POINT_DESC_MASK_NOISE_ANOMALY.json").read_text())["prediction_difference_count"] == 506
    )
    checks["all_checks_pass"] = all(checks.values())
    payload = {
        "audit_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "deterministic_output_hashes_before": before,
        "deterministic_output_hashes_after": after,
        "deterministic_hash_maps_identical": before == after,
        "status": "PASS" if checks["all_checks_pass"] else "FAIL",
    }
    out = ROOT / "audits/RQ1_SELF_AUDIT.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if checks["all_checks_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
