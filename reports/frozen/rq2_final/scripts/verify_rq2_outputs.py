#!/usr/bin/env python3
"""Independent, fail-closed verification of the final RQ2 analysis package."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ARMS = ("plain", "point", "plain_desc", "point_desc")
CONDITIONS = ("grey", "mismatch", "mask", "noise", "mirror")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--reproduction-root", required=True, type=Path)
    args = parser.parse_args()
    root = args.root
    failures: list[str] = []
    checks: dict[str, object] = {}

    integrity = json.loads((root / "audits/RQ2_ALIGNMENT_AND_INTEGRITY_AUDIT.json").read_text())
    checks["input_integrity_pass"] = integrity["status"] == "PASS" and not integrity["failures"]
    checks["primary_n"] = integrity["families"]["stage2_primary"]["output_rows"]
    checks["known_change_n"] = integrity["families"]["stage2_known_change"]["output_rows"]
    checks["ablation_n"] = integrity["families"]["stage2_description_ablation"]["output_rows"]
    checks["global_unique_logical_keys"] = integrity["global_unique_logical_keys"]
    if checks["primary_n"] != 63672 or checks["known_change_n"] != 800 or checks["ablation_n"] != 800 or checks["global_unique_logical_keys"] != 65272:
        failures.append("family/global row-count audit failed")
    for arm in ARMS:
        alignment = integrity["primary_alignment"][arm]
        if alignment["status"] != "PASS" or alignment["intersection_n"] != 2653 or alignment["union_n"] != 2653:
            failures.append(f"primary alignment failed for {arm}")

    coverage = read_tsv(root / "audits/RQ2_PARSE_COVERAGE.tsv")
    primary_coverage = [row for row in coverage if row["family"] == "stage2_primary"]
    checks["primary_cells"] = len(primary_coverage)
    checks["primary_raw_total"] = sum(int(row["raw_generation_n"]) for row in primary_coverage)
    checks["primary_tolerant_final_total"] = sum(int(row["tolerant_final_success_n"]) for row in primary_coverage)
    checks["primary_pairable_perturbation_total"] = sum(int(row["source_relative_pairable_final_n"] or 0) for row in primary_coverage if row["condition"] != "source")
    if len(primary_coverage) != 24 or any(int(row["raw_generation_n"]) != 2653 for row in primary_coverage):
        failures.append("not every primary raw cell contains 2,653 rows")
    if checks["primary_raw_total"] != 63672 or checks["primary_tolerant_final_total"] != 63664 or checks["primary_pairable_perturbation_total"] != 53049:
        failures.append("coverage totals do not reconcile")

    scoring = json.loads((root / "provenance/RQ2_BERTSCORE_SCORING_METADATA.json").read_text())
    expected_config = {
        "rows_scored": 176154,
        "bert_score_version": "0.3.12",
        "model": "roberta-large",
        "model_snapshot": "722cf37b1afa9454edce342e7895e588b6ff1d59",
        "num_layers": 17,
        "idf": False,
        "primary_baseline_rescaling": False,
        "tokenizer_class": "RobertaTokenizer",
        "use_fast_tokenizer": False,
        "output_sha256": "9b76546d1d3ed21c4dd1ba1dfa3ade85dbb57495eae6f8836e7214cd30ceb065",
    }
    checks["bertscore_configuration_exact"] = all(scoring.get(key) == value for key, value in expected_config.items())
    if not checks["bertscore_configuration_exact"] or sha(root / "data/RQ2_BERTSCORE_SCORES.tsv.gz") != expected_config["output_sha256"]:
        failures.append("BERTScore configuration or score archive SHA mismatch")

    per_record = read_tsv(root / "data/RQ2_PRIMARY_DRIFT_PER_RECORD.tsv")
    checks["primary_per_record_rows"] = len(per_record)
    checks["primary_per_record_unique_keys"] = len({(row["arm"], row["condition"], row["record_id"]) for row in per_record})
    if len(per_record) != 53049 or checks["primary_per_record_unique_keys"] != 53049:
        failures.append("primary per-record score count/uniqueness failed")
    if any(not row["source_text_sha256"] or not row["condition_text_sha256"] for row in per_record):
        failures.append("per-record text hashes missing")

    provenance = read_tsv(root / "audits/RQ2_INPUT_PROVENANCE.tsv")
    primary_provenance = [row for row in provenance if row["family"] == "stage2_primary"]
    checks["primary_provenance_cells"] = len(primary_provenance)
    checks["primary_5090_only"] = len(primary_provenance) == 24 and all("5090" in row.get("hardware_runtime_lineage", "") for row in primary_provenance)
    if not checks["primary_5090_only"]:
        failures.append("primary generation provenance is not qualified RTX-5090-only")

    known = read_tsv(root / "tables/RQ2_KNOWN_CHANGE_RESULTS.tsv")
    checks["known_change_exact_200_each"] = len(known) == 4 and all(int(row["valid_paired_n"]) == 200 for row in known)
    if not checks["known_change_exact_200_each"]:
        failures.append("known-change matched n is not 200/arm")

    description = read_tsv(root / "tables/RQ2_DESCRIPTION_AVAILABILITY_RESULTS.tsv")
    checks["description_ablation_exact_200_each"] = len(description) == 2 and all(int(row["valid_paired_n"]) == 200 for row in description)
    checks["description_checkpoints_retained"] = all(integrity["ablation_description_checkpoint_retained"].values())
    if not checks["description_ablation_exact_200_each"] or not checks["description_checkpoints_retained"]:
        failures.append("description ablation n/checkpoint audit failed")

    job_audit = json.loads((root / "audits/RQ2_BERTSCORE_JOB_PREPARATION_AUDIT.json").read_text())
    checks["no_semantic_imputation"] = job_audit.get("no_semantic_imputation") is True
    checks["unrelated_map_valid"] = job_audit.get("unrelated_map_rows") == 10612 and job_audit.get("unrelated_map_failures") == []
    if not checks["no_semantic_imputation"] or not checks["unrelated_map_valid"]:
        failures.append("semantic-imputation or unrelated-map constraint audit failed")

    # Deterministic bootstrap/statistical rerun must reproduce all scientific TSVs byte-for-byte.
    deterministic_files = (
        "data/RQ2_PRIMARY_DRIFT_PER_RECORD.tsv",
        "tables/RQ2_PRIMARY_DRIFT_RESULTS.tsv",
        "tables/RQ2_CONDITION_ORDERING.tsv",
        "tables/RQ2_PLANNED_CONTRASTS.tsv",
        "tables/RQ2_KNOWN_CHANGE_RESULTS.tsv",
        "secondary/RQ2_KNOWN_CHANGE_ECHO_SENSITIVITY.tsv",
        "tables/RQ2_UNRELATED_REFERENCE.tsv",
        "secondary/RQ2_BERTSCORE_SCALE_DIAGNOSTIC.tsv",
        "secondary/RQ2_TEXT_UNIT_SENSITIVITY.tsv",
        "tables/RQ2_DESCRIPTION_AVAILABILITY_RESULTS.tsv",
        "tables/RQ2_ANCHOR_SCALE_SUMMARY.tsv",
        "audits/RQ2_BOOTSTRAP_RESULTS.tsv",
    )
    reproduction = {}
    for relative in deterministic_files:
        original = root / relative
        rerun = args.reproduction_root / relative
        match = rerun.exists() and sha(original) == sha(rerun)
        reproduction[relative] = {"original_sha256": sha(original), "rerun_sha256": sha(rerun) if rerun.exists() else None, "match": match}
        if not match:
            failures.append(f"deterministic rerun mismatch: {relative}")
    checks["bootstrap_seed_42_exact_rerun"] = all(item["match"] for item in reproduction.values())

    # Publication TSV and CSV must have identical parsed values.
    publication_stems = (
        "RQ2_TABLE_A_PARSE_COVERAGE",
        "RQ2_TABLE_B_PRIMARY_DRIFT_BATTERY",
        "RQ2_TABLE_C_CONDITION_CONTRASTS",
        "RQ2_TABLE_D_KNOWN_CHANGE_ANCHOR",
        "RQ2_TABLE_E_ANCHOR_SCALE_SUMMARY",
        "RQ2_TABLE_F_DESCRIPTION_AVAILABILITY",
        "PRED_A_FINAL_SECONDARY_SUMMARY",
    )
    publication_matches = {}
    for stem in publication_stems:
        match = read_tsv(root / f"tables/{stem}.tsv") == read_csv(root / f"tables/{stem}.csv")
        publication_matches[stem] = match
        if not match:
            failures.append(f"TSV/CSV publication mismatch: {stem}")
    checks["publication_tsv_csv_reconcile"] = all(publication_matches.values())

    report = (root / "RQ2_FINAL_ANALYSIS_REPORT.md").read_text(encoding="utf-8")
    required_report_tokens = (
        "0.0835 [0.0821, 0.0850]",
        "0.0890 [0.0875, 0.0906]",
        "0.0254 [0.0196, 0.0312]",
        "0.0790 [0.0726, 0.0853]",
        "RQ2 measures behavioural sensitivity",
        "does not reveal the model's internal causal reasoning",
    )
    checks["report_key_values_reconcile"] = all(token in report for token in required_report_tokens)
    checks["forbidden_absolute_language_absent"] = "rationales changed very little" not in report.lower()
    if not checks["report_key_values_reconcile"] or not checks["forbidden_absolute_language_absent"]:
        failures.append("report reconciliation or claim-boundary audit failed")

    audit = {
        "schema_version": "rq2-independent-self-audit-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if not failures else "FAIL",
        "checks": checks,
        "publication_matches": publication_matches,
        "deterministic_reproduction": reproduction,
        "failures": failures,
    }
    output = root / "audits/RQ2_SELF_AUDIT.json"
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": audit["status"], "failures": failures}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
