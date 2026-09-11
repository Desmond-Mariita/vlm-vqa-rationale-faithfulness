#!/usr/bin/env python3
"""Compute frozen estimate-led RQ2 summaries and clustered bootstrap intervals."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ARMS = ("plain", "point", "plain_desc", "point_desc")
PERTURBATIONS = ("grey", "mismatch", "mask", "noise", "mirror")
BOOTSTRAP_SEED = 42
BOOTSTRAP_REPLICATES = 10_000


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"refusing empty table: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n", extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def cluster_ci(values: np.ndarray, clusters: list[str]) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    labels, inverse = np.unique(np.asarray(clusters, dtype=object), return_inverse=True)
    sums = np.bincount(inverse, weights=values, minlength=len(labels))
    counts = np.bincount(inverse, minlength=len(labels))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    chunk = 250
    for start in range(0, BOOTSTRAP_REPLICATES, chunk):
        stop = min(start + chunk, BOOTSTRAP_REPLICATES)
        picks = rng.integers(0, len(labels), size=(stop - start, len(labels)))
        bootstrap[start:stop] = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    low, high = np.percentile(bootstrap, [2.5, 97.5])
    return float(low), float(high)


def stats(values: list[float], clusters: list[str]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    low, high = cluster_ci(array, clusters)
    return {
        "n": int(array.size),
        "source_frame_n": len(set(clusters)),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "sd": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
        "ci_lower": low,
        "ci_upper": high,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True, type=Path)
    parser.add_argument("--scoring-metadata", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--masking-counts", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    for directory in ("tables", "data", "audits", "secondary"):
        (args.out / directory).mkdir(parents=True, exist_ok=True)

    scoring_metadata = json.loads(args.scoring_metadata.read_text(encoding="utf-8"))
    if scoring_metadata.get("output_sha256") != sha_file(args.scores):
        raise SystemExit("score file SHA does not match scoring metadata")
    required_config = {
        "rows_scored": 176154,
        "bert_score_version": "0.3.12",
        "model_snapshot": "722cf37b1afa9454edce342e7895e588b6ff1d59",
        "num_layers": 17,
        "idf": False,
        "primary_baseline_rescaling": False,
        "tokenizer_class": "RobertaTokenizer",
        "use_fast_tokenizer": False,
    }
    config_failures = [f"{key}: {scoring_metadata.get(key)!r} != {expected!r}" for key, expected in required_config.items() if scoring_metadata.get(key) != expected]
    if config_failures:
        raise SystemExit("; ".join(config_failures))

    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    dataset_order = {row["image_id"]: index for index, row in enumerate(dataset)}
    rows = read_tsv(args.scores)
    if len(rows) != 176154 or len({int(row["job_index"]) for row in rows}) != len(rows):
        raise SystemExit("score row count/job-index uniqueness failure")

    groups: dict[tuple[str, str, str], dict[str, dict]] = defaultdict(dict)
    for row in rows:
        key = (row["analysis"], row["arm"], row["condition"])
        rid = row["record_id"]
        if rid in groups[key]:
            raise SystemExit(f"duplicate score record in group: {key}/{rid}")
        groups[key][rid] = row

    bootstrap_rows: list[dict] = []

    def score_values(analysis: str, arm: str, condition: str, field: str = "raw_drift") -> tuple[list[str], list[float], list[str]]:
        mapping = groups[(analysis, arm, condition)]
        ids = sorted(mapping, key=lambda rid: dataset_order[rid])
        return ids, [float(mapping[rid][field]) for rid in ids], [mapping[rid]["source_frame_id"] for rid in ids]

    def add_bootstrap(estimate_id: str, estimate: str, arms: str, conditions: str, values: list[float], clusters: list[str]) -> dict:
        result = stats(values, clusters)
        bootstrap_rows.append({
            "estimate_id": estimate_id,
            "estimate": estimate,
            "arms": arms,
            "conditions": conditions,
            "record_n": result["n"],
            "source_frame_n": result["source_frame_n"],
            "point_estimate": result["mean"],
            "lower_95": result["ci_lower"],
            "upper_95": result["ci_upper"],
            "seed": BOOTSTRAP_SEED,
            "replicates": BOOTSTRAP_REPLICATES,
        })
        return result

    # Primary per-record results and arm-condition summaries.
    per_record: list[dict] = []
    primary_results: list[dict] = []
    primary_stats: dict[tuple[str, str], dict] = {}
    for arm in ARMS:
        for condition in PERTURBATIONS:
            mapping = groups[("primary_final", arm, condition)]
            ids = sorted(mapping, key=lambda rid: dataset_order[rid])
            values = [float(mapping[rid]["raw_drift"]) for rid in ids]
            clusters = [mapping[rid]["source_frame_id"] for rid in ids]
            result = add_bootstrap(
                f"primary_mean__{arm}__{condition}", "mean_raw_final_span_drift", arm, f"source->{condition}", values, clusters
            )
            primary_stats[(arm, condition)] = result
            primary_results.append({
                "arm": arm,
                "condition": condition,
                "valid_pair_n": result["n"],
                "source_frame_n": result["source_frame_n"],
                "mean_raw_drift": result["mean"],
                "median_raw_drift": result["median"],
                "sd_raw_drift": result["sd"],
                "q25_raw_drift": result["q25"],
                "q75_raw_drift": result["q75"],
                "ci_lower_95": result["ci_lower"],
                "ci_upper_95": result["ci_upper"],
                "bootstrap_seed": BOOTSTRAP_SEED,
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            })
            for rid in ids:
                row = mapping[rid]
                per_record.append({
                    "arm": arm,
                    "condition": condition,
                    "record_id": rid,
                    "source_frame_id": row["source_frame_id"],
                    "source_text_sha256": row["source_text_sha256"],
                    "condition_text_sha256": row["condition_text_sha256"],
                    "bertscore_precision": row["bertscore_precision"],
                    "bertscore_recall": row["bertscore_recall"],
                    "bertscore_f1": row["bertscore_f1"],
                    "raw_drift": row["raw_drift"],
                })

    condition_ordering: list[dict] = []
    rank_by_arm: dict[str, dict[str, int]] = {}
    for arm in ARMS:
        ordered = sorted(PERTURBATIONS, key=lambda condition: primary_stats[(arm, condition)]["mean"])
        rank_by_arm[arm] = {condition: rank + 1 for rank, condition in enumerate(ordered)}
        condition_ordering.append({
            "arm": arm,
            "lowest_to_highest_mean_drift": " < ".join(ordered),
            "highest_to_lowest_mean_drift": " > ".join(reversed(ordered)),
        })
    for row in primary_results:
        row["ascending_drift_rank"] = rank_by_arm[row["arm"]][row["condition"]]

    # Frozen within-arm and cross-arm planned contrasts.
    contrast_specs = []
    for arm in ARMS:
        contrast_specs.extend((
            (f"grey_minus_mirror__{arm}", arm, "grey", arm, "mirror", "grey - mirror", "planned"),
            (f"mismatch_minus_mirror__{arm}", arm, "mismatch", arm, "mirror", "mismatch - mirror", "planned"),
            (f"grey_minus_mismatch__{arm}", arm, "grey", arm, "mismatch", "grey - mismatch", "descriptive planned"),
            (f"mask_minus_noise__{arm}", arm, "mask", arm, "noise", "mask - noise", "descriptive planned"),
        ))
    contrast_specs.extend((
        ("plain_desc_minus_plain__grey", "plain_desc", "grey", "plain", "grey", "plain_desc - plain", "planned"),
        ("point_desc_minus_point__grey", "point_desc", "grey", "point", "grey", "point_desc - point", "planned"),
    ))
    contrasts: list[dict] = []
    for contrast_id, left_arm, left_condition, right_arm, right_condition, label, status in contrast_specs:
        left = groups[("primary_final", left_arm, left_condition)]
        right = groups[("primary_final", right_arm, right_condition)]
        ids = sorted(set(left) & set(right), key=lambda rid: dataset_order[rid])
        values = [float(left[rid]["raw_drift"]) - float(right[rid]["raw_drift"]) for rid in ids]
        clusters = [left[rid]["source_frame_id"] for rid in ids]
        if any(left[rid]["source_frame_id"] != right[rid]["source_frame_id"] for rid in ids):
            raise SystemExit(f"source-frame mismatch in contrast {contrast_id}")
        result = add_bootstrap(contrast_id, "paired_raw_final_span_drift_difference", f"{left_arm};{right_arm}", f"{left_condition};{right_condition}", values, clusters)
        contrasts.append({
            "contrast_id": contrast_id,
            "contrast": label,
            "status": status,
            "left_arm": left_arm,
            "left_condition": left_condition,
            "left_mean_drift": primary_stats[(left_arm, left_condition)]["mean"],
            "right_arm": right_arm,
            "right_condition": right_condition,
            "right_mean_drift": primary_stats[(right_arm, right_condition)]["mean"],
            "paired_n": result["n"],
            "source_frame_n": result["source_frame_n"],
            "mean_difference": result["mean"],
            "ci_lower_95": result["ci_lower"],
            "ci_upper_95": result["ci_upper"],
        })

    # Same-lineage known-change and lexical-echo sensitivities.
    masking = {row["arm"]: row for row in read_tsv(args.masking_counts)}
    known_results: list[dict] = []
    echo_results: list[dict] = []
    known_stats: dict[str, dict[str, dict]] = defaultdict(dict)
    representations = (
        ("final_span", "known_image_final", "known_answer_final"),
        ("reasoning_only", "known_image_reasoning", "known_answer_reasoning"),
        ("answer_choice_masked_final", "known_image_masked_final", "known_answer_masked_final"),
    )
    for arm in ARMS:
        for representation, image_analysis, answer_analysis in representations:
            image = groups[(image_analysis, arm, "grey")]
            answer = groups[(answer_analysis, arm, "alternate_answer")]
            ids = sorted(set(image) & set(answer), key=lambda rid: dataset_order[rid])
            image_values = [float(image[rid]["raw_drift"]) for rid in ids]
            answer_values = [float(answer[rid]["raw_drift"]) for rid in ids]
            difference = [answer_value - image_value for answer_value, image_value in zip(answer_values, image_values)]
            clusters = [image[rid]["source_frame_id"] for rid in ids]
            image_summary = stats(image_values, clusters)
            answer_summary = stats(answer_values, clusters)
            delta_summary = add_bootstrap(
                f"known_answer_minus_image__{arm}__{representation}", "paired_known_change_difference", arm, "alternate-answer minus grey", difference, clusters
            )
            known_stats[arm][representation] = {"image": image_summary, "answer": answer_summary, "delta": delta_summary}
            common = {
                "arm": arm,
                "representation": representation,
                "valid_paired_n": delta_summary["n"],
                "source_frame_n": delta_summary["source_frame_n"],
                "image_removal_mean_drift": image_summary["mean"],
                "image_removal_median_drift": image_summary["median"],
                "alternate_answer_mean_drift": answer_summary["mean"],
                "alternate_answer_median_drift": answer_summary["median"],
                "answer_minus_image_mean_difference": delta_summary["mean"],
                "difference_ci_lower_95": delta_summary["ci_lower"],
                "difference_ci_upper_95": delta_summary["ci_upper"],
                "direction_answer_greater_than_image": delta_summary["mean"] > 0,
            }
            if representation == "final_span":
                known_results.append({key: value for key, value in common.items() if key != "representation"})
            echo_results.append(common | {
                "source_rationales_with_choice_match": masking[arm]["source_rationales_with_choice_match"],
                "grey_rationales_with_choice_match": masking[arm]["grey_rationales_with_choice_match"],
                "alternate_rationales_with_choice_match": masking[arm]["alternate_rationales_with_choice_match"],
                "records_with_any_choice_match": masking[arm]["records_with_any_choice_match"],
                "total_choice_string_occurrences": masking[arm]["total_choice_string_occurrences"],
            })

    # Frozen unrelated reference.
    unrelated_results: list[dict] = []
    unrelated_stats: dict[str, dict] = {}
    for arm in ARMS:
        ids, values, clusters = score_values("unrelated_final", arm, "unrelated")
        result = add_bootstrap(f"unrelated_mean__{arm}", "mean_unrelated_final_span_drift", arm, "source->unrelated source", values, clusters)
        unrelated_stats[arm] = result
        unrelated_results.append({
            "arm": arm,
            "valid_pair_n": result["n"],
            "source_frame_n": result["source_frame_n"],
            "mean_raw_drift": result["mean"],
            "median_raw_drift": result["median"],
            "sd_raw_drift": result["sd"],
            "ci_lower_95": result["ci_lower"],
            "ci_upper_95": result["ci_upper"],
            "reference_role": "empirical scale reference; not a ceiling, threshold, or null distribution",
        })

    # Description-availability diagnostic.
    description_results: list[dict] = []
    for arm in ("plain_desc", "point_desc"):
        on = groups[("description_on_image", arm, "grey_description_on")]
        off = groups[("description_off_image", arm, "grey_description_off")]
        ids = sorted(set(on) & set(off), key=lambda rid: dataset_order[rid])
        on_values = [float(on[rid]["raw_drift"]) for rid in ids]
        off_values = [float(off[rid]["raw_drift"]) for rid in ids]
        delta = [off_value - on_value for off_value, on_value in zip(off_values, on_values)]
        clusters = [on[rid]["source_frame_id"] for rid in ids]
        on_summary = add_bootstrap(f"description_d_on__{arm}", "mean_image_removal_drift_description_on", arm, "source ON->grey ON", on_values, clusters)
        off_summary = add_bootstrap(f"description_d_off__{arm}", "mean_image_removal_drift_description_off", arm, "source OFF->grey OFF", off_values, clusters)
        delta_summary = add_bootstrap(f"description_delta__{arm}", "paired_D_off_minus_D_on", arm, "description OFF minus ON", delta, clusters)
        source_caption_ids, source_caption_values, source_caption_clusters = score_values("description_source_on_off", arm, "source_on_vs_off")
        grey_caption_ids, grey_caption_values, grey_caption_clusters = score_values("description_grey_on_off", arm, "grey_on_vs_off")
        source_caption = stats(source_caption_values, source_caption_clusters)
        grey_caption = stats(grey_caption_values, grey_caption_clusters)
        description_results.append({
            "arm": arm,
            "valid_paired_n": delta_summary["n"],
            "source_frame_n": delta_summary["source_frame_n"],
            "D_on_mean": on_summary["mean"],
            "D_on_median": on_summary["median"],
            "D_off_mean": off_summary["mean"],
            "D_off_median": off_summary["median"],
            "Delta_description_D_off_minus_D_on": delta_summary["mean"],
            "delta_ci_lower_95": delta_summary["ci_lower"],
            "delta_ci_upper_95": delta_summary["ci_upper"],
            "source_caption_removal_valid_n": len(source_caption_ids),
            "source_caption_removal_mean_drift": source_caption["mean"],
            "grey_caption_removal_valid_n": len(grey_caption_ids),
            "grey_caption_removal_mean_drift": grey_caption["mean"],
            "interpretation_boundary": "input ablation on a description-trained checkpoint; not equivalent to plain/point",
        })

    # Scale diagnostic and raw-generation/reasoning sensitivities.
    scale_rows: list[dict] = []
    for arm in ARMS:
        for condition in PERTURBATIONS:
            mapping = groups[("primary_final", arm, condition)]
            ids = sorted(mapping, key=lambda rid: dataset_order[rid])
            raw = [float(mapping[rid]["raw_drift"]) for rid in ids]
            rescaled = [float(mapping[rid]["baseline_rescaled_drift"]) for rid in ids]
            scale_rows.append({
                "analysis": "primary_final",
                "arm": arm,
                "condition": condition,
                "n": len(ids),
                "raw_mean_drift": float(np.mean(raw)),
                "baseline_rescaled_mean_drift": float(np.mean(rescaled)),
                "raw_sign": "positive" if np.mean(raw) > 0 else "zero_or_negative",
                "rescaled_sign": "positive" if np.mean(rescaled) > 0 else "zero_or_negative",
                "ordering_preserved_by_monotonic_rescaling": True,
                "role": "sensitivity/scale diagnostic only",
            })
        for analysis, condition, label in (
            ("known_answer_final", "alternate_answer", "deliberate_answer_control"),
            ("unrelated_final", "unrelated", "unrelated_reference"),
        ):
            ids, raw, _clusters = score_values(analysis, arm, condition)
            mapping = groups[(analysis, arm, condition)]
            rescaled = [float(mapping[rid]["baseline_rescaled_drift"]) for rid in ids]
            scale_rows.append({
                "analysis": label,
                "arm": arm,
                "condition": condition,
                "n": len(ids),
                "raw_mean_drift": float(np.mean(raw)),
                "baseline_rescaled_mean_drift": float(np.mean(rescaled)),
                "raw_sign": "positive" if np.mean(raw) > 0 else "zero_or_negative",
                "rescaled_sign": "positive" if np.mean(rescaled) > 0 else "zero_or_negative",
                "ordering_preserved_by_monotonic_rescaling": True,
                "role": "sensitivity/scale diagnostic only",
            })

    unit_rows: list[dict] = []
    unit_means: dict[tuple[str, str, str], float] = {}
    for arm in ARMS:
        for unit_label, analysis in (
            ("tolerant_final_primary", "primary_final"),
            ("tolerant_reasoning_sensitivity", "primary_reasoning"),
            ("raw_full_generation_sensitivity", "primary_raw_generation"),
        ):
            condition_summaries = []
            for condition in PERTURBATIONS:
                ids, values, clusters = score_values(analysis, arm, condition)
                result = stats(values, clusters)
                unit_means[(arm, unit_label, condition)] = result["mean"]
                condition_summaries.append((condition, result))
            ordered = [condition for condition, _result in sorted(condition_summaries, key=lambda item: item[1]["mean"])]
            primary_order = [condition for condition in sorted(PERTURBATIONS, key=lambda value: primary_stats[(arm, value)]["mean"])]
            for condition, result in condition_summaries:
                unit_rows.append({
                    "arm": arm,
                    "text_unit": unit_label,
                    "condition": condition,
                    "valid_pair_n": result["n"],
                    "source_frame_n": result["source_frame_n"],
                    "mean_raw_drift": result["mean"],
                    "median_raw_drift": result["median"],
                    "ascending_rank_within_unit": ordered.index(condition) + 1,
                    "exact_order_matches_final_primary": ordered == primary_order,
                    "unit_order_low_to_high": " < ".join(ordered),
                    "primary_order_low_to_high": " < ".join(primary_order),
                })

    # Compact anchor ladder with explicit unlike-N labels.
    anchor_rows: list[dict] = []
    for arm in ARMS:
        anchor_rows.append({
            "arm": arm, "anchor_or_condition": "identity", "n": "definition", "source_frame_n": "definition",
            "mean_raw_drift": 0.0, "ci_lower_95": "", "ci_upper_95": "", "sample_scope": "identity by definition",
        })
        for condition in PERTURBATIONS:
            result = primary_stats[(arm, condition)]
            anchor_rows.append({
                "arm": arm, "anchor_or_condition": condition, "n": result["n"], "source_frame_n": result["source_frame_n"],
                "mean_raw_drift": result["mean"], "ci_lower_95": result["ci_lower"], "ci_upper_95": result["ci_upper"],
                "sample_scope": "full primary population",
            })
        answer = known_stats[arm]["final_span"]["answer"]
        anchor_rows.append({
            "arm": arm, "anchor_or_condition": "deliberate_answer_control", "n": answer["n"], "source_frame_n": answer["source_frame_n"],
            "mean_raw_drift": answer["mean"], "ci_lower_95": answer["ci_lower"], "ci_upper_95": answer["ci_upper"],
            "sample_scope": "fixed matched 200-record metric anchor",
        })
        unrelated_result = unrelated_stats[arm]
        anchor_rows.append({
            "arm": arm, "anchor_or_condition": "unrelated_rationale_reference", "n": unrelated_result["n"], "source_frame_n": unrelated_result["source_frame_n"],
            "mean_raw_drift": unrelated_result["mean"], "ci_lower_95": unrelated_result["ci_lower"], "ci_upper_95": unrelated_result["ci_upper"],
            "sample_scope": "frozen derangement empirical reference",
        })

    write_tsv(args.out / "data/RQ2_PRIMARY_DRIFT_PER_RECORD.tsv", per_record)
    write_tsv(args.out / "tables/RQ2_PRIMARY_DRIFT_RESULTS.tsv", primary_results)
    write_tsv(args.out / "tables/RQ2_CONDITION_ORDERING.tsv", condition_ordering)
    write_tsv(args.out / "tables/RQ2_PLANNED_CONTRASTS.tsv", contrasts)
    write_tsv(args.out / "tables/RQ2_KNOWN_CHANGE_RESULTS.tsv", known_results)
    write_tsv(args.out / "secondary/RQ2_KNOWN_CHANGE_ECHO_SENSITIVITY.tsv", echo_results)
    write_tsv(args.out / "tables/RQ2_UNRELATED_REFERENCE.tsv", unrelated_results)
    write_tsv(args.out / "secondary/RQ2_BERTSCORE_SCALE_DIAGNOSTIC.tsv", scale_rows)
    write_tsv(args.out / "secondary/RQ2_TEXT_UNIT_SENSITIVITY.tsv", unit_rows)
    write_tsv(args.out / "tables/RQ2_DESCRIPTION_AVAILABILITY_RESULTS.tsv", description_results)
    write_tsv(args.out / "tables/RQ2_ANCHOR_SCALE_SUMMARY.tsv", anchor_rows)
    write_tsv(args.out / "audits/RQ2_BOOTSTRAP_RESULTS.tsv", bootstrap_rows)

    audit = {
        "schema_version": "rq2-statistical-computation-audit-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "scores_path": str(args.scores),
        "scores_sha256": sha_file(args.scores),
        "score_rows": len(rows),
        "primary_per_record_rows": len(per_record),
        "primary_result_cells": len(primary_results),
        "bootstrap_result_rows": len(bootstrap_rows),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "cluster_unit": "source_frame_id",
        "no_semantic_imputation": True,
        "core_p_values": 0,
        "bertscore_configuration": required_config,
        "condition_ordering": condition_ordering,
    }
    (args.out / "audits/RQ2_ANALYSIS_COMPUTATION_AUDIT.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "PASS",
        "score_rows": len(rows),
        "primary_per_record_rows": len(per_record),
        "bootstrap_result_rows": len(bootstrap_rows),
    }, indent=2))


if __name__ == "__main__":
    main()
