#!/usr/bin/env python3
"""Prepare frozen RQ2 BERTScore pairs without semantic imputation."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ARMS = ("plain", "point", "plain_desc", "point_desc")
PERTURBATIONS = ("grey", "mismatch", "mask", "noise", "mirror")


def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_outputs(root: Path, family: str) -> dict[tuple[str, str, str], dict]:
    result: dict[tuple[str, str, str], dict] = {}
    for path in sorted((root / family).rglob("records.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (row["arm"], row["condition"], row["record_id"])
                if key in result:
                    raise RuntimeError(f"duplicate output key: {key}")
                result[key] = row
    return result


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def flexible_literal_pattern(value: str) -> re.Pattern | None:
    value = unicodedata.normalize("NFKC", value or "").strip()
    if not value:
        return None
    parts = re.split(r"(\s+)", value)
    body = "".join(r"\s+" if part.isspace() else re.escape(part) for part in parts if part)
    return re.compile(body, re.IGNORECASE)


def mask_choices(text: str, choices: list[str]) -> tuple[str, list[int]]:
    output = unicodedata.normalize("NFKC", text or "")
    matches: list[int] = []
    order = sorted(range(len(choices)), key=lambda index: len(unicodedata.normalize("NFKC", choices[index] or "")), reverse=True)
    for index in order:
        pattern = flexible_literal_pattern(choices[index])
        if pattern is None:
            continue
        output, count = pattern.subn(f"<ANSWER_CHOICE_{index}>", output)
        matches.extend([index] * count)
    return output, matches


def add_job(rows: list[dict], *, analysis: str, arm: str, condition: str, unit: str,
            record_id: str, source_frame_id: str, ref: str, cand: str) -> bool:
    if not isinstance(ref, str) or not ref.strip() or not isinstance(cand, str) or not cand.strip():
        return False
    rows.append({
        "job_index": len(rows),
        "analysis": analysis,
        "arm": arm,
        "condition": condition,
        "unit": unit,
        "record_id": record_id,
        "source_frame_id": source_frame_id,
        "source_text_sha256": sha_text(ref),
        "condition_text_sha256": sha_text(cand),
        "source_text": ref,
        "condition_text": cand,
    })
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--unrelated-map", required=True, type=Path)
    parser.add_argument("--jobs", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--masking-counts", required=True, type=Path)
    args = parser.parse_args()

    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    by_id = {row["image_id"]: row for row in dataset}
    primary = read_outputs(args.outputs_root, "stage2_primary")
    known = read_outputs(args.outputs_root, "stage2_known_change")
    ablation = read_outputs(args.outputs_root, "stage2_description_ablation")
    unrelated = read_tsv(args.unrelated_map)

    jobs: list[dict] = []
    counts: Counter[tuple[str, str, str, str]] = Counter()
    masking_rows: list[dict] = []

    # Primary final span and appendix raw/reasoning sensitivities.
    for arm in ARMS:
        for record in dataset:
            rid = record["image_id"]
            source = primary[(arm, "source", rid)]
            for condition in PERTURBATIONS:
                changed = primary[(arm, condition, rid)]
                for analysis, unit, field in (
                    ("primary_final", "tolerant_final", "tolerant_final"),
                    ("primary_reasoning", "tolerant_reasoning", "tolerant_reasoning"),
                    ("primary_raw_generation", "raw_generation", "raw_generation"),
                ):
                    if add_job(
                        jobs,
                        analysis=analysis,
                        arm=arm,
                        condition=condition,
                        unit=unit,
                        record_id=rid,
                        source_frame_id=source["source_frame_id"],
                        ref=source[field],
                        cand=changed[field],
                    ):
                        counts[(analysis, arm, condition, unit)] += 1

    # Frozen known-change: matched image-removal and alternate-answer comparisons.
    known_ids_by_arm: dict[str, list[str]] = {arm: [] for arm in ARMS}
    for arm in ARMS:
        arm_known = sorted(
            (row for (row_arm, _condition, _rid), row in known.items() if row_arm == arm),
            key=lambda row: int(row["dataset_index"]),
        )
        known_ids_by_arm[arm] = [row["record_id"] for row in arm_known]
        role_contains = Counter()
        total_occurrences = 0
        records_any = 0
        for alternate in arm_known:
            rid = alternate["record_id"]
            source = primary[(arm, "source", rid)]
            grey = primary[(arm, "grey", rid)]
            for analysis, unit, field in (
                ("known_image_final", "tolerant_final", "tolerant_final"),
                ("known_answer_final", "tolerant_final", "tolerant_final"),
                ("known_image_reasoning", "tolerant_reasoning", "tolerant_reasoning"),
                ("known_answer_reasoning", "tolerant_reasoning", "tolerant_reasoning"),
            ):
                candidate = grey if "image" in analysis else alternate
                if add_job(
                    jobs,
                    analysis=analysis,
                    arm=arm,
                    condition="grey" if "image" in analysis else "alternate_answer",
                    unit=unit,
                    record_id=rid,
                    source_frame_id=source["source_frame_id"],
                    ref=source[field],
                    cand=candidate[field],
                ):
                    counts[(analysis, arm, "grey" if "image" in analysis else "alternate_answer", unit)] += 1

            choices = by_id[rid]["choices"]
            texts = {
                "source": source["tolerant_final"],
                "grey": grey["tolerant_final"],
                "alternate": alternate["tolerant_final"],
            }
            masked: dict[str, str] = {}
            matched_this_record = False
            for role, text in texts.items():
                masked[role], matches = mask_choices(text, choices)
                role_contains[role] += int(bool(matches))
                total_occurrences += len(matches)
                matched_this_record |= bool(matches)
            records_any += int(matched_this_record)
            for analysis, role, condition in (
                ("known_image_masked_final", "grey", "grey"),
                ("known_answer_masked_final", "alternate", "alternate_answer"),
            ):
                if add_job(
                    jobs,
                    analysis=analysis,
                    arm=arm,
                    condition=condition,
                    unit="answer_choice_masked_tolerant_final",
                    record_id=rid,
                    source_frame_id=source["source_frame_id"],
                    ref=masked["source"],
                    cand=masked[role],
                ):
                    counts[(analysis, arm, condition, "answer_choice_masked_tolerant_final")] += 1
        masking_rows.append({
            "arm": arm,
            "records": len(arm_known),
            "source_rationales_with_choice_match": role_contains["source"],
            "grey_rationales_with_choice_match": role_contains["grey"],
            "alternate_rationales_with_choice_match": role_contains["alternate"],
            "records_with_any_choice_match": records_any,
            "total_choice_string_occurrences": total_occurrences,
            "normalisation_rule": "NFKC; case-insensitive; flexible whitespace; literal punctuation; longest choice first",
        })

    # Frozen unrelated-source pairing map.
    unrelated_failures: list[str] = []
    for row in unrelated:
        arm = row["arm"]
        source = primary[(arm, "source", row["source_record_id"])]
        target = primary[(arm, "source", row["target_record_id"])]
        if row["source_record_id"] == row["target_record_id"]:
            unrelated_failures.append(f"self-pair: {arm}/{row['source_record_id']}")
        if row["source_frame_id"] == row["target_frame_id"]:
            unrelated_failures.append(f"same-frame pair: {arm}/{row['source_record_id']}")
        if add_job(
            jobs,
            analysis="unrelated_final",
            arm=arm,
            condition="unrelated",
            unit="tolerant_final",
            record_id=row["source_record_id"],
            source_frame_id=source["source_frame_id"],
            ref=source["tolerant_final"],
            cand=target["tolerant_final"],
        ):
            counts[("unrelated_final", arm, "unrelated", "tolerant_final")] += 1

    # Bounded description-availability diagnostic and optional caption-removal spans.
    for arm in ("plain_desc", "point_desc"):
        ids = sorted(
            {rid for row_arm, _condition, rid in ablation if row_arm == arm},
            key=lambda rid: int(primary[(arm, "source", rid)]["dataset_index"]),
        )
        if len(ids) != 200:
            raise RuntimeError(f"description ablation ID count: {arm} {len(ids)}")
        for rid in ids:
            source_on = primary[(arm, "source", rid)]
            grey_on = primary[(arm, "grey", rid)]
            source_off = ablation[(arm, "source_description_off", rid)]
            grey_off = ablation[(arm, "grey_description_off", rid)]
            comparisons = (
                ("description_on_image", "grey_description_on", source_on["tolerant_final"], grey_on["tolerant_final"]),
                ("description_off_image", "grey_description_off", source_off["tolerant_final"], grey_off["tolerant_final"]),
                ("description_source_on_off", "source_on_vs_off", source_on["tolerant_final"], source_off["tolerant_final"]),
                ("description_grey_on_off", "grey_on_vs_off", grey_on["tolerant_final"], grey_off["tolerant_final"]),
            )
            for analysis, condition, ref, cand in comparisons:
                if add_job(
                    jobs,
                    analysis=analysis,
                    arm=arm,
                    condition=condition,
                    unit="tolerant_final",
                    record_id=rid,
                    source_frame_id=source_on["source_frame_id"],
                    ref=ref,
                    cand=cand,
                ):
                    counts[(analysis, arm, condition, "tolerant_final")] += 1

    args.jobs.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "job_index", "analysis", "arm", "condition", "unit", "record_id", "source_frame_id",
        "source_text_sha256", "condition_text_sha256", "source_text", "condition_text",
    ]
    with gzip.open(args.jobs, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(jobs)
    with args.masking_counts.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(masking_rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(masking_rows)

    audit = {
        "schema_version": "rq2-bertscore-job-preparation-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if not unrelated_failures else "FAIL",
        "job_rows": len(jobs),
        "job_counts": [
            {"analysis": key[0], "arm": key[1], "condition": key[2], "unit": key[3], "n": value}
            for key, value in sorted(counts.items())
        ],
        "known_change_ids_per_arm": {arm: len(ids) for arm, ids in known_ids_by_arm.items()},
        "unrelated_map_rows": len(unrelated),
        "unrelated_map_failures": unrelated_failures,
        "no_semantic_imputation": True,
        "masking_rule": masking_rows[0]["normalisation_rule"],
        "jobs_path": str(args.jobs),
        "jobs_sha256": hashlib.sha256(args.jobs.read_bytes()).hexdigest(),
    }
    args.audit.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": audit["status"], "job_rows": len(jobs), "jobs_sha256": audit["jobs_sha256"]}, indent=2))
    raise SystemExit(0 if audit["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
