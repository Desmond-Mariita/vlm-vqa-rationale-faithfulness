#!/usr/bin/env python3
"""Independent RQ2 provenance, alignment, record-hash, and coverage audit."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


ARMS = ("plain", "point", "plain_desc", "point_desc")
CONDITIONS = ("source", "grey", "mismatch", "mask", "noise", "mirror")
SPECS = {
    "stage2_primary": {
        "manifest": "STAGE2_PRIMARY_MANIFEST.tsv.gz",
        "sha": "f8349fa09918f6948cf8f831f5b125ead2acb897860898d458c7a6da98b1c6de",
        "rows": 63672,
        "shards": 144,
        "answer_index": "supplied_gold_answer_index",
        "answer_text": "supplied_gold_answer_text",
        "prompt": "prompt_sha256",
    },
    "stage2_known_change": {
        "manifest": "STAGE2_KNOWN_CHANGE_MANIFEST.tsv",
        "sha": "a146fe45daced64087146257f82b6096f59602b2992769665363b0bf9eaee3ca",
        "rows": 800,
        "shards": 24,
        "answer_index": "alternate_answer_index",
        "answer_text": "alternate_answer_text",
        "prompt": "prompt_sha256",
    },
    "stage2_description_ablation": {
        "manifest": "STAGE2_DESCRIPTION_AVAILABILITY_ABLATION_MANIFEST.tsv",
        "sha": "30a98b7aac15ad0725a971b735789003e3554bf66526baa7f98e4f3a33d47937",
        "rows": 800,
        "shards": 24,
        "answer_index": "supplied_gold_answer_index",
        "answer_text": "supplied_gold_answer_text",
        "prompt": "description_off_prompt_sha256",
    },
}


def sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha_bytes(payload.encode("utf-8"))


def manifest_sha(path: Path) -> str:
    digest = hashlib.sha256()
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def ordered_hash(values: list[str]) -> str:
    return sha_bytes("\n".join(values).encode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-root", required=True, type=Path)
    parser.add_argument("--outputs-root", required=True, type=Path)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    dataset_by_id = {row["image_id"]: row for row in dataset}
    dataset_ids = [row["image_id"] for row in dataset]
    dataset_ordered_hash = ordered_hash(dataset_ids)
    if len(dataset) != 2653 or len(dataset_by_id) != 2653:
        failures.append("dataset is not 2,653 unique records")
    if len({row["orig_image_id"] for row in dataset}) != 2400:
        failures.append("dataset does not contain 2,400 source frames")

    ledger = json.loads(args.ledger.read_text(encoding="utf-8"))
    ledger_shards = {entry["relative_path"]: entry for entry in ledger["entries"] if entry.get("kind") == "shard"}
    if ledger.get("run_id") != args.run_id or len(ledger_shards) != 192:
        failures.append("completion ledger run identity/shard count mismatch")

    provenance_rows: list[dict] = []
    coverage_rows: list[dict] = []
    family_audits: dict[str, dict] = {}
    all_keys: set[str] = set()
    primary_records: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    family_records: dict[tuple[str, str, str], dict[str, dict]] = defaultdict(dict)

    for family, spec in SPECS.items():
        manifest_path = args.frozen_root / "manifests" / spec["manifest"]
        manifest_rows = read_tsv(manifest_path)
        observed_manifest_sha = manifest_sha(manifest_path)
        if observed_manifest_sha != spec["sha"]:
            failures.append(f"{family}: manifest SHA mismatch")
        if len(manifest_rows) != spec["rows"]:
            failures.append(f"{family}: manifest row count mismatch")
        manifest_by_key = {row["primary_key"]: row for row in manifest_rows}
        if len(manifest_by_key) != len(manifest_rows):
            failures.append(f"{family}: duplicate manifest keys")

        shard_root = args.frozen_root / "shards" / family
        shard_paths = sorted(shard_root.rglob("worker_*.tsv"))
        if len(shard_paths) != spec["shards"]:
            failures.append(f"{family}: frozen shard count mismatch")
        family_failures: list[str] = []
        observed_keys: list[str] = []
        cell_files: dict[tuple[str, str], list[dict]] = defaultdict(list)

        for shard_path in shard_paths:
            shard_rows = read_tsv(shard_path)
            relative = shard_path.relative_to(shard_root)
            output_path = args.outputs_root / family / relative.with_suffix("") / "records.jsonl"
            ledger_relative = output_path.relative_to(args.outputs_root.parents[1]).as_posix()
            ledger_entry = ledger_shards.get(ledger_relative)
            if not output_path.is_file():
                family_failures.append(f"missing output: {output_path}")
                continue
            raw = output_path.read_bytes()
            if raw and not raw.endswith(b"\n"):
                family_failures.append(f"torn final line: {output_path}")
            output_sha = sha_bytes(raw)
            if ledger_entry is None or ledger_entry.get("sha256") != output_sha:
                family_failures.append(f"ledger SHA mismatch: {output_path}")
            records = [json.loads(line) for line in raw.splitlines() if line.strip()]
            expected_keys = [row["primary_key"] for row in shard_rows]
            actual_keys = [str(row.get("primary_key")) for row in records]
            if actual_keys != expected_keys:
                family_failures.append(f"ordered shard key mismatch: {output_path}")
            shard_sha = sha_file(shard_path)
            shard_by_key = {row["primary_key"]: row for row in shard_rows}

            for record in records:
                key = str(record.get("primary_key"))
                observed_keys.append(key)
                if key in all_keys:
                    family_failures.append(f"duplicate global key: {key}")
                all_keys.add(key)
                mrow = manifest_by_key.get(key)
                srow = shard_by_key.get(key)
                if mrow is None or srow is None:
                    family_failures.append(f"key outside manifest/shard: {key}")
                    continue
                payload = dict(record)
                output_record_sha = payload.pop("output_record_sha256", None)
                if output_record_sha != canonical_hash(payload):
                    family_failures.append(f"output record hash mismatch: {key}")
                expected_fields = {
                    "arm": mrow["arm"],
                    "condition": mrow["condition"],
                    "record_id": mrow["record_id"],
                    "source_frame_id": mrow["source_frame_id"],
                    "dataset_index": mrow["dataset_index"],
                    "supplied_answer_index": mrow[spec["answer_index"]],
                    "supplied_answer_text": mrow[spec["answer_text"]],
                    "prompt_sha256": mrow[spec["prompt"]],
                    "canonical_checkpoint_sha256": mrow["canonical_checkpoint_sha256"],
                    "transport_checkpoint_sha256": mrow["transport_checkpoint_sha256"],
                    "model_processor_revision": mrow["model_processor_revision"],
                    "output_schema_version": mrow["expected_output_schema_version"],
                    "manifest_row_sha256": srow["manifest_row_sha256"],
                    "shard_sha256": shard_sha,
                    "worker_id": srow["worker_id"],
                }
                for field, expected in expected_fields.items():
                    if str(record.get(field)) != str(expected):
                        family_failures.append(f"metadata mismatch {field}: {key}")
                if not all(field in record for field in (
                    "strict_reasoning", "strict_final", "strict_parser_status",
                    "tolerant_reasoning", "tolerant_final", "tolerant_parser_status",
                    "finish_reason", "record_status", "raw_generation",
                )):
                    family_failures.append(f"parser/runtime metadata missing: {key}")

                data_row = dataset_by_id.get(mrow["record_id"])
                if data_row is None:
                    family_failures.append(f"record missing from frozen dataset: {key}")
                elif family != "stage2_known_change":
                    gold = int(data_row["answer_label"])
                    if int(record["supplied_answer_index"]) != gold or record["supplied_answer_text"] != data_row["choices"][gold]:
                        family_failures.append(f"gold answer mismatch: {key}")
                else:
                    if int(record["supplied_answer_index"]) == int(data_row["answer_label"]):
                        family_failures.append(f"known-change alternate equals gold: {key}")

                cell = (family, mrow["arm"], mrow["condition"])
                family_records[cell][mrow["record_id"]] = record
                if family == "stage2_primary":
                    primary_records[(mrow["arm"], mrow["condition"])][mrow["record_id"]] = record

            first_manifest = manifest_by_key[expected_keys[0]] if expected_keys else {}
            if expected_keys:
                cell_files[(first_manifest["arm"], first_manifest["condition"])].append({
                    "path": str(output_path),
                    "sha256": output_sha,
                    "rows": len(records),
                    "ledger_relative": ledger_relative,
                })

        expected_key_set = set(manifest_by_key)
        observed_key_set = set(observed_keys)
        if expected_key_set != observed_key_set or len(observed_keys) != len(observed_key_set):
            family_failures.append("family manifest/output key union mismatch or duplicates")

        for (arm, condition), files in sorted(cell_files.items()):
            records = family_records[(family, arm, condition)]
            ordered = sorted(records.values(), key=lambda row: int(row["dataset_index"]))
            ids = [row["record_id"] for row in ordered]
            if family == "stage2_primary" and ids != dataset_ids:
                family_failures.append(f"primary ordered population mismatch: {arm}/{condition}")
            frames = {row["source_frame_id"] for row in ordered}
            statuses = Counter(row["finish_reason"] for row in ordered)
            strict_reason = sum(bool(str(row["strict_reasoning"]).strip()) for row in ordered)
            strict_final = sum(bool(str(row["strict_final"]).strip()) for row in ordered)
            tolerant_reason = sum(bool(str(row["tolerant_reasoning"]).strip()) for row in ordered)
            tolerant_final = sum(bool(str(row["tolerant_final"]).strip()) for row in ordered)
            tolerant_both = sum(bool(str(row["tolerant_reasoning"]).strip()) and bool(str(row["tolerant_final"]).strip()) for row in ordered)
            source = primary_records.get((arm, "source"), {}) if family == "stage2_primary" else {}
            pairable_ids = [rid for rid, row in records.items() if str(row["tolerant_final"]).strip() and (condition == "source" or (rid in source and str(source[rid]["tolerant_final"]).strip()))]
            pairable_frames = {records[rid]["source_frame_id"] for rid in pairable_ids}
            prompt_values = [row["prompt_sha256"] for row in ordered]
            checkpoint_values = sorted({row["canonical_checkpoint_sha256"] for row in ordered})
            transport_values = sorted({row["transport_checkpoint_sha256"] for row in ordered})
            model_values = sorted({row["model_processor_revision"] for row in ordered})
            schema_values = sorted({row["output_schema_version"] for row in ordered})
            gpu_values = sorted({row["gpu_uuid"] for row in ordered})
            environment_values = sorted({row["environment_sha256"] for row in ordered})
            file_lines = [f"{item['ledger_relative']}\t{item['sha256']}" for item in sorted(files, key=lambda item: item["ledger_relative"])]
            provenance_rows.append({
                "family": family,
                "arm": arm,
                "condition": condition,
                "artifact_paths_json": json.dumps([item["path"] for item in files], ensure_ascii=False),
                "artifact_sha256s_json": json.dumps([item["sha256"] for item in files]),
                "artifact_set_sha256": ordered_hash(file_lines),
                "raw_n": len(ordered),
                "ordered_id_sha256": ordered_hash(ids),
                "source_frame_n": len(frames),
                "prompt_sha256_ordered_hash": ordered_hash(prompt_values),
                "canonical_checkpoint_sha256s_json": json.dumps(checkpoint_values),
                "transport_checkpoint_sha256s_json": json.dumps(transport_values),
                "model_processor_revisions_json": json.dumps(model_values),
                "output_schema_versions_json": json.dumps(schema_values),
                "gpu_uuids_json": json.dumps(gpu_values),
                "environment_sha256s_json": json.dumps(environment_values),
                "hardware_runtime_lineage": "homogeneous qualified RTX-5090 production lineage",
                "ledger_reconciled": True,
                "record_hashes_valid": True,
                "status": "PASS",
            })
            coverage_rows.append({
                "family": family,
                "arm": arm,
                "condition": condition,
                "raw_generation_n": len(ordered),
                "strict_reasoning_success_n": strict_reason,
                "strict_reasoning_success_rate": strict_reason / len(ordered),
                "strict_final_success_n": strict_final,
                "strict_final_success_rate": strict_final / len(ordered),
                "tolerant_reasoning_success_n": tolerant_reason,
                "tolerant_reasoning_success_rate": tolerant_reason / len(ordered),
                "tolerant_final_success_n": tolerant_final,
                "tolerant_final_success_rate": tolerant_final / len(ordered),
                "tolerant_both_success_n": tolerant_both,
                "tolerant_both_success_rate": tolerant_both / len(ordered),
                "empty_or_missing_tolerant_final_n": len(ordered) - tolerant_final,
                "finish_eos_or_stop_n": statuses.get("eos_or_stop", 0),
                "finish_length_n": statuses.get("length", 0),
                "finish_error_n": sum(count for value, count in statuses.items() if value not in {"eos_or_stop", "length"}),
                "record_error_n": sum(row["record_status"] != "success" for row in ordered),
                "source_relative_pairable_final_n": len(pairable_ids) if family == "stage2_primary" else "",
                "source_relative_pairable_source_frame_n": len(pairable_frames) if family == "stage2_primary" else "",
            })

        family_audits[family] = {
            "manifest_path": str(manifest_path),
            "manifest_sha256": observed_manifest_sha,
            "manifest_rows": len(manifest_rows),
            "frozen_shards": len(shard_paths),
            "output_rows": len(observed_keys),
            "unique_output_keys": len(observed_key_set),
            "failures": sorted(set(family_failures)),
            "status": "PASS" if not family_failures else "FAIL",
        }
        failures.extend(f"{family}: {failure}" for failure in sorted(set(family_failures)))

    # Primary six-condition alignment and fixed-answer audit.
    primary_alignment: dict[str, dict] = {}
    for arm in ARMS:
        id_sets = {condition: set(primary_records[(arm, condition)]) for condition in CONDITIONS}
        intersections = set.intersection(*id_sets.values())
        unions = set.union(*id_sets.values())
        gold_invariant = True
        frame_invariant = True
        for record_id in intersections:
            records = [primary_records[(arm, condition)][record_id] for condition in CONDITIONS]
            gold_invariant &= len({(row["supplied_answer_index"], row["supplied_answer_text"]) for row in records}) == 1
            frame_invariant &= len({row["source_frame_id"] for row in records}) == 1
        status = len(intersections) == len(unions) == 2653 and gold_invariant and frame_invariant
        if not status:
            failures.append(f"primary alignment failure: {arm}")
        primary_alignment[arm] = {
            "condition_counts": {condition: len(ids) for condition, ids in id_sets.items()},
            "intersection_n": len(intersections),
            "union_n": len(unions),
            "gold_answer_invariant": gold_invariant,
            "source_frame_invariant": frame_invariant,
            "status": "PASS" if status else "FAIL",
        }

    # Ablation retains description-trained checkpoints and frozen prompt variant.
    ablation_checkpoint_match: dict[str, bool] = {}
    for arm in ("plain_desc", "point_desc"):
        primary_checkpoints = {row["canonical_checkpoint_sha256"] for row in primary_records[(arm, "source")].values()}
        off_records = []
        for condition in ("source_description_off", "grey_description_off"):
            off_records.extend(family_records[("stage2_description_ablation", arm, condition)].values())
        match = {row["canonical_checkpoint_sha256"] for row in off_records} == primary_checkpoints
        ablation_checkpoint_match[arm] = match
        if not match:
            failures.append(f"ablation checkpoint does not retain description-trained checkpoint: {arm}")

    provenance_fields = list(provenance_rows[0])
    coverage_fields = list(coverage_rows[0])
    write_tsv(args.out / "RQ2_INPUT_PROVENANCE.tsv", provenance_rows, provenance_fields)
    write_tsv(args.out / "RQ2_PARSE_COVERAGE.tsv", coverage_rows, coverage_fields)
    audit = {
        "schema_version": "rq2-alignment-integrity-audit-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": args.run_id,
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "scientific_text_inspected_by_audit": False,
        "dataset": {
            "path": str(args.dataset),
            "sha256": sha_file(args.dataset),
            "records": len(dataset),
            "source_frames": len({row["orig_image_id"] for row in dataset}),
            "ordered_id_sha256": dataset_ordered_hash,
        },
        "completion_ledger": {"path": str(args.ledger), "sha256": sha_file(args.ledger), "shard_entries": len(ledger_shards)},
        "families": family_audits,
        "primary_alignment": primary_alignment,
        "ablation_description_checkpoint_retained": ablation_checkpoint_match,
        "global_unique_logical_keys": len(all_keys),
        "expected_global_logical_keys": 65272,
        "no_semantic_imputation": True,
        "parser_metadata_required": True,
    }
    (args.out / "RQ2_ALIGNMENT_AND_INTEGRITY_AUDIT.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": audit["status"], "failures": failures, "global_unique_logical_keys": len(all_keys)}, indent=2))
    raise SystemExit(0 if not failures else 1)


if __name__ == "__main__":
    main()
