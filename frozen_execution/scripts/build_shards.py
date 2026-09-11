#!/usr/bin/env python3
"""Create immutable key-only worker shards and prove disjoint exact coverage."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any


WORKERS = {
    0: {"gpu_slot": 0, "process_slot": 0},
    1: {"gpu_slot": 0, "process_slot": 1},
    2: {"gpu_slot": 0, "process_slot": 2},
    3: {"gpu_slot": 1, "process_slot": 0},
    4: {"gpu_slot": 1, "process_slot": 1},
    5: {"gpu_slot": 1, "process_slot": 2},
}


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    return sha_file(path)


def manifest_row_hash(row: dict[str, str]) -> str:
    return canonical_hash(row)


def shard_rows(
    manifest: Path,
    rows: list[dict[str, str]],
    grouping: tuple[str, ...],
    output_root: Path,
    prefix: str,
    six_workers: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_sha = sha_file(manifest)
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in grouping)].append(row)
    entries: list[dict[str, Any]] = []
    group_audits: dict[str, Any] = {}
    for group, group_rows in sorted(groups.items()):
        group_name = "__".join(group)
        worker_count = 6 if six_workers else 1
        shard_sets: list[set[str]] = []
        for worker_id in range(worker_count):
            selected = [row for row in group_rows if int(row["dataset_index"]) % worker_count == worker_id]
            shard_payload = [
                {
                    "primary_key": row["primary_key"],
                    "record_id": row["record_id"],
                    "dataset_index": row["dataset_index"],
                    "parent_manifest_sha256": manifest_sha,
                    "manifest_row_sha256": manifest_row_hash(row),
                    "worker_id": worker_id,
                    "gpu_slot": WORKERS[worker_id]["gpu_slot"] if six_workers else "RTX-3090",
                    "process_slot": WORKERS[worker_id]["process_slot"] if six_workers else 0,
                }
                for row in selected
            ]
            if not shard_payload:
                raise RuntimeError(f"empty shard {prefix}/{group_name}/worker-{worker_id}")
            path = output_root / prefix / group_name / f"worker_{worker_id}.tsv"
            shard_sha = write_tsv(path, shard_payload)
            keys = {row["primary_key"] for row in shard_payload}
            shard_sets.append(keys)
            entries.append({
                "family": prefix,
                "group": group_name,
                "worker_id": worker_id,
                "gpu_slot": WORKERS[worker_id]["gpu_slot"] if six_workers else "RTX-3090",
                "process_slot": WORKERS[worker_id]["process_slot"] if six_workers else 0,
                "path": str(path),
                "sha256": shard_sha,
                "rows": len(shard_payload),
                "parent_manifest": str(manifest),
                "parent_manifest_sha256": manifest_sha,
            })
        expected = {row["primary_key"] for row in group_rows}
        union = set().union(*shard_sets)
        overlap_pairs = sum(bool(shard_sets[i] & shard_sets[j]) for i in range(len(shard_sets)) for j in range(i + 1, len(shard_sets)))
        group_audits[group_name] = {
            "expected_n": len(expected),
            "union_n": len(union),
            "union_exact": union == expected,
            "overlapping_shard_pairs": overlap_pairs,
            "shard_sizes": [len(values) for values in shard_sets],
            "ordered_dataset_index_rule": f"dataset_index modulo {worker_count}",
        }
        if union != expected or overlap_pairs:
            raise RuntimeError(f"shard audit failed: {prefix}/{group_name}")
    return entries, {"manifest": str(manifest), "manifest_sha256": manifest_sha, "groups": group_audits}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", type=Path, required=True)
    args = ap.parse_args()
    package = args.package.resolve()
    manifests = package / "manifests"
    shard_root = package / "shards"

    specs = [
        (manifests / "STAGE2_PRIMARY_MANIFEST.tsv", ("arm", "condition"), "stage2_primary", True),
        (manifests / "STAGE2_KNOWN_CHANGE_MANIFEST.tsv", ("arm", "condition"), "stage2_known_change", True),
        (manifests / "STAGE2_DESCRIPTION_AVAILABILITY_ABLATION_MANIFEST.tsv", ("arm", "condition"), "stage2_description_ablation", True),
        (manifests / "STAGE1_FINAL_WRONG_MANIFEST.tsv", ("arm",), "stage1_wrong", False),
    ]
    all_entries: list[dict[str, Any]] = []
    audits: dict[str, Any] = {}
    for manifest, grouping, prefix, six_workers in specs:
        entries, audit = shard_rows(manifest, read_tsv(manifest), grouping, shard_root, prefix, six_workers)
        all_entries.extend(entries)
        audits[prefix] = audit

    plan = {
        "status": "PASS",
        "assignment": WORKERS,
        "rule": "Within every arm-condition cell, worker_id = canonical dataset_index modulo 6. The canonical dataset index, not shard-local position, remains the mask/noise seed source.",
        "entries": all_entries,
        "audits": audits,
    }
    plan_path = shard_root / "SHARD_PLAN.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    audit_path = package / "audits" / "SHARD_UNION_INTERSECTION_AUDIT.json"
    audit_path.write_text(json.dumps(audits, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "PASS",
        "shards": len(all_entries),
        "shard_plan": str(plan_path),
        "shard_plan_sha256": sha_file(plan_path),
        "audit": str(audit_path),
        "audit_sha256": sha_file(audit_path),
    }, indent=2))


if __name__ == "__main__":
    main()
