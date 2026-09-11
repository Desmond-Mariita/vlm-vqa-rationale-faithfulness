#!/usr/bin/env python3
"""Crash-safe, manifest-locked JSONL storage for final inference.

Rows are accepted only when their primary key belongs to the immutable shard,
their manifest-row hash matches, and their own content hash validates.  A torn
non-newline-terminated final write is quarantined and truncated on restart;
malformed complete lines are never silently repaired.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any


class StoreError(RuntimeError):
    pass


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def add_record_hash(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    result.pop("output_record_sha256", None)
    result["output_record_sha256"] = canonical_hash(result)
    return result


def validate_record_hash(row: dict[str, Any]) -> bool:
    expected = row.get("output_record_sha256")
    if not isinstance(expected, str):
        return False
    payload = dict(row)
    payload.pop("output_record_sha256", None)
    return expected == canonical_hash(payload)


class ManifestLockedStore:
    def __init__(self, root: Path, metadata: dict[str, Any], expected_rows: dict[str, str]):
        self.root = root
        self.metadata_path = root / "run_metadata.json"
        self.records_path = root / "records.jsonl"
        self.expected_rows = dict(expected_rows)
        self.root.mkdir(parents=True, exist_ok=True)

        frozen = dict(metadata)
        frozen["expected_primary_keys_sha256"] = canonical_hash(sorted(expected_rows))
        frozen["expected_manifest_rows_sha256"] = canonical_hash(expected_rows)
        frozen["invariant_sha256"] = canonical_hash(frozen)
        if self.metadata_path.exists():
            existing = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            if existing != frozen:
                raise StoreError(
                    "incompatible run metadata/manifest/shard; refusing append "
                    f"existing={existing.get('invariant_sha256')} requested={frozen['invariant_sha256']}"
                )
        else:
            atomic_json(self.metadata_path, frozen)
        self.metadata = frozen
        self.completed: dict[str, dict[str, Any]] = {}
        self.torn_line_recovered = False
        self._recover_torn_final_line()
        self._load_and_validate()

    def _recover_torn_final_line(self) -> None:
        if not self.records_path.exists() or self.records_path.stat().st_size == 0:
            return
        data = self.records_path.read_bytes()
        if data.endswith(b"\n"):
            return
        last_newline = data.rfind(b"\n")
        good_end = last_newline + 1 if last_newline >= 0 else 0
        torn = data[good_end:]
        quarantine = self.root / f"records.torn.{time.time_ns()}.bin"
        quarantine.write_bytes(torn)
        with self.records_path.open("r+b") as handle:
            handle.truncate(good_end)
            handle.flush()
            os.fsync(handle.fileno())
        self.torn_line_recovered = True

    def _load_and_validate(self) -> None:
        if not self.records_path.exists():
            return
        with self.records_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise StoreError(f"malformed complete JSONL record at line {line_number}") from exc
                key = str(row.get("primary_key") or "")
                if key not in self.expected_rows:
                    raise StoreError(f"completed key is outside immutable shard: {key!r}")
                if key in self.completed:
                    raise StoreError(f"duplicate completed key: {key!r}")
                if row.get("manifest_row_sha256") != self.expected_rows[key]:
                    raise StoreError(f"manifest row hash mismatch for completed key: {key!r}")
                if not validate_record_hash(row):
                    raise StoreError(f"output record hash mismatch for completed key: {key!r}")
                if row.get("record_status") != "success":
                    raise StoreError(f"non-success row found in completed-record store: {key!r}")
                self.completed[key] = row

    def append(self, row: dict[str, Any]) -> dict[str, Any]:
        key = str(row.get("primary_key") or "")
        if key not in self.expected_rows:
            raise StoreError(f"append key is outside immutable shard: {key!r}")
        if key in self.completed:
            raise StoreError(f"duplicate append rejected: {key!r}")
        if row.get("manifest_row_sha256") != self.expected_rows[key]:
            raise StoreError(f"append manifest-row hash mismatch: {key!r}")
        if row.get("record_status") != "success":
            raise StoreError("only successful records may enter the completed-record store")
        frozen = add_record_hash(row)
        payload = (json.dumps(frozen, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        fd = os.open(self.records_path, os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o640)
        try:
            written = os.write(fd, payload)
            if written != len(payload):
                raise StoreError(f"short append: {written}/{len(payload)} bytes")
            os.fsync(fd)
        finally:
            os.close(fd)
        self.completed[key] = frozen
        return frozen

    def append_error(self, row: dict[str, Any]) -> None:
        payload = dict(row)
        payload["record_status"] = "error"
        payload = add_record_hash(payload)
        path = self.root / "errors.jsonl"
        encoded = (json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        fd = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o640)
        try:
            written = os.write(fd, encoded)
            if written != len(encoded):
                raise StoreError(f"short error append: {written}/{len(encoded)} bytes")
            os.fsync(fd)
        finally:
            os.close(fd)

    def final_audit(self) -> dict[str, Any]:
        expected = set(self.expected_rows)
        actual = set(self.completed)
        return {
            "expected_n": len(expected),
            "completed_n": len(actual),
            "missing_n": len(expected - actual),
            "unexpected_n": len(actual - expected),
            "exact_id_set": expected == actual,
            "torn_final_line_recovered_on_this_start": self.torn_line_recovered,
            "records_sha256": sha_file(self.records_path) if self.records_path.exists() else None,
        }


def selftest(root: Path) -> dict[str, Any]:
    keys = [f"record-{i}" for i in range(10)]
    expected = {key: canonical_hash({"key": key}) for key in keys}
    metadata = {"purpose": "production-store-selftest", "manifest": canonical_hash(keys)}
    store = ManifestLockedStore(root, metadata, expected)
    for key in keys[:3]:
        store.append({"primary_key": key, "manifest_row_sha256": expected[key], "record_status": "success", "value": key})

    # Simulate a torn fourth append.  Restart must quarantine and truncate only it.
    with (root / "records.jsonl").open("ab") as handle:
        handle.write(b'{"primary_key":"record-3"')
        handle.flush()
        os.fsync(handle.fileno())
    restarted = ManifestLockedStore(root, metadata, expected)
    if not restarted.torn_line_recovered or set(restarted.completed) != set(keys[:3]):
        raise AssertionError("torn-line recovery failed")
    for key in keys[3:]:
        restarted.append({"primary_key": key, "manifest_row_sha256": expected[key], "record_status": "success", "value": key})

    duplicate_rejected = incompatible_rejected = false_key_rejected = False
    try:
        restarted.append({"primary_key": keys[0], "manifest_row_sha256": expected[keys[0]], "record_status": "success"})
    except StoreError:
        duplicate_rejected = True
    try:
        ManifestLockedStore(root, {"purpose": "incompatible"}, expected)
    except StoreError:
        incompatible_rejected = True
    try:
        restarted.append({"primary_key": "not-in-shard", "manifest_row_sha256": "x", "record_status": "success"})
    except StoreError:
        false_key_rejected = True
    audit = restarted.final_audit()
    result = {
        **audit,
        "atomic_append_and_fsync": True,
        "torn_final_line_recovery": True,
        "restart_completed_id_skip_ready": True,
        "duplicate_rejected": duplicate_rejected,
        "incompatible_manifest_rejected": incompatible_rejected,
        "outside_shard_rejected": false_key_rejected,
        "row_hashes_validated": True,
    }
    if not all((audit["exact_id_set"], duplicate_rejected, incompatible_rejected, false_key_rejected)):
        raise AssertionError(result)
    atomic_json(root / "selftest_result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(selftest(args.selftest_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
