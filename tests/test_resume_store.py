from pathlib import Path
import json
import pytest
from frozen_execution.scripts.production_store import ManifestLockedStore, StoreError, canonical_hash, selftest
from frozen_execution.scripts.local_assets import validate_local_assets

def test_restart_torn_append_and_shard_invariants(tmp_path):
    result = selftest(tmp_path / "store")
    assert result["exact_id_set"]
    assert result["completed_n"] == 10
    assert result["torn_final_line_recovery"]
    assert result["duplicate_rejected"]
    assert result["incompatible_manifest_rejected"]
    assert result["outside_shard_rejected"]

def test_complete_corruption_is_not_silently_repaired(tmp_path):
    expected = {"synthetic-0": canonical_hash({"index": 0})}
    store = ManifestLockedStore(tmp_path, {"purpose": "synthetic"}, expected)
    store.append({"primary_key": "synthetic-0", "manifest_row_sha256": expected["synthetic-0"], "record_status": "success", "value": 1})
    path = tmp_path / "records.jsonl"
    row = json.loads(path.read_text()); row["value"] = 2
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(StoreError, match="output record hash mismatch"):
        ManifestLockedStore(tmp_path, {"purpose": "synthetic"}, expected)

def test_local_assets_hash_actual_files(tmp_path):
    import hashlib
    data = tmp_path / "synthetic.txt"; data.write_text("synthetic")
    manifest = tmp_path / "assets.json"
    manifest.write_text(json.dumps({"schema_version": 1, "files": [{"path": data.name, "sha256": hashlib.sha256(data.read_bytes()).hexdigest(), "bytes": data.stat().st_size}]}))
    assert len(validate_local_assets(manifest)) == 64
    data.write_text("changed")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_local_assets(manifest)
