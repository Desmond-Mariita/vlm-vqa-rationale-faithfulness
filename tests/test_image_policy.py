import hashlib
import json
import subprocess

import pytest
from scripts.validate_release import image_audit, ROOT

PNG = bytes.fromhex("89504e470d0a1a0a") + b"synthetic invalid image"


def test_public_payload_has_zero_vcr_rasters():
    audit = image_audit()
    assert audit["status"] == "PASS", audit["failures"]
    assert audit["thesis_vcr_count"] == 0
    assert audit["visual_file_count"] == 4
    assert audit["vcr_outside_thesis_figures"] == []
    assert {r["classification"] for r in audit["visual_files"]} == {"AUTHOR_CREATED_NON_VCR", "SYNTHETIC"}
    policy = json.loads((ROOT / "provenance/image_policy.json").read_text())
    assert policy["expected_tracked_vcr_raster_count"] == 0
    history = policy["historical_vcr_files"]
    assert len(history) == 43
    graph = json.loads((ROOT / "provenance/thesis_dependencies.json").read_text())
    dependencies = {row["path"]: row for row in graph["files"]}
    for row in history:
        assert not (ROOT / row["path"]).exists()
        assert dependencies[row["path"]]["withheld"]
        assert dependencies[row["path"]]["sha256"] == row["sha256"]
        for ref in row["active_references"]:
            source = ROOT / ref["source"]
            assert hashlib.sha256(source.read_bytes()).hexdigest() == ref["source_sha256"]
            assert ref["literal"] in source.read_text()


@pytest.mark.parametrize("payload", [PNG, bytes.fromhex("ffd8ff") + b"synthetic", b"RIFF0000WEBPsynthetic"])
@pytest.mark.parametrize("location", ["data/unexpected.dat", "docs/unexpected.txt", "unexpected.bin"])
def test_unknown_visual_is_rejected_even_with_nonimage_suffix(tmp_path, payload, location):
    target = tmp_path / location
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    audit = image_audit(tmp_path, {"visual_files": []})
    assert audit["status"] == "FAIL"
    assert audit["visual_files"][0]["classification"] == "FORBIDDEN"


@pytest.mark.parametrize("directory", ["thesis/Figures/records/C", "thesis/Figures/w2c/panels"])
@pytest.mark.parametrize("suffix", [".png", ".jpg", ".jpeg", ".webp", ".PNG", ".dat"])
def test_known_vcr_paths_cannot_be_relabelled_synthetic(tmp_path, directory, suffix):
    rel = directory + "/unexpected" + suffix
    target = tmp_path / rel
    target.parent.mkdir(parents=True)
    target.write_bytes(PNG)
    policy = {"visual_files": [{"path": rel, "sha256": hashlib.sha256(PNG).hexdigest(), "classification": "SYNTHETIC"}]}
    audit = image_audit(tmp_path, policy)
    assert audit["status"] == "FAIL"
    assert audit["thesis_vcr_count"] == 1


def test_historical_vcr_hash_cannot_be_moved_and_relabelled(tmp_path):
    rel = "innocent.dat"
    (tmp_path / rel).write_bytes(PNG)
    digest = hashlib.sha256(PNG).hexdigest()
    policy = {"visual_files": [{"path": rel, "sha256": digest, "classification": "SYNTHETIC"}],
              "historical_vcr_files": [{"path": "thesis/Figures/records/C/source.png", "sha256": digest}]}
    audit = image_audit(tmp_path, policy)
    assert audit["status"] == "FAIL"
    assert audit["vcr_outside_thesis_figures"] == [rel]


def test_former_exemption_cannot_be_reenabled(tmp_path):
    rel = "thesis/Figures/sample.png"
    target = tmp_path / rel
    target.parent.mkdir(parents=True)
    target.write_bytes(PNG)
    source = tmp_path / "figure.tex"
    source.write_text(rel)
    policy = {"approved_manuscript_sha": "synthetic", "visual_files": [{
        "path": rel, "sha256": hashlib.sha256(PNG).hexdigest(),
        "classification": "THESIS_VCR_EXEMPTION", "approved_manuscript_sha": "synthetic",
        "active_references": [{"source": "figure.tex", "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "literal": rel}]}]}
    assert image_audit(tmp_path, policy)["status"] == "FAIL"


def test_synthetic_asset_allowed_but_hash_change_rejected(tmp_path):
    rel = "synthetic.png"
    (tmp_path / rel).write_bytes(PNG)
    policy = {"visual_files": [{"path": rel, "sha256": hashlib.sha256(PNG).hexdigest(), "classification": "SYNTHETIC"}]}
    assert image_audit(tmp_path, policy)["status"] == "PASS"
    (tmp_path / rel).write_bytes(PNG + b"changed")
    assert image_audit(tmp_path, policy)["status"] == "FAIL"


def test_forced_git_addition_in_ignored_cache_is_rejected(tmp_path):
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text(".venv/\n")
    target = tmp_path / ".venv/unexpected.dat"
    target.parent.mkdir()
    target.write_bytes(PNG)
    subprocess.run(["git", "-C", str(tmp_path), "add", "--force", ".venv/unexpected.dat"], check=True)
    audit = image_audit(tmp_path, {"visual_files": []})
    assert audit["status"] == "FAIL"
    assert audit["visual_files"][0]["path"] == ".venv/unexpected.dat"


def test_nonzero_public_vcr_expectation_is_rejected(tmp_path):
    assert image_audit(tmp_path, {"visual_files": [], "expected_tracked_vcr_raster_count": 43})["status"] == "FAIL"


def test_secret_fallback_reports_only_redacted_fingerprint(tmp_path):
    from scripts.validate_release import secret_scan
    synthetic = "gh" + "p_" + "SYNTHETIC" * 5
    (tmp_path / "fixture.txt").write_text(synthetic)
    result = secret_scan(tmp_path)
    assert result["status"] == "FAIL"
    assert len(result["findings"]) == 1
    assert synthetic not in json.dumps(result)
    assert len(result["findings"][0]["fingerprint"]) == 12
