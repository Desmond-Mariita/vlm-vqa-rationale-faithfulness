from pathlib import Path
import ast
import csv
import gzip
import json
import yaml
from scripts.validate_release import validate, thesis_status, TITLE
from utils.yaml_config import load_yaml

ROOT = Path(__file__).resolve().parents[1]

def test_title_and_two_stage_configs():
    assert (ROOT / "README.md").read_text().splitlines()[0] == "# " + TITLE
    assert yaml.safe_load((ROOT / "CITATION.cff").read_text())["title"] == TITLE
    ev = yaml.safe_load((ROOT / "configs/evaluation/final.yaml").read_text())
    assert ev["arms"] == ["plain", "point", "plain_desc", "point_desc"]
    assert ev["conditions"] == ["source", "grey", "mismatch", "mask", "noise", "mirror"]
    assert ev["stage2"]["answer_source"] == "gold" and ev["stage2"]["max_new_tokens"] == 256
    assert ev["pred_a"]["role"] == "secondary"
    assert ev["bootstrap"]["replicates"] == 10000
    for arm in ev["arms"]:
        a = load_yaml(str(ROOT / "configs/stage1" / (arm + ".yaml")))
        b = load_yaml(str(ROOT / "configs/stage2" / (arm + ".yaml")))
        assert a["train"]["max_epochs_stage1"] == 3
        assert b["train_stage2"]["max_epochs"] == 6
        assert b["train_stage2"]["answer_source"] == "gold"
        assert b["experiment"]["arm"] == arm
        assert b["train"]["stage1_ckpt"] == a["project"]["save_dir"] + "model_best.pt"

def test_payload_hashes_rights_and_secrets():
    result = validate()
    assert result["status"] == "PASS", result["failures"]

def test_exact_pdf_hold_is_honest():
    result = thesis_status()
    assert result["status"] == "PASS"
    assert result["exact_build"] == "ON_HOLD"
    assert result["approved_pdf_sha256"] == "59b9a0da63f099ad7a6b8ad36e9f948f4c49bd4792e41b32212ce721b33339ad"
    assert not (ROOT / "LICENSE").exists()
    assert (ROOT / "LICENSE_PENDING.md").exists()

def test_projections_have_only_allowed_columns_and_numeric_ids():
    manifest = json.loads((ROOT / "provenance/numeric_projection.json").read_text())
    for item in manifest["files"]:
        path = ROOT / item["path"]
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", newline="") as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            assert reader.fieldnames == item["columns"]
            assert not set(reader.fieldnames) & {"question", "choices", "rationale", "rationale_gen", "description", "image", "image_path"}
            count = 0
            for row in reader:
                assert row["record_id"].isdigit() and row["source_frame_id"].isdigit()
                count += 1
            assert count == item["rows"]

def test_cpu_only_interfaces_import():
    from cli import _common
    from frozen_execution.scripts import production_store
    from scripts import build_public_tables
    assert callable(_common.resolve_config)
    assert callable(production_store.ManifestLockedStore)
    assert callable(build_public_tables.render_tables)


def test_workflows_are_cpu_only_and_thesis_is_manual():
    ci = yaml.load((ROOT / ".github/workflows/ci.yml").read_text(), Loader=yaml.BaseLoader)
    thesis = yaml.load((ROOT / ".github/workflows/thesis.yml").read_text(), Loader=yaml.BaseLoader)
    assert ci["permissions"] == {"contents": "read"}
    assert set(thesis["on"]) == {"workflow_dispatch"}
    assert thesis["permissions"] == {"contents": "read"}
    assert all("train" not in step.get("run", "") for step in ci["jobs"]["cpu"]["steps"])
    assert all("latexmk" not in step.get("run", "") for step in thesis["jobs"]["source-status"]["steps"])
    dev = json.loads((ROOT / ".devcontainer/devcontainer.json").read_text())
    assert dev["remoteUser"] == "researcher"
