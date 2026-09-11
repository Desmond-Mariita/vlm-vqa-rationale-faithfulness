"""Tests for config inheritance and mode resolution.

No GPU, no training. Pure-Python, fast.
"""

from __future__ import annotations

import os
import sys
import tempfile
import textwrap
import unittest
from unittest.mock import patch
import cli._common as common
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.yaml_config import load_yaml
from cli._common import resolve_config, prepare_run_dir


class IsolatedRunTest(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.root_patch = patch.object(common, "PROJECT_ROOT", Path(self.scratch.name))
        self.root_patch.start()
        cfg = load_yaml(str(PROJECT_ROOT / "configs/plain.yaml"))
        for key in ("json_train_by_mode", "json_val_by_mode"):
            for relative in cfg["data"][key].values():
                path = Path(self.scratch.name) / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("[]\n")

    def tearDown(self):
        self.root_patch.stop()
        self.scratch.cleanup()

class TestBaseInheritance(unittest.TestCase):
    def test_base_key_is_resolved_and_removed(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            (td_path / "base.yaml").write_text(textwrap.dedent("""\
                mode: thesis
                project:
                  name: "base_proj"
                  seed: 1
                train:
                  lr: 0.001
                  batch: 16
            """))
            (td_path / "child.yaml").write_text(textwrap.dedent("""\
                _base: base.yaml
                project:
                  seed: 42
                train:
                  lr: 0.0005
                experiment:
                  arm: "plain"
            """))
            cfg = load_yaml(str(td_path / "child.yaml"))
            self.assertNotIn("_base", cfg)
            # Inherited
            self.assertEqual(cfg["project"]["name"], "base_proj")
            self.assertEqual(cfg["train"]["batch"], 16)
            # Overridden
            self.assertEqual(cfg["project"]["seed"], 42)
            self.assertEqual(cfg["train"]["lr"], 0.0005)
            # New in child
            self.assertEqual(cfg["experiment"]["arm"], "plain")


class TestModeResolution(IsolatedRunTest):
    def test_thesis_mode_resolves_expected_splits(self):
        cfg_path = str(PROJECT_ROOT / "configs" / "plain.yaml")
        cfg = resolve_config(cfg_path, mode_override="thesis")
        self.assertEqual(cfg["mode"], "thesis")
        self.assertTrue(cfg["data"]["json_train_path"].endswith("vcr_train_5_pct.json"))
        self.assertTrue(cfg["data"]["json_val_path"].endswith("vcr_val_10_pct.json"))

    def test_debug_mode_resolves_expected_splits(self):
        cfg_path = str(PROJECT_ROOT / "configs" / "plain.yaml")
        cfg = resolve_config(cfg_path, mode_override="debug")
        self.assertTrue(cfg["data"]["json_train_path"].endswith("vcr_train_0.1_pct.json"))
        self.assertTrue(cfg["data"]["json_val_path"].endswith("vcr_val_1_pct.json"))

    def test_invalid_mode_raises(self):
        cfg_path = str(PROJECT_ROOT / "configs" / "plain.yaml")
        with self.assertRaises(ValueError):
            resolve_config(cfg_path, mode_override="bogus")


class TestPrepareRunDir(IsolatedRunTest):
    def test_sets_output_paths_and_writes_manifest(self):
        cfg = resolve_config(str(PROJECT_ROOT / "configs" / "plain.yaml"),
                             mode_override="debug")
        run = prepare_run_dir(cfg, suffix="stage1")
        run_dir = Path(run["run_dir"])
        try:
            self.assertTrue(run_dir.exists())
            self.assertTrue((run_dir / "manifest.json").exists())
            self.assertTrue((run_dir / "config.resolved.yaml").exists())
            # Paths are rewritten to point into run_dir
            self.assertIn(run_dir.name, cfg["project"]["save_dir"])
            self.assertIn(run_dir.name, cfg["project"]["output_dir"])
            self.assertIn(run_dir.name, cfg["paths"]["rationales_jsonl"])
        finally:
            # cleanup
            import shutil
            shutil.rmtree(run_dir, ignore_errors=True)


class TestGoldVsPredSeparation(IsolatedRunTest):
    """Structural test: gold and pred runs must go to different directories."""

    def test_different_suffixes_yield_different_dirs(self):
        cfg1 = resolve_config(str(PROJECT_ROOT / "configs" / "plain.yaml"),
                              mode_override="debug")
        cfg2 = resolve_config(str(PROJECT_ROOT / "configs" / "plain.yaml"),
                              mode_override="debug")
        r1 = prepare_run_dir(cfg1, suffix="stage2_gold")
        r2 = prepare_run_dir(cfg2, suffix="stage2_pred")
        try:
            self.assertIn("stage2_gold", r1["run_dir"])
            self.assertIn("stage2_pred", r2["run_dir"])
            self.assertNotEqual(r1["run_dir"], r2["run_dir"])
        finally:
            import shutil
            shutil.rmtree(r1["run_dir"], ignore_errors=True)
            shutil.rmtree(r2["run_dir"], ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
