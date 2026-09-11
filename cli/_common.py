"""Shared helpers for CLI entrypoints.

Responsibilities:
    * resolve config inheritance and mode-dependent paths
    * set deterministic seeds
    * build a run directory under reports/runs/<arm>/<suffix>/<run_id>/
    * emit a resolved config to disk for downstream scripts
    * emit a run manifest (JSON) with seed, git SHA, device, precision
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.yaml_config import load_yaml  # noqa: E402


VALID_MODES = {"debug", "thesis", "full", "replication", "replication_sample"}


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or "unknown"
    except Exception:
        return "unknown"


def _device_summary() -> Dict[str, Any]:
    info: Dict[str, Any] = {"cuda_available": False}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_device_count"] = torch.cuda.device_count()
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
            info["cuda_capability"] = list(torch.cuda.get_device_capability(0))
    except ImportError:
        pass
    return info


def set_all_seeds(seed: int) -> None:
    """Set Python, NumPy and torch seeds. Also forces cuDNN deterministic mode
    (benchmark=False, deterministic=True) and sets CUBLAS_WORKSPACE_CONFIG so that
    callers can optionally enable torch.use_deterministic_algorithms(True).

    Fail closed if torch is missing.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Needed by torch.use_deterministic_algorithms(True) on CUDA. Harmless to set
    # unconditionally; we don't force-enable the strict algorithm check here,
    # but exporting the workspace config keeps the door open.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError as e:
        raise RuntimeError("numpy is required for deterministic seeding") from e
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # cuDNN: pick deterministic algos and disable autotuner.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError as e:
        raise RuntimeError("torch is required for deterministic seeding") from e


def make_dataloader_seed_kwargs(seed: int) -> Dict[str, Any]:
    """Return kwargs to thread through `DataLoader(...)` for reproducible
    shuffling and per-worker RNG seeding.

    Usage:
        dl = DataLoader(ds, batch_size=..., shuffle=True,
                        **make_dataloader_seed_kwargs(cfg["project"]["seed"]))

    Returns a dict with `worker_init_fn` and `generator` keys. Safe when
    torch is unavailable (returns empty dict).
    """
    try:
        import numpy as np
        import torch
    except ImportError:
        return {}

    def _worker_init_fn(worker_id: int) -> None:
        worker_seed = (seed + worker_id) % (2**32)
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    gen = torch.Generator()
    gen.manual_seed(seed)
    return {"worker_init_fn": _worker_init_fn, "generator": gen}


def _env_versions() -> Dict[str, Any]:
    """Capture versions of key libraries and cuDNN for the run manifest."""
    versions: Dict[str, Any] = {}
    for mod_name in (
        "torch",
        "transformers",
        "peft",
        "accelerate",
        "bitsandbytes",
        "numpy",
        "bert_score",
        "sacrebleu",
    ):
        try:
            mod = __import__(mod_name)
            versions[mod_name] = getattr(mod, "__version__", "unknown")
        except Exception:
            versions[mod_name] = "not-installed"
    try:
        import torch

        versions["cudnn"] = (
            torch.backends.cudnn.version() if torch.cuda.is_available() else None
        )
        versions["cuda_runtime"] = (
            torch.version.cuda if torch.cuda.is_available() else None
        )
    except Exception:
        pass
    return versions


def _config_hash(cfg: Dict[str, Any]) -> str:
    payload = yaml.safe_dump(cfg, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def resolve_config(
    cfg_path: str,
    mode_override: Optional[str] = None,
) -> Dict[str, Any]:
    """Load a YAML config (with `_base` inheritance) and resolve mode-dependent paths.

    Mutates `data.json_train_path` and `data.json_val_path` to concrete files
    based on the selected mode.

    Args:
        cfg_path: path to the arm YAML config (relative or absolute)
        mode_override: if given, overrides `mode` in the loaded config

    Returns:
        resolved config dict

    Raises:
        FileNotFoundError: if the config or any resolved JSON split is missing
        ValueError: if `mode` is not one of {debug, thesis, full}
    """
    if not os.path.isabs(cfg_path):
        cfg_path = str(PROJECT_ROOT / cfg_path)
    cfg = load_yaml(cfg_path)

    mode = (mode_override or cfg.get("mode", "thesis")).strip().lower()
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")
    cfg["mode"] = mode

    train_map = cfg["data"].get("json_train_by_mode", {})
    val_map = cfg["data"].get("json_val_by_mode", {})
    if mode not in train_map or mode not in val_map:
        raise KeyError(
            f"mode '{mode}' not present in data.json_train_by_mode / json_val_by_mode"
        )
    cfg["data"]["json_train_path"] = train_map[mode]
    cfg["data"]["json_val_path"] = val_map[mode]

    # Debug mode: cap epochs so smoke tests finish in minutes.
    if mode == "debug":
        cfg.setdefault("train", {})["max_epochs_stage1"] = 1
        cfg["train"]["max_epochs_stage2"] = 1
        cfg.setdefault("train_stage2", {})["max_epochs"] = 1

    for key in ("json_train_path", "json_val_path"):
        p = cfg["data"][key]
        abs_p = p if os.path.isabs(p) else str(PROJECT_ROOT / p)
        if not os.path.exists(abs_p):
            raise FileNotFoundError(
                f"Resolved data split missing for mode={mode}: {abs_p}"
            )

    return cfg


def prepare_run_dir(cfg: Dict[str, Any], suffix: str,
                    reuse_run_dir: Optional[str] = None) -> Dict[str, Any]:
    """Create reports/runs/<arm>/<suffix>/<run_id>/ and write the resolved config.

    Also sets cfg["project"]["output_dir"|"log_dir"|"save_dir"] and
    cfg["paths"] so the downstream training scripts write into this run dir.

    Args:
        cfg: resolved config (from `resolve_config`)
        suffix: short run-kind tag, e.g. "stage1", "stage2_gold", "stage2_pred", "eval"
        reuse_run_dir: if given, reuse this exact existing run dir instead of
            minting a fresh timestamped one. This is the resume path: the
            trainer, pointed at the reused dir's save_dir, finds its existing
            checkpoints and continues. Creating it if absent is fine (first
            launch of a pinned-dir run).

    Returns:
        dict with run_id, run_dir, resolved_cfg_path, manifest_path, resumed
    """
    arm = cfg["experiment"]["arm"]
    resumed = False
    if reuse_run_dir:
        run_dir = Path(reuse_run_dir)
        if not run_dir.is_absolute():
            run_dir = PROJECT_ROOT / run_dir
        resumed = run_dir.exists() and (run_dir / "checkpoints").exists()
        run_id = run_dir.name
    else:
        run_id = time.strftime("%Y%m%d_%H%M%S")
        run_dir = PROJECT_ROOT / "reports" / "runs" / arm / suffix / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "outputs").mkdir(parents=True, exist_ok=True)

    # Arm configs hardcode experiment.name as "qwen_stage2_<arm>_gold"; rewrite
    # it for pred passes so downstream artifacts and W&B labels are honest.
    base_name = cfg["experiment"].get("base_name", f"qwen_stage2_{arm}")
    if suffix == "stage2_pred":
        cfg["experiment"]["name"] = f"{base_name}_pred"
    elif suffix == "stage2_gold":
        cfg["experiment"]["name"] = f"{base_name}_gold"
    elif suffix == "stage1":
        cfg["experiment"]["name"] = f"qwen_stage1_{arm}"

    # Rewrite paths so downstream scripts write into this run dir
    rel = str(run_dir.relative_to(PROJECT_ROOT))
    cfg.setdefault("project", {})
    cfg["project"]["output_dir"] = rel + "/"
    cfg["project"]["log_dir"] = rel + "/logs/"
    cfg["project"]["save_dir"] = rel + "/checkpoints/"
    cfg.setdefault("paths", {})
    cfg["paths"]["predictions_dir"] = rel + "/outputs/"
    cfg["paths"]["plots_dir"] = rel + "/plots/"
    cfg["paths"]["tables_dir"] = rel + "/tables/"
    cfg["paths"]["rationales_jsonl"] = rel + "/outputs/Route2/generated_rationales.jsonl"
    # Ask the dataset loader to persist its kept-IDs / drop-counts manifest into
    # this run's outputs/ so the CONSORT-style accounting is per-run auditable.
    os.environ["VQA_DATASET_MANIFEST_DIR"] = str(run_dir / "outputs")

    resolved_cfg_path = run_dir / "config.resolved.yaml"
    with open(resolved_cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=True)

    manifest = {
        "arm": arm,
        "suffix": suffix,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "mode": cfg.get("mode"),
        "seed": cfg["project"]["seed"],
        "config_hash": _config_hash(cfg),
        "git_sha": _git_sha(),
        "device": _device_summary(),
        "precision": cfg.get("train", {}).get("mixed_precision", "unknown"),
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "env_versions": _env_versions(),
        "backbone": {
            "model_name": cfg.get("backbone", {}).get("model_name"),
            "revision": cfg.get("backbone", {}).get("revision"),
        },
        "stage1_ckpt": cfg.get("train", {}).get("stage1_ckpt"),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "resumed": resumed,
    }
    manifest_path = run_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    # Also emit a JSON log so the caller can grep/parse
    log_line = {"event": "run_resumed" if resumed else "run_prepared", **manifest}
    with open(run_dir / "logs" / "cli.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(log_line) + "\n")
    if resumed:
        print(f"[cli] RESUMING existing run dir: {run_dir}", flush=True)

    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "resolved_cfg_path": str(resolved_cfg_path),
        "manifest_path": str(manifest_path),
        "resumed": resumed,
    }


def run_subprocess(cmd: list[str], cwd: Optional[str] = None, extra_env: Optional[Dict[str, str]] = None) -> int:
    """Run a command, stream stdout/stderr, return exit code. Fail-closed on non-zero.

    Wall-clock timing is appended to reports/runs/_meta/timings.jsonl so that
    a post-hoc compute-budget report can be produced for the thesis (which
    subset choices cost what, etc.).
    """
    env = os.environ.copy()
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    if "PYTORCH_CUDA_ALLOC_CONF" in env and "PYTORCH_ALLOC_CONF" not in env:
        env["PYTORCH_ALLOC_CONF"] = env["PYTORCH_CUDA_ALLOC_CONF"]
    if extra_env:
        env.update(extra_env)
    print(f"[cli] $ {' '.join(cmd)}", flush=True)

    timing_dir = PROJECT_ROOT / "reports" / "runs" / "_meta"
    timing_dir.mkdir(parents=True, exist_ok=True)
    timing_log = timing_dir / "timings.jsonl"
    start_ts = time.time()
    start_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(start_ts))
    proc = subprocess.run(cmd, cwd=cwd or str(PROJECT_ROOT), env=env)
    end_ts = time.time()
    end_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(end_ts))
    try:
        with open(timing_log, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "cmd": cmd,
                "start_utc": start_iso,
                "end_utc": end_iso,
                "duration_s": round(end_ts - start_ts, 2),
                "exit_code": int(proc.returncode),
            }) + "\n")
    except Exception as e:
        print(f"[cli] WARN: failed to write timing log: {e}", flush=True)

    if proc.returncode != 0:
        raise RuntimeError(f"Subprocess failed (exit={proc.returncode}): {' '.join(cmd)}")
    return proc.returncode


def print_resolved(cfg: Dict[str, Any]) -> None:
    """Print the resolved config to stdout for operator visibility."""
    print("=" * 72, flush=True)
    print("[cli] Resolved config:", flush=True)
    print(yaml.safe_dump(cfg, sort_keys=True), flush=True)
    print("=" * 72, flush=True)
