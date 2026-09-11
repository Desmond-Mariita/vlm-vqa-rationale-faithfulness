#!/usr/bin/env python3
"""Dedicated manifest-locked RTX-3090 runner for the final Stage-1 wrong cell."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training

from local_assets import validate_local_assets
from production_store import ManifestLockedStore, atomic_json, canonical_hash, sha_file


MODEL_REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"
DATASET_SHA = "0288740b785021e667c12eb08ab2c222a65f80b2702d598a7ebb7b942b3a5fdc"
MANIFEST_SHA = "e5db9cd53751b7b7b9b5c0f3429c1dd518a5edb6cc56580207a531327934ff12"
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "mlp.up_proj", "mlp.down_proj"]


def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def gpu_uuid() -> str | None:
    try:
        return subprocess.check_output(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True).splitlines()[0].strip()
    except Exception:
        return None


def runtime_metadata() -> dict[str, Any]:
    result = {
        "hostname": platform.node(),
        "python": sys.version.replace("\n", " "),
        "packages": {name: package_version(name) for name in (
            "torch", "torchvision", "transformers", "peft", "accelerate", "bitsandbytes",
            "numpy", "Pillow", "PyYAML", "safetensors", "huggingface-hub", "tokenizers",
        )},
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_capability": list(torch.cuda.get_device_capability(0)),
        "gpu_uuid": gpu_uuid(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    result["environment_sha256"] = canonical_hash(result)
    return result


def freeze_runtime() -> None:
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    os.environ.setdefault("PYTHONHASHSEED", "42")


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def configure_project(project_root: Path):
    os.chdir(project_root)
    sys.path.insert(0, str(project_root))
    from cli._common import resolve_config
    from scripts.sufficiency_blindfold import (
        build_model_inputs_from_prompts, format_prompt, pool_decoder, prepare_lora,
    )
    from models.mc_classifier import MCClassifier
    return resolve_config, build_model_inputs_from_prompts, format_prompt, pool_decoder, prepare_lora, MCClassifier


def quantisation_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def load_model(model_dir: Path, checkpoint: Path, expected_transport: str, expected_canonical: str, prepare_lora, MCClassifier):
    if sha_file(checkpoint) != expected_transport:
        raise SystemExit("transport checkpoint SHA-256 mismatch")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state.get("canonical_source_sha256") != expected_canonical:
        raise SystemExit("transport checkpoint canonical-source identity mismatch")
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(str(model_dir), revision=MODEL_REVISION, trust_remote_code=True, local_files_only=True)
    backbone = AutoModelForVision2Seq.from_pretrained(
        str(model_dir), revision=MODEL_REVISION, device_map="auto", trust_remote_code=True,
        torch_dtype=None, quantization_config=quantisation_config(), local_files_only=True,
    )
    backbone = prepare_model_for_kbit_training(backbone)
    if hasattr(backbone.config, "use_cache"):
        backbone.config.use_cache = False
    backbone = prepare_lora(backbone, {
        "use_lora": True, "r": 32, "alpha": 64, "dropout": 0.05, "target_modules": TARGET_MODULES,
    })
    hidden_size = getattr(backbone.config, "hidden_size", getattr(backbone.config, "d_model", 4096))
    head = MCClassifier(input_dim=hidden_size, num_classes=4)
    missing, unexpected = backbone.load_state_dict(state["state_dict_model"], strict=False)
    missing_head, unexpected_head = head.load_state_dict(state["state_dict_head"], strict=False)
    backbone.to(torch.device("cuda:0")); head.to(torch.device("cuda:0"))
    backbone.eval(); head.eval(); torch.cuda.synchronize()
    return backbone, head, processor, {
        "load_seconds": time.perf_counter() - started,
        "missing_checkpoint_keys": len(missing), "unexpected_checkpoint_keys": len(unexpected),
        "missing_head_keys": len(missing_head), "unexpected_head_keys": len(unexpected_head),
        "load_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--shard", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--project-root", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--transport-root", type=Path, required=True)
    ap.add_argument("--local-assets", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--max-records", type=int)
    ap.add_argument("--force-interrupt-after", type=int)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if sha_file(args.manifest) != MANIFEST_SHA:
        raise SystemExit("Stage-1 manifest is not the frozen allow-listed manifest")
    if sha_file(args.dataset) != DATASET_SHA:
        raise SystemExit("dataset SHA-256 mismatch")
    local_assets_sha = validate_local_assets(args.local_assets)
    manifest_rows = read_tsv(args.manifest)
    by_key = {row["primary_key"]: row for row in manifest_rows}
    shard_rows = read_tsv(args.shard)
    selected: list[dict[str, str]] = []
    expected_rows: dict[str, str] = {}
    for shard in shard_rows:
        if shard["parent_manifest_sha256"] != MANIFEST_SHA or shard["primary_key"] not in by_key:
            raise SystemExit("shard is incompatible with frozen Stage-1 manifest")
        row = by_key[shard["primary_key"]]
        row_hash = canonical_hash(row)
        if row_hash != shard["manifest_row_sha256"]:
            raise SystemExit("Stage-1 shard row hash mismatch")
        selected.append(row); expected_rows[row["primary_key"]] = row_hash
    if args.max_records is not None and len(selected) > args.max_records:
        raise SystemExit("shard exceeds requested max-records bound")
    arms = {row["arm"] for row in selected}
    if len(arms) != 1:
        raise SystemExit("one Stage-1 process may serve only one arm shard")
    arm = next(iter(arms))
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    for row in selected:
        if dataset[int(row["dataset_index"])]["image_id"] != row["record_id"]:
            raise SystemExit("dataset/manifest ID mismatch")

    freeze_runtime()
    resolve_config, build_inputs, format_prompt, pool_decoder, prepare_lora, MCClassifier = configure_project(args.project_root)
    cfg = resolve_config(f"configs/{arm}.yaml", mode_override="thesis")
    template = cfg["prompts"]["stage1_template"]
    first = selected[0]
    checkpoint = args.transport_root / Path(first["transport_checkpoint"]).name
    runtime = runtime_metadata()
    metadata = {
        "purpose": "final-stage1-wrong-production", "manifest_sha256": MANIFEST_SHA,
        "shard_sha256": sha_file(args.shard), "arm": arm, "condition": "mismatch",
        "batch_size": 2, "seed": 42, "model_processor_revision": MODEL_REVISION,
        "canonical_checkpoint_sha256": first["canonical_checkpoint_sha256"],
        "transport_checkpoint_sha256": first["transport_checkpoint_sha256"],
        "runner_sha256": sha_file(Path(__file__)),
        "store_sha256": sha_file(Path(__file__).with_name("production_store.py")),
        "local_assets_sha256": local_assets_sha, "runtime": runtime,
    }
    store = ManifestLockedStore(args.output_dir, metadata, expected_rows)
    backbone, head, processor, load_meta = load_model(
        args.model_dir, checkpoint, first["transport_checkpoint_sha256"], first["canonical_checkpoint_sha256"], prepare_lora, MCClassifier,
    )
    appended = skipped = 0
    inference_seconds = 0.0
    image_cache: dict[str, str] = {}
    for offset in range(0, len(selected), 2):
        pending = [row for row in selected[offset:offset + 2] if row["primary_key"] not in store.completed]
        skipped += len(selected[offset:offset + 2]) - len(pending)
        if not pending:
            continue
        images, prompts = [], []
        for row in pending:
            target = args.project_root / row["target_asset"]
            image_cache.setdefault(str(target), sha_file(target))
            if image_cache[str(target)] != row["target_asset_sha256"]:
                raise SystemExit(f"target asset hash mismatch: {target}")
            drow = dataset[int(row["dataset_index"])]
            cap = (drow.get("image_descriptions") or {}).get("caption") if arm.endswith("_desc") else None
            prompt = format_prompt(template, drow["question"], drow["choices"], cap)
            if sha_text(prompt) != row["prompt_sha256"]:
                raise SystemExit(f"Stage-1 prompt hash mismatch: {row['primary_key']}")
            images.append(Image.open(target).convert("RGB")); prompts.append(prompt)
        inputs = build_inputs(processor, images, prompts, torch.device("cuda:0"))
        torch.cuda.synchronize(); started = time.perf_counter()
        with torch.no_grad():
            pooled = pool_decoder(backbone, inputs, torch.device("cuda:0")); logits = head(pooled)
        torch.cuda.synchronize(); elapsed = time.perf_counter() - started
        inference_seconds += elapsed
        probs = torch.softmax(logits, dim=-1).cpu().tolist(); predictions = logits.argmax(dim=-1).cpu().tolist(); raw_logits = logits.cpu().tolist()
        for row, prompt, pred, probability, row_logits in zip(pending, prompts, predictions, probs, raw_logits):
            gold = int(row["gold_label"])
            store.append({
                "primary_key": row["primary_key"], "manifest_row_sha256": expected_rows[row["primary_key"]],
                "record_status": "success", "dataset_index": int(row["dataset_index"]),
                "record_id": row["record_id"], "source_frame_id": row["source_frame_id"],
                "arm": arm, "condition": "mismatch", "gold_label": gold,
                "prediction": int(pred), "correct": int(pred) == gold,
                "logits": row_logits, "probabilities": probability,
                "prompt_sha256": row["prompt_sha256"], "target_asset": row["target_asset"],
                "target_asset_sha256": row["target_asset_sha256"],
                "inference_seconds": elapsed / len(pending), "gpu_uuid": runtime["gpu_uuid"],
                "canonical_checkpoint_sha256": first["canonical_checkpoint_sha256"],
                "transport_checkpoint_sha256": first["transport_checkpoint_sha256"],
                "environment_sha256": runtime["environment_sha256"], "output_schema_version": row["output_schema_version"],
            })
            appended += 1
            if args.force_interrupt_after and appended >= args.force_interrupt_after:
                os._exit(99)
    audit = store.final_audit()
    summary = {**load_meta, **audit, "appended_records": appended, "skipped_validated_records": skipped,
               "inference_seconds": inference_seconds, "seconds_per_example": inference_seconds / appended if appended else 0.0,
               "inference_peak_reserved_bytes": torch.cuda.max_memory_reserved()}
    atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
