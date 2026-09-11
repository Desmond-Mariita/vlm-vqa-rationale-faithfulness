#!/usr/bin/env python3
"""Manifest-locked production runner for final Stage-2 inference.

The runner accepts only frozen primary, known-change, or bounded
description-availability manifests. Local assets are hash-validated before use.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import re
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
from production_store import ManifestLockedStore, StoreError, atomic_json, canonical_hash, sha_file


MODEL_REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"
DATASET_SHA = "0288740b785021e667c12eb08ab2c222a65f80b2702d598a7ebb7b942b3a5fdc"
ALLOWED_MANIFESTS = {
    "f8349fa09918f6948cf8f831f5b125ead2acb897860898d458c7a6da98b1c6de": "primary",
    "a146fe45daced64087146257f82b6096f59602b2992769665363b0bf9eaee3ca": "known_change",
    "30a98b7aac15ad0725a971b735789003e3554bf66526baa7f98e4f3a33d47937": "description_ablation",
}
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "mlp.up_proj", "mlp.down_proj"]
STRICT_REASONING = re.compile(r"<reasoning>\s*(.*?)\s*</reasoning>", re.I | re.S)
STRICT_FINAL = re.compile(r"<final>\s*(.*?)\s*</final>", re.I | re.S)


def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def gpu_uuid() -> str | None:
    try:
        uuids = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True
        ).splitlines()
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
        return uuids[int(visible)].strip() if visible.isdigit() else uuids[0].strip()
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
        "historical_determinism": "seed-only; strict deterministic flags intentionally not added",
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
    from scripts.rationale_drift import prepare_lora
    from utils.cot_parser import parse_cot
    return prepare_lora, parse_cot


def caption(row: dict[str, Any]) -> str:
    return str((row.get("image_descriptions") or {}).get("caption") or "")[:256]


def build_prompt(row: dict[str, Any], arm: str, answer_index: int, description_mode: str) -> str:
    choices = list(row["choices"][:4])
    while len(choices) < 4:
        choices.append("")
    answer = choices[answer_index]
    prompt = f"""You are an expert visual commonsense reasoner.

You see an image and a multiple-choice question with four options.
Your task is to explain, step by step, why the given answer is correct,
using only what can reasonably be inferred from the image and question.

Question: {row['question']}

Choices:
  (0) {choices[0]}
  (1) {choices[1]}
  (2) {choices[2]}
  (3) {choices[3]}

The correct answer is choice index {answer_index}: "{answer}".

Explain your reasoning step by step, focusing only on what can be seen
in the image (and what follows directly from it).
Then give a short final justification.

Use the following format exactly:

<reasoning>
...detailed step-by-step reasoning about the image, question, and answer...
</reasoning>
<final>
...short 2-3 sentence justification...
</final>"""
    if arm.endswith("_desc"):
        if description_mode == "on" and caption(row):
            prompt += f"\n\nImage caption: {caption(row)}"
        elif description_mode == "off":
            prompt += "\n\nImage caption:"
    return prompt


def strict_parse(text: str) -> tuple[str, str, str]:
    reasoning = STRICT_REASONING.search(text)
    final = STRICT_FINAL.search(text)
    status = "both" if reasoning and final else "reasoning_only" if reasoning else "final_only" if final else "none"
    return reasoning.group(1).strip() if reasoning else "", final.group(1).strip() if final else "", status


def quantisation_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def load_model(model_dir: Path, checkpoint: Path, expected_checkpoint_sha: str, expected_canonical_sha: str, prepare_lora):
    if sha_file(checkpoint) != expected_checkpoint_sha:
        raise SystemExit("transport checkpoint SHA-256 mismatch")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state.get("canonical_source_sha256") != expected_canonical_sha:
        raise SystemExit("transport checkpoint canonical-source identity mismatch")
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(
        str(model_dir), revision=MODEL_REVISION, trust_remote_code=True, local_files_only=True,
    )
    model = AutoModelForVision2Seq.from_pretrained(
        str(model_dir), revision=MODEL_REVISION, device_map="auto", trust_remote_code=True,
        torch_dtype=None, quantization_config=quantisation_config(), local_files_only=True,
    )
    model = prepare_model_for_kbit_training(model)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True
    model = prepare_lora(model, {
        "use_lora": True, "r": 32, "alpha": 64, "dropout": 0.05, "target_modules": TARGET_MODULES,
    })
    current = model.state_dict()
    model_state = state.get("model_state") or state.get("state_dict_model")
    filtered = {key: value for key, value in model_state.items() if key in current and current[key].shape == value.shape}
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    model.to(torch.device("cuda:0"))
    model.eval()
    torch.cuda.synchronize()
    return model, processor, {
        "load_seconds": time.perf_counter() - started,
        "loaded_checkpoint_keys": len(filtered),
        "missing_checkpoint_keys": len(missing),
        "unexpected_checkpoint_keys": len(unexpected),
        "load_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def generate_one(model, processor, image: Image.Image, prompt: str) -> tuple[str, int, str, float]:
    chat = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}],
        add_generation_prompt=True, tokenize=False,
    )
    inputs = processor(text=[chat], images=[image], return_tensors="pt", padding=True)
    inputs = {key: value.to("cuda:0") if isinstance(value, torch.Tensor) else value for key, value in inputs.items()}
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.no_grad():
        generated = model.generate(
            **inputs, max_new_tokens=256, do_sample=False, num_beams=1, temperature=1.0, top_p=1.0,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    new_ids = generated[0][inputs["input_ids"].shape[1]:].tolist()
    eos = processor.tokenizer.eos_token_id
    pad = processor.tokenizer.pad_token_id
    effective: list[int] = []
    for token in new_ids:
        if token == pad and effective:
            break
        effective.append(token)
        if token == eos:
            break
    raw = processor.decode(effective, skip_special_tokens=True).strip() if effective else ""
    finish = "length" if len(effective) >= 256 and (not effective or effective[-1] != eos) else "eos_or_stop"
    return raw, len(effective), finish, elapsed


def condition_image(manifest_kind: str, mrow: dict[str, str], row: dict[str, Any], project_root: Path, image_cache: dict[str, str]) -> tuple[Image.Image, str, str]:
    source_path = project_root / mrow["source_image_asset"]
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    source_key = str(source_path)
    image_cache.setdefault(source_key, sha_file(source_path))
    if image_cache[source_key] != mrow["source_image_asset_sha256"]:
        raise RuntimeError(f"source asset hash mismatch: {source_path}")
    source = Image.open(source_path).convert("RGB")
    condition = mrow.get("condition", "")
    base_condition = mrow.get("base_image_condition", condition)
    if manifest_kind == "known_change":
        base_condition = "source"
    if base_condition == "source":
        return source, str(source_path), mrow["source_image_asset_sha256"]
    if base_condition == "grey":
        return Image.new("RGB", (256, 256), (128, 128, 128)), "constructor:grey_256_rgb_128", mrow.get("constructor_specification_sha256", "")
    if base_condition == "mismatch":
        target = project_root / mrow["intervention_asset"]
        target_key = str(target)
        image_cache.setdefault(target_key, sha_file(target))
        if image_cache[target_key] != mrow["intervention_asset_sha256"]:
            raise RuntimeError(f"mismatch asset hash mismatch: {target}")
        return Image.open(target).convert("RGB"), str(target), image_cache[target_key]
    if base_condition == "mirror":
        return source.transpose(Image.Transpose.FLIP_LEFT_RIGHT), "constructor:horizontal_mirror", mrow["constructor_specification_sha256"]
    array = np.array(source)
    index = int(mrow["dataset_index"])
    if base_condition == "mask":
        rng = np.random.default_rng(index)
        result = array.copy()
        result[rng.random(array.shape[:2]) < 0.5] = 0
        return Image.fromarray(result), f"constructor:mask:seed={index}", mrow["constructor_specification_sha256"]
    if base_condition == "noise":
        seed = 1_000_000 + index
        rng = np.random.default_rng(seed)
        result = np.clip(array.astype(np.int16) + rng.normal(0, 80, array.shape).astype(np.int16), 0, 255).astype(np.uint8)
        return Image.fromarray(result), f"constructor:noise:seed={seed}", mrow["constructor_specification_sha256"]
    raise ValueError(f"unsupported condition {base_condition!r}")



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
    ap.add_argument("--worker-id", type=int, choices=range(6), required=True)
    ap.add_argument("--max-records", type=int)
    ap.add_argument("--force-interrupt-after", type=int)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    manifest_sha = sha_file(args.manifest)
    if manifest_sha not in ALLOWED_MANIFESTS:
        raise SystemExit(f"manifest is not in frozen allow-list: {manifest_sha}")
    manifest_kind = ALLOWED_MANIFESTS[manifest_sha]
    if sha_file(args.dataset) != DATASET_SHA:
        raise SystemExit("dataset SHA-256 mismatch")
    local_assets_sha = validate_local_assets(args.local_assets)
    manifest_rows = read_tsv(args.manifest)
    by_key = {row["primary_key"]: row for row in manifest_rows}
    if len(by_key) != len(manifest_rows):
        raise SystemExit("duplicate primary key in full manifest")
    shard_rows = read_tsv(args.shard)
    if not shard_rows or any(int(row["worker_id"]) != args.worker_id for row in shard_rows):
        raise SystemExit("shard worker identity mismatch")
    if any(row["parent_manifest_sha256"] != manifest_sha for row in shard_rows):
        raise SystemExit("shard parent-manifest hash mismatch")
    selected: list[tuple[dict[str, str], dict[str, str]]] = []
    expected_rows: dict[str, str] = {}
    for shard_row in shard_rows:
        key = shard_row["primary_key"]
        if key not in by_key:
            raise SystemExit(f"shard key not found in manifest: {key}")
        row_hash = canonical_hash(by_key[key])
        if row_hash != shard_row["manifest_row_sha256"]:
            raise SystemExit(f"shard manifest-row hash mismatch: {key}")
        expected_rows[key] = row_hash
        selected.append((by_key[key], shard_row))
    if len(expected_rows) != len(shard_rows):
        raise SystemExit("duplicate primary key in shard")
    if args.max_records is not None and len(selected) > args.max_records:
        raise SystemExit("shard exceeds requested max-records bound")
    arm_values = {row[0]["arm"] for row in selected}
    condition_values = {row[0]["condition"] for row in selected}
    if len(arm_values) != 1 or len(condition_values) != 1:
        raise SystemExit("one production process may serve only one arm-condition shard")
    arm = next(iter(arm_values))
    condition = next(iter(condition_values))

    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    for mrow, _ in selected:
        index = int(mrow["dataset_index"])
        if dataset[index]["image_id"] != mrow["record_id"]:
            raise SystemExit(f"dataset/manifest ID mismatch at index {index}")
        if manifest_kind == "primary":
            if sha_text(dataset[index]["question"]) != mrow["question_sha256"]:
                raise SystemExit("question hash mismatch")
            if sha_text(canonical_json(dataset[index]["choices"])) != mrow["choices_sha256"]:
                raise SystemExit("choices hash mismatch")

    freeze_runtime()
    prepare_lora, parse_cot = configure_project(args.project_root)
    first = selected[0][0]
    transport_name = Path(first["transport_checkpoint"]).name
    checkpoint = args.transport_root / transport_name
    runtime = runtime_metadata()
    metadata = {
        "purpose": "final-stage2-production",
        "manifest_kind": manifest_kind,
        "manifest_path": str(args.manifest),
        "manifest_sha256": manifest_sha,
        "shard_path": str(args.shard),
        "shard_sha256": sha_file(args.shard),
        "worker_id": args.worker_id,
        "arm": arm,
        "condition": condition,
        "batch_size": 1,
        "seed": 42,
        "model_processor_revision": MODEL_REVISION,
        "canonical_checkpoint_sha256": first["canonical_checkpoint_sha256"],
        "transport_checkpoint_sha256": first["transport_checkpoint_sha256"],
        "runner_sha256": sha_file(Path(__file__)),
        "store_sha256": sha_file(Path(__file__).with_name("production_store.py")),
        "local_assets_sha256": local_assets_sha,
        "runtime": runtime,
    }
    store = ManifestLockedStore(args.output_dir, metadata, expected_rows)
    model, processor, load_meta = load_model(
        args.model_dir, checkpoint, first["transport_checkpoint_sha256"], first["canonical_checkpoint_sha256"], prepare_lora,
    )
    image_hash_cache: dict[str, str] = {}
    appended = skipped = generated_tokens = 0
    generation_seconds = 0.0
    for mrow, shard_row in selected:
        key = mrow["primary_key"]
        if key in store.completed:
            skipped += 1
            continue
        index = int(mrow["dataset_index"])
        drow = dataset[index]
        if manifest_kind == "known_change":
            answer_index = int(mrow["alternate_answer_index"])
            description_mode = "on"
            expected_prompt_sha = mrow["prompt_sha256"]
            answer_source = "deliberate_alternate_control"
        elif manifest_kind == "description_ablation":
            answer_index = int(mrow["supplied_gold_answer_index"])
            description_mode = "off"
            expected_prompt_sha = mrow["description_off_prompt_sha256"]
            answer_source = "gold"
        else:
            answer_index = int(mrow["supplied_gold_answer_index"])
            description_mode = "on"
            expected_prompt_sha = mrow["prompt_sha256"]
            answer_source = "gold"
        prompt = build_prompt(drow, arm, answer_index, description_mode)
        if sha_text(prompt) != expected_prompt_sha:
            raise SystemExit(f"prompt hash mismatch before generation: {key}")
        try:
            image, intervention_ref, intervention_sha = condition_image(manifest_kind, mrow, drow, args.project_root, image_hash_cache)
            raw, token_count, finish_reason, elapsed = generate_one(model, processor, image, prompt)
            strict_reasoning, strict_final, strict_status = strict_parse(raw)
            tolerant_reasoning, tolerant_final = parse_cot(raw)
            tolerant_status = "both" if tolerant_reasoning and tolerant_final else "reasoning_only" if tolerant_reasoning else "final_only" if tolerant_final else "none"
            result = {
                "primary_key": key,
                "manifest_row_sha256": expected_rows[key],
                "record_status": "success",
                "dataset_index": index,
                "record_id": mrow["record_id"],
                "source_frame_id": mrow["source_frame_id"],
                "arm": arm,
                "condition": condition,
                "answer_source": answer_source,
                "supplied_answer_index": answer_index,
                "supplied_answer_text": drow["choices"][answer_index],
                "prompt_sha256": expected_prompt_sha,
                "source_image_asset": mrow["source_image_asset"],
                "source_image_asset_sha256": mrow["source_image_asset_sha256"],
                "intervention_reference": intervention_ref,
                "intervention_sha256": intervention_sha,
                "raw_generation": raw,
                "strict_reasoning": strict_reasoning,
                "strict_final": strict_final,
                "strict_parser_status": strict_status,
                "tolerant_reasoning": tolerant_reasoning,
                "tolerant_final": tolerant_final,
                "tolerant_parser_status": tolerant_status,
                "generated_token_count": token_count,
                "finish_reason": finish_reason,
                "generation_seconds": elapsed,
                "gpu_uuid": runtime["gpu_uuid"],
                "worker_id": args.worker_id,
                "shard_sha256": metadata["shard_sha256"],
                "canonical_checkpoint_sha256": first["canonical_checkpoint_sha256"],
                "transport_checkpoint_sha256": first["transport_checkpoint_sha256"],
                "environment_sha256": runtime["environment_sha256"],
                "model_processor_revision": MODEL_REVISION,
                "output_schema_version": mrow["expected_output_schema_version"],
            }
            store.append(result)
            appended += 1
            generated_tokens += token_count
            generation_seconds += elapsed
            if args.force_interrupt_after and appended >= args.force_interrupt_after:
                os._exit(99)
        except Exception as exc:
            store.append_error({
                "primary_key": key,
                "manifest_row_sha256": expected_rows[key],
                "dataset_index": index,
                "record_id": mrow["record_id"],
                "arm": arm,
                "condition": condition,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "gpu_uuid": runtime["gpu_uuid"],
                "worker_id": args.worker_id,
                "environment_sha256": runtime["environment_sha256"],
            })
            raise
        print(f"worker={args.worker_id} arm={arm} condition={condition} completed={len(store.completed)}/{len(selected)}", flush=True)
    final_audit = store.final_audit()
    summary = {
        **load_meta,
        **final_audit,
        "appended_records": appended,
        "skipped_validated_records": skipped,
        "generation_seconds": generation_seconds,
        "generated_tokens": generated_tokens,
        "seconds_per_generation": generation_seconds / appended if appended else 0.0,
        "tokens_per_second": generated_tokens / generation_seconds if generation_seconds else 0.0,
        "generation_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
