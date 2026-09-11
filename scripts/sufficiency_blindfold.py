#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sufficiency / Blindfold Test for Stage-1 (Answer Prediction)

Goal:
  Measure how much the model truly needs the image vs just the text.

Metrics:
  - accuracy_image:     accuracy with the real image
  - accuracy_no_image:  accuracy when the image is replaced by a blank dummy
  - sufficiency_gap:    accuracy_image - accuracy_no_image

Expected pattern (hypothesis):
  point_desc > point > plain_desc > plain
    i.e., caption + polygons may reduce "pure visual" dependency.

Outputs:
  reports/<arm>/<exp_name>/sufficiency_blindfold/epoch_xx/:
    - sufficiency_examples.csv
    - sufficiency_summary.json
    - sufficiency_table.tex
"""

from __future__ import annotations

import os
import sys
_PR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PR not in sys.path:
    sys.path.insert(0, _PR)
from utils import silence  # noqa: E402, F401

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ---- project imports ----
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.vqa_dataset import VQADataset, vqa_collate_fn
from models.mc_classifier import MCClassifier
from utils.yaml_config import load_yaml


# ---------------------------------------------------------
# LoRA helper (same pattern as Stage-1 / Stage-2)
# ---------------------------------------------------------
def prepare_lora(backbone: nn.Module, peft_cfg: Dict[str, Any]) -> nn.Module:
    """Wrap a backbone model with LoRA adapters.

    Args:
        backbone: Pre-trained vision-language model.
        peft_cfg: LoRA configuration dict with keys r, alpha,
            target_modules, dropout, use_lora.

    Returns:
        The backbone wrapped with LoRA (or unchanged if use_lora is False).
    """
    if not peft_cfg.get("use_lora", True):
        return backbone

    lora = LoraConfig(
        r=peft_cfg.get("r", 32),
        lora_alpha=peft_cfg.get("alpha", 64),
        target_modules=peft_cfg.get("target_modules", None),
        lora_dropout=peft_cfg.get("dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(backbone, lora)
    return model


# ---------------------------------------------------------
# Prompt + inputs + pooling (copied from Stage-1 logic)
# ---------------------------------------------------------
def format_prompt(stage1_template: str, question: str, choices: List[str], caption: Optional[str]) -> str:
    """Format a Stage-1 multiple-choice prompt from template and fields.

    Args:
        stage1_template: Format string with {question}, {c0}..{c3} placeholders.
        question: The VQA question text.
        choices: List of 4 answer choice strings.
        caption: Optional image caption, truncated to 256 chars if provided.

    Returns:
        Formatted prompt string ready for the model.
    """
    c0, c1, c2, c3 = choices
    prompt = stage1_template.format(question=question, c0=c0, c1=c1, c2=c2, c3=c3)
    if caption:
        prompt += f"\nCaption: {caption[:256]}"
    return prompt


def build_model_inputs_from_prompts(processor, images, prompts, device):
    """Build batched model inputs from images and text prompts.

    Args:
        processor: HuggingFace processor (tokenizer + image processor).
        images: List of PIL images for the batch.
        prompts: List of text prompt strings for the batch.
        device: Torch device to move tensors to.

    Returns:
        Dict of batched, device-placed model input tensors.
    """
    chats = []
    for p in prompts:
        chats.append(
            processor.apply_chat_template(
                [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": p}]}],
                add_generation_prompt=False,
                tokenize=False,
            )
        )
    mi = processor(text=chats, images=images, return_tensors="pt", padding=True)
    for k, v in list(mi.items()):
        if isinstance(v, torch.Tensor):
            mi[k] = v.to(device)
    return mi


def _get_nested_attr(obj, *names):
    """Return the first matching attribute from obj, or None."""
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return None


_last_dec_output: Dict[str, torch.Tensor] = {}


def pool_decoder(backbone: nn.Module, model_inputs: Dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    """
    Forward once WITHOUT output_hidden_states and pool from the text/decoder side.
    Returns pooled [B, H].

    This mirrors the logic used in train_answer_qwen.py
    (simplified for evaluation only).
    """
    _last_dec_output.clear()
    outputs = backbone(**model_inputs, use_cache=False)

    dec_last = (
        getattr(outputs, "decoder_last_hidden_state", None)
        or getattr(outputs, "last_hidden_state", None)
        or _get_nested_attr(outputs, "language_model_output", "last_hidden_state")
        or _last_dec_output.get("hidden", None)
    )

    if dec_last is None:
        # Fallback with hidden_states if needed
        outputs = backbone(**model_inputs, use_cache=False, output_hidden_states=True)
        if hasattr(outputs, "decoder_hidden_states") and outputs.decoder_hidden_states is not None:
            dec_last = outputs.decoder_hidden_states[-1]
        elif hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            dec_last = outputs.hidden_states[-1]
        else:
            raise RuntimeError("Cannot find decoder/last_hidden_state for pooling.")

    # Last-token pooling — must match Stage-1 exactly
    # (train_answer_qwen.py uses attention_mask.sum - 1). Mean-pooling here
    # would evaluate a different representation than the classifier was
    # trained on, invalidating the sufficiency-gap metric.
    attention_mask = model_inputs.get("attention_mask", None)
    if attention_mask is not None:
        idx = attention_mask.sum(dim=1) - 1  # [B]
        pooled = dec_last[torch.arange(dec_last.size(0), device=device), idx]
    else:
        pooled = dec_last[:, -1, :]
    return pooled


# ---------------------------------------------------------
# Stage-1 checkpoint loader (minimal)
# ---------------------------------------------------------
def load_stage1_backbone_and_head(cfg: Dict[str, Any], device: torch.device):
    """
    Build Qwen backbone + MCClassifier head and load Stage-1 best checkpoint.
    Mirrors train_answer_qwen.py but only for inference.
    """
    model_name = cfg["backbone"]["model_name"]
    train_cfg = cfg["train"]
    peft_cfg = cfg.get("peft", {})

    use_4bit = bool(train_cfg.get("load_in_4bit", False))
    use_8bit = bool(train_cfg.get("load_in_8bit", False)) and not use_4bit
    dtype = torch.bfloat16 if train_cfg.get("mixed_precision") == "bf16" else torch.float16

    if use_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
    elif use_8bit:
        quant_cfg = BitsAndBytesConfig(load_in_8bit=True)
    else:
        quant_cfg = None

    # Backbone
    backbone = AutoModelForVision2Seq.from_pretrained(
        model_name,
        device_map="auto" if (use_4bit or use_8bit) else None,
        trust_remote_code=True,
        torch_dtype=None if (use_4bit or use_8bit) else dtype,
        quantization_config=quant_cfg,
    )

    if use_4bit or use_8bit:
        backbone = prepare_model_for_kbit_training(backbone)

    # Disable cache / enable gradient checkpointing if needed
    if hasattr(backbone.config, "use_cache"):
        backbone.config.use_cache = False

    # Apply LoRA
    backbone = prepare_lora(backbone, peft_cfg)

    # Build classification head
    hidden_size = getattr(backbone.config, "hidden_size", getattr(backbone.config, "d_model", 4096))
    head = MCClassifier(input_dim=hidden_size, num_classes=4)

    backbone.to(device)
    head.to(device)

    # Load checkpoint. Prefer the new-layout Stage-1 checkpoints dir.
    paths_cfg = cfg.get("paths", {}) or {}
    stage1_ckpt_rel = paths_cfg.get("stage1_checkpoints_dir")
    if stage1_ckpt_rel:
        ckpt_dir = Path(stage1_ckpt_rel)
        if not ckpt_dir.is_absolute():
            ckpt_dir = PROJECT_ROOT / ckpt_dir
    else:
        ckpt_dir = Path(cfg["project"]["save_dir"])
        if not ckpt_dir.is_absolute():
            ckpt_dir = PROJECT_ROOT / ckpt_dir
    ckpt_path = ckpt_dir / "model_best.pt"
    if ckpt_path.exists():
        print(f"[Sufficiency] Loading Stage-1 checkpoint: {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cpu")
        model_sd = state.get("state_dict_model") or state.get("model_state")
        head_sd = state.get("state_dict_head") or {}

        if model_sd is not None:
            missing, unexpected = backbone.load_state_dict(model_sd, strict=False)
            print(f"[Sufficiency] Backbone loaded (missing={len(missing)}, unexpected={len(unexpected)})")
        else:
            print("[Sufficiency] WARNING: checkpoint missing model weights; using base backbone.")

        if head_sd:
            missing_h, unexpected_h = head.load_state_dict(head_sd, strict=False)
            print(f"[Sufficiency] Head loaded (missing={len(missing_h)}, unexpected={len(unexpected_h)})")
        else:
            print("[Sufficiency] WARNING: no head weights in checkpoint; head is random.")
    else:
        print(f"[Sufficiency] WARNING: Stage-1 checkpoint not found at {ckpt_path}; using base model weights.")

    backbone.eval()
    head.eval()
    return backbone, head, ckpt_path


# ---------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------
def main():
    """Run the sufficiency/blindfold test on the validation set.

    Compares Stage-1 answer prediction accuracy with real images vs.
    blank dummy images, then saves per-example CSV, summary JSON,
    and a LaTeX table to the reports directory.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True, help="Path to base.yaml")
    parser.add_argument("--epoch", type=int, default=None,
                        help="Epoch label (only used for naming outputs).")
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="Limit number of examples for speed.")
    args = parser.parse_args()

    cfg = load_yaml(args.cfg)
    data_cfg = cfg["data"]
    exp_cfg = cfg["experiment"]
    arm = exp_cfg["arm"]

    # For naming outputs; if None, use 0
    epoch_tag = args.epoch if args.epoch is not None else 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset & loader (VAL split)
    ds_val = VQADataset(
        json_train_path=data_cfg["json_train_path"],
        json_val_path=data_cfg["json_val_path"],
        images_plain_train=data_cfg["images_plain_train"],
        images_plain_val=data_cfg["images_plain_val"],
        images_point_train=data_cfg["images_point_train"],
        images_point_val=data_cfg["images_point_val"],
        split=data_cfg["split_val"],
        arm=arm,
        caption_key=exp_cfg.get("caption_key", "image_descriptions.caption"),
        drop_missing=True,
    )

    dl_val = DataLoader(
        ds_val,
        batch_size=max(1, cfg["loader"].get("batch_size", 8)),
        shuffle=False,
        num_workers=max(1, cfg["loader"].get("num_workers", 2)),
        pin_memory=cfg["loader"].get("pin_memory", True),
        drop_last=False,
        collate_fn=vqa_collate_fn,
    )

    # Processor + Stage-1 model
    model_name = cfg["backbone"]["model_name"]
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    backbone, head, ckpt_path = load_stage1_backbone_and_head(cfg, device)

    stage1_tmpl = cfg["prompts"]["stage1_template"]
    use_caption = arm.endswith("_desc")

    # Dummy grey image for blindfold condition
    dummy_img = Image.new("RGB", (256, 256), (128, 128, 128))

    total = 0
    correct_image = 0
    correct_blind = 0

    rows: List[Dict[str, Any]] = []

    print(f"[Sufficiency] Evaluating arm={arm} on VAL split (blindfold test)...")

    for batch in tqdm(dl_val):
        images = batch["images"]
        questions = batch["questions"]
        choices_list = batch["choices"]
        labels = batch["labels"]
        captions = batch["captions"]
        image_ids = batch.get("image_id", [None] * len(images))

        B = len(images)
        if args.max_samples > 0 and total >= args.max_samples:
            break

        # Prompts
        prompts = [
            format_prompt(
                stage1_tmpl,
                questions[i],
                choices_list[i],
                captions[i] if (use_caption and i < len(captions)) else None,
            )
            for i in range(B)
        ]

        # 1) With real image
        model_inputs_img = build_model_inputs_from_prompts(processor, images, prompts, device)
        with torch.no_grad():
            pooled_img = pool_decoder(backbone, model_inputs_img, device)
            logits_img = head(pooled_img)
            preds_img = logits_img.argmax(dim=-1)

        # 2) Blindfold (dummy image)
        dummy_batch = [dummy_img] * B
        model_inputs_blind = build_model_inputs_from_prompts(processor, dummy_batch, prompts, device)
        with torch.no_grad():
            pooled_blind = pool_decoder(backbone, model_inputs_blind, device)
            logits_blind = head(pooled_blind)
            preds_blind = logits_blind.argmax(dim=-1)

        labels_t = torch.tensor(labels, device=device, dtype=torch.long)

        # Cap to max_samples BEFORE scoring so a partial last batch doesn't
        # inflate correct/total counts past the intended sample size.
        if args.max_samples > 0:
            remaining = args.max_samples - total
            if remaining <= 0:
                break
            effective_B = min(B, remaining)
        else:
            effective_B = B

        correct_image += (preds_img[:effective_B] == labels_t[:effective_B]).sum().item()
        correct_blind += (preds_blind[:effective_B] == labels_t[:effective_B]).sum().item()
        total += effective_B

        for i in range(effective_B):
            rows.append(
                {
                    "image_id": image_ids[i],
                    "question": questions[i],
                    "choices": choices_list[i],
                    "label": int(labels[i]),
                    "pred_image": int(preds_img[i].item()),
                    "pred_blind": int(preds_blind[i].item()),
                    "correct_image": int(preds_img[i].item() == labels[i]),
                    "correct_blind": int(preds_blind[i].item() == labels[i]),
                }
            )

        if args.max_samples > 0 and total >= args.max_samples:
            break

    accuracy_image = correct_image / max(1, total)
    accuracy_no_image = correct_blind / max(1, total)
    sufficiency_gap = accuracy_image - accuracy_no_image

    print(f"[Sufficiency] total={total}")
    print(f"[Sufficiency] accuracy_image    = {accuracy_image:.4f}")
    print(f"[Sufficiency] accuracy_no_image = {accuracy_no_image:.4f}")
    print(f"[Sufficiency] sufficiency_gap   = {sufficiency_gap:.4f}")

    # -----------------------------------------------------
    # Save outputs
    # -----------------------------------------------------
    exp_name = exp_cfg.get("name", f"qwen_stage1_{arm}")
    # Prefer eval_run_dir so this metric's outputs live under the current eval run.
    paths_cfg = cfg.get("paths", {}) or {}
    eval_run_rel = paths_cfg.get("eval_run_dir")
    if eval_run_rel:
        eval_run_dir = Path(eval_run_rel)
        if not eval_run_dir.is_absolute():
            eval_run_dir = PROJECT_ROOT / eval_run_dir
        out_dir = eval_run_dir / "sufficiency_blindfold"
    else:
        out_dir = (
            Path("reports")
            / arm
            / exp_name
            / "sufficiency_blindfold"
            / f"epoch_{epoch_tag:02d}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "sufficiency_examples.csv", index=False)

    summary = {
        "arm": arm,
        "epoch": epoch_tag,
        "num_samples": int(total),
        "accuracy_image": float(accuracy_image),
        "accuracy_no_image": float(accuracy_no_image),
        "sufficiency_gap": float(sufficiency_gap),
        "stage1_checkpoint": str(ckpt_path),
        "checkpoint_existed": bool(ckpt_path.exists()),
    }

    with open(out_dir / "sufficiency_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # LaTeX table
    latex = f"""
\\begin{{table}}[h]
\\centering
\\begin{{tabular}}{{l c}}
\\hline
Arm & {arm} \\\\
Epoch & {epoch_tag} \\\\
Samples & {total} \\\\
Accuracy (with image) & {accuracy_image:.4f} \\\\
Accuracy (no image) & {accuracy_no_image:.4f} \\\\
Sufficiency gap & {sufficiency_gap:.4f} \\\\
\\hline
\\end{{tabular}}
\\caption{{Sufficiency / Blindfold Test for arm={arm}, epoch={epoch_tag}.}}
\\end{{table}}
"""
    with open(out_dir / "sufficiency_table.tex", "w") as f:
        f.write(latex)

    print(f"[Sufficiency] Saved outputs → {out_dir}")


if __name__ == "__main__":
    from utils.perf import time_main
    raise SystemExit(time_main(main, "sufficiency_blindfold"))
