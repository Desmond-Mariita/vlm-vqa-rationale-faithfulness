#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 2 — Rationale Generation (Qwen2.5-VL + LoRA, CoT-style)

– Reuses VQADataset with the same 'arm' knob as Stage 1
– Loads Stage-1 best checkpoint (backbone + LoRA), ignores MC head
– Conditions on (image, question, answer, optional caption)
– Trains Qwen as a causal LM to generate CoT-style rationales:
    <reasoning>...step-by-step...</reasoning>
    <final>...short justification...</final>
– Saves per-epoch JSONL of:
    {
      image_id, question, choices, answer_idx(_gold/_pred),
      rationale_gold,
      rationale_reasoning,
      rationale_final,
      rationale_gen,              # alias of rationale_final
      bleu,
      cot_final_agreement,
      cot_reasoning_jaccard,
      ...
    }
  plus BLEU + analysis assets.

Run:
  cd .
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  accelerate launch scripts/train_rationale_qwen.py
"""

from __future__ import annotations
import os
import sys

# Ensure utils/ is importable, then suppress library noise before heavy imports.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from utils import silence  # noqa: E402, F401
from utils.cot_parser import parse_cot as _shared_parse_cot  # noqa: E402

import json
import math
import textwrap
import random
import subprocess
import time
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont

import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm

from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    get_cosine_schedule_with_warmup,
    BitsAndBytesConfig,
)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ---------- project imports ----------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.vqa_dataset import VQADataset, vqa_collate_fn
from utils.yaml_config import load_yaml


# ---------- Image + tile utilities (mirrors Stage 1 style) ----------


def _safe_open_image(path: str, size_max: int = 512) -> Image.Image | None:
    """Open an image safely and shrink large images for faster rendering."""
    if not path:
        return None
    try:
        im = Image.open(path).convert("RGB")
        im.thumbnail((size_max, size_max), Image.Resampling.LANCZOS)
        return im
    except Exception:
        return None


def _build_image_path(image_id: str | None, root_dir: str | None) -> str | None:
    """Join image_id with root_dir and return the path if it exists, else None."""
    if not image_id or not root_dir:
        return None
    p = os.path.join(root_dir, image_id)
    return p if os.path.exists(p) else None


def _render_rationale_tile(
    img: Image.Image | None,
    meta: dict,
    out_path: str,
    title: str,
    dpi: int = 300,
) -> None:
    """Render a Stage-2 tile with the image and rationale metadata.

    The top row shows the image; the bottom row displays identifiers,
    gold and generated rationales, and metric scores (BLEU, BERTScore,
    BLEURT).
    """

    def safe(v):
        return str(v) if v is not None else "None"

    # Basic identifiers
    image_id = safe(meta.get("image_id"))
    orig_image_id = safe(meta.get("orig_image_id"))
    question = safe(meta.get("question"))
    caption = meta.get("caption") or ""

    # Choices + chosen answer text
    choices = meta.get("choices") or []
    choice_str = (
        " | ".join([f"[{i}] {c}" for i, c in enumerate(choices)]) if choices else "N/A"
    )
    answer_idx = meta.get("answer_idx", None)
    if isinstance(answer_idx, int) and 0 <= answer_idx < len(choices):
        answer_text = choices[answer_idx]
    else:
        answer_text = "N/A"

    # Rationale + metrics
    gold_rat = meta.get("rationale_gold") or ""
    gen_rat = meta.get("rationale_gen") or ""
    bleu = meta.get("bleu", float("nan"))
    bert_f1 = meta.get("bertscore_f1", float("nan"))
    bleurt = meta.get("bleurt", float("nan"))

    caption_wrapped = "\n".join(textwrap.wrap(caption, width=90))
    gold_wrapped = "\n".join(textwrap.wrap(gold_rat, width=90))
    gen_wrapped = "\n".join(textwrap.wrap(gen_rat, width=90))

    metric_parts = []
    if not math.isnan(bleu):
        metric_parts.append(f"BLEU: {bleu:.4f}")
    if isinstance(bert_f1, (int, float)) and not math.isnan(float(bert_f1)):
        metric_parts.append(f"BERTScore F1: {float(bert_f1):.4f}")
    if isinstance(bleurt, (int, float)) and not math.isnan(float(bleurt)):
        metric_parts.append(f"BLEURT: {float(bleurt):.4f}")
    metrics_line = " | ".join(metric_parts) if metric_parts else "BLEU: N/A"

    info = (
        f"image_id: {image_id}\n"
        f"orig_image_id: {orig_image_id}\n"
        f"answer_idx: {answer_idx} ({answer_text})\n"
        f"{metrics_line}\n"
        f"question: {question}\n"
        f"choices: {choice_str}\n"
        f"caption: {caption_wrapped}\n"
        f"gold rationale: {gold_wrapped}\n"
        f"generated rationale: {gen_wrapped}"
    )

    fig, axes = plt.subplots(
        2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [3, 2]}
    )
    ax_img, ax_text = axes
    for ax in axes:
        ax.axis("off")

    if img is not None:
        ax_img.imshow(img)
        ax_img.set_title(title, fontsize=13, pad=8, fontweight="bold")
    else:
        ax_img.text(
            0.5,
            0.5,
            "Image not found",
            ha="center",
            va="center",
            fontsize=14,
        )

    ax_text.text(
        0,
        1.0,
        info,
        va="top",
        ha="left",
        fontsize=9,
        family="monospace",
        linespacing=1.3,
        wrap=True,
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _render_contact_sheet(
    tile_paths: list[str],
    out_path: str,
    cols: int = 4,
    title: str = "",
    dpi: int = 300,
) -> None:
    """Assemble individual tile PNGs into a grid contact sheet image."""
    tiles = [Image.open(p).convert("RGB") for p in tile_paths if os.path.exists(p)]
    if not tiles:
        return

    w, h = tiles[0].size
    rows = int(math.ceil(len(tiles) / max(1, cols)))
    sheet = Image.new("RGB", (cols * w, rows * h), (255, 255, 255))

    for i, im in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet.paste(im, (c * w, r * h))

    if title:
        band_h = 60
        title_band = Image.new("RGB", (sheet.width, band_h), (240, 240, 240))
        draw = ImageDraw.Draw(title_band)
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", 32)
        except Exception:
            font = ImageFont.load_default()

        try:
            bbox = draw.textbbox((0, 0), title, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            tw, th = font.getsize(title)

        draw.text(
            ((sheet.width - tw) // 2, (band_h - th) // 2),
            title,
            fill=(0, 0, 0),
            font=font,
        )

        sheet = ImageOps.expand(
            sheet, border=(0, band_h, 0, 0), fill=(255, 255, 255)
        )
        sheet.paste(title_band, (0, 0))

    try:
        sheet.save(out_path, dpi=(dpi, dpi))
    except TypeError:
        sheet.save(out_path)


def _export_rationale_samples_json(samples: list[dict], out_path: str) -> None:
    """Save per-sample rationale metadata (Top/Bottom-50) to a JSON file.

    Each record includes image IDs, question, choices, caption, answer,
    gold and generated rationales, metric scores, and image path existence
    flags.
    """
    records = []
    for s in samples:
        choices = s.get("choices") or []
        answer_idx = s.get("answer_idx", None)
        if isinstance(answer_idx, int) and 0 <= answer_idx < len(choices):
            answer_text = choices[answer_idx]
        else:
            answer_text = None

        rec = {
            "image_id": s.get("image_id"),
            "orig_image_id": s.get("orig_image_id"),
            "question": s.get("question"),
            "choices": choices,
            "caption": s.get("caption"),
            "answer_idx": answer_idx,
            "answer_text": answer_text,
            "rationale_gold": s.get("rationale_gold"),
            "rationale_gen": s.get("rationale_gen"),
            "bleu": s.get("bleu"),
            "bertscore_f1": s.get("bertscore_f1"),
            "bleurt": s.get("bleurt"),
            "image_path_plain": s.get("_plain_path"),
            "image_path_point": s.get("_point_path"),
            "image_path_visual": s.get("_visual_path"),
        }

        for k in ("_plain_path", "_point_path", "_visual_path"):
            path = s.get(k)
            rec[f"exists_{k[1:]}"] = bool(path and os.path.exists(path))

        records.append(rec)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


# --- optional BLEU (will silently disable if nltk not installed) -------------
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

    _nltk_bleu = True
    _smooth = SmoothingFunction().method1
except Exception:
    _nltk_bleu = False
    _smooth = None


# ---------- small utils ----------


def set_seed(seed: int):
    """Set random seeds for reproducibility across all backends.

    Args:
        seed: Integer seed value for random number generators.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _ensure_dir(p: str | Path) -> Path:
    """Ensure directory exists and return it as a Path."""
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _save_jsonl(path: str | Path, rows: List[Dict[str, Any]]):
    """Write a list of dicts as newline-delimited JSON (JSONL)."""
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# Set by main() when cfg["train_stage2"]["stage1_preds_val"] is provided.
_STAGE1_PREDS_VAL_OVERRIDE: Dict[str, Any] = {}


def _load_stage1_pred_map(
    project_cfg: Dict[str, Any],
) -> Dict[Tuple[Any, Any], Dict[str, Any]]:
    """Load Stage-1 validation predictions as a lookup mapping.

    Scans the latest epoch directory under the Stage-1 analysis folder and
    reads the per-sample JSONL predictions file.

    Args:
        project_cfg: The "project" section of the YAML config, which must
            contain "save_dir" pointing to the Stage-1 report root.

    Returns:
        Dict mapping (image_id, question) tuples to their full prediction
        records. Returns an empty dict if files are missing.
    """
    # NEW: prefer the explicit CLI-provided stage1 val preds path when set.
    # cli/run_stage2.py writes cfg["train_stage2"]["stage1_preds_val"] = <abs path>.
    # We read it here via a module-level lookup into the outer cfg (passed below).
    explicit_path = _STAGE1_PREDS_VAL_OVERRIDE.get("path")
    if explicit_path:
        p = Path(explicit_path)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if p.exists():
            print(f"[Stage2] Loading Stage-1 val preds (explicit): {p}")
            mapping: Dict[Tuple[Any, Any], Dict[str, Any]] = {}
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # Accept both per-line JSONL and a single JSON list/dict.
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(rec, list):
                        for r in rec:
                            mapping[(r.get("image_id"), r.get("question"))] = r
                        continue
                    mapping[(rec.get("image_id"), rec.get("question"))] = rec
            print(f"[Stage2] Explicit-path preds loaded: {len(mapping)} examples")
            return mapping
        print(f"[Stage2] WARN: explicit stage1 preds path not found: {p}; falling back to legacy scan.")

    save_dir = project_cfg.get("save_dir")
    if not save_dir:
        print("[Stage2] WARN: project.save_dir not set; cannot load Stage-1 preds.")
        return {}

    analysis_root = PROJECT_ROOT / save_dir / "analysis"
    if not analysis_root.exists():
        print(
            f"[Stage2] WARN: Stage-1 analysis dir {analysis_root} not found; using gold answers."
        )
        return {}

    epoch_dirs = [d for d in analysis_root.glob("epoch_*") if d.is_dir()]
    if not epoch_dirs:
        print(
            f"[Stage2] WARN: no epoch_* dirs found under {analysis_root}; using gold answers."
        )
        return {}

    def _ep(d: Path) -> int:
        try:
            return int(d.name.split("_")[1])
        except Exception:
            return -1

    epoch_dirs = sorted(epoch_dirs, key=_ep)
    best_dir = epoch_dirs[-1]
    epoch_num = _ep(best_dir)
    if epoch_num < 0:
        print(
            f"[Stage2] WARN: could not parse epoch number from {best_dir}; using gold answers."
        )
        return {}

    preds_path = best_dir / f"val_preds_epoch{epoch_num}.jsonl"
    if not preds_path.exists():
        print(f"[Stage2] WARN: {preds_path} not found; using gold answers.")
        return {}

    mapping: Dict[Tuple[Any, Any], Dict[str, Any]] = {}
    with preds_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            key = (rec.get("image_id"), rec.get("question"))
            mapping[key] = rec

    print(f"[Stage2] Loaded {len(mapping)} Stage-1 val preds from {preds_path}")
    return mapping


# ---------- dataloaders (reuse arm knob) ----------


def build_dataloaders(cfg: Dict[str, Any]) -> Dict[str, DataLoader]:
    """Construct train and validation DataLoaders from the YAML config.

    Uses the same arm-aware VQADataset as Stage 1, with batch sizes
    sourced from the ``batch_size_stage2`` config key.

    Args:
        cfg: Full YAML configuration dictionary containing "data",
            "experiment", and "loader" sections.

    Returns:
        Dict with keys "train" and "val" mapping to DataLoader instances.
    """
    data, exp, ld = cfg["data"], cfg["experiment"], cfg["loader"]
    seed = int(cfg.get("project", {}).get("seed", 42))
    from cli._common import make_dataloader_seed_kwargs  # local import to avoid cycles
    seed_kwargs = make_dataloader_seed_kwargs(seed)

    ds_train = VQADataset(
        json_train_path=data["json_train_path"],
        json_val_path=data["json_val_path"],
        images_plain_train=data["images_plain_train"],
        images_plain_val=data["images_plain_val"],
        images_point_train=data["images_point_train"],
        images_point_val=data["images_point_val"],
        split=data["split_train"],
        arm=exp["arm"],
        caption_key=exp.get("caption_key", "image_descriptions.caption"),
        drop_missing=True,
    )

    ds_val = VQADataset(
        json_train_path=data["json_train_path"],
        json_val_path=data["json_val_path"],
        images_plain_train=data["images_plain_train"],
        images_plain_val=data["images_plain_val"],
        images_point_train=data["images_point_train"],
        images_point_val=data["images_point_val"],
        split=data["split_val"],
        arm=exp["arm"],
        caption_key=exp.get("caption_key", "image_descriptions.caption"),
        drop_missing=True,
    )

    dl_train = DataLoader(
        ds_train,
        batch_size=ld.get("batch_size_stage2", ld.get("batch_size", 1)),
        shuffle=True,
        num_workers=ld.get("num_workers", 2),
        pin_memory=ld.get("pin_memory", True),
        drop_last=ld.get("drop_last", False),
        collate_fn=vqa_collate_fn,
        **seed_kwargs,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=max(1, ld.get("batch_size_stage2", ld.get("batch_size", 1))),
        shuffle=False,
        num_workers=max(1, ld.get("num_workers", 2)),
        pin_memory=ld.get("pin_memory", True),
        drop_last=False,
        collate_fn=vqa_collate_fn,
        **seed_kwargs,
    )
    return {"train": dl_train, "val": dl_val}


# ---------- LoRA ----------


def prepare_lora(model, peft_cfg: Dict[str, Any]):
    """Wrap a model with LoRA adapters based on the PEFT configuration.

    Args:
        model: Base model to apply LoRA to.
        peft_cfg: Dict with LoRA hyperparameters (r, alpha, target_modules,
            dropout, use_lora).

    Returns:
        The LoRA-wrapped model, or the original model if use_lora is False.
    """
    if not peft_cfg.get("use_lora", True):
        return model
    lora = LoraConfig(
        r=peft_cfg.get("r", 32),
        lora_alpha=peft_cfg.get("alpha", 64),
        target_modules=peft_cfg.get("target_modules", None),
        lora_dropout=peft_cfg.get("dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora)


# ---------- prompt helpers ----------


def format_stage2_prompt(
    template: str,
    question: str,
    choices: List[str],
    answer_idx: int,
    caption: Optional[str],
) -> str:
    """Format a Stage-2 rationale prompt with question, choices, answer, and optional caption.

    Args:
        template: Format string with placeholders {question}, {c0}-{c3},
            {answer}, {answer_idx}.
        question: The VQA question text.
        choices: List of exactly 4 answer choice strings.
        answer_idx: Index (0-3) of the answer to condition on.
        caption: Optional image caption; appended with instructions not to
            copy it into the rationale.

    Returns:
        The fully formatted prompt string.
    """
    c0, c1, c2, c3 = choices
    answer_txt = choices[answer_idx] if 0 <= answer_idx < len(choices) else ""
    prompt = template.format(
        question=question,
        c0=c0,
        c1=c1,
        c2=c2,
        c3=c3,
        answer=answer_txt,
        answer_idx=answer_idx,
    )
    if caption:
        prompt += (
            "\nCaption (context only — do NOT copy this text in your reasoning): "
            f"{caption[:256]}"
            "\nYour explanation MUST NOT reuse the caption. Use only the image + question + answer."
        )


    return prompt

def pack_chat_batch(
    processor,
    images: List[Any],
    prompts: List[str],
    rationales: List[str],
    device: torch.device,
):
    """Build Qwen chat-format training inputs with user prompts and assistant rationales.

    Labels = input_ids (with padding positions masked to -100). The pre-
    archive runs trained on the full sequence and produced functional
    rationales with low empty-fraction; an attempt at masking prompt
    tokens cut gradient signal too aggressively for the configured 3
    epochs, so we are deliberately back to the archive regime here.
    """
    chats = []
    for p, rat in zip(prompts, rationales):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": p},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": rat},
                ],
            },
        ]
        chats.append(
            processor.apply_chat_template(
                messages,
                add_generation_prompt=False,
                tokenize=False,
            )
        )

    model_inputs = processor(
        text=chats,
        images=images,
        return_tensors="pt",
        padding=True,
    )
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    model_inputs["labels"] = labels

    for k, v in list(model_inputs.items()):
        if isinstance(v, torch.Tensor):
            model_inputs[k] = v.to(device)
    return model_inputs


def pack_chat_for_generation(processor, images, prompts, device):
    """Build Qwen chat-format inputs for generation (user turn only).

    Applies the chat template with ``add_generation_prompt=True`` so the
    model can generate the assistant (rationale) response.

    Args:
        processor: HuggingFace processor with chat template support.
        images: List of PIL images, one per sample.
        prompts: List of text prompts, one per sample.
        device: Target torch device for the returned tensors.

    Returns:
        Dict of batched model input tensors ready for ``model.generate()``.
    """
    chats = []
    for p in prompts:
        chats.append(
            processor.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": p},
                        ],
                    }
                ],
                add_generation_prompt=True,  # assistant prefix for generation
                tokenize=False,
            )
        )
    model_inputs = processor(
        text=chats,
        images=images,
        return_tensors="pt",
        padding=True,
    )
    for k, v in list(model_inputs.items()):
        if isinstance(v, torch.Tensor):
            model_inputs[k] = v.to(device)
    return model_inputs


# ---------- Stage-2 checkpoint helpers (mirrors Stage-1 logic) ----------


def _save_stage2_checkpoint(
    path,
    accelerator,
    model,
    optimizer,
    scheduler,
    epoch,
    best_bleu,
    cfg,
    global_step: int = 0,
    epoch_step: int = 0,
):
    """Save a Stage-2 checkpoint with model weights and optimizer/scheduler state.

    Also writes a lightweight weights-only sidecar file alongside the main
    checkpoint. Used for end-of-epoch, latest, and intra-epoch step saves.

    Args:
        path: Destination file path for the checkpoint.
        accelerator: HuggingFace Accelerator for distributed state extraction.
        model: The LoRA-wrapped backbone model.
        optimizer: Optimizer whose state will be saved.
        scheduler: LR scheduler whose state will be saved.
        epoch: Current epoch number.
        best_bleu: Best validation BLEU score seen so far.
        cfg: Full YAML config dict to store in the checkpoint.
        global_step: Current global training step count.
        epoch_step: Current step within the epoch (for intra-epoch resume).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model_sd = accelerator.get_state_dict(model)

    ckpt = {
        "epoch": epoch,
        "best_bleu": best_bleu,
        "model_state": model_sd,
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "cfg": cfg,
        "global_step": int(global_step),
        "epoch_step": int(epoch_step),
    }
    accelerator.save(ckpt, path)

    # Lightweight weights-only sidecar
    weights_path = Path(path).with_suffix(".weights.pt")
    accelerator.save({"state_dict": model_sd}, weights_path)


def _load_stage2_checkpoint(path, accelerator, model, optimizer, scheduler):
    """Load a Stage-2 epoch-level checkpoint and restore training state.

    Args:
        path: Path to the checkpoint file. Skipped if empty or missing.
        accelerator: Accelerator instance for logging.
        model: Model to load weights into.
        optimizer: Optimizer to restore state for.
        scheduler: LR scheduler to restore state for.

    Returns:
        A tuple of (start_epoch, best_bleu) where start_epoch is 1-based.
    """
    if (not path) or (not os.path.exists(path)):
        accelerator.print(f"[Stage2] No resume checkpoint found at {path}; starting fresh.")
        return 1, -1.0

    accelerator.print(f"[Stage2] Resuming Stage-2 training from: {path}")
    state = torch.load(path, map_location="cpu")

    if "model_state" in state:
        missing, unexpected = model.load_state_dict(state["model_state"], strict=False)
        accelerator.print(
            f"[Stage2] Loaded model_state (missing={len(missing)}, unexpected={len(unexpected)})"
        )
    else:
        accelerator.print("[Stage2] WARNING: checkpoint missing model_state")

    if optimizer and ("optimizer" in state) and (state["optimizer"] is not None):
        try:
            optimizer.load_state_dict(state["optimizer"])
        except Exception as e:
            accelerator.print(f"[Stage2] WARNING: failed to load optimizer state: {e}")

    if scheduler and ("scheduler" in state) and (state["scheduler"] is not None):
        try:
            scheduler.load_state_dict(state["scheduler"])
        except Exception as e:
            accelerator.print(f"[Stage2] WARNING: failed to load scheduler state: {e}")

    start_epoch = int(state.get("epoch", 0)) + 1
    best_bleu = float(state.get("bleu", state.get("best_bleu", -1.0)))

    accelerator.print(
        f"[Stage2] Will resume at epoch {start_epoch} (best BLEU so far = {best_bleu:.4f})"
    )
    return start_epoch, best_bleu


def _load_stage2_step_checkpoint(path, accelerator, model, optimizer, scheduler):
    """Load an intra-epoch step-level checkpoint for mid-epoch resume.

    Args:
        path: Path to the step checkpoint file. Returns None if missing.
        accelerator: Accelerator instance for logging.
        model: Model to load weights into.
        optimizer: Optimizer to restore state for.
        scheduler: LR scheduler to restore state for.

    Returns:
        A dict with keys (epoch, epoch_step, global_step, best_bleu) if
        the checkpoint exists, or None otherwise.
    """
    if (not path) or (not os.path.exists(path)):
        return None

    accelerator.print(f"[Stage2] Resuming from intra-epoch step checkpoint: {path}")
    state = torch.load(path, map_location="cpu")

    if "model_state" in state:
        missing, unexpected = model.load_state_dict(state["model_state"], strict=False)
        accelerator.print(
            f"[Stage2] Loaded model_state (missing={len(missing)}, unexpected={len(unexpected)})"
        )
    else:
        accelerator.print("[Stage2] WARNING: step checkpoint missing model_state")

    if optimizer and ("optimizer" in state):
        try:
            optimizer.load_state_dict(state["optimizer"])
        except Exception as e:
            accelerator.print(f"[Stage2] WARNING: optimizer restore failed: {e}")

    if scheduler and ("scheduler" in state):
        try:
            scheduler.load_state_dict(state["scheduler"])
        except Exception as e:
            accelerator.print(f"[Stage2] WARNING: scheduler restore failed: {e}")

    return {
        "epoch": int(state.get("epoch", 1)),
        "epoch_step": int(state.get("epoch_step", 0)),
        "global_step": int(state.get("global_step", 0)),
        "best_bleu": float(state.get("best_bleu", -1.0)),
    }


# ---------- BLEU (simple) ----------


def _bleu_safe(ref: str, hyp: str) -> float:
    """Compute sentence-level BLEU with smoothing, returning NaN if nltk is unavailable."""
    if not _nltk_bleu:
        return float("nan")
    ref_toks = ref.split()
    hyp_toks = hyp.split()
    if not ref_toks or not hyp_toks:
        return 0.0
    return float(sentence_bleu([ref_toks], hyp_toks, smoothing_function=_smooth))


def _clean_generated_rationale(text: str) -> str:
    """Strip obvious prompt and system boilerplate from a generated rationale."""
    if not isinstance(text, str):
        return ""

    text = text.replace("\r", " ").strip()
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]

    drop_prefixes = (
        "system",
        "System",
        "You are a helpful assistant",
        "You are an expert visual commonsense reasoner",
        "Consider the following multiple-choice",
    )
    while lines and any(lines[0].startswith(p) for p in drop_prefixes):
        lines.pop(0)

    cleaned = " ".join(lines).strip()
    return cleaned or text

def safe_batch_decode(processor, sequences, skip_special_tokens: bool = True) -> List[str]:
    """Decode token sequences with a fallback for Qwen2 None-token edge cases.

    Attempts standard ``processor.batch_decode`` first. If that raises a
    TypeError (due to None tokens), falls back to per-sequence token
    conversion that drops None entries.

    Args:
        processor: HuggingFace processor whose tokenizer will be used.
        sequences: Tensor or list-of-lists of token IDs.
        skip_special_tokens: Whether to omit special tokens in the output.

    Returns:
        List of decoded strings, one per sequence.
    """
    try:
        return processor.batch_decode(sequences, skip_special_tokens=skip_special_tokens)
    except TypeError as e:
        # Fallback: convert ids -> tokens, drop None, then join.
        tok = processor.tokenizer
        out: List[str] = []

        if isinstance(sequences, torch.Tensor):
            seq_list = sequences.detach().cpu().tolist()
        else:
            # list of lists already
            seq_list = [list(s) for s in sequences]

        for ids in seq_list:
            # convert ids to tokens; skip special tokens if asked
            toks = tok.convert_ids_to_tokens(ids, skip_special_tokens=skip_special_tokens)
            # drop None tokens defensively
            toks = [t for t in toks if isinstance(t, str)]
            text = tok.convert_tokens_to_string(toks)
            out.append(text)

        return out

# ---------- CoT helpers ----------


def _build_cot_target(gold_rationale: str) -> str:
    """Convert a gold rationale into a CoT-style target with XML tags.

    Splits the rationale into reasoning (all but the last sentence) and
    final justification (last sentence), wrapping each in
    ``<reasoning>...</reasoning>`` and ``<final>...</final>`` tags.

    Args:
        gold_rationale: The reference rationale text.

    Returns:
        A string with ``<reasoning>`` and ``<final>`` XML sections.
    """
    if not isinstance(gold_rationale, str):
        gold_rationale = ""
    text = gold_rationale.strip()

    if not text:
        return "<reasoning></reasoning>\n<final></final>"

    # --- Heuristic split: first 1–2 sentences = reasoning; last sentence = final
    sentences = re.split(r'(?<=[.!?]) +', text)
    if len(sentences) == 1:
        reasoning = sentences[0]
        final = sentences[0]
    else:
        reasoning = " ".join(sentences[:-1])
        final = sentences[-1]

    return (
        "<reasoning>\n"
        f"{reasoning}\n"
        "</reasoning>\n"
        "<final>\n"
        f"{final}\n"
        "</final>"
    )


def _parse_cot(text: str) -> Tuple[str, str]:
    """Parse CoT ``<reasoning>``/``<final>`` spans from a generated rationale.

    Thin delegate to the shared, malformation-tolerant parser in
    ``utils/cot_parser.py`` (single source of truth per
    docs/EXPERIMENTAL_FINDINGS.md §5.8.1 / §11). The shared parser tolerates
    whitespace drift, tag-name misspellings (``<reasonning>``), stray in-tag
    punctuation (``<final'>``), and tags missing the closing ``>``
    (``<final) ...``); it falls back to a last-sentence split when no usable
    tags are present.

    Returns:
        A tuple of (reasoning_text, final_text).
    """
    return _shared_parse_cot(text)


def _jaccard_tokens(a: str, b: str) -> float:
    """Compute token-level Jaccard similarity between two strings."""
    a_tokens = set(a.lower().split())
    b_tokens = set(b.lower().split())
    if not a_tokens and not b_tokens:
        return 1.0
    if not a_tokens or not b_tokens:
        return 0.0
    inter = len(a_tokens & b_tokens)
    union = len(a_tokens | b_tokens)
    return inter / max(1, union)


def _cot_stats(reasonings: List[str], finals: List[str]) -> Dict[str, float]:
    """Compute CoT consistency statistics from multiple sampled outputs.

    Measures majority agreement among final justifications and average
    pairwise Jaccard similarity among reasoning sections.

    Args:
        reasonings: List of reasoning texts from multiple CoT samples.
        finals: List of final justification texts from multiple CoT samples.

    Returns:
        Dict with keys "cot_final_agreement" and "cot_reasoning_jaccard".
    """
    k = len(finals)
    if k == 0:
        return {
            "cot_final_agreement": float("nan"),
            "cot_reasoning_jaccard": float("nan"),
        }
    if k == 1:
        return {
            "cot_final_agreement": 1.0,
            "cot_reasoning_jaccard": 1.0,
        }

    from collections import Counter

    counts = Counter([f.strip() for f in finals if f.strip()])
    if counts:
        majority = max(counts.values())
        final_agreement = majority / k
    else:
        final_agreement = 0.0

    sims: List[float] = []
    for i in range(k):
        for j in range(i + 1, k):
            sims.append(_jaccard_tokens(reasonings[i], reasonings[j]))
    reasoning_jaccard = float(sum(sims) / max(1, len(sims))) if sims else 1.0

    return {
        "cot_final_agreement": final_agreement,
        "cot_reasoning_jaccard": reasoning_jaccard,
    }


# ---------- main ----------


def main():
    """Entry point for Stage-2 rationale generation training.

    Loads configuration from --cfg (required), builds dataloaders,
    initialises the Qwen2.5-VL backbone with LoRA (warm-started from
    Stage-1 weights), then runs the CoT-style rationale generation
    training loop with per-epoch BLEU evaluation, checkpointing, and
    analysis report generation.
    """
    import argparse
    parser = argparse.ArgumentParser(description="Stage-2 rationale generation training")
    parser.add_argument("--cfg", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--answer-source", type=str, default=None,
                        choices=["gold", "pred"],
                        help="Override train_stage2.answer_source from config")
    args, _ = parser.parse_known_args()
    cfg_path_str = args.cfg
    if not os.path.isabs(cfg_path_str):
        cfg_path_str = str(PROJECT_ROOT / cfg_path_str)
    if not os.path.exists(cfg_path_str):
        raise FileNotFoundError(f"Config not found: {cfg_path_str}")
    print(f"[train_rationale_qwen] Loading config: {cfg_path_str}", flush=True)
    cfg = load_yaml(cfg_path_str)
    if args.answer_source is not None:
        cfg.setdefault("train_stage2", {})["answer_source"] = args.answer_source
        print(f"[train_rationale_qwen] Override answer_source -> {args.answer_source}", flush=True)

    project_cfg = cfg["project"]
    train_cfg = cfg.get("train", {})
    stage2_cfg = cfg.get("train_stage2", {})
    data_cfg = cfg["data"]
    exp_cfg = cfg["experiment"]
    eval_stage2_cfg = cfg.get("eval_stage2", {})

    # gold vs predicted answer source for Stage-2 conditioning
    answer_source = stage2_cfg.get("answer_source", "gold").lower()
    use_pred_answers = answer_source == "pred"

    # Inference-only mode: skip training, load a pre-trained Stage-2 checkpoint,
    # run a single eval+generation pass. Used for the pred alternate-inference
    # pass so stage2_pred reuses the gold-trained checkpoint instead of
    # retraining a separate model.
    inference_only = bool(stage2_cfg.get("inference_only", False))
    stage2_ckpt_load_path = stage2_cfg.get("stage2_ckpt") if inference_only else None
    if inference_only and not stage2_ckpt_load_path:
        raise ValueError(
            "train_stage2.inference_only=True requires train_stage2.stage2_ckpt "
            "(path to the gold-trained Stage-2 checkpoint)."
        )
    if inference_only and not os.path.exists(stage2_ckpt_load_path):
        raise FileNotFoundError(
            f"train_stage2.stage2_ckpt not found: {stage2_ckpt_load_path}"
        )

    # Prompt template for *validation* generation
    prompt_template_eval = stage2_cfg["stage2_prompt_template"]
    if use_pred_answers:
        prompt_template_eval = stage2_cfg.get(
            "stage2_prompt_template_pred", stage2_cfg["stage2_prompt_template"]
        )

    set_seed(project_cfg.get("seed", 1234))

    # Stage-2 specific mixed precision, fallback to global train if missing
    mp = stage2_cfg.get("mixed_precision", train_cfg.get("mixed_precision", "no"))
    grad_accum = int(stage2_cfg.get("grad_accum", 1))

    accelerator = Accelerator(
        mixed_precision=mp,
        gradient_accumulation_steps=grad_accum,
    )
    device = accelerator.device

    # Optional Stage-1 val predictions map for answer_source="pred"
    # Register the CLI-provided explicit path (if any) so _load_stage1_pred_map sees it.
    _STAGE1_PREDS_VAL_OVERRIDE.clear()
    explicit_preds_val = stage2_cfg.get("stage1_preds_val")
    if explicit_preds_val:
        _STAGE1_PREDS_VAL_OVERRIDE["path"] = explicit_preds_val

    stage1_pred_map: Dict[Tuple[Any, Any], Dict[str, Any]] = {}
    if use_pred_answers:
        stage1_pred_map = _load_stage1_pred_map(project_cfg)
        accelerator.print(
            f"[Stage2] answer_source='pred'; "
            f"conditioning val rationales on Stage-1 predicted answers "
            f"({len(stage1_pred_map)} mapped examples)."
        )
    else:
        accelerator.print(
            "[Stage2] answer_source='gold'; conditioning on gold answers."
        )

    # paths
    exp_name = exp_cfg.get("name")
    base_exp_name = exp_cfg.get("base_name", exp_name)

    # Root folder for STAGE 2.
    # If the CLI set project.output_dir (new run-dir layout), use it. This keeps
    # stage2_gold and stage2_pred in separate, explicit run directories and
    # prevents gold/pred collision. Fall back to the legacy layout for scripts
    # invoked outside the new CLI.
    _new_output_dir = project_cfg.get("output_dir")
    if _new_output_dir:
        _out = Path(_new_output_dir)
        if not _out.is_absolute():
            _out = PROJECT_ROOT / _out
        stage2_root = _ensure_dir(_out)
        # In the new layout, save_dir already points to <run>/checkpoints/
        _save_dir = project_cfg.get("save_dir")
        if _save_dir:
            _sd = Path(_save_dir)
            if not _sd.is_absolute():
                _sd = PROJECT_ROOT / _sd
            stage2_ckpt_dir = _ensure_dir(_sd)
        else:
            stage2_ckpt_dir = _ensure_dir(stage2_root / "checkpoints")
    else:
        stage2_root = _ensure_dir(
            PROJECT_ROOT / "reports" / exp_cfg.get("arm", "point") / exp_name
        )
        stage2_ckpt_dir = _ensure_dir(stage2_root / "checkpoints_stage2")

    # ---------------------- Stage-2 resume config ----------------------
    # Epoch-level resume hint (YAML: train.resume_from_stage2)
    resume_path = (train_cfg.get("resume_from_stage2") or "").strip()
    if not resume_path:
        auto = stage2_ckpt_dir / "latest.pt"
        if auto.exists():
            resume_path = str(auto)

    # Step-level resume checkpoint (auto-managed by this script)
    step_resume_path = stage2_ckpt_dir / "latest_step.pt"

    accelerator.print(
        f"[Stage2] Epoch-level resume checkpoint = {resume_path or '<none>'}"
    )
    accelerator.print(
        f"[Stage2] Step-level resume checkpoint  = {step_resume_path if step_resume_path.exists() else '<none>'}"
    )

    # Where Stage 2 will save rationales.
    # New layout: cfg["paths"]["rationales_jsonl"] points at <run>/outputs/Route2/generated_rationales.jsonl
    # so results_root = parent of that file.
    _paths = cfg.get("paths", {}) or {}
    _rj = _paths.get("rationales_jsonl")
    if _rj:
        _rjp = Path(_rj)
        if not _rjp.is_absolute():
            _rjp = PROJECT_ROOT / _rjp
        results_root = _ensure_dir(_rjp.parent)
    else:
        results_root = _ensure_dir(stage2_root / "rationales")

    # data
    loaders = build_dataloaders(cfg)
    dl_train, dl_val = loaders["train"], loaders["val"]

    # backbone + LoRA — must match Stage 1 exactly
    model_name = cfg["backbone"]["model_name"]
    processor_name = cfg["backbone"].get("processor_name", model_name)
    revision = cfg["backbone"].get("revision") or None

    # --- quantization flags: SAME as train_answer_qwen.py ---
    use_4bit = bool(train_cfg.get("load_in_4bit", False))
    use_8bit = bool(train_cfg.get("load_in_8bit", False)) and not use_4bit
    dtype = (
        torch.bfloat16
        if train_cfg.get("mixed_precision") == "bf16"
        else torch.float16
    )

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

    # --- processor: same as Stage 1 (and shrink images if needed) ---
    processor = AutoProcessor.from_pretrained(
        processor_name,
        trust_remote_code=True,
        use_fast=False,
        revision=revision,
    )

    # Avoid "decoder-only right-padding" warnings
    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        processor.tokenizer.padding_side = "left"
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token

    try:
        if hasattr(processor, "image_processor") and hasattr(
            processor.image_processor, "size"
        ):
            size = processor.image_processor.size
            if isinstance(size, dict):
                if "shortest_edge" in size:
                    processor.image_processor.size["shortest_edge"] = 256
                else:
                    processor.image_processor.size["height"] = 256
                    processor.image_processor.size["width"] = 256
            else:
                processor.image_processor.size = 256
    except Exception:
        pass

    # Qwen2.5-VL emits a variable number of vision tokens proportional to
    # (pixels / 28**2). On thesis-scale data, occasional high-res images
    # produce token counts large enough to OOM the backward pass at bs=1.
    # Cap max_pixels to ~512 patches (512 * 28 * 28 = 401408).
    try:
        ip = getattr(processor, "image_processor", None)
        if ip is not None:
            patch = 28 * 28
            if hasattr(ip, "max_pixels"):
                ip.max_pixels = min(getattr(ip, "max_pixels", 512 * patch), 512 * patch)
            if hasattr(ip, "min_pixels"):
                ip.min_pixels = max(getattr(ip, "min_pixels", 4 * patch), 4 * patch)
    except Exception:
        pass

    # --- model: mirror train_answer_qwen.py ---
    try:
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            device_map="auto" if (use_4bit or use_8bit) else None,
            trust_remote_code=True,
            torch_dtype=None if (use_4bit or use_8bit) else dtype,
            quantization_config=quant_cfg,
            attn_implementation="sdpa",
            revision=revision,
        )
    except Exception:
        from transformers import AutoModel

        model = AutoModel.from_pretrained(
            model_name,
            device_map="auto" if (use_4bit or use_8bit) else None,
            trust_remote_code=True,
            torch_dtype=None if (use_4bit or use_8bit) else dtype,
            quantization_config=quant_cfg,
            attn_implementation="sdpa",
            revision=revision,
        )

    if use_4bit or use_8bit:
        model = prepare_model_for_kbit_training(model)

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if train_cfg.get("gradient_checkpointing", True) and hasattr(
        model, "gradient_checkpointing_enable"
    ):
        model.gradient_checkpointing_enable()

    # LoRA must see the SAME base module structure as Stage 1
    model = prepare_lora(model, cfg.get("peft", {}))

    # ---- load Stage-1 best checkpoint (only backbone/LoRA weights) ----
    # In inference-only mode we skip this entirely: the Stage-2 checkpoint
    # we're about to load already contains the warm-started weights.
    if inference_only:
        accelerator.print(
            "[Stage2] inference-only mode: skipping Stage-1 warm-start "
            "(Stage-2 checkpoint loaded later already contains it)."
        )
    else:
        # If `train.stage1_ckpt` is set in the resolved config (the CLI always sets it),
        # the file MUST exist — silent fallback to base Qwen was the prior bug.
        # If absent (direct-script invocation), fall back to the legacy default path
        # and keep the old warning behavior.
        explicit_stage1_ckpt = train_cfg.get("stage1_ckpt")
        if explicit_stage1_ckpt:
            stage1_ckpt = explicit_stage1_ckpt
            if not os.path.exists(stage1_ckpt):
                raise FileNotFoundError(
                    f"[Stage2] train.stage1_ckpt is set but file is missing: {stage1_ckpt}. "
                    "Stage 2 cannot warm-start from Stage 1."
                )
        else:
            stage1_ckpt = str(PROJECT_ROOT / project_cfg["save_dir"] / "model_best.pt")

        if os.path.exists(stage1_ckpt):
            state = torch.load(stage1_ckpt, map_location="cpu")

            if "state_dict_model" in state:
                sd = state["state_dict_model"]
            elif "model_state" in state:
                sd = state["model_state"]
            elif "model" in state:
                sd = state["model"]
            else:
                sd = state

            missing, unexpected = model.load_state_dict(sd, strict=False)
            accelerator.print(
                f"[Stage2] Loaded Stage-1 backbone from {stage1_ckpt}\n"
                f"         missing keys: {len(missing)}, unexpected keys: {len(unexpected)}"
            )
        else:
            accelerator.print(
                f"[Stage2] WARNING: stage1_ckpt not found at {stage1_ckpt}; "
                "starting from base Qwen weights."
            )

    # optimizer + scheduler (Stage-2 specific)
    lr = float(stage2_cfg.get("lr", 5e-5))
    wd = float(stage2_cfg.get("weight_decay", 0.0))
    max_epochs = int(stage2_cfg.get("max_epochs", 6))

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=wd)

    total_steps = max_epochs * math.ceil(len(dl_train) / max(1, grad_accum))
    warmup_steps = int(stage2_cfg.get("warmup_ratio", 0.05) * total_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    model, optimizer, dl_train, dl_val, scheduler = accelerator.prepare(
        model, optimizer, dl_train, dl_val, scheduler
    )

    # ---- Resume Stage-2 if possible (step-level first, then epoch-level) ----
    start_epoch = 1
    start_step = 0
    global_step = 0
    best_bleu = -1.0

    if inference_only:
        # Load the gold-trained Stage-2 checkpoint and run a single eval pass.
        accelerator.print(
            f"[Stage2] inference-only: loading Stage-2 weights from "
            f"{stage2_ckpt_load_path}"
        )
        _load_stage2_checkpoint(
            stage2_ckpt_load_path, accelerator, model, optimizer, scheduler
        )
        # Force a single-epoch "loop" so the eval block runs once with
        # is_last_epoch=True (heavy artifacts + full val).
        max_epochs = 1
        start_epoch = 1
        start_step = 0
    else:
        step_state = None
        if step_resume_path.exists():
            # Highest priority: intra-epoch resume
            step_state = _load_stage2_step_checkpoint(
                step_resume_path, accelerator, model, optimizer, scheduler
            )
            if step_state is not None:
                start_epoch = step_state["epoch"]
                start_step = step_state["epoch_step"]
                global_step = step_state["global_step"]
                best_bleu = step_state["best_bleu"]

        if (not step_state) and resume_path:
            # Fallback: epoch-level resume
            start_epoch, best_bleu = _load_stage2_checkpoint(
                resume_path, accelerator, model, optimizer, scheduler
            )

    # ---- Epoch recovery (only when NOT resuming mid-epoch) ----
    analysis_root = Path(stage2_root) / "analysis_stage2"
    prev_dir = analysis_root / f"epoch_{start_epoch-1:02d}"
    if (start_epoch > 1) and (not prev_dir.exists()) and (start_step == 0):
        accelerator.print(
            f"[Stage2] WARNING: missing analysis for epoch {start_epoch-1}; "
            f"retrying that epoch instead."
        )
        start_epoch = start_epoch - 1

    # ---- CoT evaluation config ----
    cot_samples = int(eval_stage2_cfg.get("cot_samples", 1))
    cot_temperature = float(eval_stage2_cfg.get("cot_temperature", 0.7))
    cot_top_p = float(eval_stage2_cfg.get("cot_top_p", 0.9))
    cot_max_examples = int(eval_stage2_cfg.get("cot_max_examples", -1))
    # Per-epoch val-gen subsampling: on non-final epochs, cap val gen to this
    # many samples for a cheap BLEU signal; 0 means use the full val set.
    # Only takes effect when max_epochs > 1.
    val_gen_subsample_per_epoch = int(
        eval_stage2_cfg.get("val_gen_subsample_per_epoch", 0) or 0
    )
    # Heavy artifacts (tiles, contact sheets, length hists, deep_rationale_analysis,
    # make_rationale_metrics subprocess) are rendered either every epoch or only
    # on the final epoch. "last_epoch" is the big saving when multi-epoch runs.
    val_gen_heavy_artifacts = str(
        eval_stage2_cfg.get("val_gen_heavy_artifacts", "every_epoch")
    ).lower()
    if val_gen_heavy_artifacts not in ("every_epoch", "last_epoch"):
        val_gen_heavy_artifacts = "every_epoch"

    # ---- training loop ----
    grad_accum = int(stage2_cfg.get("grad_accum", 1))
    save_every_steps = int(stage2_cfg.get("save_every_steps", 50))
    gen_max_new_tokens = int(stage2_cfg.get("max_new_tokens", 160))

    # Resume-surviving per-epoch wall-clock timing (each completed epoch is
    # persisted; a power-off loses at most the current epoch). Inference-only
    # pred passes are timed under a separate stage label.
    from utils.perf import StageTimer  # lazy import
    _timing_path = os.environ.get("PIPELINE_TIMING_PATH") or os.path.join(
        str(PROJECT_ROOT), cfg["project"]["output_dir"], "pipeline_timing.json")
    _timer = StageTimer(_timing_path) if accelerator.is_main_process else None
    _stage2_label = "stage2_pred_infer" if inference_only else "stage2_gold_train"
    _ep_t0 = None  # prev-epoch start; timed at the top of the NEXT iteration + after the loop

    for epoch in range(start_epoch, max_epochs + 1):
        if _timer is not None and _ep_t0 is not None:
            _timer.add(_stage2_label, time.time() - _ep_t0, {"last_epoch": epoch - 1})
        _ep_t0 = time.time()
        if inference_only:
            accelerator.print(
                f"\n[Stage2] inference-only: skipping training epoch {epoch}, "
                f"running eval/generation only."
            )
        else:
            model.train()
            accelerator.print(f"\n[Stage2] Epoch {epoch}/{max_epochs}")
        is_last_epoch = (epoch == max_epochs)
        emit_heavy_artifacts = (
            val_gen_heavy_artifacts == "every_epoch" or is_last_epoch
        )
        # Non-final epochs may cap val-gen to a subsample for a cheap BLEU signal.
        if val_gen_subsample_per_epoch > 0 and not is_last_epoch:
            epoch_cot_max = val_gen_subsample_per_epoch
            accelerator.print(
                f"[Stage2] Epoch {epoch}: val-gen capped to "
                f"{epoch_cot_max} samples (subsample mode)"
            )
        else:
            epoch_cot_max = cot_max_examples
        running_loss = 0.0
        n_seen = 0

        # If we resumed mid-epoch, start from that step once; afterwards start from 0
        epoch_start_step = start_step if epoch == start_epoch else 0

        for step, batch in enumerate(
            [] if inference_only else tqdm(dl_train, disable=not accelerator.is_local_main_process)
        ):
            # Skip already-completed batches when resuming mid-epoch
            if step < epoch_start_step:
                continue

            images = batch["images"]
            questions = batch["questions"]
            choices_list = batch["choices"]
            labels = batch["labels"]
            captions = batch["captions"]
            gold_rats = batch["rationale_gold"]

            # filter out examples without rationales
            keep_idx = [
                i for i, r in enumerate(gold_rats) if isinstance(r, str) and r.strip()
            ]
            if not keep_idx:
                continue

            images_f = [images[i] for i in keep_idx]
            questions_f = [questions[i] for i in keep_idx]
            choices_f = [choices_list[i] for i in keep_idx]
            labels_f = [labels[i] for i in keep_idx]
            captions_f = [captions[i] for i in keep_idx]
            rats_f = [gold_rats[i] for i in keep_idx]

            prompts = [
                format_stage2_prompt(
                    stage2_cfg["stage2_prompt_template"],
                    q,
                    ch,
                    int(lbl),
                    cap,
                )
                for q, ch, lbl, cap in zip(
                    questions_f, choices_f, labels_f, captions_f
                )
            ]

            # ---- CoT-style targets for training ----
            cot_targets = [_build_cot_target(r) for r in rats_f]

            inputs = pack_chat_batch(
                processor, images_f, prompts, cot_targets, device
            )

            with accelerator.accumulate(model):
                outputs = model(**inputs)
                loss = outputs.loss
                accelerator.backward(loss)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            bs = len(keep_idx)
            running_loss += loss.item() * bs
            n_seen += bs

            # --------- step-level bookkeeping & checkpointing ----------
            global_step += 1

            if (
                accelerator.is_main_process
                and save_every_steps > 0
                and (global_step % save_every_steps == 0)
            ):
                latest_step_path = Path(stage2_ckpt_dir) / "latest_step.pt"
                _save_stage2_checkpoint(
                    latest_step_path,
                    accelerator,
                    model,
                    optimizer,
                    scheduler,
                    epoch=epoch,
                    best_bleu=best_bleu,
                    cfg=cfg,
                    global_step=global_step,
                    epoch_step=step,
                )
                accelerator.print(
                    f"[Stage2] Saved step checkpoint at "
                    f"epoch={epoch}, step={step}, global_step={global_step} "
                    f"→ {latest_step_path}"
                )

        # reset for later epochs; only the first epoch may start mid-way
        start_step = 0

        train_loss = running_loss / max(1, n_seen)
        accelerator.print(f"[Stage2] Epoch {epoch} train_loss={train_loss:.4f}")

        # ---- validation: generate rationales + BLEU + CoT stats ----
        model.eval()
        all_rows = []
        bleu_scores = []
        n_cot_examples = 0

        eos_id = processor.tokenizer.eos_token_id
        pad_id = processor.tokenizer.pad_token_id or eos_id

        if cot_samples > 1:
            gen_kwargs = dict(
                max_new_tokens=gen_max_new_tokens,
                do_sample=True,
                num_beams=1,
                temperature=cot_temperature,
                top_p=cot_top_p,
                no_repeat_ngram_size=3,
                eos_token_id=eos_id,
                pad_token_id=pad_id,
            )
        else:
            gen_kwargs = dict(
                max_new_tokens=gen_max_new_tokens,
                do_sample=False,
                num_beams=1,
                no_repeat_ngram_size=3,
                eos_token_id=eos_id,
                pad_token_id=pad_id,
            )

        # Config-equivalence guard (panel item 18 / Ch8 schema-malformation): the heavy per-epoch
        # artifacts persisted this epoch are exactly what the offline faithfulness eval
        # (rationale_drift.py, cross_modal_consistency.py) later scores, so their generation config
        # must match the eval's deterministic greedy decoding. Sampling (cot_samples>1) would silently
        # diverge the persisted rationales from what the eval assumes and re-introduce the
        # schema-malformation regression. Fail loudly rather than let a future run inherit it.
        if emit_heavy_artifacts:
            assert gen_kwargs["do_sample"] is False, (
                "Per-epoch heavy-artifact generation must be greedy (do_sample=False) to match the "
                "offline eval config; set cot_samples<=1 when emitting persisted artifacts. "
                "See docs/EXPERIMENTAL_FINDINGS.md (schema malformation)."
            )

        with torch.no_grad():
            for batch in tqdm(
                dl_val, disable=not accelerator.is_local_main_process
            ):
                if epoch_cot_max > 0 and n_cot_examples >= epoch_cot_max:
                    break

                images = batch["images"]
                questions = batch["questions"]
                choices_list = batch["choices"]
                labels = batch["labels"]
                captions = batch["captions"]
                gold_rats = batch["rationale_gold"]
                img_ids = batch.get("image_id", [None] * len(images))
                orig_ids = batch.get("orig_image_id", [None] * len(images))

                keep_idx = [
                    i for i, r in enumerate(gold_rats) if isinstance(r, str) and r.strip()
                ]
                if not keep_idx:
                    continue

                images_f = [images[i] for i in keep_idx]
                questions_f = [questions[i] for i in keep_idx]
                choices_f = [choices_list[i] for i in keep_idx]
                labels_f = [labels[i] for i in keep_idx]
                captions_f = [captions[i] for i in keep_idx]
                rats_f = [gold_rats[i] for i in keep_idx]
                img_ids_f = [img_ids[i] for i in keep_idx]
                orig_ids_f = [orig_ids[i] for i in keep_idx]

                # Decide which answer index to condition on:
                #   - gold index (labels_f) if answer_source="gold"
                #   - Stage-1 predicted index if answer_source="pred"
                #     (falls back to gold if no prediction available)
                answer_indices: List[int] = []
                answer_indices_gold: List[int] = []
                answer_indices_pred: List[int] = []

                for q, lbl, iid in zip(questions_f, labels_f, img_ids_f):
                    lbl_int = int(lbl)
                    ans_idx = lbl_int
                    pred_idx = lbl_int

                    if use_pred_answers and stage1_pred_map:
                        key = (iid, q)
                        s1 = stage1_pred_map.get(key)
                        if s1 is not None:
                            try:
                                pred_idx = int(s1.get("pred", lbl_int))
                            except Exception:
                                pred_idx = lbl_int
                            ans_idx = pred_idx

                    answer_indices.append(ans_idx)
                    answer_indices_gold.append(lbl_int)
                    answer_indices_pred.append(pred_idx)

                prompts = [
                    format_stage2_prompt(
                        prompt_template_eval,
                        q,
                        ch,
                        ans_idx,
                        cap,
                    )
                    for q, ch, ans_idx, cap in zip(
                        questions_f, choices_f, answer_indices, captions_f
                    )
                ]

                gen_inputs = pack_chat_for_generation(
                    processor, images_f, prompts, device
                )

                B = gen_inputs["input_ids"].shape[0]
                input_len = gen_inputs["input_ids"].shape[1]

                if cot_samples > 1:
                    generated = model.generate(
                        **gen_inputs,
                        **gen_kwargs,
                        #eos_token_id=None,                       
                        #pad_token_id=processor.tokenizer.pad_token_id or 0,
                        num_return_sequences=cot_samples,
                    )
                    generated_cont = generated[:, input_len:]
                    raw_all = safe_batch_decode(processor, generated_cont, skip_special_tokens=True)
                    grouped_texts: List[List[str]] = [
                        raw_all[i * cot_samples : (i + 1) * cot_samples]
                        for i in range(B)
                    ]
                else:
                    generated = model.generate(
                    **gen_inputs,
                    **gen_kwargs,   # eos_token_id & pad_token_id already included
                    )


                    generated_cont = generated[:, input_len:]
                    raw_texts = safe_batch_decode(processor, generated_cont, skip_special_tokens=True)
                    grouped_texts = [[t] for t in raw_texts]

                for idx, (
                    p_txt,
                    gold,
                    cot_texts,
                    q,
                    ch,
                    ans_idx,
                    ans_idx_gold,
                    ans_idx_pred,
                    cap,
                    iid,
                    oid,
                ) in enumerate(
                    zip(
                        prompts,
                        rats_f,
                        grouped_texts,
                        questions_f,
                        choices_f,
                        answer_indices,
                        answer_indices_gold,
                        answer_indices_pred,
                        captions_f,
                        img_ids_f,
                        orig_ids_f,
                    )
                ):
                    cleaned = [_clean_generated_rationale(t) for t in cot_texts]

                    reasonings: List[str] = []
                    finals: List[str] = []
                    for t in cleaned:
                        r_txt, f_txt = _parse_cot(t)
                        reasonings.append(r_txt)
                        finals.append(f_txt)

                    rationale_reasoning = reasonings[0] if reasonings else ""
                    rationale_final = finals[0] if finals else ""
                    # Remove stray tags or leftover XML
                    rationale_final = re.sub(r"</?reasoning>|</?final>", "", rationale_final).strip()

                    # ensure rationale_gen always equals the final justification ---
                    rationale_gen = rationale_final

                    # BLEU is computed on the <final> part only
                    bleu = _bleu_safe(gold, rationale_final)
                    if not math.isnan(bleu):
                        bleu_scores.append(bleu)

                    stats = _cot_stats(reasonings, finals)
                    cot_final_agreement = stats["cot_final_agreement"]
                    cot_reasoning_jaccard = stats["cot_reasoning_jaccard"]

                    row = {
                        "image_id": iid,
                        "orig_image_id": oid,
                        "question": q,
                        "choices": ch,
                        "answer_idx": int(ans_idx),
                        "answer_idx_gold": int(ans_idx_gold),
                        "answer_idx_pred": int(ans_idx_pred),
                        "caption": cap,
                        "prompt": p_txt,
                        "rationale_gold": gold,
                        "rationale_reasoning": rationale_reasoning,
                        "rationale_final": rationale_final,
                        # backwards-compatible: rationale_gen == <final>
                        "rationale_gen": rationale_gen,
                        "bleu": bleu,
                        "cot_final_agreement": cot_final_agreement,
                        "cot_reasoning_jaccard": cot_reasoning_jaccard,
                    }
                    all_rows.append(row)

                    n_cot_examples += 1
                    if epoch_cot_max > 0 and n_cot_examples >= epoch_cot_max:
                        break

                if epoch_cot_max > 0 and n_cot_examples >= epoch_cot_max:
                    break

        mean_bleu = (
            float(sum(bleu_scores) / max(1, len(bleu_scores)))
            if bleu_scores
            else float("nan")
        )
        # "Full-val" means this epoch's BLEU is apples-to-apples comparable to
        # other full-val passes. When subsampling is on, only the last epoch
        # runs on the full val set, so only the last epoch may update best
        # and write the canonical rationales file. This keeps best-checkpoint
        # selection coherent and prevents the downstream metric scripts from
        # consuming a subsampled JSONL.
        val_is_full = (val_gen_subsample_per_epoch <= 0) or is_last_epoch
        accelerator.print(
            f"[Stage2] Epoch {epoch} val BLEU={mean_bleu:.4f} "
            f"(valid={len(bleu_scores)}, full_val={val_is_full})"
        )

        # save JSONL of val rationales + CKPTs + thesis-quality reports
        if accelerator.is_main_process:
            out_jsonl = Path(results_root) / f"val_rationales_epoch{epoch}.jsonl"
            _save_jsonl(out_jsonl, all_rows)
            accelerator.print(f"[Stage2] Saved val rationales to {out_jsonl}")

            # ----------------- 1) Save Stage-2 checkpoints -----------------
            # In inference-only mode we do not write Stage-2 checkpoints: the
            # source checkpoint is authoritative and must not be overwritten.
            if not inference_only:
                tag = f"stage2_epoch{epoch}_bleu_{mean_bleu:.4f}.pt"
                ckpt_path = Path(stage2_ckpt_dir) / tag
                accelerator.save(
                    {
                        "model_state": accelerator.get_state_dict(model),
                        "cfg": cfg,
                        "epoch": epoch,
                        "bleu": mean_bleu,
                    },
                    ckpt_path,
                )
                accelerator.print(f"[Stage2] Saved checkpoint {ckpt_path}")

            # Checkpoint save fires only in training mode, gated on new best.
            best_improved = val_is_full and mean_bleu > best_bleu
            if (not inference_only) and best_improved:
                best_bleu = mean_bleu
                best_path = Path(stage2_ckpt_dir) / "stage2_best.pt"
                accelerator.save(
                    {
                        "model_state": accelerator.get_state_dict(model),
                        "cfg": cfg,
                        "epoch": epoch,
                        "bleu": mean_bleu,
                    },
                    best_path,
                )
                accelerator.print(
                    f"[Stage2] Updated best checkpoint: {best_path} (BLEU={best_bleu:.4f})"
                )

            # Canonical rationales file consumed by downstream metric scripts.
            # Fires (a) in training mode when BLEU improved on full val, or
            # (b) always in inference-only mode (single pass IS the canonical output).
            if (inference_only and val_is_full) or (
                (not inference_only) and best_improved
            ):
                if inference_only:
                    best_bleu = mean_bleu
                try:
                    canonical = Path(results_root) / "generated_rationales.jsonl"
                    _save_jsonl(canonical, all_rows)
                    accelerator.print(f"[Stage2] Wrote canonical rationales: {canonical}")
                    # Stamp answer_source into a sidecar for downstream verification.
                    meta_path = Path(results_root) / "generated_rationales.meta.json"
                    with open(meta_path, "w", encoding="utf-8") as _mf:
                        json.dump(
                            {
                                "answer_source": answer_source,
                                "best_epoch": int(epoch),
                                "best_bleu": float(best_bleu),
                                "arm": exp_cfg.get("arm"),
                                "exp_name": exp_name,
                                "inference_only": inference_only,
                                "stage2_ckpt": stage2_ckpt_load_path,
                            },
                            _mf,
                            indent=2,
                        )
                except Exception as _e:
                    accelerator.print(f"[Stage2] WARN: failed to write canonical rationales: {_e}")

            # ----------------- 2) Enrich rows with image paths -----------------
            arm = exp_cfg.get("arm", "plain")
            plain_root = data_cfg.get("images_plain_val") or data_cfg.get(
                "images_baseline_val"
            )
            point_root = data_cfg.get("images_point_val")

            for rec in all_rows:
                img_id = rec.get("image_id")
                orig_id = rec.get("orig_image_id", img_id)

                # Build plain & point paths
                plain_path = _build_image_path(orig_id, plain_root)
                point_path = _build_image_path(img_id, point_root)

                rec["_plain_path"] = plain_path if plain_path and os.path.exists(plain_path) else None
                rec["_point_path"] = point_path if point_path and os.path.exists(point_path) else None

                # plain & plain_desc always use plain images
                if arm in ("plain", "plain_desc"):
                    visual = rec["_plain_path"] or rec["_point_path"]
                else:  # point & point_desc
                    visual = rec["_point_path"] or rec["_plain_path"]

                rec["_visual_path"] = visual
                
            rows_valid = [
                r
                for r in all_rows
                if isinstance(r.get("bleu", None), (int, float))
                and not math.isnan(r["bleu"])
            ]
            rows_sorted = sorted(rows_valid, key=lambda x: x["bleu"])
            bottom50 = rows_sorted[:50]
            top50 = list(reversed(rows_sorted))[:50]

            # ----------------- 3) Per-epoch analysis folder -----------------
            analysis_root = _ensure_dir(stage2_root / "analysis_stage2")
            epoch_dir = _ensure_dir(analysis_root / f"epoch_{epoch:02d}")
            top_dir = _ensure_dir(epoch_dir / "top50_best_bleu")
            bot_dir = _ensure_dir(epoch_dir / "bottom50_worst_bleu")

            # ----------------- 4) Export JSON sidecars -----------------
            _export_rationale_samples_json(
                top50,
                epoch_dir / f"top50_best_bleu_epoch{epoch}.json",
            )
            _export_rationale_samples_json(
                bottom50,
                epoch_dir / f"bottom50_worst_bleu_epoch{epoch}.json",
            )

            # ----------------- 5) Render tiles + contact sheets -----------------
            def _emit_rationale_tiles(samples: list[dict], out_dir: Path, tag: str):
                paths = []
                for k, rec in enumerate(samples, 1):
                    img_path = (
                        rec.get("_visual_path")
                        or rec.get("_point_path")
                        or rec.get("_plain_path")
                    )
                    img = _safe_open_image(img_path, size_max=700)
                    outfile = out_dir / f"{tag}_{k:02d}.png"
                    _render_rationale_tile(
                        img,
                        rec,
                        str(outfile),
                        title=f"{tag.replace('_', ' ').title()} #{k}",
                    )
                    paths.append(str(outfile))

                _render_contact_sheet(
                    paths,
                    str(out_dir / f"{tag}_contact_sheet.png"),
                    cols=4,
                    title=f"{tag.replace('_', ' ').title()} — epoch {epoch}",
                )

            if emit_heavy_artifacts:
                _emit_rationale_tiles(top50, top_dir, "top50_best_bleu")
                _emit_rationale_tiles(bottom50, bot_dir, "bottom50_worst_bleu")
            else:
                accelerator.print(
                    f"[Stage2] Epoch {epoch}: skipping tile/contact-sheet render "
                    f"(val_gen_heavy_artifacts=last_epoch)"
                )

            # ----------------- 6) BLEU histogram -----------------
            if rows_valid:
                bleu_arr = np.array(
                    [r["bleu"] for r in rows_valid], dtype=np.float32
                )
                plt.figure(figsize=(6, 4))
                plt.hist(bleu_arr, bins=20, range=(0.0, 1.0))
                plt.xlabel("BLEU score")
                plt.ylabel("Count")
                plt.title(f"BLEU distribution — epoch {epoch}")
                plt.tight_layout()
                plt.savefig(
                    epoch_dir / f"bleu_hist_epoch{epoch}.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close()

            # ----------------- 7) Rationale length histograms -----------------
            gold_lens = [
                len((r.get("rationale_gold") or "").split()) for r in rows_valid
            ]
            gen_lens = [
                len((r.get("rationale_gen") or "").split()) for r in rows_valid
            ]

            if gold_lens and gen_lens:
                plt.figure(figsize=(6, 4))
                plt.hist(gold_lens, bins=20, alpha=0.5, label="gold")
                plt.hist(gen_lens, bins=20, alpha=0.5, label="generated")
                plt.xlabel("Rationale length (words)")
                plt.ylabel("Count")
                plt.title(
                    f"Rationale length distribution — epoch {epoch}"
                )
                plt.legend()
                plt.tight_layout()
                plt.savefig(
                    epoch_dir / f"rationale_length_hist_epoch{epoch}.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close()

            # ----------------- 8) SAVE LATEST CHECKPOINT FOR RESUME -----------------
            if not inference_only:
                latest_path = Path(stage2_ckpt_dir) / "latest.pt"
                _save_stage2_checkpoint(
                    latest_path,
                    accelerator,
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    best_bleu,
                    cfg,
                    global_step=global_step,
                    epoch_step=-1,
                )
                accelerator.print(f"[Stage2] Saved latest resume checkpoint → {latest_path}")

            # After a *successful* end-of-epoch, any step-level checkpoint
            # is obsolete. Remove it so next resume uses epoch-level.
            step_ckpt = Path(stage2_ckpt_dir) / "latest_step.pt"
            if step_ckpt.exists():
                try:
                    step_ckpt.unlink()
                    accelerator.print(
                        f"[Stage2] Removed stale step checkpoint → {step_ckpt}"
                    )
                except Exception as e:
                    accelerator.print(
                        f"[Stage2] WARNING: could not remove step checkpoint: {e}"
                    )

            # ---- 9) Offline rationale metrics (BLEU/BERTScore/BLEURT) ----
            eval_cfg = cfg.get("eval", {})

            if not emit_heavy_artifacts:
                accelerator.print(
                    f"[Stage2] Epoch {epoch}: skipping offline BERTScore/BLEURT "
                    f"metrics (val_gen_heavy_artifacts=last_epoch)"
                )
            elif eval_cfg.get("compute_bertscore", False) or eval_cfg.get(
                "compute_bleurt", False
            ):
                script_path = PROJECT_ROOT / "scripts" / "make_rationale_metrics.py"
                if script_path.exists():
                    cmd = [
                        sys.executable,
                        str(script_path),
                        "--cfg",
                        str(cfg_path_str),
                        "--epoch",
                        str(epoch),
                    ]

                    bert_model = eval_cfg.get("bertscore_model")
                    if eval_cfg.get("compute_bertscore", False) and bert_model:
                        cmd.extend(["--bertscore_model", str(bert_model)])

                    bleurt_ckpt = eval_cfg.get("bleurt_ckpt")
                    if eval_cfg.get("compute_bleurt", False) and bleurt_ckpt:
                        cmd.extend(["--bleurt_ckpt", str(bleurt_ckpt)])
                        bleurt_device = eval_cfg.get("bleurt_device")
                        if bleurt_device:
                            cmd.extend(["--bleurt_device", str(bleurt_device)])

                    accelerator.print(
                        f"[Stage2] Launching offline rationale metrics: {' '.join(cmd)}"
                    )
                    try:
                        subprocess.run(cmd, check=True)
                    except Exception as e:
                        accelerator.print(
                            f"[Stage2] WARNING: offline metrics script failed: {e}"
                        )
                else:
                    accelerator.print(
                        f"[Stage2] Offline rationale metrics script not found at {script_path}; skipping."
                    )

            # ---- 10) Deep rationale analysis ----
            deep_script = PROJECT_ROOT / "scripts" / "deep_rationale_analysis.py"
            if not emit_heavy_artifacts:
                accelerator.print(
                    f"[Stage2] Epoch {epoch}: skipping deep_rationale_analysis "
                    f"(val_gen_heavy_artifacts=last_epoch)"
                )
            elif deep_script.exists():
                cmd2 = [
                    sys.executable,
                    str(deep_script),
                    "--cfg",
                    str(cfg_path),
                    "--epoch",
                    str(epoch),
                ]
                accelerator.print(
                    f"[Stage2] Launching deep rationale analysis: {' '.join(cmd2)}"
                )
                try:
                    subprocess.run(cmd2, check=True)
                except Exception as e:
                    accelerator.print(
                        f"[Stage2] WARNING: deep rationale analysis script failed: {e}"
                    )
            else:
                accelerator.print(
                    f"[Stage2] Deep rationale analysis script not found at {deep_script}; skipping."
                )

    # Finalise timing for the last completed epoch (the per-epoch add above
    # records each epoch at the start of the next iteration; this catches the
    # final one, including the early-stop / single-epoch / inference cases).
    if _timer is not None and _ep_t0 is not None:
        _timer.add(_stage2_label, time.time() - _ep_t0, {"last_epoch": "final"})
        accelerator.print(
            f"[timing] {_stage2_label} cumulative: "
            f"{_timer.data['stages'][_stage2_label]['human']}")


if __name__ == "__main__":
    main()
