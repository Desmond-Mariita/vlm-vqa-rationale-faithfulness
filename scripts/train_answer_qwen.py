#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 1 — Answer Prediction (Qwen2.5-VL + LoRA), memory-stable + resumable.

Features:
  • Arms: plain | point | point_desc | plain_desc
      - plain:       plain image only
      - point:       polygon image only
      - point_desc:  polygon image + caption
      - plain_desc:  plain image + caption
  • Never requests output_hidden_states by default (OOM-safe).
  • Pools from decoder/text last hidden state (cheap).
  • Resume from checkpoint (model + head + optimizer + scheduler).
  • Detects missing epoch outputs and re-runs that epoch instead of skipping.
  • Top/Bottom-50 visuals + calibration + confusion matrix per epoch.
"""

from __future__ import annotations
import os
import sys

# Ensure utils/ is importable, then suppress library noise before heavy imports.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from utils import silence  # noqa: E402, F401

import math
import time
import json
import csv
import glob
import yaml
import random
import tempfile
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm

from transformers import (
    AutoModelForVision2Seq,
    AutoModel,
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
from models.mc_classifier import MCClassifier
from utils.yaml_config import load_yaml


# ---------- small utils ----------
def set_seed(seed: int):
    """Set random seeds for reproducibility across all backends.

    Args:
        seed: Integer seed value for random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _ensure_dir(p: str) -> str:
    """Create directory if it does not exist and return the path."""
    os.makedirs(p, exist_ok=True)
    return p


# ---------- metrics / reports ----------
def _confusion_matrix(labels: torch.Tensor, preds: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Build a confusion matrix from label and prediction tensors."""
    cm = torch.zeros((num_classes, num_classes), dtype=torch.long, device=labels.device)
    idx = labels * num_classes + preds
    cm += torch.bincount(idx, minlength=num_classes * num_classes).view(num_classes, num_classes)
    return cm


def _ascii_confusion(cm: torch.Tensor) -> str:
    """Format a confusion matrix as an ASCII table string."""
    nc = cm.size(0)
    width = max(5, len(str(int(cm.max().item()))))
    hdr = "gold\\pred | " + " ".join([f"{i:>{width}d}" for i in range(nc)])
    sep = "-" * len(hdr)
    rows = [hdr, sep]
    for i in range(nc):
        rows.append(f"{i:>9d} | " + " ".join([f"{int(cm[i, j]):>{width}d}" for j in range(nc)]))
    return "\n".join(rows)


def _write_csv_matrix(path: str, cm: torch.Tensor):
    """Write a confusion matrix tensor to a CSV file."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        for i in range(cm.size(0)):
            w.writerow([int(x) for x in cm[i].tolist()])


def _save_json(path: str, obj: dict):
    """Serialize a dictionary to a JSON file with pretty printing."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _softmax_confidence(logits: torch.Tensor) -> torch.Tensor:
    """Compute max softmax probability as a confidence score per sample."""
    probs = torch.softmax(logits.float(), dim=-1)
    return probs.max(dim=-1).values  # [B]


def _per_class_metrics_from_cm(cm: torch.Tensor):
    """Derive macro and micro precision/recall/F1 from a confusion matrix."""
    eps = 1e-12
    tp = torch.diag(cm).float()
    fp = cm.sum(0).float() - tp
    fn = cm.sum(1).float() - tp
    tn = cm.sum().float() - (tp + fp + fn)
    prec = (tp / (tp + fp + eps)).tolist()
    rec = (tp / (tp + fn + eps)).tolist()
    f1 = [(2 * p * r / (p + r + eps)) for p, r in zip(prec, rec)]
    acc_class = ((tp + tn) / (tp + fp + fn + tn + eps)).tolist()
    macro = {
        "precision": float(sum(prec) / len(prec)),
        "recall": float(sum(rec) / len(rec)),
        "f1": float(sum(f1) / len(f1)),
        "per_class_precision": prec,
        "per_class_recall": rec,
        "per_class_f1": f1,
        "per_class_accuracy": acc_class,
    }
    micro = {
        "precision": float(tp.sum() / (tp.sum() + fp.sum() + eps)),
        "recall": float(tp.sum() / (tp.sum() + fn.sum() + eps)),
        "f1": float(2 * tp.sum() / (2 * tp.sum() + fp.sum() + fn.sum() + eps)),
        "accuracy": float(tp.sum() / (cm.sum() + eps)),
    }
    return macro, micro


def _safe_open_image(path: Optional[str], size_max: int = 512) -> Optional[Image.Image]:
    """Open and thumbnail-resize an image, returning None on failure."""
    if not path:
        return None
    try:
        im = Image.open(path).convert("RGB")
        im.thumbnail((size_max, size_max), Image.Resampling.LANCZOS)
        return im
    except Exception:
        return None


def _build_image_path(image_id: Optional[str], root_dir: Optional[str]) -> Optional[str]:
    """Join image_id with root_dir and return the path if it exists, else None."""
    if not image_id or not root_dir:
        return None
    p = os.path.join(root_dir, image_id)
    return p if os.path.exists(p) else None


def _render_sample_tile(img: Optional[Image.Image], meta: dict, out_path: str, title: str, dpi: int = 300):
    """Render a 2-row tile with the image on top and formatted metadata below."""
    def safe(v): return str(v) if v is not None else "None"
    choices = meta.get("choices") or []
    choice_str = " | ".join([f"[{i}] {c}" for i, c in enumerate(choices)]) if choices else "N/A"

    caption = meta.get("caption") or ""
    caption_wrapped = "\n".join(textwrap.wrap(caption, width=90))

    info = (
        f"image_id: {safe(meta.get('image_id'))}\n"
        f"orig_image_id: {safe(meta.get('orig_image_id'))}\n"
        f"pred / gold: {meta.get('pred')} / {meta.get('label')}    "
        f"conf: {meta.get('confidence'):.3f}\n"
        f"question: {safe(meta.get('question'))}\n"
        f"choices: {choice_str}\n"
        f"caption: {caption_wrapped}"
    )

    fig, axes = plt.subplots(
        2, 1, figsize=(10, 8),
        gridspec_kw={'height_ratios': [3, 2]}
    )
    ax_img, ax_text = axes
    for ax in axes:
        ax.axis("off")

    if img is not None:
        ax_img.imshow(img)
        ax_img.set_title(title, fontsize=13, pad=8, fontweight="bold")
    else:
        ax_img.text(0.5, 0.5, "Image not found", ha="center", va="center", fontsize=14)

    ax_text.text(
        0, 1.0, info, va="top", ha="left", fontsize=10,
        family="monospace", linespacing=1.3, wrap=True
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _render_contact_sheet(tile_paths: List[str], out_path: str, cols: int = 4, title: str = "", dpi: int = 300):
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
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", 36)
        except Exception:
            font = ImageFont.load_default()
        try:
            bbox = draw.textbbox((0, 0), title, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            try:
                tw, th = font.getsize(title)
            except Exception:
                tw, th = len(title) * 9, 36
        draw.text(((sheet.width - tw) // 2, (band_h - th) // 2), title, fill=(0, 0, 0), font=font)
        sheet = ImageOps.expand(sheet, border=(0, band_h, 0, 0), fill=(255, 255, 255))
        sheet.paste(title_band, (0, 0))

    try:
        sheet.save(out_path, dpi=(dpi, dpi))
    except TypeError:
        sheet.save(out_path)


def _export_samples_json(samples: List[dict], out_path: str):
    """Save per-sample metadata for Top-50 / Bottom-50 analyses to JSON."""
    def to_record(s):
        return {
            "image_id": s.get("image_id"),
            "orig_image_id": s.get("orig_image_id"),
            "question": s.get("question"),
            "choices": s.get("choices"),
            "caption": s.get("caption"),
            "chosen_answer_idx": s.get("pred"),
            "correct_answer_idx": s.get("label"),
            "confidence": s.get("confidence"),
            "image_path_plain": s.get("_plain_path"),
            "image_path_point": s.get("_point_path"),
            "image_exists_plain": bool(s.get("_plain_path") and os.path.exists(s["_plain_path"])),
            "image_exists_point": bool(s.get("_point_path") and os.path.exists(s["_point_path"])),
            "meta": {k: v for k, v in s.items() if k not in {
                "image_id", "orig_image_id", "question", "choices", "caption",
                "pred", "label", "confidence", "_plain_path", "_point_path", "correct"
            }},
        }
    _save_json(out_path, {"count": len(samples), "items": [to_record(s) for s in samples]})


# ---------- Calibration helpers ----------
def _ece_from_preds(labels: torch.Tensor, preds: torch.Tensor, conf: torch.Tensor, n_bins: int = 15):
    """Compute Expected Calibration Error using equal-width bins on [0, 1]."""
    eps = 1e-12
    bins = torch.linspace(0.0, 1.0, steps=n_bins + 1, device=conf.device)
    idx = torch.bucketize(conf, bins) - 1
    ece = 0.0
    table = []
    N = labels.numel()
    for b in range(n_bins):
        mask = (idx == b)
        cnt = int(mask.sum().item())
        if cnt == 0:
            table.append({"bin": b, "count": 0, "acc": None, "conf": None})
            continue
        acc_b = float((labels[mask] == preds[mask]).float().mean().item())
        conf_b = float(conf[mask].mean().item())
        w_b = float(cnt / max(1, N))
        ece += w_b * abs(acc_b - conf_b)
        table.append({"bin": b, "count": cnt, "acc": acc_b, "conf": conf_b})
    return float(ece), table


def _ece_per_class_predicted(labels: torch.Tensor, preds: torch.Tensor, conf: torch.Tensor,
                             num_classes: int = 4, n_bins: int = 10):
    """Compute per-class ECE, grouping samples by predicted class."""
    out = {}
    for k in range(num_classes):
        mask = (preds == k)
        if not mask.any():
            out[str(k)] = {"ece": None, "bins": []}
            continue
        ece_k, table_k = _ece_from_preds(labels[mask], preds[mask], conf[mask], n_bins=n_bins)
        out[str(k)] = {"ece": ece_k, "bins": table_k}
    return out


# ---------- checkpointing ----------
def _atomic_torch_save(obj: dict, final_path: str) -> None:
    """Save a checkpoint atomically via temp file to prevent corruption."""
    d = os.path.dirname(final_path)
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_ckpt_", suffix=".pt", dir=d)
    os.close(fd)
    try:
        torch.save(obj, tmp)
        os.replace(tmp, final_path)
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass


def _strip_quant_aux_keys(state_dict: dict) -> dict:
    """Remove bitsandbytes quantization auxiliary keys from a state dict."""
    DROP_SUFFIXES = (".absmax", ".quant_map", ".nested_absmax", ".nested_quant_map")
    DROP_CONTAINS = (".quant_state.", "bitsandbytes__nf4")
    return {
        k: v for k, v in state_dict.items()
        if not any(k.endswith(s) for s in DROP_SUFFIXES) and not any(t in k for t in DROP_CONTAINS)
    }


def _prune_checkpoints_pre_save(dirpath: str, keep_after_save: int = 4,
                                protect_patterns=("model_best.pt", "*best*.pt")):
    """Delete oldest checkpoints to keep at most ``keep_after_save`` files."""
    all_pts = sorted(glob.glob(os.path.join(dirpath, "*.pt")), key=os.path.getmtime)
    protected = {os.path.realpath(p) for pat in protect_patterns for p in glob.glob(os.path.join(dirpath, pat))}
    cands = [p for p in all_pts if os.path.realpath(p) not in protected]
    target_pre = max(0, keep_after_save - 1)
    if len(all_pts) <= target_pre:
        return
    excess = max(0, len(all_pts) - target_pre)
    for p in cands[:excess]:
        try:
            os.remove(p)
        except Exception:
            pass


def save_checkpoint_atomic(
    save_dir: str, tag: str, accelerator: "Accelerator",
    model: nn.Module, head: nn.Module,
    optimizer: torch.optim.Optimizer, scheduler,
    step: int, epoch: int, best_val_acc: float, cfg: dict, keep: int = 4,
):
    """Save a full training checkpoint atomically with automatic pruning.

    Writes both a main checkpoint (model, head, optimizer, scheduler state)
    and a lightweight weights-only sidecar file.

    Args:
        save_dir: Directory to store checkpoint files.
        tag: Checkpoint tag -- "best", "latest", or an epoch/step identifier.
        accelerator: HuggingFace Accelerator instance for distributed state.
        model: Backbone model (with LoRA adapters).
        head: Classification head module.
        optimizer: Optimizer whose state will be saved.
        scheduler: Learning rate scheduler whose state will be saved.
        step: Current global training step.
        epoch: Current epoch number.
        best_val_acc: Best validation accuracy seen so far.
        cfg: Full YAML config dict, stored as a fingerprint.
        keep: Maximum number of checkpoint files to retain.

    Returns:
        Path to the saved checkpoint file.
    """
    _ensure_dir(save_dir)
    _prune_checkpoints_pre_save(save_dir, keep_after_save=keep)
    model_state = accelerator.get_state_dict(model)
    head_state = accelerator.get_state_dict(head)
    sidecar_state = _strip_quant_aux_keys({
        **{f"model.{k}": v for k, v in model_state.items()},
        **{f"head.{k}": v for k, v in head_state.items()},
    })
    checkpoint = {
        "epoch": epoch,
        "step": step,
        "best_val_acc": best_val_acc,
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "state_dict_model": model_state,
        "state_dict_head": head_state,
        "cfg_fingerprint": yaml.safe_dump(cfg, sort_keys=True),
        "time": time.time(),
    }
    fname = "model_best.pt" if tag == "best" else ("latest.pt" if tag == "latest" else f"checkpoint_{tag}.pt")
    path = os.path.join(save_dir, fname)
    _atomic_torch_save(checkpoint, path)
    _atomic_torch_save({"state_dict": sidecar_state}, path.replace(".pt", ".weights.pt"))
    return path


def _load_stage1_checkpoint(
    ckpt_path: str,
    backbone: nn.Module,
    head: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    accelerator: Optional["Accelerator"] = None,
) -> tuple[int, float, int]:
    """Load a Stage-1 checkpoint and restore model, head, optimizer, and scheduler.

    Args:
        ckpt_path: Path to the checkpoint file. If empty or missing, starts fresh.
        backbone: Backbone model to load weights into.
        head: Classification head to load weights into.
        optimizer: Optimizer to restore state for. Skipped if None.
        scheduler: LR scheduler to restore state for. Skipped if None.
        accelerator: Accelerator instance used for logging.

    Returns:
        A tuple of (start_epoch, best_val_acc, global_step) where start_epoch
        is 1-based.
    """
    logger = accelerator.print if accelerator is not None else print

    if not ckpt_path or not os.path.exists(ckpt_path):
        logger(f"[Stage1] No checkpoint found at '{ckpt_path}'; starting from scratch.")
        return 1, 0.0, 0

    logger(f"[Stage1] Resuming from checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")

    # Backbone / LoRA weights
    model_sd = state.get("state_dict_model") or state.get("model_state")
    if model_sd is not None:
        missing, unexpected = backbone.load_state_dict(model_sd, strict=False)
        logger(
            f"[Stage1] Loaded backbone (missing={len(missing)}, "
            f"unexpected={len(unexpected)})"
        )
    else:
        logger("[Stage1] WARNING: no model weights in checkpoint.")

    # Classifier head
    head_sd = state.get("state_dict_head") or state.get("head_state")
    if head_sd is not None:
        head.load_state_dict(head_sd, strict=False)
        logger("[Stage1] Loaded classifier head.")
    else:
        logger("[Stage1] WARNING: no head weights in checkpoint.")

    # Optimizer / scheduler
    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
        logger("[Stage1] Restored optimizer state.")
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])
        logger("[Stage1] Restored scheduler state.")

    last_epoch = int(state.get("epoch", 0))
    best_val_acc = float(state.get("best_val_acc", 0.0))
    global_step = int(state.get("step", 0))
    start_epoch = last_epoch + 1

    logger(
        f"[Stage1] Resuming at epoch {start_epoch} "
        f"(best_val_acc={best_val_acc:.4f}, global_step={global_step})."
    )
    return start_epoch, best_val_acc, global_step


# ---------- Qwen chat packing ----------
def build_model_inputs_from_prompts(processor, images, prompts, device):
    """Build Qwen chat-format model inputs from images and text prompts.

    Applies the processor's chat template to each (image, prompt) pair,
    tokenizes the batch, and moves tensors to the target device.

    Args:
        processor: HuggingFace processor with chat template support.
        images: List of PIL images, one per sample.
        prompts: List of text prompts, one per sample.
        device: Target torch device for the returned tensors.

    Returns:
        Dict of batched model input tensors ready for forward pass.
    """
    chats = []
    for p in prompts:
        chats.append(
            processor.apply_chat_template(
                [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": p}]}],
                add_generation_prompt=False, tokenize=False,
            )
        )
    mi = processor(text=chats, images=images, return_tensors="pt", padding=True)
    for k, v in list(mi.items()):
        if isinstance(v, torch.Tensor):
            mi[k] = v.to(device)
    return mi


# ---------- prompt format ----------
def format_prompt(stage1_template: str, question: str, choices: List[str], caption: Optional[str]) -> str:
    """Format a Stage-1 multiple-choice prompt from question, choices, and optional caption.

    Args:
        stage1_template: Format string with placeholders {question}, {c0}-{c3}.
        question: The VQA question text.
        choices: List of exactly 4 answer choice strings.
        caption: Optional image caption; appended if provided (truncated to 256 chars).

    Returns:
        The fully formatted prompt string.
    """
    c0, c1, c2, c3 = choices
    prompt = stage1_template.format(question=question, c0=c0, c1=c1, c2=c2, c3=c3)
    if caption:
        prompt += f"\nCaption: {caption[:256]}"
    return prompt


# ---------- dataloaders ----------
def build_dataloaders(cfg: Dict[str, Any]) -> Dict[str, DataLoader]:
    """Construct train and validation DataLoaders from the YAML config.

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
        json_train_path=data["json_train_path"], json_val_path=data["json_val_path"],
        images_plain_train=data["images_plain_train"], images_plain_val=data["images_plain_val"],
        images_point_train=data["images_point_train"], images_point_val=data["images_point_val"],
        split=data["split_train"], arm=exp["arm"],
        caption_key=exp.get("caption_key", "image_descriptions.caption"),
        drop_missing=True,
    )
    ds_val = VQADataset(
        json_train_path=data["json_train_path"], json_val_path=data["json_val_path"],
        images_plain_train=data["images_plain_train"], images_plain_val=data["images_plain_val"],
        images_point_train=data["images_point_train"], images_point_val=data["images_point_val"],
        split=data["split_val"], arm=exp["arm"],
        caption_key=exp.get("caption_key", "image_descriptions.caption"),
        drop_missing=True,
    )
    dl_train = DataLoader(
        ds_train,
        batch_size=ld.get("batch_size", 1),
        shuffle=ld.get("shuffle", True),
        num_workers=ld.get("num_workers", 2),
        pin_memory=ld.get("pin_memory", True),
        drop_last=ld.get("drop_last", False),
        collate_fn=vqa_collate_fn,
        **seed_kwargs,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=max(1, ld.get("batch_size", 1)),
        shuffle=False,
        num_workers=max(1, ld.get("num_workers", 2)),
        pin_memory=ld.get("pin_memory", True),
        drop_last=False,
        collate_fn=vqa_collate_fn,
        **seed_kwargs,
    )
    return {"train": dl_train, "val": dl_val}


# ---------- LoRA ----------
def prepare_lora(model: nn.Module, peft_cfg: Dict[str, Any]) -> nn.Module:
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


# ---------- optional hook backup (registered AFTER prepare) ----------
_last_dec_output: Dict[str, torch.Tensor] = {}


def _capture_final_decoder_hidden(module, inp, out):
    """Forward hook that stores the final decoder layer's hidden state."""
    _last_dec_output["hidden"] = out[0] if isinstance(out, (tuple, list)) else out


# ---------- forward pooling ----------
def _get_nested_attr(obj, *path):
    """Traverse nested attributes, returning None if any step is missing."""
    cur = obj
    for p in path:
        if cur is None:
            return None
        cur = getattr(cur, p, None)
    return cur


def pool_decoder(backbone: nn.Module, model_inputs: Dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    """Run a forward pass and pool the last decoder hidden state per sample.

    Avoids requesting output_hidden_states by default (OOM-safe). Falls back
    to output_hidden_states only when the primary extraction paths fail.

    Args:
        backbone: The VLM backbone model.
        model_inputs: Tokenized and preprocessed model inputs dict.
        device: Target torch device.

    Returns:
        Pooled tensor of shape [B, H] using last-token pooling.

    Raises:
        RuntimeError: If no text-side hidden state can be extracted.
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
        # Fallback with hidden_states (should be rare, ideally batch small)
        outputs = backbone(**model_inputs, use_cache=False, output_hidden_states=True)
        if hasattr(outputs, "decoder_hidden_states") and outputs.decoder_hidden_states is not None:
            dec_last = outputs.decoder_hidden_states[-1]
        elif hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            dec_last = outputs.hidden_states[-1]
        else:
            lmo = getattr(outputs, "language_model_output", None)
            if lmo is not None and getattr(lmo, "hidden_states", None) is not None:
                dec_last = lmo.hidden_states[-1]
        if dec_last is None:
            raise RuntimeError(
                "Could not obtain text-side last hidden state from model outputs."
            )

    attention_mask = model_inputs.get("attention_mask", None)
    if attention_mask is not None:
        idx = attention_mask.sum(dim=1) - 1  # [B]
        pooled = dec_last[torch.arange(dec_last.size(0), device=device), idx]
    else:
        pooled = dec_last[:, -1, :]
    return pooled


# ---------- main ----------
def main():
    """Entry point for Stage-1 answer prediction training.

    Loads configuration from --cfg (required), builds dataloaders,
    initialises the Qwen2.5-VL backbone with LoRA and a 4-way
    classification head, then runs the training loop with periodic
    checkpointing and per-epoch validation reports.
    """
    import argparse
    parser = argparse.ArgumentParser(description="Stage-1 answer prediction training")
    parser.add_argument("--cfg", type=str, required=True, help="Path to YAML config")
    args, _ = parser.parse_known_args()
    cfg_path = args.cfg
    if not os.path.isabs(cfg_path):
        cfg_path = str(PROJECT_ROOT / cfg_path)
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    print(f"[train_answer_qwen] Loading config: {cfg_path}", flush=True)
    cfg = load_yaml(cfg_path)
    set_seed(cfg["project"]["seed"])

    # TF32 trades bit-exact determinism for matmul throughput. Off by default so
    # it doesn't silently override cudnn.deterministic=True set by set_all_seeds.
    if cfg["train"].get("allow_tf32", False):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    accelerator = Accelerator(mixed_precision=cfg["train"].get("mixed_precision", "no"))
    device = accelerator.device

    # Resume checkpoint path (optional)
    train_cfg = cfg["train"]
    resume_from = (train_cfg.get("resume_from") or "").strip()
    if not resume_from:
        default_latest = os.path.join(cfg["project"]["save_dir"], "latest.pt")
        resume_from = default_latest if os.path.exists(default_latest) else ""

    loaders = build_dataloaders(cfg)

    model_name = cfg["backbone"]["model_name"]
    processor_name = cfg["backbone"].get("processor_name", model_name)
    revision = cfg["backbone"].get("revision") or None
    processor = AutoProcessor.from_pretrained(
        processor_name, trust_remote_code=True, use_fast=False, revision=revision
    )

    # shrink image size a bit to reduce VRAM
    try:
        if hasattr(processor, "image_processor") and hasattr(processor.image_processor, "size"):
            size = processor.image_processor.size
            if isinstance(size, dict):
                if "shortest_edge" in size:
                    processor.image_processor.size["shortest_edge"] = 192
                else:
                    processor.image_processor.size["height"] = 256
                    processor.image_processor.size["width"] = 256
            else:
                processor.image_processor.size = 256
    except Exception:
        pass

    # k-bit loading
    use_4bit = bool(cfg["train"].get("load_in_4bit", False))
    use_8bit = bool(cfg["train"].get("load_in_8bit", False)) and not use_4bit
    dtype = torch.bfloat16 if cfg["train"].get("mixed_precision") == "bf16" else torch.float16

    quant_cfg = None
    if use_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
    elif use_8bit:
        quant_cfg = BitsAndBytesConfig(load_in_8bit=True)

    try:
        backbone = AutoModelForVision2Seq.from_pretrained(
            model_name,
            device_map="auto" if (use_4bit or use_8bit) else None,
            trust_remote_code=True,
            torch_dtype=None if (use_4bit or use_8bit) else dtype,
            quantization_config=quant_cfg,
            revision=revision,
        )
    except Exception:
        backbone = AutoModel.from_pretrained(
            model_name,
            device_map="auto" if (use_4bit or use_8bit) else None,
            trust_remote_code=True,
            torch_dtype=None if (use_4bit or use_8bit) else dtype,
            quantization_config=quant_cfg,
            revision=revision,
        )

    if use_4bit or use_8bit:
        backbone = prepare_model_for_kbit_training(backbone)

    if hasattr(backbone.config, "use_cache"):
        backbone.config.use_cache = False
    if hasattr(backbone, "gradient_checkpointing_enable"):
        backbone.gradient_checkpointing_enable()

    # Freeze vision/encoder blocks
    for n, p in backbone.named_parameters():
        if (".vision" in n) or (".encoder" in n):
            p.requires_grad = False

    # LoRA + head
    backbone = prepare_lora(backbone, cfg["peft"])
    hidden_size = getattr(backbone.config, "hidden_size", getattr(backbone.config, "d_model", 4096))
    head = MCClassifier(input_dim=hidden_size, num_classes=4)

    lr, wd = cfg["train"]["lr_lora"], cfg["train"]["weight_decay"]
    params = list(backbone.parameters()) + list(head.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wd)

    grad_accum = max(1, cfg["train"].get("gradient_accumulation_steps", 1))
    steps_per_epoch = math.ceil(len(loaders["train"]) / grad_accum)
    total_steps = cfg["train"]["max_epochs_stage1"] * steps_per_epoch
    warmup_steps = int(cfg["train"]["warmup_ratio"] * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    criterion = nn.CrossEntropyLoss()

    # Resume (before accelerator.prepare)
    start_epoch, best_val, global_step = _load_stage1_checkpoint(
        resume_from,
        backbone=backbone,
        head=head,
        optimizer=optimizer,
        scheduler=scheduler,
        accelerator=accelerator,
    )

    # Missing-epoch detection (OOM before validation): re-run last epoch if no analysis folder
    analysis_root = os.path.join(cfg["project"]["save_dir"], "analysis")
    expected_prev_epoch_dir = os.path.join(analysis_root, f"epoch_{start_epoch - 1:02d}")
    if (start_epoch > 1) and (not os.path.exists(expected_prev_epoch_dir)):
        accelerator.print(
            f"[resume-fix] Missing analysis for epoch {start_epoch-1}. "
            f"Resuming from epoch {start_epoch-1} instead."
        )
        start_epoch = start_epoch - 1

    # Prepare with accelerate
    backbone, head, optimizer, loaders["train"], loaders["val"], scheduler = accelerator.prepare(
        backbone, head, optimizer, loaders["train"], loaders["val"], scheduler
    )
    accelerator.print(
        f"Backbone params: {sum(p.numel() for p in backbone.parameters()) / 1e6:.1f}M | "
        f"Head: {sum(p.numel() for p in head.parameters()) / 1e6:.2f}M"
    )

    # Hook on unwrapped model (after prepare)
    try:
        real_model = accelerator.unwrap_model(backbone)
        final_block = None
        if hasattr(real_model, "base_model") and hasattr(real_model.base_model, "model") and hasattr(real_model.base_model.model, "decoder"):
            final_block = real_model.base_model.model.decoder.layers[-1]
        elif hasattr(real_model, "model") and hasattr(real_model.model, "decoder"):
            final_block = real_model.model.decoder.layers[-1]
        if final_block is not None:
            final_block.register_forward_hook(_capture_final_decoder_hidden)
    except Exception:
        pass

    stage1_tmpl = cfg["prompts"]["stage1_template"]
    arm = cfg["experiment"]["arm"]

    save_every = int(cfg["train"].get("save_every_steps", 500))
    max_epochs = cfg["train"]["max_epochs_stage1"]

    es_cfg = cfg["train"].get("early_stopping") or {}
    es_patience = int(es_cfg.get("patience", 0) or 0)
    es_mode = (es_cfg.get("mode") or "max").lower()
    epochs_since_improve = 0

    # Resume-surviving per-epoch wall-clock timing (each completed epoch is
    # persisted, so a power-off loses at most the current epoch).
    from utils.perf import StageTimer  # lazy import; keeps top-of-file imports intact
    _timing_path = os.environ.get("PIPELINE_TIMING_PATH") or os.path.join(
        PROJECT_ROOT, cfg["project"]["output_dir"], "pipeline_timing.json")
    _timer = StageTimer(_timing_path) if accelerator.is_main_process else None

    for epoch in range(start_epoch, max_epochs + 1):
        _ep_t0 = time.time()
        backbone.train()
        head.train()
        running_loss = 0.0
        correct = 0
        seen = 0

        pbar = tqdm(
            loaders["train"],
            disable=not accelerator.is_local_main_process,
            desc=f"Epoch {epoch}/{max_epochs}",
        )
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(pbar):
            images, questions = batch["images"], batch["questions"]
            choices_list, captions = batch["choices"], batch["captions"]
            labels = torch.tensor(batch["labels"], device=device, dtype=torch.long)

            use_caption = arm.endswith("_desc")
            prompts = [
                format_prompt(
                    stage1_tmpl,
                    questions[i],
                    choices_list[i],
                    captions[i] if (use_caption and i < len(captions)) else None,
                )
                for i in range(len(images))
            ]

            model_inputs = build_model_inputs_from_prompts(processor, images, prompts, device)

            with accelerator.autocast():
                try:
                    pooled = pool_decoder(backbone, model_inputs, device)
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        accelerator.print("[OOM] Batch too large; retrying sample-by-sample.")
                        torch.cuda.empty_cache()
                        pooled_list = []
                        for j in range(len(images)):
                            try:
                                mi_j = build_model_inputs_from_prompts(
                                    processor, [images[j]], [prompts[j]], device
                                )
                                pj = pool_decoder(backbone, mi_j, device)
                                pooled_list.append(pj)
                            except RuntimeError as e2:
                                accelerator.print(f"[OOM] Skipping sample {j}: {e2}")
                                continue
                        if len(pooled_list) == 0:
                            raise RuntimeError("[OOM] Unable to process any sample in this batch.")
                        pooled = torch.cat(pooled_list, dim=0)
                        # Adjust labels/logits length to match pooled batch
                        if pooled.size(0) != labels.size(0):
                            labels = labels[: pooled.size(0)]
                    else:
                        raise

                logits = head(pooled)
                loss = criterion(logits, labels) / grad_accum

            accelerator.backward(loss)
            running_loss += loss.item() * grad_accum

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    list(backbone.parameters()) + list(head.parameters()),
                    cfg["train"]["max_grad_norm"],
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if accelerator.is_main_process and save_every > 0 and (global_step % save_every) == 0:
                    ckpt = save_checkpoint_atomic(
                        cfg["project"]["save_dir"],
                        "latest",
                        accelerator,
                        backbone,
                        head,
                        optimizer,
                        scheduler,
                        global_step,
                        epoch,
                        best_val,
                        cfg,
                    )
                    accelerator.print(f"[ckpt] saved latest at step {global_step}: {ckpt}")

            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            seen += labels.numel()
            pbar.set_postfix(
                loss=f"{running_loss / max(1, (step + 1)):.4f}",
                acc=f"{correct / max(1, seen):.3f}",
            )

        # ----- validation -----
        val_acc, val_loss, cm = evaluate_with_reports(
            accelerator,
            backbone,
            head,
            processor,
            loaders["val"],
            cfg,
            device,
            epoch=epoch,
        )

        # metrics & confusion
        if accelerator.is_main_process:
            rep_root = _ensure_dir(os.path.join(cfg["project"]["save_dir"], "analysis"))
            _write_csv_matrix(os.path.join(rep_root, f"confusion_epoch{epoch}.csv"), cm.cpu())
            # metrics JSON now written by evaluate_with_reports in epoch-specific dir
            accelerator.print("\n" + _ascii_confusion(cm.cpu()))

        # best
        improved = (val_acc > best_val) if es_mode == "max" else (val_acc < best_val)
        if accelerator.is_main_process and val_acc >= best_val:
            best_val = val_acc
            ckpt = save_checkpoint_atomic(
                cfg["project"]["save_dir"],
                "best",
                accelerator,
                backbone,
                head,
                optimizer,
                scheduler,
                global_step,
                epoch,
                best_val,
                cfg,
            )
            accelerator.print(f"[ckpt] saved BEST at epoch {epoch}: {ckpt}")

        # epoch-end latest
        if accelerator.is_main_process:
            ckpt = save_checkpoint_atomic(
                cfg["project"]["save_dir"],
                "latest",
                accelerator,
                backbone,
                head,
                optimizer,
                scheduler,
                global_step,
                epoch,
                best_val,
                cfg,
            )
            accelerator.print(f"[ckpt] epoch-end latest saved: {ckpt}")

        if _timer is not None:
            _timer.add("stage1_train", time.time() - _ep_t0, {"last_epoch": epoch})
            accelerator.print(
                f"[timing] stage1_train cumulative: "
                f"{_timer.data['stages']['stage1_train']['human']}")

        # early stopping (fail-closed: identical on all ranks via broadcast)
        if es_patience > 0:
            epochs_since_improve = 0 if improved else (epochs_since_improve + 1)
            if epochs_since_improve >= es_patience:
                accelerator.print(
                    f"[early-stop] No improvement for {epochs_since_improve} epoch(s) "
                    f"(patience={es_patience}); stopping at epoch {epoch}."
                )
                break

    accelerator.print("Training complete.")
    torch.cuda.empty_cache()


# ---------- eval ----------
@torch.no_grad()
def evaluate_with_reports(
    accelerator: Accelerator,
    backbone: nn.Module,
    head: nn.Module,
    processor: AutoProcessor,
    loader: DataLoader,
    cfg: Dict[str, Any],
    device: torch.device,
    epoch: int,
):
    """Run validation and generate per-epoch analysis reports.

    Produces confusion matrix (CSV + PNG), macro/micro metrics JSON,
    calibration statistics, per-sample JSONL predictions, per-class bar
    plots, and Top-50 correct / Bottom-50 wrong tile visualisations.

    Args:
        accelerator: HuggingFace Accelerator for distributed evaluation.
        backbone: The VLM backbone model.
        head: The 4-way classification head.
        processor: Tokenizer/image processor for building model inputs.
        loader: Validation DataLoader.
        cfg: Full YAML config dict.
        device: Target torch device.
        epoch: Current epoch number (used for file naming).

    Returns:
        A tuple of (val_acc, val_loss, confusion_matrix_tensor).
    """
    backbone.eval()
    head.eval()
    stage1_tmpl = cfg["prompts"]["stage1_template"]
    arm = cfg["experiment"]["arm"]
    criterion = nn.CrossEntropyLoss()

    point_root = cfg["data"].get("images_point_val")
    plain_root = cfg["data"].get("images_plain_val")

    all_labels, all_preds, all_conf, all_meta = [], [], [], []
    total_loss, total_samples = 0.0, 0

    for batch in tqdm(loader, disable=not accelerator.is_local_main_process, desc="Validate"):
        images, questions = batch["images"], batch["questions"]
        choices_list, captions = batch["choices"], batch["captions"]
        labels = torch.tensor(batch["labels"], device=device, dtype=torch.long)

        use_caption = arm.endswith("_desc")
        prompts = [
            format_prompt(
                stage1_tmpl,
                questions[i],
                choices_list[i],
                captions[i] if (use_caption and i < len(captions)) else None,
            )
            for i in range(len(images))
        ]

        model_inputs = build_model_inputs_from_prompts(processor, images, prompts, device)
        pooled = pool_decoder(backbone, model_inputs, device)
        logits = head(pooled)
        loss = criterion(logits, labels)

        preds = logits.argmax(dim=-1)
        confs = torch.softmax(logits.float(), dim=-1).max(dim=-1).values

        total_loss += loss.item() * labels.size(0)
        total_samples += labels.numel()

        all_labels.append(labels.detach().cpu())
        all_preds.append(preds.detach().cpu())
        all_conf.append(confs.detach().cpu())

        B = len(images)
        for i in range(B):
            meta_item = (batch.get("meta", [None] * B)[i] if "meta" in batch else {})
            img_id = (
                batch.get("image_ids", [None] * B)[i]
                or batch.get("image_id", [None] * B)[i]
                or (meta_item.get("image_id") if meta_item else None)
            )
            orig_id = (
                batch.get("orig_image_id", [None] * B)[i]
                or (meta_item.get("orig_image_id") if meta_item else None)
            )
            guess_point = _build_image_path(img_id, point_root)
            guess_plain = _build_image_path(orig_id, plain_root)

            path_point = (batch.get("image_path_point", [None] * B)[i]
                          if "image_path_point" in batch else None)
            path_plain = (batch.get("image_path_plain", [None] * B)[i]
                          if "image_path_plain" in batch else None)
            path_any = (batch.get("image_path", [None] * B)[i]
                        if "image_path" in batch else None)
            path_orig = (batch.get("orig_image_path", [None] * B)[i]
                         if "orig_image_path" in batch else None)

            _point_path = path_point or path_any or guess_point
            _plain_path = path_plain or path_orig or guess_plain

            rec = {
                "image_id": img_id,
                "orig_image_id": orig_id,
                "question": questions[i],
                "choices": choices_list[i],
                "caption": captions[i] if i < len(captions) else None,
                "_point_path": _point_path,
                "_plain_path": _plain_path,
            }
            if "meta" in batch and isinstance(batch["meta"][i], dict):
                rec.update(batch["meta"][i])

            if arm == "plain" or arm == "plain_desc":
                visual_path = rec.get("_plain_path") or rec.get("_point_path")
            else:
                visual_path = rec.get("_point_path") or rec.get("_plain_path")

            rec["_visual_path"] = visual_path
            rec["arm"] = arm
            all_meta.append(rec)

    labels = torch.cat(all_labels)
    preds = torch.cat(all_preds)
    conf = torch.cat(all_conf)

    ece_overall, ece_bins = _ece_from_preds(labels, preds, conf, n_bins=15)
    ece_per_class = _ece_per_class_predicted(labels, preds, conf, num_classes=4, n_bins=10)

    cm = _confusion_matrix(labels.to(device), preds.to(device), num_classes=4).cpu()
    val_acc = float((preds == labels).sum().item() / max(1, len(labels)))
    val_loss = float(total_loss / max(1, total_samples))
    macro, micro = _per_class_metrics_from_cm(cm)

    correct_mask = (preds == labels)
    avg_conf_all = float(conf.mean().item())
    avg_conf_correct = float(conf[correct_mask].mean().item()) if correct_mask.any() else 0.0
    avg_conf_wrong = float(conf[~correct_mask].mean().item()) if (~correct_mask).any() else 0.0

    per_sample = []
    for i in range(len(labels)):
        m = dict(all_meta[i])
        m.update({
            "pred": int(preds[i].item()),
            "label": int(labels[i].item()),
            "confidence": float(conf[i].item()),
            "correct": bool(preds[i].item() == labels[i].item()),
        })
        per_sample.append(m)

    top50_correct = sorted(
        [x for x in per_sample if x["correct"]],
        key=lambda x: -x["confidence"],
    )[:50]
    bottom50_wrong = sorted(
        [x for x in per_sample if not x["correct"]],
        key=lambda x: x["confidence"],
    )[:50]

    report_root = _ensure_dir(os.path.join(cfg["project"]["save_dir"], "analysis"))
    epoch_dir = _ensure_dir(os.path.join(report_root, f"epoch_{epoch:02d}"))
    top_dir = _ensure_dir(os.path.join(epoch_dir, "top50_correct"))
    bot_dir = _ensure_dir(os.path.join(epoch_dir, "bottom50_wrong"))

    preds_path = os.path.join(epoch_dir, f"val_preds_epoch{epoch}.jsonl")
    with open(preds_path, "w", encoding="utf-8") as f:
        for x in per_sample:
            f.write(json.dumps(x) + "\n")

    # ALSO emit the canonical file under the new-layout run dir's outputs/.
    # cli/run_all.py reads outputs/stage1_preds_val.json to feed Stage-2 (pred).
    # Overwrite every epoch so the "final" epoch wins; if early-stopping picks
    # the best epoch, it is the one whose val_preds are in the last epoch run.
    try:
        _paths = cfg.get("paths", {}) or {}
        _pred_dir = _paths.get("predictions_dir")
        if _pred_dir:
            _pd = _pred_dir if os.path.isabs(_pred_dir) else os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(cfg["project"]["save_dir"].rstrip("/") + "/x"))),
                _pred_dir,
            )
            # Simpler: use project root + relative path
            from pathlib import Path as _P
            _project_root = _P(__file__).resolve().parents[1]
            _pdir = _P(_pred_dir)
            if not _pdir.is_absolute():
                _pdir = _project_root / _pdir
            _pdir.mkdir(parents=True, exist_ok=True)
            _canonical = _pdir / "stage1_preds_val.json"
            with open(_canonical, "w", encoding="utf-8") as _f:
                for x in per_sample:
                    _f.write(json.dumps(x) + "\n")
            print(f"[Stage1] Wrote canonical val predictions: {_canonical}")
    except Exception as _e:
        print(f"[Stage1] WARN: failed to write canonical val predictions: {_e}")

    _export_samples_json(top50_correct, os.path.join(epoch_dir, f"top50_correct_epoch{epoch}.json"))
    _export_samples_json(bottom50_wrong, os.path.join(epoch_dir, f"bottom50_wrong_epoch{epoch}.json"))

    _write_csv_matrix(os.path.join(epoch_dir, f"confusion_epoch{epoch}.csv"), cm)
    _save_json(os.path.join(epoch_dir, f"metrics_epoch{epoch}.json"), {
        "val_acc": val_acc,
        "val_loss": val_loss,
        "macro": macro,
        "micro": micro,
        "avg_conf_all": avg_conf_all,
        "avg_conf_correct": avg_conf_correct,
        "avg_conf_wrong": avg_conf_wrong,
        "class_support": [int(x) for x in cm.sum(1).tolist()],
        "pred_distribution": [int(x) for x in cm.sum(0).tolist()],
        "calibration": {
            "ece": ece_overall,
            "bins": ece_bins,
            "per_class_pred": ece_per_class,
            "n_bins_overall": 15,
            "n_bins_class": 10,
        },
    })

    # Confusion matrix PNG
    try:
        plt.figure(figsize=(6, 5))
        plt.imshow(cm.numpy(), interpolation="nearest")
        plt.title(f"Confusion Matrix — epoch {epoch}")
        plt.colorbar()
        ticks = np.arange(cm.size(0))
        plt.xticks(ticks, ticks)
        plt.yticks(ticks, ticks)
        for i in range(cm.size(0)):
            for j in range(cm.size(1)):
                plt.text(j, i, str(int(cm[i, j].item())),
                         ha="center", va="center", fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(epoch_dir, f"confusion_epoch{epoch}.png"),
                    dpi=300, bbox_inches="tight")
        plt.close()
    except Exception:
        pass

    # Per-class bars
    try:
        idx = list(range(len(macro["per_class_f1"])))
        width = 0.25
        plt.figure(figsize=(8, 5))
        plt.bar([i - width for i in idx], macro["per_class_precision"], width, label="Precision")
        plt.bar(idx, macro["per_class_recall"], width, label="Recall")
        plt.bar([i + width for i in idx], macro["per_class_f1"], width, label="F1")
        plt.xlabel("Class")
        plt.ylabel("Score")
        plt.ylim(0, 1.0)
        plt.title(f"Per-class Metrics — epoch {epoch}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(epoch_dir, f"per_class_metrics_epoch{epoch}.png"),
                    dpi=300, bbox_inches="tight")
        plt.close()
    except Exception:
        pass

    def _emit_tiles(samples: List[dict], out_dir: str, tag: str):
        paths = []
        for k, item in enumerate(samples, 1):
            img_path = (
                item.get("_visual_path")
                or item.get("_point_path")
                or item.get("_plain_path")
            )
            img = _safe_open_image(img_path, size_max=700)
            outfile = os.path.join(out_dir, f"{tag}_{k:02d}.png")
            _render_sample_tile(img, item, outfile, title=f"{tag.replace('_', ' ').title()} #{k}")
            paths.append(outfile)
        _render_contact_sheet(
            paths,
            os.path.join(out_dir, f"{tag}_contact_sheet.png"),
            cols=4,
            title=f"{tag.replace('_', ' ').title()} — epoch {epoch}",
        )

    _emit_tiles(top50_correct, top_dir, "top50_correct")
    _emit_tiles(bottom50_wrong, bot_dir, "bottom50_wrong")

    if accelerator.is_main_process:
        accelerator.print("\n" + _ascii_confusion(cm))
        accelerator.print(
            f"[VAL] epoch={epoch} acc={val_acc:.4f} loss={val_loss:.4f} "
            f"macroF1={macro['f1']:.4f} microF1={micro['f1']:.4f} "
            f"avgConf(all/corr/wrong)={avg_conf_all:.3f}/"
            f"{avg_conf_correct:.3f}/{avg_conf_wrong:.3f}"
        )

    return val_acc, val_loss, cm


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
