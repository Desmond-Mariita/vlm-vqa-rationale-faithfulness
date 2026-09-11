#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rationale Drift Test for Stage-2 Hallucination Detection.

This module measures whether the Stage-2 rationale generation model hallucinates
visual details without actually using the image. It compares rationales generated
with the actual image vs. a blank dummy image.

Metric - Drift Score:
    Drift = 1 - BERTScore(Rationale_with_image, Rationale_without_image)

Interpretation:
    - HIGH drift: Rationale changes significantly when image removed -> FAITHFUL
    - LOW drift: Same rationale regardless of image -> HALLUCINATING

Example:
    $ python scripts/rationale_drift.py --cfg configs/point_desc.yaml --epoch 6

Outputs:
    reports/{arm}/qwen_stage2_{arm}_gold/rationale_drift/
        - drift_results.json         # Summary statistics
        - drift_samples.csv          # Per-sample drift scores
        - drift_histogram.png        # Distribution of drift scores
        - low_drift_samples.csv      # Examples of potential hallucination
        - drift_table.tex            # LaTeX summary table
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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ---- project imports ----
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.vqa_dataset import VQADataset, vqa_collate_fn
from utils.yaml_config import load_yaml

# BERTScore
try:
    from bert_score import score as bert_score
    HAS_BERTSCORE = True
except ImportError:
    HAS_BERTSCORE = False


def prepare_lora(backbone: nn.Module, peft_cfg: Dict[str, Any]) -> nn.Module:
    """Wrap a backbone model with LoRA adapters if configured.

    Args:
        backbone: The base model to wrap with LoRA.
        peft_cfg: Configuration dictionary containing LoRA parameters:
            - use_lora (bool): Whether to apply LoRA (default: True)
            - r (int): LoRA rank (default: 32)
            - alpha (int): LoRA alpha scaling (default: 64)
            - dropout (float): LoRA dropout (default: 0.05)
            - target_modules (List[str]): Modules to apply LoRA to

    Returns:
        The model with LoRA adapters applied, or original model if use_lora=False.
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
    return get_peft_model(backbone, lora)


def load_stage2_model(
    cfg: Dict[str, Any],
    epoch: Optional[int],
    device: torch.device
) -> Tuple[nn.Module, Any, Path]:
    """Load Stage-2 Qwen2.5-VL + LoRA model for rationale generation.

    Args:
        cfg: Configuration dictionary containing backbone, train, experiment,
            and peft settings.
        epoch: Specific epoch checkpoint to load, or None for best/latest.
        device: PyTorch device to load the model onto.

    Returns:
        Tuple containing:
            - model: The loaded and configured model
            - processor: The tokenizer/processor for the model
            - stage2_root: Path to the Stage-2 experiment directory

    Raises:
        RuntimeError: If checkpoint format is unexpected.
    """
    backbone_cfg = cfg["backbone"]
    train_cfg = cfg["train"]
    exp_cfg = cfg["experiment"]
    peft_cfg = cfg.get("peft", {})

    model_name = backbone_cfg["model_name"]
    processor_name = backbone_cfg.get("processor_name", model_name)

    # Use 8-bit for memory safety
    use_4bit = bool(train_cfg.get("load_in_4bit", False))
    use_8bit = not use_4bit
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

    processor = AutoProcessor.from_pretrained(processor_name, trust_remote_code=True)

    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        device_map="auto" if (use_4bit or use_8bit) else None,
        trust_remote_code=True,
        torch_dtype=None if (use_4bit or use_8bit) else dtype,
        quantization_config=quant_cfg,
    )

    if use_4bit or use_8bit:
        model = prepare_model_for_kbit_training(model)

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True

    model = prepare_lora(model, peft_cfg)

    # Load Stage-2 checkpoint
    arm = exp_cfg.get("arm", "plain")
    base_exp_name = exp_cfg.get("base_name", f"qwen_stage2_{arm}")

    # New CLI path layout (preferred): cfg["paths"] is set by cli/run_eval.py.
    paths_cfg = cfg.get("paths", {}) or {}
    stage2_ckpts_p = paths_cfg.get("stage2_checkpoints_dir")
    stage2_run_p = paths_cfg.get("stage2_run_dir")

    def _abs_path(p: str) -> Path:
        pp = Path(p)
        return pp if pp.is_absolute() else (PROJECT_ROOT / pp)

    if stage2_ckpts_p:
        ckpt_dir = _abs_path(stage2_ckpts_p)
        stage2_root = _abs_path(stage2_run_p) if stage2_run_p else ckpt_dir.parent
    else:
        # Legacy fallback: gold-answer source for drift-test consistency.
        exp_name_stage2 = f"{base_exp_name}_gold"
        stage2_root = PROJECT_ROOT / "reports" / arm / exp_name_stage2
        ckpt_dir = stage2_root / "checkpoints_stage2"

    ckpt_path: Optional[Path] = None
    if ckpt_dir.exists():
        candidates: List[Path] = []
        if epoch is not None:
            candidates.append(ckpt_dir / f"epoch_{epoch:02d}.pt")
        candidates.append(ckpt_dir / "stage2_best.pt")
        candidates.append(ckpt_dir / "latest.pt")

        for p in candidates:
            if p.exists():
                ckpt_path = p
                break

    if ckpt_path:
        print(f"[RationaleDrift] Loading Stage-2 checkpoint: {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cpu")

        sd: Optional[Dict[str, torch.Tensor]] = None
        if isinstance(state, dict):
            for key in ["model_state", "state_dict", "state_dict_model", "model_state_stage2"]:
                if key in state and isinstance(state[key], dict):
                    sd = state[key]
                    break
            if sd is None and all(isinstance(v, torch.Tensor) for v in state.values()):
                sd = state

        if sd:
            # Filter matching keys
            model_sd = model.state_dict()
            filtered = {k: v for k, v in sd.items() if k in model_sd and model_sd[k].shape == v.shape}
            missing, unexpected = model.load_state_dict(filtered, strict=False)
            print(f"[RationaleDrift] Loaded {len(filtered)} keys (missing={len(missing)}, unexpected={len(unexpected)})")
    else:
        print(f"[RationaleDrift] WARNING: No Stage-2 checkpoint found under {ckpt_dir}")

    model.to(device)
    model.eval()
    return model, processor, stage2_root


def build_rationale_prompt(
    question: str,
    choices: List[str],
    answer_idx: int,
    caption: Optional[str] = None,
) -> str:
    """Build a Stage-2 style prompt for rationale generation.

    Args:
        question: The VQA question text.
        choices: List of 4 answer choices.
        answer_idx: Index of the correct answer (0-3).
        caption: Optional image caption to include.

    Returns:
        Formatted prompt string for the model.
    """
    c0, c1, c2, c3 = choices[:4] if len(choices) >= 4 else (choices + [""] * 4)[:4]
    answer = choices[answer_idx] if 0 <= answer_idx < len(choices) else ""

    prompt = f"""You are an expert visual commonsense reasoner.

You see an image and a multiple-choice question with four options.
Your task is to explain, step by step, why the given answer is correct,
using only what can reasonably be inferred from the image and question.

Question: {question}

Choices:
  (0) {c0}
  (1) {c1}
  (2) {c2}
  (3) {c3}

The correct answer is choice index {answer_idx}: "{answer}".

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

    if caption:
        prompt += f"\n\nImage caption: {caption[:256]}"

    return prompt


def generate_rationale(
    model: nn.Module,
    processor: Any,
    image: Image.Image,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 256,
) -> str:
    """Generate a rationale given an image and prompt.

    Args:
        model: The VLM model for generation.
        processor: The tokenizer/processor for the model.
        image: PIL Image to condition on.
        prompt: Text prompt for generation.
        device: PyTorch device for inference.
        max_new_tokens: Maximum tokens to generate.

    Returns:
        Generated rationale text.
    """
    # Build chat template
    chat = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}],
        add_generation_prompt=True,
        tokenize=False,
    )

    model_inputs = processor(
        text=[chat],
        images=[image],
        return_tensors="pt",
        padding=True,
    )

    for k, v in model_inputs.items():
        if isinstance(v, torch.Tensor):
            model_inputs[k] = v.to(device)

    with torch.no_grad():
        gen_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
        )

    # Decode only generated tokens
    input_len = model_inputs["input_ids"].shape[1]
    generated = gen_ids[0, input_len:]
    if generated.numel() == 0:
        return ""
    try:
        rationale = processor.decode(generated, skip_special_tokens=True)
    except (TypeError, ValueError):
        # Fallback for invalid token sequences
        gen_list = [t for t in generated.tolist() if t is not None and isinstance(t, int)]
        rationale = processor.decode(gen_list, skip_special_tokens=True) if gen_list else ""
    return rationale.strip()


def compute_bertscore_f1(
    refs: List[str],
    hyps: List[str],
    model_type: str = "roberta-large",
    batch_size: int = 16,
    device: str = "cpu",
) -> np.ndarray:
    """Compute BERTScore F1 between reference and hypothesis lists."""
    if not HAS_BERTSCORE:
        raise RuntimeError("bert_score not installed. Install with: pip install bert-score")

    P, R, F1 = bert_score(
        hyps, refs,
        model_type=model_type,
        lang="en",
        device=device,
        batch_size=batch_size,
    )
    return F1.cpu().numpy()


def main() -> None:
    """Main entry point for rationale drift evaluation."""
    parser = argparse.ArgumentParser(
        description="Rationale Drift Test for Stage-2 hallucination detection"
    )
    parser.add_argument(
        "--cfg", type=str, required=True,
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--epoch", type=int, default=None,
        help="Stage-2 epoch to use (default: best/latest)"
    )
    parser.add_argument(
        "--max_samples", type=int, default=-1,
        help="Limit number of samples (-1 = all)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    if not HAS_BERTSCORE:
        print("[ERROR] bert_score not installed. Install with: pip install bert-score")
        return

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = load_yaml(args.cfg)
    data_cfg = cfg["data"]
    exp_cfg = cfg["experiment"]
    arm = exp_cfg["arm"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[RationaleDrift] arm={arm}, device={device}")

    # Load model
    model, processor, stage2_root = load_stage2_model(cfg, args.epoch, device)

    # Dataset
    caption_key = exp_cfg.get("caption_key", "image_descriptions.caption")
    use_caption = arm.endswith("_desc")

    ds_val = VQADataset(
        json_train_path=data_cfg["json_train_path"],
        json_val_path=data_cfg["json_val_path"],
        images_plain_train=data_cfg["images_plain_train"],
        images_plain_val=data_cfg["images_plain_val"],
        images_point_train=data_cfg["images_point_train"],
        images_point_val=data_cfg["images_point_val"],
        split=data_cfg.get("split_val", "val"),
        arm=arm,
        caption_key=caption_key,
        drop_missing=True,
    )

    total_samples = len(ds_val)
    if args.max_samples > 0:
        total_samples = min(total_samples, args.max_samples)

    print(f"[RationaleDrift] Evaluating {total_samples} samples...")

    # Dummy grey image for blindfold condition
    dummy_img = Image.new("RGB", (512, 512), (128, 128, 128))

    # Collect rationales
    records: List[Dict[str, Any]] = []
    rationales_with_image: List[str] = []
    rationales_without_image: List[str] = []

    for idx in tqdm(range(total_samples), desc="Generating rationales"):
        sample = ds_val[idx]
        image = sample["image"]
        question = sample["question"]
        choices = sample["choices"]
        label = sample["label"]
        caption = sample.get("caption", "") if use_caption else None
        image_id = sample.get("image_id", f"sample_{idx}")

        # Build prompt (using gold answer for consistency)
        prompt = build_rationale_prompt(question, choices, label, caption)

        # Generate with real image
        try:
            rat_with_img = generate_rationale(model, processor, image, prompt, device)
        except Exception as e:
            print(f"[WARN] Failed to generate with image for sample {idx}: {e}")
            rat_with_img = ""

        # Generate with dummy image (blindfolded)
        try:
            rat_without_img = generate_rationale(model, processor, dummy_img, prompt, device)
        except Exception as e:
            print(f"[WARN] Failed to generate without image for sample {idx}: {e}")
            rat_without_img = ""

        rationales_with_image.append(rat_with_img)
        rationales_without_image.append(rat_without_img)

        records.append({
            "idx": idx,
            "image_id": image_id,
            "question": question,
            "choices": choices,
            "label": label,
            "caption": caption or "",
            "rationale_with_image": rat_with_img,
            "rationale_without_image": rat_without_img,
        })

    # Compute BERTScore between rationale pairs
    print("[RationaleDrift] Computing BERTScore similarity...")

    # Filter out empty rationales
    valid_indices = [
        i for i in range(len(records))
        if rationales_with_image[i].strip() and rationales_without_image[i].strip()
    ]

    if not valid_indices:
        print("[ERROR] No valid rationale pairs found.")
        return

    valid_with = [rationales_with_image[i] for i in valid_indices]
    valid_without = [rationales_without_image[i] for i in valid_indices]

    eval_cfg = cfg.get("eval", {})
    bertscore_f1 = compute_bertscore_f1(
        valid_with, valid_without,
        model_type=eval_cfg.get("bertscore_model", "roberta-large"),
        batch_size=int(eval_cfg.get("bertscore_batch", 16)),
    )

    # Compute drift = 1 - BERTScore
    drift_scores = 1.0 - bertscore_f1

    # Add drift scores to records
    drift_map = {valid_indices[i]: drift_scores[i] for i in range(len(valid_indices))}
    for rec in records:
        idx = rec["idx"]
        if idx in drift_map:
            rec["bertscore_f1"] = float(1.0 - drift_map[idx])  # similarity
            rec["drift_score"] = float(drift_map[idx])
        else:
            rec["bertscore_f1"] = float("nan")
            rec["drift_score"] = float("nan")

    # Statistics
    mean_drift = float(np.mean(drift_scores))
    std_drift = float(np.std(drift_scores))
    median_drift = float(np.median(drift_scores))
    min_drift = float(np.min(drift_scores))
    max_drift = float(np.max(drift_scores))

    # Identify low-drift (potential hallucination) samples
    low_drift_threshold = 0.3
    low_drift_indices = [i for i, d in enumerate(drift_scores) if d < low_drift_threshold]
    high_drift_indices = [i for i, d in enumerate(drift_scores) if d >= low_drift_threshold]

    print(f"\n[RationaleDrift] Results:")
    print(f"  Total samples: {len(records)}")
    print(f"  Valid pairs: {len(valid_indices)}")
    print(f"  Mean drift: {mean_drift:.4f}")
    print(f"  Std drift: {std_drift:.4f}")
    print(f"  Median drift: {median_drift:.4f}")
    print(f"  Min/Max drift: {min_drift:.4f} / {max_drift:.4f}")
    print(f"  Low drift (<{low_drift_threshold}): {len(low_drift_indices)} ({100*len(low_drift_indices)/len(drift_scores):.1f}%)")
    print(f"  High drift (>={low_drift_threshold}): {len(high_drift_indices)} ({100*len(high_drift_indices)/len(drift_scores):.1f}%)")

    # Output directory
    epoch_tag = f"epoch_{args.epoch:02d}" if args.epoch is not None else "latest"
    paths_cfg = cfg.get("paths", {}) or {}
    eval_run_dir_p = paths_cfg.get("eval_run_dir")
    if eval_run_dir_p:
        ern = Path(eval_run_dir_p)
        if not ern.is_absolute():
            ern = PROJECT_ROOT / ern
        out_dir = ern / "rationale_drift" / epoch_tag
    else:
        out_dir = stage2_root / "rationale_drift" / epoch_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save per-sample CSV
    df = pd.DataFrame(records)
    df.to_csv(out_dir / "drift_samples.csv", index=False)

    # Save low-drift samples (potential hallucination)
    low_drift_records = [records[valid_indices[i]] for i in low_drift_indices]
    df_low = pd.DataFrame(low_drift_records)
    df_low.to_csv(out_dir / "low_drift_samples.csv", index=False)

    # Save summary JSON
    summary = {
        "arm": arm,
        "epoch": args.epoch,
        "total_samples": len(records),
        "valid_pairs": len(valid_indices),
        "mean_drift": mean_drift,
        "std_drift": std_drift,
        "median_drift": median_drift,
        "min_drift": min_drift,
        "max_drift": max_drift,
        "low_drift_threshold": low_drift_threshold,
        "low_drift_count": len(low_drift_indices),
        "low_drift_pct": 100 * len(low_drift_indices) / len(drift_scores) if drift_scores.size > 0 else 0,
        "high_drift_count": len(high_drift_indices),
        "high_drift_pct": 100 * len(high_drift_indices) / len(drift_scores) if drift_scores.size > 0 else 0,
    }
    with open(out_dir / "drift_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Histogram
    plt.figure(figsize=(8, 5))
    plt.hist(drift_scores, bins=30, edgecolor="black", alpha=0.7)
    plt.axvline(mean_drift, color="red", linestyle="--", label=f"Mean: {mean_drift:.3f}")
    plt.axvline(low_drift_threshold, color="orange", linestyle=":", label=f"Threshold: {low_drift_threshold}")
    plt.xlabel("Drift Score (1 - BERTScore)")
    plt.ylabel("Count")
    plt.title(f"Rationale Drift Distribution (arm={arm})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "drift_histogram.png", dpi=300)
    plt.close()

    # LaTeX table
    latex = f"""
\\begin{{table}}[h]
\\centering
\\begin{{tabular}}{{l c}}
\\hline
Arm & {arm} \\\\
Epoch & {args.epoch or 'latest'} \\\\
Total samples & {len(records)} \\\\
Valid pairs & {len(valid_indices)} \\\\
\\hline
Mean drift & {mean_drift:.4f} \\\\
Std drift & {std_drift:.4f} \\\\
Median drift & {median_drift:.4f} \\\\
Min / Max drift & {min_drift:.4f} / {max_drift:.4f} \\\\
\\hline
Low drift (<{low_drift_threshold}) & {len(low_drift_indices)} ({100*len(low_drift_indices)/len(drift_scores):.1f}\\%) \\\\
High drift (>={low_drift_threshold}) & {len(high_drift_indices)} ({100*len(high_drift_indices)/len(drift_scores):.1f}\\%) \\\\
\\hline
\\end{{tabular}}
\\caption{{Rationale Drift Test for arm={arm}. High drift indicates faithful visual grounding; low drift suggests potential hallucination.}}
\\label{{tab:drift_{arm}}}
\\end{{table}}
"""
    with open(out_dir / "drift_table.tex", "w") as f:
        f.write(latex)

    print(f"\n[RationaleDrift] Saved outputs to: {out_dir}")


if __name__ == "__main__":
    from utils.perf import time_main
    raise SystemExit(time_main(main, "rationale_drift"))
