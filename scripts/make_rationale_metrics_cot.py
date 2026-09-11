#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_rationale_metrics_cot.py

CoT (Chain-of-Thought) rationale visualisation for Stage-2 (Qwen2.5-VL + LoRA).

- Reads Stage-2 val rationales JSONL for a given epoch.
- Assumes each record contains at least:
      image_id, orig_image_id, question, choices,
      answer_idx_pred, answer_idx_gold,
      rationale_final, rationale_gold, caption (optional),
      plus a scalar score: bleurt / bleurt_final / bleurt_score / bleu.

- Creates PNG tiles with:
      • VCR image on the LEFT
      • Caption in a RIGHT column (top → bottom)
      • Below the image (not under the caption):
          – CORRECT / INCORRECT banner
          – question
          – predicted + gold answers
          – final CoT justification
          – gold rationale

- Saves tiles to:

      reports/<arm>/<exp_name>/analysis_stage2/epoch_XX/cot/correct/
      reports/<arm>/<exp_name>/analysis_stage2/epoch_XX/cot/incorrect/

Usage (from project root):

    python scripts/make_rationale_metrics_cot.py \
        --cfg configs/base.yaml \
        --epoch 1 \
        --topk 50
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------- project imports ----------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from utils.yaml_config import load_yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _ensure_dir(p: Path | str) -> Path:
    """Create directory (and parents) if it does not exist, then return it."""
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read a JSONL file and return a list of parsed dictionaries.

    Args:
        path: Path to the JSONL file.

    Returns:
        List of dictionaries, one per non-empty line.
    """
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def get_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    """Load a DejaVuSans TrueType font, falling back to the default font.

    Args:
        size: Font size in points.
        bold: If True, load the bold variant.

    Returns:
        A Pillow ImageFont instance.
    """
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def wrap_and_draw(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    x: int,
    y: int,
    max_width: int,
    line_height: int,
    fill=(0, 0, 0),
) -> int:
    """
    Word-wrap and draw multi-line text using Pillow >=10 API.
    Returns the y position under the last line.
    """
    if not text:
        return y

    words = text.split()
    line = ""

    for w in words:
        test = f"{line}{w} "
        bbox = draw.textbbox((0, 0), test, font=font)
        w_px = bbox[2] - bbox[0]
        if w_px <= max_width:
            line = test
        else:
            draw.text((x, y), line, font=font, fill=fill)
            y += line_height
            line = f"{w} "

    if line:
        draw.text((x, y), line, font=font, fill=fill)
        y += line_height

    return y


def clean_text(x: Any) -> str:
    """
    Remove CoT markup (<reason...>, <final...>), LaTeX-style escapes,
    and tokenizer apostrophe artifacts.
    """
    if x is None:
        return ""
    if not isinstance(x, str):
        x = str(x)

    import re

    # Remove ANY tag that starts with <reason...> or <final...>
    x = re.sub(r"<\/?reason[^>]*>", "", x, flags=re.IGNORECASE)
    x = re.sub(r"<\/?final[^>]*>", "", x, flags=re.IGNORECASE)

    # LaTeX-style underscore escape
    x = x.replace("\\_", "_")

    # Fix common apostrophe spacing from tokenizer
    fixes = {
        " ' s": "'s",
        " ' t": "'t",
        " ' re": "'re",
        " ' ve": "'ve",
        " ' ll": "'ll",
        " ’ s": "'s",
    }
    for k, v in fixes.items():
        x = x.replace(k, v)

    # Remove spaces around apostrophes in general
    x = re.sub(r"\s*'\s*", "'", x)

    return x.strip()


def safe_open_image(path: str | None, max_size: int = 900) -> Image.Image:
    """
    Open image at path and resize so max side ≤ max_size.
    If path is None or missing, return a gray placeholder.
    """
    if not path or not os.path.exists(path):
        img = Image.new("RGB", (max_size, int(max_size * 0.6)), (230, 230, 230))
        d = ImageDraw.Draw(img)
        d.text((20, 20), "IMAGE\nNOT\nFOUND", fill=(0, 0, 0))
        return img

    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = min(max_size / max(w, h), 1.0)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img


# ---------------------------------------------------------------------------
# Robust VCR image path resolution (from old working script)
# ---------------------------------------------------------------------------

def resolve_image_path_for_record(
    arm: str,
    data_cfg: Dict[str, Any],
    rec: Dict[str, Any],
) -> str | None:
    """
    Reproduce the old, working VCR image resolution logic:

    - PLAIN / PLAIN_DESC use images_plain_val (orig_image_id)
    - POINT / POINT_DESC use images_point_val (image_id)
    - Fallback: if not found in the main root, try the other root.
    """

    plain_root = data_cfg.get("images_plain_val") or data_cfg.get("images_baseline_val")
    point_root = data_cfg.get("images_point_val")

    img_id = rec.get("image_id")
    orig_id = rec.get("orig_image_id", img_id)

    p_plain = Path(plain_root) / orig_id if plain_root else None
    p_point = Path(point_root) / img_id if point_root else None

    def maybe(path: Path | None) -> Path | None:
        if path and path.exists():
            return path
        # try adding .jpg if missing
        if path and not path.suffix:
            cand = path.with_suffix(".jpg")
            if cand.exists():
                return cand
        return None

    p_plain = maybe(p_plain)
    p_point = maybe(p_point)

    if arm in ("plain", "plain_desc"):
        chosen = p_plain if p_plain is not None else p_point
    else:
        chosen = p_point if p_point is not None else p_plain

    return str(chosen) if chosen is not None else None


# ---------------------------------------------------------------------------
# Tile rendering (keep NEW layout / formatting)
# ---------------------------------------------------------------------------

def render_tile(
    img: Image.Image,
    rec: Dict[str, Any],
    out_path: Path,
    is_correct: bool,
) -> None:
    """
    Layout (as per NEW script that you liked):

        +----------------------------------------------------+-------------+
        | image_id (small title)                             |  caption    |
        | VCR frame (left, ~70%)                             |  (right)    |
        +----------------------------------------------------+-------------+
        | CORRECT/INCORRECT + question + answers + CoT final + gold rat.   |
        | (this block spans only under the IMAGE, not under caption)       |
        +------------------------------------------------------------------+
    """

    # --- Canvas / columns ---
    CANVAS_W = 1400
    IMG_COL_W = int(CANVAS_W * 0.70)
    CAP_COL_W = CANVAS_W - IMG_COL_W
    BOTTOM_H = 420

    # --- Resize image to fit left column ---
    img_w, img_h = img.size
    scale = IMG_COL_W / img_w
    IMG_H = int(img_h * scale)
    img_resized = img.resize((IMG_COL_W, IMG_H), Image.LANCZOS)

    CANVAS_H = IMG_H + BOTTOM_H
    canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # --- Fonts ---
    font_title = get_font(22, bold=True)
    font_body = get_font(18)
    font_body_small = get_font(16)
    font_bold = get_font(18, bold=True)
    font_status = get_font(20, bold=True)
    line_h = 24

    # --- image_id title ---
    image_id = str(rec.get("image_id") or rec.get("orig_image_id") or "")
    draw.text((10, 5), image_id, font=font_title, fill=(0, 0, 0))

    # --- paste image slightly below title ---
    canvas.paste(img_resized, (0, 35))

    # --- caption column (right) ---
    caption = clean_text(rec.get("caption", ""))

    cap_x = IMG_COL_W + 10
    cap_y = 70     # push down to avoid horizontal overlap with title
    cap_w = CAP_COL_W - 20
    cap_h = IMG_H - 40

    # soft background, NO border (avoid horizontal line artefacts)
    draw.rectangle([cap_x, cap_y, cap_x + cap_w, cap_y + cap_h], fill=(245, 245, 245))

    cx = cap_x + 10
    cy = cap_y + 10

    draw.text((cx, cy), "caption:", font=font_bold, fill=(70, 70, 70))
    cy += line_h

    cy = wrap_and_draw(
        draw,
        caption,
        font_body_small,
        cx,
        cy,
        cap_w - 20,
        line_h,
        fill=(0, 0, 0),
    )

    # --- bottom text block (under image only) ---
    x = 20
    y = IMG_H + 55
    max_text_w = IMG_COL_W - 40

    # correctness banner
    label = "CORRECT ✓" if is_correct else "INCORRECT ✗"
    color = (5, 140, 5) if is_correct else (196, 0, 0)
    draw.text((x, y), label, font=font_status, fill=color)
    y += line_h + 4

    # question
    q = clean_text(rec.get("question", ""))
    y = wrap_and_draw(draw, f"question: {q}", font_body, x, y, max_text_w, line_h)

    # answers
    choices = rec.get("choices") or []
    pred_idx = rec.get("answer_idx_pred", rec.get("answer_idx"))
    gold_idx = rec.get("answer_idx_gold", rec.get("answer_idx"))

    def choice_text(idx: Any) -> str:
        try:
            i = int(idx)
            if 0 <= i < len(choices):
                return clean_text(choices[i])
        except Exception:
            pass
        return ""

    pred_txt = choice_text(pred_idx)
    gold_txt = choice_text(gold_idx)

    y = wrap_and_draw(
        draw,
        f"answer (pred): [{pred_idx}] {pred_txt}",
        font_body,
        x,
        y,
        max_text_w,
        line_h,
    )
    y = wrap_and_draw(
        draw,
        f"answer (gold): [{gold_idx}] {gold_txt}",
        font_body,
        x,
        y,
        max_text_w,
        line_h,
    )

    # final justification
    y += 8
    draw.text((x, y), "final justification:", font=font_bold, fill=(0, 0, 0))
    y += line_h

    final = clean_text(rec.get("rationale_final") or rec.get("rationale_gen") or "")
    final_words = final.split()
    if len(final_words) > 130:
        final = " ".join(final_words[:130]) + " …"
    y = wrap_and_draw(draw, final, font_body_small, x + 20, y, max_text_w, line_h)

    # gold rationale
    y += 8
    draw.text((x, y), "gold rationale:", font=font_bold, fill=(0, 0, 0))
    y += line_h

    gold = clean_text(rec.get("rationale_gold", ""))
    y = wrap_and_draw(draw, gold, font_body_small, x + 20, y, max_text_w, line_h)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


# ---------------------------------------------------------------------------
# Rationale file discovery (same spirit as previous robust version)
# ---------------------------------------------------------------------------

def find_rationales_file(
    arm: str,
    split: str,
    epoch: int,
    exp_name_hint: str | None,
) -> Path:
    """
    Look for  reports/<arm>/<exp_name>/rationales/{split}_rationales_epoch{epoch}.jsonl

    - Prefer exp_name_hint if given.
    - Otherwise, search under reports/<arm>/.
    """
    pattern = f"{split}_rationales_epoch{epoch}.jsonl"
    arm_root = PROJECT_ROOT / "reports" / arm

    candidates: List[Path] = []

    if exp_name_hint:
        p = arm_root / exp_name_hint / "rationales" / pattern
        if p.exists():
            return p

    if arm_root.exists():
        for root, _, files in os.walk(arm_root):
            if pattern in files:
                candidates.append(Path(root) / pattern)

    if not candidates:
        raise FileNotFoundError(f"No rationale file named '{pattern}' under {arm_root}")

    if exp_name_hint:
        for c in candidates:
            # .../<exp_name>/rationales/file
            if c.parents[1].name == exp_name_hint:
                return c

    return candidates[0]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point: render CoT rationale tiles for correct/incorrect samples."""
    parser = argparse.ArgumentParser(description="CoT rationale visualisation.")
    parser.add_argument("--cfg", type=str, default="configs/base.yaml")
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["val", "train"])
    parser.add_argument("--topk", type=int, default=50)
    args = parser.parse_args()

    cfg = load_yaml(args.cfg)
    data_cfg = cfg.get("data", {})
    exp_cfg = cfg.get("experiment", {})

    arm = exp_cfg.get("arm", "plain")
    exp_name_hint = exp_cfg.get("exp_name")

    # ---- locate rationale file robustly ----
    rationales_path = find_rationales_file(arm, args.split, args.epoch, exp_name_hint)
    print(f"[CoT] Using rationale file: {rationales_path}")

    rows = load_jsonl(rationales_path)
    if not rows:
        print("[CoT] No rows in rationale file; aborting.")
        return

    # ---- attach scores + correctness ----
    def get_score(rec: Dict[str, Any]) -> float:
        for k in ("bleurt", "bleurt_final", "bleurt_score", "bleu"):
            if k in rec and rec[k] is not None:
                try:
                    return float(rec[k])
                except Exception:
                    pass
        return float("nan")

    for r in rows:
        r["_score"] = get_score(r)
        pred = r.get("answer_idx_pred", r.get("answer_idx"))
        gold = r.get("answer_idx_gold", r.get("answer_idx"))
        try:
            r["_correct"] = int(pred) == int(gold)
        except Exception:
            r["_correct"] = False

    rows = [r for r in rows if not math.isnan(r["_score"])]
    if not rows:
        print("[CoT] All rows have NaN scores; nothing to visualise.")
        return

    correct_rows = [r for r in rows if r["_correct"]]
    incorrect_rows = [r for r in rows if not r["_correct"]]

    topk = args.topk
    correct_sorted = sorted(correct_rows, key=lambda r: r["_score"], reverse=True)[:topk]
    incorrect_sorted = sorted(incorrect_rows, key=lambda r: r["_score"], reverse=True)[:topk]

    # ---- figure out actual exp_name from path ----
    exp_name_actual = rationales_path.parents[1].name  # .../<exp_name>/rationales/file

    stage2_root = PROJECT_ROOT / "reports" / arm / exp_name_actual
    epoch_dir = _ensure_dir(stage2_root / "analysis_stage2" / f"epoch_{args.epoch:02d}" / "cot")
    correct_dir = _ensure_dir(epoch_dir / "correct")
    incorrect_dir = _ensure_dir(epoch_dir / "incorrect")

    print(f"[CoT] Saving {len(correct_sorted)} correct tiles to   {correct_dir}")
    print(f"[CoT] Saving {len(incorrect_sorted)} incorrect tiles to {incorrect_dir}")

    # ---- render tiles with OLD image resolution logic + NEW layout ----
    for i, r in enumerate(correct_sorted):
        img_path = resolve_image_path_for_record(arm, data_cfg, r)
        img = safe_open_image(img_path)
        out = correct_dir / f"cot_correct_{i:03d}.png"
        render_tile(img, r, out, is_correct=True)

    for i, r in enumerate(incorrect_sorted):
        img_path = resolve_image_path_for_record(arm, data_cfg, r)
        img = safe_open_image(img_path)
        out = incorrect_dir / f"cot_incorrect_{i:03d}.png"
        render_tile(img, r, out, is_correct=False)

    # save JSON of selections for inspection
    with (epoch_dir / "cot_correct_examples.json").open("w", encoding="utf-8") as f:
        for r in correct_sorted:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with (epoch_dir / "cot_incorrect_examples.json").open("w", encoding="utf-8") as f:
        for r in incorrect_sorted:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[CoT] Done. Outputs in {epoch_dir}")


if __name__ == "__main__":
    main()
