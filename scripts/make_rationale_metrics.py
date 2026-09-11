#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Offline rationale metrics analysis for Stage 2 (Qwen2.5-VL + LoRA).

Reads:
  - configs/base.yaml
  - reports/<arm>/<exp_name>/rationales/val_rationales_epochN.jsonl

Computes (on CPU by default):
  - BLEU (recomputed, in case NLTK wasn't available during training)
  - BERTScore (F1)  [optional, if bert_score is installed]
  - BLEURT          [optional, if bleurt is installed AND --bleurt_ckpt
                     or eval.bleurt_ckpt is provided]

Writes, under:
  reports/<arm>/<exp_name>/analysis_stage2/epoch_XX/ :

  - rationale_metrics_epochN.json           # summary stats
  - *_hist_epochN.png                       # histograms

  - top50_best_<metric>_epochN.json
  - bottom50_worst_<metric>_epochN.json
  - top50_best_<metric>/   [tiles + contact sheet]
  - bottom50_worst_<metric>/
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
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Reuse Stage-2 helpers for consistent tiles and path resolution
from train_rationale_qwen import (  # type: ignore
    _safe_open_image,
    _render_rationale_tile,
    _render_contact_sheet,
    _build_image_path,
    _export_rationale_samples_json,
    _bleu_safe,
)

# ---------- BLEURT backend (Hugging Face Transformers, Elron models) ----------
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    HAS_BLEURT = True
except Exception:
    HAS_BLEURT = False

# ---------- project imports ----------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from utils.yaml_config import load_yaml  # type: ignore


# ---------- optional metric backends ----------
try:
    from bert_score import score as bert_score
    HAS_BERTSCORE = True
except Exception:
    HAS_BERTSCORE = False


# ---------- small utils ----------
def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _find_available_epochs(results_root: Path) -> List[int]:
    """Discover epoch numbers from val_rationales_epoch*.jsonl files."""
    epochs: List[int] = []
    if not results_root.exists():
        return epochs
    for fn in results_root.glob("val_rationales_epoch*.jsonl"):
        stem = fn.stem  # e.g., "val_rationales_epoch3"
        if "epoch" not in stem:
            continue
        try:
            n = int(stem.split("epoch", 1)[1])
            epochs.append(n)
        except Exception:
            continue
    return sorted(set(epochs))


def _load_val_rationales(results_root: Path, epoch: int) -> List[Dict[str, Any]]:
    """Load validation rationale records from a JSONL file for a given epoch."""
    path = results_root / f"val_rationales_epoch{epoch}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. "
            "Make sure Stage 2 has run and saved val_rationales_epochN.jsonl."
        )
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _recompute_bleu(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], float]:
    """Fill/refresh 'bleu' scores in-place and return mean over valid samples."""
    vals: List[float] = []
    for r in rows:
        ref = r.get("rationale_gold") or ""
        hyp = r.get("rationale_gen") or ""
        bleu = _bleu_safe(ref, hyp)
        r["bleu"] = float(bleu)
        if not math.isnan(bleu):
            vals.append(float(bleu))
    mean_bleu = float(sum(vals) / max(1, len(vals))) if vals else float("nan")
    return rows, mean_bleu


def _enrich_paths(rows: List[Dict[str, Any]], cfg: Dict[str, Any]) -> None:
    """
    Add _point_path, _plain_path, _visual_path to each row,
    mirroring the logic inside train_rationale_qwen.py.
    """
    data_cfg = cfg["data"]
    exp_cfg = cfg["experiment"]
    arm = exp_cfg.get("arm", "plain")

    plain_root = data_cfg.get("images_plain_val") or data_cfg.get("images_baseline_val")
    point_root = data_cfg.get("images_point_val")

    for r in rows:
        img_id = r.get("image_id")
        orig_id = r.get("orig_image_id")

        r["_point_path"] = _build_image_path(img_id, point_root)
        r["_plain_path"] = _build_image_path(orig_id, plain_root)

        if arm in ("plain", "plain_desc"):
            visual = r["_plain_path"] or r["_point_path"]
        else:  # "point" or "point_desc"
            visual = r["_point_path"] or r["_plain_path"]
        r["_visual_path"] = visual


def _filter_valid_rationales(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep only rows with non-empty gold and generated rationales."""
    return [
        r
        for r in rows
        if isinstance(r.get("rationale_gold"), str)
        and r["rationale_gold"].strip()
        and isinstance(r.get("rationale_gen"), str)
        and r["rationale_gen"].strip()
    ]


def _compute_bertscore(
    rows: List[Dict[str, Any]],
    model_type: str | None,
    batch_size: int,
) -> Tuple[np.ndarray, float]:
    """Compute BERTScore F1 on CPU for all rows."""
    refs = [r["rationale_gold"] for r in rows]
    hyps = [r["rationale_gen"] for r in rows]

    kwargs: Dict[str, Any] = {
        "lang": "en",
        "device": "cpu",
        "batch_size": batch_size,
    }
    if model_type:
        kwargs["model_type"] = model_type

    P, R, F1 = bert_score(hyps, refs, **kwargs)
    f1_np = F1.cpu().numpy()
    mean_f1 = float(f1_np.mean()) if f1_np.size > 0 else float("nan")
    for r, f1 in zip(rows, f1_np.tolist()):
        r["bertscore_f1"] = float(f1)
    return f1_np, mean_f1

def _compute_bleurt_hf(
    rows: List[Dict[str, Any]],
    model_name: str,
    device: str,
    batch_size: int,
) -> Tuple[np.ndarray, float]:
    """Compute BLEURT-like scores using a HF classification model (e.g. Elron/bleurt-base-128)."""
    refs = [r["rationale_gold"] for r in rows]
    hyps = [r["rationale_gen"] for r in rows]

    # Normalize device: "cpu" / "gpu" / "cuda"
    dev = (device or "cpu").lower()
    if dev in {"gpu", "cuda"} and torch.cuda.is_available():
        torch_device = "cuda"
    else:
        torch_device = "cpu"

    # Elron BLEURT models work with AutoTokenizer + AutoModelForSequenceClassification
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(torch_device)
    model.eval()

    scores: List[float] = []
    with torch.no_grad():
        for i in range(0, len(rows), batch_size):
            batch_refs = refs[i : i + batch_size]
            batch_hyps = hyps[i : i + batch_size]

            inputs = tokenizer(
                batch_refs,
                batch_hyps,
                padding="longest",
                truncation=True,
                return_tensors="pt",
            ).to(torch_device)

            logits = model(**inputs).logits  # [B, 1] or [B]
            if logits.ndim > 1:
                logits = logits.squeeze(-1)
            scores.extend(logits.detach().cpu().tolist())

    vals_np = np.asarray(scores, dtype=np.float32)
    mean_val = float(vals_np.mean()) if vals_np.size > 0 else float("nan")
    for r, v in zip(rows, vals_np.tolist()):
        r["bleurt"] = float(v)
    return vals_np, mean_val

def _rank_by_metric(
    rows: List[Dict[str, Any]],
    key: str,
    top_k: int = 50,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
    """Return (topK, bottomK, valid_count) for a given metric key."""
    def is_valid(v: Any) -> bool:
        try:
            x = float(v)
        except Exception:
            return False
        return not math.isnan(x)

    metric_rows = [r for r in rows if is_valid(r.get(key))]
    if not metric_rows:
        return [], [], 0

    sorted_rows = sorted(metric_rows, key=lambda r: float(r[key]))
    bottom = sorted_rows[:top_k]
    top = list(reversed(sorted_rows))[:top_k]
    return top, bottom, len(metric_rows)


def _emit_metric_tiles(
    epoch_dir: Path,
    metric_name: str,
    top_rows: List[Dict[str, Any]],
    bot_rows: List[Dict[str, Any]],
    epoch: int,
) -> None:
    """
    For a metric, generate tiles + contact sheets for top/bottom samples.
    Uses the same layout as Stage-2 BLEU tiles.
    """
    if not top_rows and not bot_rows:
        return

    top_dir = _ensure_dir(epoch_dir / f"top50_best_{metric_name}")
    bot_dir = _ensure_dir(epoch_dir / f"bottom50_worst_{metric_name}")

    def _emit(samples: List[Dict[str, Any]], out_dir: Path, tag: str) -> None:
        if not samples:
            return
        paths: List[str] = []
        for k, rec in enumerate(samples, 1):
            img_path = (
                rec.get("_visual_path")
                or rec.get("_point_path")
                or rec.get("_plain_path")
            )
            img = _safe_open_image(img_path, size_max=700)
            outfile = out_dir / f"{tag}_{k:02d}.png"
            title = f"{tag.replace('_', ' ').title()} #{k}"
            _render_rationale_tile(img, rec, str(outfile), title=title)
            paths.append(str(outfile))

        _render_contact_sheet(
            paths,
            str(out_dir / f"{tag}_contact_sheet.png"),
            cols=4,
            title=f"{tag.replace('_', ' ').title()} — epoch {epoch}",
        )

    _emit(top_rows, top_dir, f"top50_best_{metric_name}")
    _emit(bot_rows, bot_dir, f"bottom50_worst_{metric_name}")


def _histogram(values: np.ndarray, out_path: Path, title: str, xlabel: str) -> None:
    """Save a histogram of metric values as a PNG."""
    if values.size == 0:
        return
    plt.figure(figsize=(6, 4))
    plt.hist(values, bins=20)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


# ---------- main ----------
def main() -> None:
    """CLI entry point: compute offline rationale metrics (BLEU, BERTScore, BLEURT).

    Reads Stage-2 val rationale JSONL files, recomputes text-quality metrics,
    generates histograms and top/bottom sample tiles, and writes a summary JSON.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "base.yaml"),
        help="Path to base YAML config (default: configs/base.yaml)",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="Epoch number to analyze (default: latest available in rationales/).",
    )
    parser.add_argument(
        "--bertscore_model",
        type=str,
        default=None,
        help="Optional model_type for bert_score (e.g. roberta-large). If omitted, uses library default.",
    )
    parser.add_argument(
        "--bertscore_batch",
        type=int,
        default=16,
        help="Batch size for BERTScore on CPU.",
    )
    parser.add_argument(
        "--bleurt_ckpt",
        type=str,
        default=None,
        help="BLEURT-PyTorch model name or path (e.g. 'lucadiliello/BLEURT-20-D3'). "
             "If not set, will use eval.bleurt_ckpt from the YAML.",
    )
    parser.add_argument(
        "--bleurt_device",
        type=str,
        default=None,
        choices=["cpu", "gpu"],
        help="Device preference for BLEURT: 'cpu' or 'gpu'. If omitted, uses eval.bleurt_device or defaults to 'cpu'.",
    )
    args = parser.parse_args()

    # --- load config + derive Stage-2 paths ---
    cfg = load_yaml(args.cfg)
    exp_cfg = cfg["experiment"]
    arm = exp_cfg.get("arm", "plain")

    # --- mirror Stage-2 naming logic (gold vs predicted) ---
    train_stage2_cfg = cfg.get("train_stage2", {})
    answer_source = train_stage2_cfg.get("answer_source", "gold").lower()

    # Base experiment name (gold-answer run)
    base_exp_name = exp_cfg.get("base_name", f"qwen_stage2_{arm}")

    if answer_source == "pred":
        # Use separate folder for predicted-answer rationales
        # This matches train_rationale_qwen.py:
        #   exp_name = exp_cfg.get("name_pred", base_exp_name + "_predicted")
        exp_name = exp_cfg.get("name_pred", base_exp_name + "_predicted")
    else:
        exp_name = exp_cfg.get("name", base_exp_name + "_gold")

    # New CLI path layout (preferred): cfg["paths"] is set by cli/run_eval.py.
    paths_cfg = cfg.get("paths", {}) or {}
    rationales_jsonl_p = paths_cfg.get("rationales_jsonl")
    eval_run_dir_p = paths_cfg.get("eval_run_dir")

    def _abs(p: str) -> Path:
        pp = Path(p)
        return pp if pp.is_absolute() else (PROJECT_ROOT / pp)

    if rationales_jsonl_p:
        # Stage-2 rationales live alongside cfg.paths.rationales_jsonl.
        results_root = _abs(rationales_jsonl_p).parent
    else:
        # Legacy fallback.
        stage2_root = PROJECT_ROOT / "reports" / arm / exp_name
        results_root = stage2_root / "rationales"

    if eval_run_dir_p:
        analysis_root = _abs(eval_run_dir_p) / "rationale_quality"
    else:
        analysis_root = PROJECT_ROOT / "reports" / arm / exp_name / "analysis_stage2"

    print(f"[INFO] answer_source={answer_source!r}, arm={arm!r}")
    print(f"[INFO] results_root={results_root}")
    print(f"[INFO] analysis_root={analysis_root}")

    if not results_root.exists():
        raise SystemExit(
            f"[ERROR] results_root '{results_root}' does not exist.\n"
            "Run Stage-2 training/validation first so val_rationales_epochN.jsonl is written."
        )

    available_epochs = _find_available_epochs(results_root)
    if not available_epochs:
        raise SystemExit(
            f"[ERROR] No val_rationales_epochN.jsonl files found in {results_root}."
        )

    if args.epoch is None:
        epoch = available_epochs[-1]
    else:
        if args.epoch not in available_epochs:
            raise SystemExit(
                f"[ERROR] Requested epoch {args.epoch} not found. "
                f"Available: {available_epochs}"
            )
        epoch = args.epoch

    print(f"[INFO] Using epoch {epoch} (results_root={results_root})")

    # --- load and prepare rows ---
    rows = _load_val_rationales(results_root, epoch)
    rows = _filter_valid_rationales(rows)
    if not rows:
        raise SystemExit("[ERROR] No valid rationales with non-empty gold/gen texts.")

    rows, mean_bleu = _recompute_bleu(rows)
    _enrich_paths(rows, cfg)

    epoch_dir = _ensure_dir(analysis_root / f"epoch_{epoch:02d}")

    summary: Dict[str, Any] = {
        "epoch": epoch,
        "n_samples": len(rows),
        "metrics": {},
    }

    # ---------- BLEU ----------
    bleu_vals = np.asarray(
        [float(r.get("bleu", float("nan"))) for r in rows],
        dtype=np.float32,
    )
    valid_bleu = bleu_vals[~np.isnan(bleu_vals)]
    bleu_mean = float(valid_bleu.mean()) if valid_bleu.size > 0 else float("nan")
    summary["metrics"]["bleu"] = {
        "mean": bleu_mean,
        "valid_count": int(valid_bleu.size),
    }

    top_bleu, bot_bleu, n_bleu = _rank_by_metric(rows, "bleu", top_k=50)
    if n_bleu > 0:
        _export_rationale_samples_json(
            top_bleu,
            epoch_dir / f"top50_best_bleu_epoch{epoch}.json",
        )
        _export_rationale_samples_json(
            bot_bleu,
            epoch_dir / f"bottom50_worst_bleu_epoch{epoch}.json",
        )
        _emit_metric_tiles(epoch_dir, "bleu", top_bleu, bot_bleu, epoch)
        _histogram(
            valid_bleu,
            epoch_dir / f"bleu_hist_epoch{epoch}.png",
            title=f"BLEU distribution — epoch {epoch}",
            xlabel="BLEU",
        )

    # ---------- BERTScore ----------
    bertscores_np = np.asarray([], dtype=np.float32)
    if HAS_BERTSCORE:
        print("[INFO] Computing BERTScore F1 on CPU...")
        eval_cfg = cfg.get("eval", {})
        model_type = args.bertscore_model or eval_cfg.get("bertscore_model")
        batch_size = int(eval_cfg.get("bertscore_batch", args.bertscore_batch))

        bertscores_np, mean_bertscore = _compute_bertscore(
            rows,
            model_type=model_type,
            batch_size=batch_size,
        )
        summary["metrics"]["bertscore_f1"] = {
            "mean": mean_bertscore,
            "valid_count": int(bertscores_np.size),
        }

        top_bs, bot_bs, n_bs = _rank_by_metric(rows, "bertscore_f1", top_k=50)
        if n_bs > 0:
            _export_rationale_samples_json(
                top_bs,
                epoch_dir / f"top50_best_bertscore_epoch{epoch}.json",
            )
            _export_rationale_samples_json(
                bot_bs,
                epoch_dir / f"bottom50_worst_bertscore_epoch{epoch}.json",
            )
            _emit_metric_tiles(epoch_dir, "bertscore", top_bs, bot_bs, epoch)
            _histogram(
                bertscores_np,
                epoch_dir / f"bertscore_hist_epoch{epoch}.png",
                title=f"BERTScore(F1) distribution — epoch {epoch}",
                xlabel="BERTScore F1",
            )
    else:
        print("[WARN] bert_score not installed; skipping BERTScore computation.")

    # ---------- BLEURT (HF classification model) ----------
    bleurt_np = np.asarray([], dtype=np.float32)
    eval_cfg = cfg.get("eval", {})

    bleurt_model = args.bleurt_ckpt or eval_cfg.get("bleurt_ckpt")
    bleurt_device = args.bleurt_device or eval_cfg.get("bleurt_device", "cpu")
    bleurt_batch = int(eval_cfg.get("bleurt_batch", 16))

    if HAS_BLEURT and bleurt_model:
        print(f"[INFO] Computing BLEURT (HF) on device={bleurt_device} (model={bleurt_model})...")
        bleurt_np, mean_bleurt = _compute_bleurt_hf(
            rows,
            model_name=bleurt_model,
            device=bleurt_device,
            batch_size=bleurt_batch,
        )
        summary["metrics"]["bleurt"] = {
            "mean": mean_bleurt,
            "valid_count": int(bleurt_np.size),
        }

        top_bt, bot_bt, n_bt = _rank_by_metric(rows, "bleurt", top_k=50)
        if n_bt > 0:
            _export_rationale_samples_json(
                top_bt,
                epoch_dir / f"top50_best_bleurt_epoch{epoch}.json",
            )
            _export_rationale_samples_json(
                bot_bt,
                epoch_dir / f"bottom50_worst_bleurt_epoch{epoch}.json",
            )
            _emit_metric_tiles(epoch_dir, "bleurt", top_bt, bot_bt, epoch)
            _histogram(
                bleurt_np,
                epoch_dir / f"bleurt_hist_epoch{epoch}.png",
                title=f"BLEURT distribution — epoch {epoch}",
                xlabel="BLEURT",
            )
    elif not HAS_BLEURT and bleurt_model:
        print("[WARN] transformers/torch not installed; skipping BLEURT computation.")
    else:
        print("[INFO] BLEURT not requested (no model name/checkpoint); skipping.")


    # ---------- write summary JSON ----------
    out_json = epoch_dir / f"rationale_metrics_epoch{epoch}.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[OK] Wrote rationale metrics summary to: {out_json}")
    print("[OK] Done.")


if __name__ == "__main__":
    from utils.perf import time_main
    raise SystemExit(time_main(main, "make_rationale_metrics"))
