"""Sufficiency battery: Stage-1 accuracy under 6 image-perturbation regimes (all arms).

Extends the grey-only Sufficiency Gap to a battery. For each arm, iterates the val
set once and scores the Stage-1 classifier under:
  real      identity                         reference
  grey      uniform (128,128,128) 256x256     info-destroying  (faithful -> DROP)
  wrong     a different real scene            misleading       (faithful -> DROP)
  occlude   ~50% pixels masked to black       info-destroying  (faithful -> DROP)
  noise     heavy Gaussian noise (std 80)     info-destroying  (faithful -> DROP)
  hflip     horizontal mirror                 semantics-preserving (faithful -> HOLD)

Per regime we report accuracy and the gap vs real. occlude/noise are seeded by a
per-record index so the perturbation is identical across arms; 'wrong' is drawn
from the previous batch (different image_id), as in the 7c script. Classification-
only (forward passes), no generation. Writes per-arm summary JSON + per-record CSV.
"""

from __future__ import annotations
import argparse, csv, json, sys, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.vqa_dataset import VQADataset, vqa_collate_fn  # noqa: E402
from cli._common import resolve_config  # noqa: E402
from scripts.sufficiency_blindfold import (  # noqa: E402
    format_prompt, build_model_inputs_from_prompts, pool_decoder, load_stage1_backbone_and_head,
)
from transformers import AutoProcessor  # noqa: E402

from utils.checkpoints import STAGE1_RUNS  # canonical, single source of truth
REGIMES = ["real", "grey", "wrong", "occlude", "noise", "hflip"]
GREY = Image.new("RGB", (256, 256), (128, 128, 128))


def transform(name, img, idx, wrong_img):
    if name == "real":
        return img
    if name == "grey":
        return GREY
    if name == "wrong":
        return wrong_img
    if name == "hflip":
        return img.convert("RGB").transpose(Image.FLIP_LEFT_RIGHT)
    a = np.array(img.convert("RGB"))
    if name == "occlude":
        rng = np.random.default_rng(idx)
        b = a.copy(); b[rng.random(a.shape[:2]) < 0.5] = 0
        return Image.fromarray(b)
    if name == "noise":
        rng = np.random.default_rng(1_000_000 + idx)
        b = np.clip(a.astype(np.int16) + rng.normal(0, 80, a.shape).astype(np.int16), 0, 255).astype(np.uint8)
        return Image.fromarray(b)
    raise ValueError(name)


def run_arm(arm, device, max_samples, out_dir):
    cfg = resolve_config(f"configs/{arm}.yaml", mode_override="thesis")
    cfg.setdefault("paths", {})["stage1_checkpoints_dir"] = str(PROJECT_ROOT / STAGE1_RUNS[arm] / "checkpoints")
    data_cfg, exp_cfg = cfg["data"], cfg["experiment"]
    ds = VQADataset(
        json_train_path=data_cfg["json_train_path"], json_val_path=data_cfg["json_val_path"],
        images_plain_train=data_cfg["images_plain_train"], images_plain_val=data_cfg["images_plain_val"],
        images_point_train=data_cfg["images_point_train"], images_point_val=data_cfg["images_point_val"],
        split=data_cfg["split_val"], arm=arm,
        caption_key=exp_cfg.get("caption_key", "image_descriptions.caption"), drop_missing=True)
    dl = DataLoader(ds, batch_size=max(1, cfg["loader"].get("batch_size", 8)), shuffle=False,
                    num_workers=max(1, cfg["loader"].get("num_workers", 2)),
                    pin_memory=cfg["loader"].get("pin_memory", True), drop_last=False, collate_fn=vqa_collate_fn)
    processor = AutoProcessor.from_pretrained(cfg["backbone"]["model_name"], trust_remote_code=True)
    backbone, head, ckpt = load_stage1_backbone_and_head(cfg, device)
    stage1_tmpl = cfg["prompts"]["stage1_template"]
    use_caption = arm.endswith("_desc")

    correct = {r: 0 for r in REGIMES}
    total = 0
    rows = []
    prev_imgs, prev_ids = [], []
    t0 = time.time()

    def predict(images, prompts):
        mi = build_model_inputs_from_prompts(processor, images, prompts, device)
        with torch.no_grad():
            return head(pool_decoder(backbone, mi, device)).argmax(dim=-1)

    for batch in tqdm(dl, desc=f"battery:{arm}"):
        images, questions = batch["images"], batch["questions"]
        choices_list, labels, captions = batch["choices"], batch["labels"], batch["captions"]
        ids = batch.get("image_id", [None] * len(images))
        B = len(images)
        if max_samples > 0 and total >= max_samples:
            break
        prompts = [format_prompt(stage1_tmpl, questions[i], choices_list[i],
                                 captions[i] if (use_caption and i < len(captions)) else None) for i in range(B)]
        # per-record 'wrong' image from previous batch (different image_id), else grey
        wrong_for = []
        for i in range(B):
            cand = GREY
            for img_c, id_c in zip(prev_imgs, prev_ids):
                if id_c != ids[i]:
                    cand = img_c; break
            wrong_for.append(cand)
        eff = B if max_samples <= 0 else min(B, max_samples - total)
        labels_t = torch.tensor(labels, device=device, dtype=torch.long)
        preds = {}
        for r in REGIMES:
            imgs = [transform(r, images[i], total + i, wrong_for[i]) for i in range(B)]
            p = predict(imgs, prompts)
            preds[r] = p
            correct[r] += (p[:eff] == labels_t[:eff]).sum().item()
        for i in range(eff):
            row = {"image_id": ids[i], "question": questions[i], "gold": int(labels[i])}
            for r in REGIMES:
                row[f"pred_{r}"] = int(preds[r][i])
                row[f"correct_{r}"] = int(int(preds[r][i]) == int(labels[i]))
            rows.append(row)
        total += eff
        prev_imgs, prev_ids = list(images), list(ids)
        if total % 200 < B:
            print(f"[battery:{arm}] {total} done, {total/(time.time()-t0):.2f} rec/s", flush=True)

    acc = {r: correct[r] / max(1, total) for r in REGIMES}
    gaps = {r: acc["real"] - acc[r] for r in REGIMES if r != "real"}
    with (out_dir / f"{arm}_examples.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["image_id"])
        w.writeheader(); w.writerows(rows)
    summary = {"arm": arm, "n": total, "stage1_checkpoint": str(ckpt),
               "accuracy": acc, "gap_vs_real": gaps}
    (out_dir / f"{arm}.json").write_text(json.dumps(summary, indent=2))
    print(f"[battery:{arm}] acc=" + " ".join(f"{r}:{acc[r]:.3f}" for r in REGIMES), flush=True)
    del backbone, head
    torch.cuda.empty_cache()
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=list(STAGE1_RUNS))
    ap.add_argument("--max-samples", type=int, default=-1)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / "reports/runs/_meta/sufficiency_battery" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    results = [run_arm(a, device, args.max_samples, out_dir) for a in args.arms]
    (out_dir / "summary_all_arms.json").write_text(json.dumps(results, indent=2))
    print("\n==================== SUFFICIENCY BATTERY (accuracy by regime) ====================")
    hdr = "arm".ljust(12) + "".join(r.rjust(9) for r in REGIMES)
    print(hdr)
    for s in results:
        print(s["arm"].ljust(12) + "".join(f"{s['accuracy'][r]:>9.3f}" for r in REGIMES))
    print(f"-> {out_dir/'summary_all_arms.json'}")
    return 0


if __name__ == "__main__":
    from utils.perf import time_main
    raise SystemExit(time_main(main, "sufficiency_battery"))
