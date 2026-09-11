"""Stage-2 runner: wraps scripts/train_rationale_qwen.py with config resolution.

Stage-2 is trained ONCE on gold answers. The predicted-answer condition
is alternate inference from the same gold-trained checkpoint, not a
second training run. Two entry modes:

    # Train gold (required first):
    python -m cli.run_stage2 --cfg configs/stage2/plain.yaml --answer-source gold \
        --stage1-ckpt .../stage1/<run>/checkpoints/model_best.pt

    # Inference-only pred pass reusing the gold Stage-2 checkpoint:
    python -m cli.run_stage2 --cfg configs/stage2/plain.yaml --answer-source pred \
        --stage1-preds-val .../stage1/<run>/outputs/stage1_preds_val.json \
        --from-stage2-ckpt .../stage2_gold/<run>/checkpoints/stage2_best.pt

The stage-1 predictions file and stage-2 checkpoint are explicitly passed
through the resolved config so there is NO implicit "latest" lookup.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from cli._common import (
    PROJECT_ROOT,
    VALID_MODES,
    prepare_run_dir,
    print_resolved,
    resolve_config,
    run_subprocess,
    set_all_seeds,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage-2 training runner")
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--mode", default=None, choices=sorted(VALID_MODES))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--answer-source",
        required=True,
        choices=["gold", "pred"],
        help="conditioning answer at Stage-2 eval/generation time",
    )
    parser.add_argument(
        "--stage1-preds-train",
        default=None,
        help="path to stage1_preds_train.json (required when --answer-source pred)",
    )
    parser.add_argument(
        "--stage1-preds-val",
        default=None,
        help="path to stage1_preds_val.json (required when --answer-source pred)",
    )
    parser.add_argument(
        "--stage1-ckpt",
        default=None,
        help="path to Stage-1 model_best.pt (required for gold training; "
             "optional in inference-only pred mode since Stage-2 weights "
             "already include the warm-started backbone)",
    )
    parser.add_argument(
        "--from-stage2-ckpt",
        default=None,
        help="path to a Stage-2 best checkpoint (e.g. stage2_gold/.../stage2_best.pt). "
             "When set, training is skipped and rationales are generated from "
             "this checkpoint with the current --answer-source. Intended for "
             "the pred alternate-inference pass.",
    )
    parser.add_argument("--resume-dir", default=None,
                        help="reuse this exact run dir and resume from its checkpoints "
                             "(epoch/step-level resume; for the gold training pass)")
    args = parser.parse_args()

    inference_only = bool(args.from_stage2_ckpt)

    if inference_only:
        if args.answer_source != "pred":
            raise ValueError(
                "--from-stage2-ckpt is only supported with --answer-source pred "
                "(gold rationales come from the original training run)."
            )
        stage2_ckpt_abs = os.path.abspath(args.from_stage2_ckpt)
        if not os.path.exists(stage2_ckpt_abs):
            raise FileNotFoundError(
                f"Stage-2 checkpoint not found: {stage2_ckpt_abs}"
            )
    else:
        stage2_ckpt_abs = None

    stage1_ckpt_abs: str | None = None
    if args.stage1_ckpt:
        stage1_ckpt_abs = os.path.abspath(args.stage1_ckpt)
        if not os.path.exists(stage1_ckpt_abs):
            raise FileNotFoundError(f"Stage-1 checkpoint not found: {stage1_ckpt_abs}")
    elif not inference_only:
        raise ValueError(
            "--stage1-ckpt is required for Stage-2 training "
            "(omit it only when using --from-stage2-ckpt for inference)."
        )

    if args.answer_source == "pred":
        # Stage-2 is trained on gold answers; predicted answers are only consumed
        # at validation/generation time, so only the VAL prediction file is required.
        # (--stage1-preds-train is accepted for backwards compat but optional.)
        if not args.stage1_preds_val:
            raise ValueError(
                "--answer-source pred requires --stage1-preds-val"
            )
        if not os.path.exists(args.stage1_preds_val):
            raise FileNotFoundError(f"stage1 val predictions not found: {args.stage1_preds_val}")
        if args.stage1_preds_train and not os.path.exists(args.stage1_preds_train):
            raise FileNotFoundError(f"stage1 train predictions not found: {args.stage1_preds_train}")

    cfg = resolve_config(args.cfg, mode_override=args.mode)
    if args.seed is not None:
        cfg["project"]["seed"] = args.seed
    cfg.setdefault("train_stage2", {})["answer_source"] = args.answer_source
    if args.answer_source == "pred":
        if args.stage1_preds_train:
            cfg["train_stage2"]["stage1_preds_train"] = args.stage1_preds_train
        cfg["train_stage2"]["stage1_preds_val"] = args.stage1_preds_val
    if stage1_ckpt_abs:
        cfg.setdefault("train", {})["stage1_ckpt"] = stage1_ckpt_abs
    if inference_only:
        cfg["train_stage2"]["inference_only"] = True
        cfg["train_stage2"]["stage2_ckpt"] = stage2_ckpt_abs

    set_all_seeds(cfg["project"]["seed"])
    suffix = f"stage2_{args.answer_source}"
    # Resume is only meaningful for the gold TRAINING pass; the pred pass is
    # inference-only from a fixed checkpoint, so a fresh dir each time is fine.
    reuse = args.resume_dir if (args.resume_dir and not inference_only) else None
    run = prepare_run_dir(cfg, suffix=suffix, reuse_run_dir=reuse)
    print_resolved(cfg)

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "train_rationale_qwen.py"),
        "--cfg",
        run["resolved_cfg_path"],
        "--answer-source",
        args.answer_source,
    ]
    run_subprocess(cmd)
    print(f"[cli] Stage-2 ({args.answer_source}) completed. Run dir: {run['run_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
