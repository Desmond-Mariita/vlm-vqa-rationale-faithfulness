"""Stage-1 runner: wraps scripts/train_answer_qwen.py with config resolution.

Usage:
    python -m cli.run_stage1 --cfg configs/plain.yaml [--mode thesis] [--seed 42]

Writes outputs under reports/runs/<arm>/stage1/<run_id>/.
"""

from __future__ import annotations

import argparse
import sys

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
    parser = argparse.ArgumentParser(description="Stage-1 training runner")
    parser.add_argument("--cfg", required=True, help="arm config (e.g., configs/plain.yaml)")
    parser.add_argument("--mode", default=None, choices=sorted(VALID_MODES))
    parser.add_argument("--seed", type=int, default=None, help="override project.seed")
    parser.add_argument("--resume-dir", default=None,
                        help="reuse this exact run dir and resume from its checkpoints "
                             "(instead of minting a fresh timestamped dir)")
    args = parser.parse_args()

    cfg = resolve_config(args.cfg, mode_override=args.mode)
    if args.seed is not None:
        cfg["project"]["seed"] = args.seed
    set_all_seeds(cfg["project"]["seed"])

    run = prepare_run_dir(cfg, suffix="stage1", reuse_run_dir=args.resume_dir)
    print_resolved(cfg)

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "train_answer_qwen.py"),
        "--cfg",
        run["resolved_cfg_path"],
    ]
    run_subprocess(cmd)
    print(f"[cli] Stage-1 completed. Run dir: {run['run_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
