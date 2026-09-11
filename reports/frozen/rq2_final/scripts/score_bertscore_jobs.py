#!/usr/bin/env python3
"""Score immutable RQ2 text pairs with the frozen BERTScore configuration."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


MODEL_REVISION = "722cf37b1afa9454edce342e7895e588b6ff1d59"
BASELINE_SHA256 = "08e2248310d0c25d8e22ef65e0a6be15060269f1a9084e560c258f09a9e122ae"
BASELINE_F = 0.831226


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", required=True, type=Path)
    parser.add_argument("--jobs-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--model-snapshot", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--max-rows", type=int)
    args = parser.parse_args()

    import bert_score
    import torch
    import transformers
    from bert_score import BERTScorer

    if sha_file(args.jobs) != args.jobs_sha256:
        raise SystemExit("scoring job SHA-256 mismatch")
    if bert_score.__version__ != "0.3.12":
        raise SystemExit(f"wrong bert_score version: {bert_score.__version__}")
    if args.model_snapshot.name != MODEL_REVISION or not args.model_snapshot.is_dir():
        raise SystemExit("pinned model snapshot unavailable or wrong revision")
    baseline_path = Path(bert_score.__file__).parent / "rescale_baseline/en/roberta-large.tsv"
    if sha_file(baseline_path) != BASELINE_SHA256:
        raise SystemExit("BERTScore baseline resource SHA mismatch")
    baseline_rows = list(csv.DictReader(baseline_path.open(encoding="utf-8")))
    layer17_f = float(next(row["F"] for row in baseline_rows if row["LAYER"] == "17"))
    if abs(layer17_f - BASELINE_F) > 5e-7:
        raise SystemExit(f"unexpected layer-17 baseline: {layer17_f}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")

    scorer = BERTScorer(
        model_type=str(args.model_snapshot),
        num_layers=17,
        batch_size=args.batch_size,
        nthreads=4,
        all_layers=False,
        idf=False,
        device="cuda",
        rescale_with_baseline=False,
        use_fast_tokenizer=False,
    )

    input_fields = [
        "job_index", "analysis", "arm", "condition", "unit", "record_id", "source_frame_id",
        "source_text_sha256", "condition_text_sha256", "source_text", "condition_text",
    ]
    output_fields = [
        "job_index", "analysis", "arm", "condition", "unit", "record_id", "source_frame_id",
        "source_text_sha256", "condition_text_sha256", "bertscore_precision", "bertscore_recall",
        "bertscore_f1", "raw_drift", "baseline_rescaled_f1", "baseline_rescaled_drift",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{args.output.name}.", dir=args.output.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    started = time.time()
    rows_scored = 0
    try:
        with gzip.open(args.jobs, "rt", encoding="utf-8", newline="") as source, gzip.open(temporary, "wt", encoding="utf-8", newline="") as target:
            reader = csv.DictReader(source, delimiter="\t")
            if reader.fieldnames != input_fields:
                raise RuntimeError(f"unexpected input fields: {reader.fieldnames}")
            writer = csv.DictWriter(target, fieldnames=output_fields, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            chunk: list[dict[str, str]] = []

            def score_chunk(items: list[dict[str, str]]) -> None:
                nonlocal rows_scored
                if not items:
                    return
                refs = [row["source_text"] for row in items]
                cands = [row["condition_text"] for row in items]
                precision, recall, f1 = scorer.score(cands, refs, batch_size=args.batch_size)
                p_values = precision.cpu().numpy().astype(float)
                r_values = recall.cpu().numpy().astype(float)
                f_values = f1.cpu().numpy().astype(float)
                for row, p_value, r_value, f_value in zip(items, p_values, r_values, f_values):
                    rescaled_f = (f_value - BASELINE_F) / (1.0 - BASELINE_F)
                    writer.writerow({
                        "job_index": row["job_index"],
                        "analysis": row["analysis"],
                        "arm": row["arm"],
                        "condition": row["condition"],
                        "unit": row["unit"],
                        "record_id": row["record_id"],
                        "source_frame_id": row["source_frame_id"],
                        "source_text_sha256": row["source_text_sha256"],
                        "condition_text_sha256": row["condition_text_sha256"],
                        "bertscore_precision": format(float(p_value), ".17g"),
                        "bertscore_recall": format(float(r_value), ".17g"),
                        "bertscore_f1": format(float(f_value), ".17g"),
                        "raw_drift": format(1.0 - float(f_value), ".17g"),
                        "baseline_rescaled_f1": format(float(rescaled_f), ".17g"),
                        "baseline_rescaled_drift": format(1.0 - float(rescaled_f), ".17g"),
                    })
                rows_scored += len(items)
                target.flush()
                print(json.dumps({"rows_scored": rows_scored, "elapsed_seconds": time.time() - started}), flush=True)

            for row in reader:
                if args.max_rows is not None and rows_scored + len(chunk) >= args.max_rows:
                    break
                chunk.append(row)
                if len(chunk) >= args.chunk_size:
                    score_chunk(chunk)
                    chunk = []
            score_chunk(chunk)
        os.replace(temporary, args.output)
    finally:
        if temporary.exists():
            temporary.unlink()

    metadata = {
        "schema_version": "rq2-bertscore-scoring-metadata-v1",
        "jobs_path": str(args.jobs),
        "jobs_sha256": sha_file(args.jobs),
        "output_path": str(args.output),
        "output_sha256": sha_file(args.output),
        "rows_scored": rows_scored,
        "bert_score_version": bert_score.__version__,
        "model": "roberta-large",
        "model_snapshot": MODEL_REVISION,
        "model_snapshot_path": str(args.model_snapshot),
        "num_layers": 17,
        "idf": False,
        "primary_baseline_rescaling": False,
        "diagnostic_baseline_rescaling": True,
        "baseline_resource": str(baseline_path),
        "baseline_resource_sha256": BASELINE_SHA256,
        "baseline_layer17_f_resource": layer17_f,
        "baseline_layer17_f_frozen_diagnostic_constant": BASELINE_F,
        "tokenizer_class": type(scorer._tokenizer).__name__,
        "use_fast_tokenizer": False,
        "device": str(scorer.device),
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_uuid": subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader", "--id=0"],
            check=True, capture_output=True, text=True,
        ).stdout.strip(),
        "batch_size": args.batch_size,
        "chunk_size": args.chunk_size,
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers_version": transformers.__version__,
        "python_version": sys.version,
        "elapsed_seconds": time.time() - started,
    }
    args.metadata.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"rows_scored": rows_scored, "output_sha256": metadata["output_sha256"], "elapsed_seconds": metadata["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
