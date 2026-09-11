# Reproducibility

Run every command from the repository root. The minimum supported path is numeric Level B: `make check` and `python scripts/build_public_tables.py --output-dir reports/generated/tables --check`. It reads frozen numeric inputs, derives rates from recorded integer counts, formats stored confidence intervals and matches all 12 active TeX table files byte for byte. It does not recompute inference, scores or bootstrap intervals. `make figures` requires the optional plotting requirements and system LaTeX packages; it renders three result PDFs in a temporary directory and checks against approved hashes.

Level A: active manuscript source, TeX figure definitions and author-created numeric plots are present. All 43 VCR raster dependencies, the thesis PDF, signature.png and uni_potsdam_logo.png remain withheld. The declaration source remains unsigned. `make thesis-status` verifies source hashes and explicitly reports withheld assets; the manual GitHub thesis workflow validates status only. `make -C thesis pdf` is the original build command and requires separately acquired/reconstructed VCR imagery and the local logo for a full image-inclusive build. See [Data access](DATA_ACCESS.md). No replacement PDF or substitute images are supplied.

Level C: obtain the exact external VCR validation subset, frozen descriptions, original manifests, model snapshots and checkpoints described in the external stubs. Source-relative paths in original hash-bound manifests may be environment-specific: preserve manifest bytes, supply the documented project/model/transport roots, and reconstruct a licensed external layout. The public numeric projections are not interchangeable with original byte-bound inference or scoring inputs. Existing historical analysis verifiers expect the original external archive, not the public projection.

Frozen runners use `--manifest`, `--shard`, `--dataset`, `--project-root`, `--model-dir`, `--transport-root`, `--local-assets`, `--output-dir`; Stage 2 additionally requires `--worker-id`. `--local-assets` is a local JSON document with `schema_version: 1` and a nonempty `files` list of `{path, sha256, bytes}` entries, relative to that document or absolute locally. Every listed file is actually hashed. Independent dataset, manifest, shard, prompt, image and checkpoint checks remain in the runners. Original full-manifest hashes are enforced. An optional `--max-records` rejects larger shards. See each script's argument parser for required scoring inputs; full inference was not executed in this release task.

Level D: externally acquire VCR, prepare plain/polygon images, and supply the frozen licensed descriptions before constructing the fixed subsets. The subset commands reconstructed and byte-replayed during discovery were:

```bash
python data/subset_balanced_json.py --input data/final/vcr_train_final_with_orig.json --output data/final/vcr_train_5_pct.json --pct 5 --label-key answer_label --seed 42
python data/subset_balanced_json.py --input data/final/vcr_val_final_with_orig.json --output data/final/vcr_val_10_pct.json --pct 10 --label-key answer_label --seed 42
```

These are reconstructed invocations, not recovered original command lines. For each of the four arms, after installing the GPU environment and preparing licensed inputs:

```bash
accelerate launch scripts/train_answer_qwen.py --cfg configs/stage1/plain.yaml
accelerate launch scripts/train_rationale_qwen.py --cfg configs/stage2/plain.yaml --answer-source gold
```

Repeat with `point`, `plain_desc`, `point_desc`. The explicit Stage 2 configurations preserve the frozen six-epoch gold-answer regime. Stage 2 warm-starts from the corresponding Stage 1 output at reports/generated/<arm>/stage1/checkpoints/model_best.pt. Archived resolved configs identify the historical retained external checkpoint paths. New output directories are separate from historical resolved configurations. Historical validation generation used 128 tokens; final primary inference used 256. Historical bootstrap defaults in archived run configs do not define final analysis: `configs/evaluation/final.yaml` records 10,000 source-frame cluster replicates and seed 42.

PRED-A is secondary and conditioned on historical Stage 1 epoch-3 predictions. Do not replace them with the epoch-2 RQ1 predictions and claim the same experiment. Exact GPT-4o description regeneration is not guaranteed. GPU evaluation and retraining remain externally dependent and are not represented as end-to-end validated by CPU CI.
