#!/usr/bin/env python3
"""Build the immutable pre-inference manifests outside the canonical repository.

This program is deliberately standard-library only.  It reads the canonical
dataset, images, checkpoints, and historical artifacts but writes only to the
caller-supplied external execution-package directory.  It performs no model
inference and does not alter any canonical file.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Iterable


ARMS = ("plain", "point", "plain_desc", "point_desc")
CONDITIONS = ("source", "grey", "mismatch", "mask", "noise", "mirror")
MODEL_REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"
DATASET_SHA = "0288740b785021e667c12eb08ab2c222a65f80b2702d598a7ebb7b942b3a5fdc"
CANDIDATE_SHA = "99d2a03b6ab3ba434a146f191471b9ef849c494fa3550a5b2cdcb31dad29da6d"
CANDIDATE_GENERATOR_SHA = "0e4694bd7e5a1e84ac7d443032de05882bbb520aee34bae11509f0136299e62b"
EXPECTED_ID_HASH = "65cd7a6aa4665d475f3663a6aec054a1eeb99bc283ba841cc20384abb257aa17"
SEED = 20260829

PLAIN_ROOT_REL = Path("data/processed/vcr/val/images")
POINT_ROOT_REL = Path("data/final/val/images")
DATASET_REL = Path("data/final/vcr_val_10_pct.json")

STAGE1 = {
    "plain": (
        "reports/runs/plain/stage1/scratch_v1/checkpoints/model_best.pt",
        "ee80fd5e56cb8fd6625765a6d17aef65761b7dc5762c729afe5d94484fc7e9a9",
        "plain_stage1_transport.pt",
        "7b570b1642881e2706425ad288d84b1c099b38c9e68ca02c8a1dfe634b7024ee",
    ),
    "point": (
        "reports/runs/point/stage1/20260429_234801/checkpoints/model_best.pt",
        "a46b2cd79f6abe45da612916a2a9f4ac25dcc462f7762cd6ce81f0fa8c4f96e8",
        "point_stage1_transport.pt",
        "e3bcef3417160f07a164994c3617cd5ec94952bbdc819983642e7bf2c88fa114",
    ),
    "plain_desc": (
        "reports/runs/plain_desc/stage1/20260507_203440/checkpoints/model_best.pt",
        "928bc4bf3ed514d750884ffc8b5e25b01733ae59e10e05dfa7cedf452d767de7",
        "plain_desc_stage1_transport.pt",
        "10c1d1264bf313e0cc8c2bf70723d276413c3c6c46f0906a53b955bcf9cde220",
    ),
    "point_desc": (
        "reports/runs/point_desc/stage1/20260511_211533/checkpoints/model_best.pt",
        "869bc1d38d259512340ebb12f7bca813b4e3f5c820c7b4b510f298d47f824ab5",
        "point_desc_stage1_transport.pt",
        "7b78fc4ce7d7257820438abe927fcc93848306f92d4ca4df77348739f81cede5",
    ),
}

STAGE2 = {
    "plain": (
        "reports/runs/plain/stage2_gold/scratch_v1/checkpoints/stage2_best.pt",
        "e629c5c818e527b810e6df8d6707e7c634f17ff984280479a992fb2b558569cb",
        "plain_stage2_transport.pt",
        "5c5627554f044d10b9c7c891aab756a81245e70bdee213d27648887b0e48079b",
    ),
    "point": (
        "reports/runs/point/stage2_gold/20260505_141209/checkpoints/stage2_best.pt",
        "5cfa9a692280ff79ad93192e55c10292eacf69b2a05123f9a0d8957217f1931a",
        "point_stage2_transport.pt",
        "873dc3cf20a34f66b8456e5863b443353ed170a93a6a4ada334663b22afce44f",
    ),
    "plain_desc": (
        "reports/runs/plain_desc/stage2_gold/20260509_174550/checkpoints/stage2_best.pt",
        "b6016869c99b4ef27099f0e40aba3f28a6ea9528820144203b0ba06dbcb9a6b8",
        "plain_desc_stage2_transport.pt",
        "05a32ade19c26dc61eb5eaa3202c20bb3e084aa36eb944ed0d16627125b02e09",
    ),
    "point_desc": (
        "reports/runs/point_desc/stage2_gold/20260513_183118/checkpoints/stage2_best.pt",
        "4735e4a954069f07f0d22e0228a3839d1f678b7c86f898d64f1a4a7a38546e12",
        "point_desc_stage2_transport.pt",
        "0bdc606d92f115533dbc22704697217ca856f4ee7739174ae77a7de15fadf38a",
    ),
}

PRED = {
    "plain": (
        "reports/runs/plain/stage2_pred/20260704_145350/outputs/Route2/generated_rationales.jsonl",
        "39ff955c3778c8e6e0ae1b0049b1bea2289e6028dc5d2061f4cadb270b7d6937",
        "reports/runs/plain/stage1/scratch_v1/outputs/stage1_preds_val.json",
        2437, 2136, 517, 0.574887156339762, 0.5280814172634754,
        {"correct": {"conditional": 0.596, "pipeline": 0.545}, "wrong": {"conditional": 0.490, "pipeline": 0.457}},
    ),
    "point": (
        "reports/runs/point/stage2_pred/20260506_100946/outputs/Route2/generated_rationales.jsonl",
        "fba34525d1f8695e822cdf44fa17288d5d7499743afc8773035d5ca6599dc163",
        "reports/runs/point/stage1/20260429_234801/outputs/stage1_preds_val.json",
        1057, 2144, 509, 0.588, 0.234,
        {"correct": {"conditional": 0.607, "pipeline": 0.246}, "wrong": {"conditional": 0.497, "pipeline": 0.185}},
    ),
    "plain_desc": (
        "reports/runs/plain_desc/stage2_pred/20260510_144738/outputs/Route2/generated_rationales.jsonl",
        "c356042dbd27ded5c9a75d275cad16562232fc8c5019b53d041ee692b603b646",
        "reports/runs/plain_desc/stage1/20260507_203440/outputs/stage1_preds_val.json",
        1624, 2153, 500, 0.5369458128078818, 0.32868450810403316,
        {"correct": {"conditional": 0.538, "pipeline": 0.330}, "wrong": {"conditional": 0.533, "pipeline": 0.322}},
    ),
    "point_desc": (
        "reports/runs/point_desc/stage2_pred/20260514_152333/outputs/Route2/generated_rationales.jsonl",
        "0ac82cc018fda8bd47edb1d2f18435ed9fa7474a787183d0c778db1b3d63a991",
        "reports/runs/point_desc/stage1/20260511_211533/outputs/stage1_preds_val.json",
        2574, 2161, 492, 0.631, 0.612,
        {"correct": {"conditional": 0.639, "pipeline": 0.620}, "wrong": {"conditional": 0.593, "pipeline": 0.575}},
    ),
}

BATTERY = {
    "plain": "reports/runs/_meta/drift_battery/20260708_015123/plain_samples.csv",
    "point": "reports/runs/_meta/drift_battery/20260707_191545/point_samples.csv",
    "plain_desc": "reports/runs/_meta/drift_battery/20260707_212705/plain_desc_samples.csv",
    "point_desc": "reports/runs/_meta/drift_battery/20260707_234157/point_desc_samples.csv",
}

KNOWN = {arm: f"reports/runs/_meta/drift_matched_control/{arm}_gen.jsonl" for arm in ARMS}

DECODE = {
    "batch_size": 1,
    "do_sample": False,
    "num_beams": 1,
    "max_new_tokens": 256,
    "seed": 42,
    "quantisation": "4-bit NF4",
    "double_quantisation": True,
    "compute_dtype": "bfloat16",
    "temperature": 1.0,
    "top_p": 1.0,
    "historical_determinism": "seed-only; do not add strict deterministic flags",
}

CONSTRUCTORS = {
    "source": {"name": "source", "rule": "load the current record through the arm-specific asset family; convert RGB"},
    "grey": {"name": "grey", "mode": "RGB", "size": [256, 256], "fill": [128, 128, 128]},
    "mismatch": {"name": "mismatch", "rule": "load final Option-B target through the arm-specific asset family; convert RGB"},
    "mask": {"name": "random-pixel mask", "rule": "RGB source; np.random.default_rng(dataset_index); draw 2-D U[0,1); set all channels to 0 where draw < 0.5"},
    "noise": {"name": "Gaussian noise", "rule": "RGB source; np.random.default_rng(1000000+dataset_index); add elementwise integer-cast N(0,80); clip [0,255]; uint8 RGB"},
    "mirror": {"name": "horizontal mirror", "rule": "RGB source; PIL Image.Transpose.FLIP_LEFT_RIGHT"},
}


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_text(value: str) -> str:
    return sha_bytes(value.encode("utf-8"))


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def ordered_hash(values: Iterable[str]) -> str:
    return sha_text("\n".join(values))


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> str:
    if not rows:
        raise RuntimeError(f"refusing empty TSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    return sha_file(path)


def write_json(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.write_text(payload, encoding="utf-8")
    return sha_file(path)


def caption(row: dict[str, Any]) -> str:
    value = row.get("image_descriptions") or {}
    return str(value.get("caption") or "")[:256]


def stage1_prompt(row: dict[str, Any], arm: str) -> str:
    choices = row["choices"]
    prompt = f"""<image>
Question: {row['question']}
Choices:
  (A) {choices[0]}
  (B) {choices[1]}
  (C) {choices[2]}
  (D) {choices[3]}
Instruction: Select the best option and answer with a single letter (A/B/C/D).
"""
    if arm.endswith("_desc") and caption(row):
        prompt += f"\nCaption: {caption(row)}"
    return prompt


def stage2_prompt(row: dict[str, Any], arm: str, answer_idx: int, description_mode: str = "on") -> str:
    choices = list(row["choices"][:4])
    while len(choices) < 4:
        choices.append("")
    answer = choices[answer_idx] if 0 <= answer_idx < len(choices) else ""
    prompt = f"""You are an expert visual commonsense reasoner.

You see an image and a multiple-choice question with four options.
Your task is to explain, step by step, why the given answer is correct,
using only what can reasonably be inferred from the image and question.

Question: {row['question']}

Choices:
  (0) {choices[0]}
  (1) {choices[1]}
  (2) {choices[2]}
  (3) {choices[3]}

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
    if arm.endswith("_desc"):
        if description_mode == "on" and caption(row):
            prompt += f"\n\nImage caption: {caption(row)}"
        elif description_mode == "off":
            prompt += "\n\nImage caption:"
    return prompt


def asset_paths(root: Path, row: dict[str, Any]) -> tuple[Path, Path]:
    return root / PLAIN_ROOT_REL / row["orig_image_id"], root / POINT_ROOT_REL / row["image_id"]


def relative(root: Path, path: Path) -> str:
    return str(path.relative_to(root))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--canonical-root", type=Path, required=True)
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--candidate-generator", type=Path, required=True)
    ap.add_argument("--transport-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    root = args.canonical_root.resolve()
    out = args.output.resolve()
    manifests = out / "manifests"
    audits = out / "audits"
    provenance = out / "provenance"
    for directory in (manifests, audits, provenance):
        directory.mkdir(parents=True, exist_ok=True)

    dataset_path = root / DATASET_REL
    if sha_file(dataset_path) != DATASET_SHA:
        raise SystemExit("canonical dataset SHA-256 mismatch")
    if sha_file(args.candidate) != CANDIDATE_SHA:
        raise SystemExit("candidate mismatch SHA-256 mismatch")
    if sha_file(args.candidate_generator) != CANDIDATE_GENERATOR_SHA:
        raise SystemExit("candidate generator SHA-256 mismatch")

    records: list[dict[str, Any]] = json.loads(dataset_path.read_text(encoding="utf-8"))
    if len(records) != 2653:
        raise SystemExit(f"expected 2,653 records, found {len(records)}")
    if ordered_hash(str(r["image_id"]) for r in records) != EXPECTED_ID_HASH:
        raise SystemExit("ordered dataset ID hash mismatch")
    by_id = {str(row["image_id"]): (index, row) for index, row in enumerate(records)}
    if len(by_id) != 2653:
        raise SystemExit("record IDs are not unique")

    # Hash every unique asset exactly once on the canonical host.
    asset_cache: dict[str, str] = {}
    inventory: list[dict[str, Any]] = []
    for index, row in enumerate(records):
        plain, point = asset_paths(root, row)
        if not plain.is_file() or not point.is_file():
            raise SystemExit(f"missing source asset at dataset index {index}: {plain} / {point}")
        for path in (plain, point):
            key = str(path)
            if key not in asset_cache:
                asset_cache[key] = sha_file(path)
        inventory.append({
            "dataset_index": index,
            "record_id": row["image_id"],
            "source_frame_id": row["orig_image_id"],
            "plain_asset": relative(root, plain),
            "plain_asset_sha256": asset_cache[str(plain)],
            "plain_asset_bytes": plain.stat().st_size,
            "point_asset": relative(root, point),
            "point_asset_sha256": asset_cache[str(point)],
            "point_asset_bytes": point.stat().st_size,
        })
    inventory_sha = write_tsv(manifests / "evaluation_asset_inventory.tsv", inventory)

    # Independently validate candidate and enrich it into the adopted final map.
    candidate = read_tsv(args.candidate)
    if len(candidate) != 2653:
        raise SystemExit("candidate does not contain exactly 2,653 rows")
    sources = {r["image_id"] for r in candidate}
    targets = {r["target_image_id"] for r in candidate}
    if len(sources) != 2653 or len(targets) != 2653:
        raise SystemExit("candidate is not a one-to-one record derangement")
    final_mismatch: list[dict[str, Any]] = []
    for expected_index, cand in enumerate(candidate):
        source_index = int(cand["index"])
        target_index = int(cand["target_index"])
        if source_index != expected_index:
            raise SystemExit("candidate is not ordered by canonical dataset index")
        source = records[source_index]
        target = records[target_index]
        if cand["image_id"] != source["image_id"] or cand["target_image_id"] != target["image_id"]:
            raise SystemExit(f"candidate/dataset mismatch at row {expected_index}")
        if source_index == target_index or source["image_id"] == target["image_id"]:
            raise SystemExit(f"fixed point at row {expected_index}")
        if source["orig_image_id"] == target["orig_image_id"]:
            raise SystemExit(f"same-source-frame target at row {expected_index}")
        plain_target, point_target = asset_paths(root, target)
        payload = {
            "source_dataset_index": source_index,
            "source_image_id": source["image_id"],
            "source_orig_image_id": source["orig_image_id"],
            "target_dataset_index": target_index,
            "target_image_id": target["image_id"],
            "target_orig_image_id": target["orig_image_id"],
            "plain_target_asset": relative(root, plain_target),
            "plain_target_asset_sha256": asset_cache[str(plain_target)],
            "point_target_asset": relative(root, point_target),
            "point_target_asset_sha256": asset_cache[str(point_target)],
            "generation_seed": int(cand["seed"]),
            "permutation_attempt": int(cand["permutation_attempt"]),
            "candidate_manifest_sha256": CANDIDATE_SHA,
            "manifest_generation_script_sha256": CANDIDATE_GENERATOR_SHA,
            "finalisation_script_sha256": sha_file(Path(__file__)),
            "constraint_audit_result": "PASS:no_fixed_point;source_frame_distinct;assets_complete;outcome_blind",
        }
        payload["row_sha256"] = sha_text(canonical_json(payload))
        final_mismatch.append(payload)
    mismatch_path = manifests / "FINAL_OPTION_B_MISMATCH_MANIFEST.tsv"
    mismatch_sha = write_tsv(mismatch_path, final_mismatch)
    mismatch_by_id = {row["source_image_id"]: row for row in final_mismatch}

    generation_rule = (
        "Load the canonical 2,653 records in dataset order. Initialise random.Random(20260829). "
        "For each permutation attempt, copy indices 0..2652 and call rng.shuffle using the continuing "
        "RNG state. Accept the first permutation for which every source index differs from its target "
        "index and every source orig_image_id differs from the target orig_image_id. The first accepted "
        "permutation is attempt 3. No prediction, rationale, metric, correctness, or other outcome enters "
        "construction or acceptance."
    )

    # Verify canonical and transport checkpoint identities before recording them.
    checkpoint_audit: dict[str, Any] = {"stage1": {}, "stage2": {}}
    for stage, table in (("stage1", STAGE1), ("stage2", STAGE2)):
        for arm, (canonical_rel, canonical_sha, transport_name, transport_sha) in table.items():
            canonical_path = root / canonical_rel
            transport_path = args.transport_root / transport_name
            actual_canonical = sha_file(canonical_path)
            actual_transport = sha_file(transport_path)
            if actual_canonical != canonical_sha or actual_transport != transport_sha:
                raise SystemExit(f"{stage}/{arm} checkpoint hash mismatch")
            checkpoint_audit[stage][arm] = {
                "canonical_path": canonical_rel,
                "canonical_sha256": actual_canonical,
                "canonical_bytes": canonical_path.stat().st_size,
                "transport_path": str(transport_path),
                "transport_sha256": actual_transport,
                "transport_bytes": transport_path.stat().st_size,
            }

    # Stage-1 wrong-only manifest, exactly four arms x 2,653 rows.
    stage1_rows: list[dict[str, Any]] = []
    for arm in ARMS:
        canonical_rel, canonical_sha, transport_name, transport_sha = STAGE1[arm]
        for index, row in enumerate(records):
            mismatch = mismatch_by_id[row["image_id"]]
            target_asset = mismatch["plain_target_asset"] if arm.startswith("plain") else mismatch["point_target_asset"]
            target_sha = mismatch["plain_target_asset_sha256"] if arm.startswith("plain") else mismatch["point_target_asset_sha256"]
            cap = caption(row) if arm.endswith("_desc") else ""
            prompt = stage1_prompt(row, arm)
            stage1_rows.append({
                "primary_key": f"{arm}|{row['image_id']}",
                "arm": arm,
                "record_id": row["image_id"],
                "source_frame_id": row["orig_image_id"],
                "gold_label": int(row["answer_label"]),
                "question_choices_sha256": sha_text(canonical_json({"question": row["question"], "choices": row["choices"]})),
                "description_text_sha256": sha_text(cap),
                "mismatch_target_dataset_index": mismatch["target_dataset_index"],
                "mismatch_target_image_id": mismatch["target_image_id"],
                "mismatch_target_orig_image_id": mismatch["target_orig_image_id"],
                "target_asset": target_asset,
                "target_asset_sha256": target_sha,
                "final_mismatch_manifest_sha256": mismatch_sha,
                "final_mismatch_row_sha256": mismatch["row_sha256"],
                "canonical_checkpoint": canonical_rel,
                "canonical_checkpoint_sha256": canonical_sha,
                "transport_checkpoint": str(args.transport_root / transport_name),
                "transport_checkpoint_sha256": transport_sha,
                "prompt_sha256": sha_text(prompt),
                "expected_constructor_mode": "final_option_b_mismatch_arm_specific_asset",
                "dataset_index": index,
                "batch_size": 2,
                "required_hardware_lineage": "live RTX-3090 historical Stage-1 environment",
                "output_schema_version": "stage1-wrong-v1",
            })
    stage1_path = manifests / "STAGE1_FINAL_WRONG_MANIFEST.tsv"
    stage1_sha = write_tsv(stage1_path, stage1_rows)

    # Stage-2 primary manifest, exactly four arms x six cells x 2,653 rows.
    stage2_rows: list[dict[str, Any]] = []
    for arm in ARMS:
        canonical_rel, canonical_sha, transport_name, transport_sha = STAGE2[arm]
        for condition in CONDITIONS:
            constructor_json = canonical_json(CONSTRUCTORS[condition])
            for index, row in enumerate(records):
                plain_source, point_source = asset_paths(root, row)
                source_path = plain_source if arm.startswith("plain") else point_source
                source_rel = relative(root, source_path)
                source_sha = asset_cache[str(source_path)]
                mismatch = mismatch_by_id[row["image_id"]]
                intervention_path = ""
                intervention_sha = ""
                if condition == "source":
                    intervention_path, intervention_sha = source_rel, source_sha
                elif condition == "mismatch":
                    intervention_path = mismatch["plain_target_asset"] if arm.startswith("plain") else mismatch["point_target_asset"]
                    intervention_sha = mismatch["plain_target_asset_sha256"] if arm.startswith("plain") else mismatch["point_target_asset_sha256"]
                cap = caption(row) if arm.endswith("_desc") else ""
                prompt = stage2_prompt(row, arm, int(row["answer_label"]), "on")
                stage2_rows.append({
                    "primary_key": f"{arm}|{condition}|{row['image_id']}",
                    "arm": arm,
                    "condition": condition,
                    "record_id": row["image_id"],
                    "source_frame_id": row["orig_image_id"],
                    "dataset_index": index,
                    "supplied_gold_answer_index": int(row["answer_label"]),
                    "supplied_gold_answer_text": row["choices"][int(row["answer_label"])],
                    "question_sha256": sha_text(str(row["question"])),
                    "choices_sha256": sha_text(canonical_json(row["choices"])),
                    "caption_text": cap,
                    "caption_sha256": sha_text(cap),
                    "source_image_asset": source_rel,
                    "source_image_asset_sha256": source_sha,
                    "intervention_asset": intervention_path,
                    "intervention_asset_sha256": intervention_sha,
                    "constructor_specification": constructor_json,
                    "constructor_specification_sha256": sha_text(constructor_json),
                    "mismatch_target_dataset_index": mismatch["target_dataset_index"] if condition == "mismatch" else "",
                    "mismatch_target_row_sha256": mismatch["row_sha256"] if condition == "mismatch" else "",
                    "final_mismatch_manifest_sha256": mismatch_sha if condition == "mismatch" else "",
                    "mask_seed": index if condition == "mask" else "",
                    "noise_seed": 1_000_000 + index if condition == "noise" else "",
                    "canonical_checkpoint": canonical_rel,
                    "canonical_checkpoint_sha256": canonical_sha,
                    "transport_checkpoint": str(args.transport_root / transport_name),
                    "transport_checkpoint_sha256": transport_sha,
                    "prompt_sha256": sha_text(prompt),
                    "model_processor_revision": MODEL_REVISION,
                    "decode_configuration": canonical_json(DECODE),
                    "expected_output_schema_version": "stage2-primary-v1",
                })
    stage2_path = manifests / "STAGE2_PRIMARY_MANIFEST.tsv"
    stage2_manifest_sha = write_tsv(stage2_path, stage2_rows)

    # Known-change: freeze historical alternate values only after exact rule verification.
    known_rows: list[dict[str, Any]] = []
    known_ids_by_arm: dict[str, list[str]] = {}
    known_audit: dict[str, Any] = {}
    for arm in ARMS:
        known_path = root / KNOWN[arm]
        battery_path = root / BATTERY[arm]
        historical = [json.loads(line) for line in known_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        with battery_path.open(newline="", encoding="utf-8") as handle:
            battery = list(csv.DictReader(handle))[:200]
        if len(historical) != 200 or len(battery) != 200:
            raise SystemExit(f"known-change {arm} is not exactly 200 records")
        battery_by_id = {row["image_id"]: row for row in battery}
        arm_mismatches: list[str] = []
        known_ids_by_arm[arm] = []
        canonical_rel, canonical_sha, transport_name, transport_sha = STAGE2[arm]
        for position, old in enumerate(historical):
            record_id = str(old["image_id"])
            if record_id not in by_id or record_id not in battery_by_id:
                raise SystemExit(f"known-change ID missing from dataset/battery: {arm}/{record_id}")
            index, row = by_id[record_id]
            b = battery_by_id[record_id]
            gold = int(float(b["gold"]))
            pred = int(float(b["pred_real"])) if b.get("pred_real") not in (None, "", "nan") else gold
            recomputed = pred if pred != gold and 0 <= pred < 4 else (gold + 2) % 4
            persisted = int(old["alt"])
            if persisted != recomputed or int(old["gold"]) != gold:
                arm_mismatches.append(record_id)
            source_path = asset_paths(root, row)[0 if arm.startswith("plain") else 1]
            prompt = stage2_prompt(row, arm, persisted, "on")
            known_rows.append({
                "primary_key": f"{arm}|alternate_answer|{record_id}",
                "arm": arm,
                "condition": "alternate_answer_source_image",
                "record_id": record_id,
                "source_frame_id": row["orig_image_id"],
                "dataset_index": index,
                "control_position": position,
                "gold_answer_index": gold,
                "alternate_answer_index": persisted,
                "alternate_answer_text": row["choices"][persisted],
                "historical_rule_recomputed_index": recomputed,
                "historical_rule_match": int(persisted == recomputed),
                "historical_source_prediction": pred,
                "historical_source_prediction_artifact": BATTERY[arm],
                "historical_source_prediction_artifact_sha256": sha_file(battery_path),
                "persisted_alternate_artifact": KNOWN[arm],
                "persisted_alternate_artifact_sha256": sha_file(known_path),
                "source_image_asset": relative(root, source_path),
                "source_image_asset_sha256": asset_cache[str(source_path)],
                "prompt_sha256": sha_text(prompt),
                "canonical_checkpoint": canonical_rel,
                "canonical_checkpoint_sha256": canonical_sha,
                "transport_checkpoint": str(args.transport_root / transport_name),
                "transport_checkpoint_sha256": transport_sha,
                "model_processor_revision": MODEL_REVISION,
                "decode_configuration": canonical_json(DECODE),
                "matched_primary_source_key": f"{arm}|source|{record_id}",
                "matched_primary_grey_key": f"{arm}|grey|{record_id}",
                "expected_output_schema_version": "stage2-known-change-v1",
            })
            known_ids_by_arm[arm].append(record_id)
        if arm_mismatches:
            raise SystemExit(f"STOP: historical alternate indices fail recomputation for {arm}: {arm_mismatches[:5]}")
        known_audit[arm] = {
            "n": 200,
            "persisted_vs_recomputed_mismatches": 0,
            "ordered_id_sha256": ordered_hash(known_ids_by_arm[arm]),
            "persisted_artifact_sha256": sha_file(known_path),
            "source_prediction_artifact_sha256": sha_file(battery_path),
        }
    if len({tuple(ids) for ids in known_ids_by_arm.values()}) != 1:
        raise SystemExit("known-change arms do not use the same ordered 200 IDs")
    known_path = manifests / "STAGE2_KNOWN_CHANGE_MANIFEST.tsv"
    known_sha = write_tsv(known_path, known_rows)

    # Description-availability ablation: only source/grey with description OFF.
    ablation_rows: list[dict[str, Any]] = []
    fixed_ids = known_ids_by_arm["plain"]
    for arm in ("plain_desc", "point_desc"):
        canonical_rel, canonical_sha, transport_name, transport_sha = STAGE2[arm]
        for condition in ("source_description_off", "grey_description_off"):
            base_condition = "source" if condition.startswith("source") else "grey"
            for position, record_id in enumerate(fixed_ids):
                index, row = by_id[record_id]
                source_path = asset_paths(root, row)[0 if arm.startswith("plain") else 1]
                off_prompt = stage2_prompt(row, arm, int(row["answer_label"]), "off")
                on_prompt = stage2_prompt(row, arm, int(row["answer_label"]), "on")
                constructor_json = canonical_json(CONSTRUCTORS[base_condition])
                ablation_rows.append({
                    "primary_key": f"{arm}|{condition}|{record_id}",
                    "arm": arm,
                    "condition": condition,
                    "base_image_condition": base_condition,
                    "record_id": record_id,
                    "source_frame_id": row["orig_image_id"],
                    "dataset_index": index,
                    "diagnostic_position": position,
                    "supplied_gold_answer_index": int(row["answer_label"]),
                    "supplied_gold_answer_text": row["choices"][int(row["answer_label"])],
                    "source_image_asset": relative(root, source_path),
                    "source_image_asset_sha256": asset_cache[str(source_path)],
                    "constructor_specification": constructor_json,
                    "constructor_specification_sha256": sha_text(constructor_json),
                    "description_on_text": caption(row),
                    "description_on_text_sha256": sha_text(caption(row)),
                    "description_off_payload": "",
                    "description_off_literal_suffix": "Image caption:",
                    "description_on_prompt_sha256": sha_text(on_prompt),
                    "description_off_prompt_sha256": sha_text(off_prompt),
                    "matched_primary_on_key": f"{arm}|{base_condition}|{record_id}",
                    "canonical_checkpoint": canonical_rel,
                    "canonical_checkpoint_sha256": canonical_sha,
                    "transport_checkpoint": str(args.transport_root / transport_name),
                    "transport_checkpoint_sha256": transport_sha,
                    "model_processor_revision": MODEL_REVISION,
                    "decode_configuration": canonical_json(DECODE),
                    "expected_output_schema_version": "stage2-description-ablation-v1",
                })
    ablation_path = manifests / "STAGE2_DESCRIPTION_AVAILABILITY_ABLATION_MANIFEST.tsv"
    ablation_sha = write_tsv(ablation_path, ablation_rows)

    # Freeze full-record unrelated-rationale pairing independently of final parse outcomes.
    unrelated_rows: list[dict[str, Any]] = []
    unrelated_audit: dict[str, Any] = {}
    indices = list(range(len(records)))
    for arm_index, arm in enumerate(ARMS):
        seed = 20260829 + arm_index
        rng = random.Random(seed)
        attempts = 0
        while True:
            attempts += 1
            perm = indices.copy()
            rng.shuffle(perm)
            if all(i != j and records[i]["orig_image_id"] != records[j]["orig_image_id"] for i, j in enumerate(perm)):
                break
        for source_index, target_index in enumerate(perm):
            source, target = records[source_index], records[target_index]
            unrelated_rows.append({
                "arm": arm,
                "source_dataset_index": source_index,
                "source_record_id": source["image_id"],
                "source_frame_id": source["orig_image_id"],
                "target_dataset_index": target_index,
                "target_record_id": target["image_id"],
                "target_frame_id": target["orig_image_id"],
                "seed": seed,
                "permutation_attempt": attempts,
            })
        unrelated_audit[arm] = {"n": 2653, "seed": seed, "attempt": attempts, "fixed_points": 0, "same_frame_pairs": 0}
    unrelated_path = manifests / "FINAL_UNRELATED_RATIONALE_PAIRING_MAP.tsv"
    unrelated_sha = write_tsv(unrelated_path, unrelated_rows)

    # PRED-A provenance: hashes and denominators are frozen, no new generation.
    pred_provenance: dict[str, Any] = {
        "plan": "PRED-A SUFFICIENT",
        "methods_justification": (
            "Gold-conditioned perturbation is primary because changing the image under predicted conditioning can also change the Stage-1 answer supplied to Stage 2, confounding image and answer interventions. Predicted conditioning is therefore retained as a source-image end-to-end pipeline analysis rather than duplicated across the perturbation battery."
        ),
        "interpretation_rules": [
            "PRED-A is secondary end-to-end pipeline evidence.",
            "It is source-only.",
            "It is historical RTX-3090 output.",
            "It is not numerically merged with primary RTX-5090 RQ2.",
            "No direct gold-versus-predicted rationale content comparison is interpreted as an answer-source effect because prompt framing also changes.",
            "Do not make a generic model-level empty-rationale claim; every coverage figure remains tied to its historical generation family and eligibility rule.",
        ],
        "arms": {},
    }
    for arm, values in PRED.items():
        artifact_rel, expected_artifact_sha, stage1_rel, usable, correct_n, wrong_n, conditional, pipeline, split = values
        artifact = root / artifact_rel
        stage1_artifact = root / stage1_rel
        actual_artifact_sha = sha_file(artifact)
        if actual_artifact_sha != expected_artifact_sha:
            raise SystemExit(f"predicted artifact hash mismatch: {arm}")
        pred_rows = [json.loads(line) for line in artifact.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(pred_rows) != 2653 or ordered_hash(str(r["image_id"]) for r in pred_rows) != EXPECTED_ID_HASH:
            raise SystemExit(f"predicted artifact ID audit failed: {arm}")
        prompts = [str(row.get("prompt") or "") for row in pred_rows]
        stage2_rel, stage2_checkpoint_sha, _, _ = STAGE2[arm]
        pred_provenance["arms"][arm] = {
            "source_artifact": artifact_rel,
            "source_artifact_sha256": actual_artifact_sha,
            "ordered_id_sha256": EXPECTED_ID_HASH,
            "ordered_prompt_sha256": ordered_hash(prompts),
            "gold_trained_stage2_checkpoint": stage2_rel,
            "gold_trained_stage2_checkpoint_sha256": stage2_checkpoint_sha,
            "stage1_prediction_artifact": stage1_rel,
            "stage1_prediction_artifact_sha256": sha_file(stage1_artifact),
            "source_generation_hardware_runtime": "historical RTX-3090 generation family",
            "records": 2653,
            "hardened_usable_final_n": usable,
            "hardened_coverage": usable / 2653,
            "stage1_correct_n": correct_n,
            "stage1_wrong_n": wrong_n,
            "conditional_cmc": conditional,
            "pipeline_cmc": pipeline,
            "correct_wrong_cmc_split": split,
            "prompt_difference_from_gold": "predicted prompt reframes the task as reflection on the model's own decision and changes completion/caption instructions; it is not an answer-only substitution",
        }
    pred_path = manifests / "PREDICTED_REGIME_PROVENANCE.json"
    pred_sha = write_json(pred_path, pred_provenance)

    audit = {
        "status": "PASS",
        "canonical_root": str(root),
        "dataset": {"path": str(DATASET_REL), "sha256": DATASET_SHA, "records": 2653, "source_frames": len({r["orig_image_id"] for r in records}), "ordered_id_sha256": EXPECTED_ID_HASH},
        "asset_inventory": {"path": str(inventory_sha and manifests / "evaluation_asset_inventory.tsv"), "sha256": inventory_sha, "rows": len(inventory), "unique_plain_assets": len({r["plain_asset"] for r in inventory}), "unique_point_assets": len({r["point_asset"] for r in inventory})},
        "mismatch": {
            "path": str(mismatch_path), "sha256": mismatch_sha, "rows": len(final_mismatch),
            "unique_sources": len({r["source_image_id"] for r in final_mismatch}),
            "unique_targets": len({r["target_image_id"] for r in final_mismatch}),
            "fixed_points": 0, "same_source_frame": 0, "all_assets_exist": True,
            "outcome_blind": True, "candidate_sha256": CANDIDATE_SHA,
            "generation_script_sha256": CANDIDATE_GENERATOR_SHA,
            "exact_generation_rule": generation_rule,
        },
        "stage1": {"path": str(stage1_path), "sha256": stage1_sha, "rows": len(stage1_rows), "unique_primary_keys": len({r["primary_key"] for r in stage1_rows})},
        "stage2_primary": {"path": str(stage2_path), "sha256": stage2_manifest_sha, "rows": len(stage2_rows), "unique_primary_keys": len({r["primary_key"] for r in stage2_rows})},
        "known_change": {"path": str(known_path), "sha256": known_sha, "rows": len(known_rows), "audit": known_audit},
        "description_ablation": {"path": str(ablation_path), "sha256": ablation_sha, "rows": len(ablation_rows), "fixed_ids_sha256": ordered_hash(fixed_ids)},
        "unrelated_pairing": {"path": str(unrelated_path), "sha256": unrelated_sha, "rows": len(unrelated_rows), "arms": unrelated_audit},
        "predicted_provenance": {"path": str(pred_path), "sha256": pred_sha},
        "checkpoint_identity": checkpoint_audit,
        "planned_stage2_generations": {"primary": 63672, "known_change": 800, "description_ablation": 800, "total": 65272},
        "builder_sha256": sha_file(Path(__file__)),
    }
    audit_path = audits / "MANIFEST_BUILD_AUDIT.json"
    audit_sha = write_json(audit_path, audit)
    print(json.dumps({"status": "PASS", "audit": str(audit_path), "audit_sha256": audit_sha, "manifest_hashes": {"mismatch": mismatch_sha, "stage1": stage1_sha, "stage2": stage2_manifest_sha, "known": known_sha, "ablation": ablation_sha, "unrelated": unrelated_sha, "predicted": pred_sha}}, indent=2))


if __name__ == "__main__":
    main()
