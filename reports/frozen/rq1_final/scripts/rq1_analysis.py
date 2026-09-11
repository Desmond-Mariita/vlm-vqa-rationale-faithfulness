#!/usr/bin/env python3
"""Frozen, estimate-led RQ1 analysis over the final six-condition Stage-1 battery.

This script reads only Stage-1 artifacts.  Historical source/grey/mask/noise/
mirror predictions come from the canonical RTX-3090 sufficiency-battery CSVs;
the mismatch column comes only from the final Option-B production outputs.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path("reports/frozen/rq1_final")
V11 = Path(".")
RUN_ID = "v11-final-2653-20260830-f8349fa0"
LOCAL_RUN = Path("data/external/execution") / RUN_ID
EXTERNAL_RUN = Path("data/external/execution") / RUN_ID
HIST = ROOT / "provenance/inputs/historical_sufficiency_battery_20260625_201107"
STAGE1_MANIFEST = V11 / "frozen_execution/manifests/STAGE1_FINAL_WRONG_MANIFEST.tsv"
OPTION_B_MANIFEST = V11 / "frozen_execution/manifests/FINAL_OPTION_B_MISMATCH_MANIFEST.tsv"

ARMS = ["plain", "point", "plain_desc", "point_desc"]
CONDITIONS = ["source", "grey", "mismatch", "mask", "noise", "mirror"]
PERTURBATIONS = CONDITIONS[1:]
HIST_FIELD = {
    "source": "real",
    "grey": "grey",
    "mask": "occlude",
    "noise": "noise",
    "mirror": "hflip",
}
EXPECTED_MISMATCH_SHA = {
    "plain": "013e98dc586ab044d7e54579ebf66569b2bf9d917fb4f43e0ab992d9bf63c398",
    "point": "42ce5a06ad5a24717b591cc5d86e2855fc956baed72043a0f749e17f77b287c8",
    "plain_desc": "a3998049d2ddce805c34d1d50644f725e3c165697fb03802ac91cfd85017b83d",
    "point_desc": "60a897a5e37bd55927882f9332808db8e97bc4d682ed8d27f0d8d8983795a84d",
}
EXPECTED_HIST_SHA = {
    "plain": "bc6ce9c30f91c313ec40bb26e4c706768f78f243c36c096eeab8ac6bdc09a92a",
    "point": "e5431fa4f9ff4007f2d0f7df65aed9302f12a5da6211f61db8cc1e07acff1015",
    "plain_desc": "6ea9785a2f49943ceddf6b3cf6b3a5d38c579f8ee7a8f99a5e47d63c1b5aa32e",
    "point_desc": "78fb8cc6169b3bb97242d738246bd080fd670e5ed5705ac4f35744cd7931e0f1",
}
ANALYSIS_PLAN_SHA = "f4a34f6fa8a62479471467c56764691fba5471502775cdf03d2739569f7ae899"
OPTION_B_SHA = "dbeeb0a0eaad8f1d6d79c9558e8748cd0f6f9854156f979969f145a8f3a4b22d"
BOOTSTRAP_SEED = 42
BOOTSTRAP_REPLICATES = 10_000


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ordered_hash(values: Iterable[str]) -> str:
    return hashlib.sha256(("\n".join(map(str, values)) + "\n").encode()).hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_delimited(path: Path, rows: list[dict], delimiter: str = "\t") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter=delimiter, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def pct_ci(values: np.ndarray) -> tuple[float, float]:
    lo, hi = np.quantile(values, [0.025, 0.975], method="linear")
    return float(lo), float(hi)


def latex_escape(value: str) -> str:
    return value.replace("_", r"\_").replace("→", r"$\rightarrow$")


def make_latex_table(headers: list[str], rows: list[list[str]], align: str, caption: str, label: str) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        f"\\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    lines.extend(" & ".join(row) + r" \\" for row in rows)
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\end{table}",
        "",
    ])
    return "\n".join(lines)


def load_inputs():
    assert sha256(V11 / "frozen_execution/FINAL_ANALYSIS_PLAN_PREINFERENCE.md") == ANALYSIS_PLAN_SHA
    assert sha256(OPTION_B_MANIFEST) == OPTION_B_SHA

    manifest_rows = read_tsv(STAGE1_MANIFEST)
    by_arm_manifest = {arm: [r for r in manifest_rows if r["arm"] == arm] for arm in ARMS}
    all_data: dict[str, dict] = {}
    provenance: list[dict] = []
    input_audit: dict = {
        "analysis_plan_sha256": ANALYSIS_PLAN_SHA,
        "option_b_manifest_sha256": OPTION_B_SHA,
        "arms": {},
    }

    for arm in ARMS:
        manifest = by_arm_manifest[arm]
        hist_path = HIST / f"{arm}_examples.csv"
        mismatch_path = LOCAL_RUN / f"shards/stage1_wrong/{arm}/worker_0/records.jsonl"
        assert len(manifest) == 2653
        assert sha256(hist_path) == EXPECTED_HIST_SHA[arm]
        assert sha256(mismatch_path) == EXPECTED_MISMATCH_SHA[arm]

        hist = read_csv(hist_path)
        mismatch = read_jsonl(mismatch_path)
        ids = [r["record_id"] for r in manifest]
        frames = [r["source_frame_id"] for r in manifest]
        gold = np.asarray([int(r["gold_label"]) for r in manifest], dtype=np.int8)

        checks = {
            "manifest_rows": len(manifest),
            "historical_rows": len(hist),
            "mismatch_rows": len(mismatch),
            "unique_record_ids": len(set(ids)),
            "unique_source_frames": len(set(frames)),
            "historical_order_match": [r["image_id"] for r in hist] == ids,
            "mismatch_order_match": [r["record_id"] for r in mismatch] == ids,
            "historical_gold_match": [int(r["gold"]) for r in hist] == gold.tolist(),
            "mismatch_gold_match": [int(r["gold_label"]) for r in mismatch] == gold.tolist(),
            "mismatch_unique_ids": len({r["record_id"] for r in mismatch}),
            "mismatch_unique_primary_keys": len({r["primary_key"] for r in mismatch}),
            "mismatch_all_success": all(r["record_status"] == "success" for r in mismatch),
            "mismatch_all_condition": all(r["condition"] == "mismatch" for r in mismatch),
            "mismatch_all_arm": all(r["arm"] == arm for r in mismatch),
            "ordered_id_sha256": ordered_hash(ids),
        }
        assert checks["manifest_rows"] == checks["historical_rows"] == checks["mismatch_rows"] == 2653
        assert checks["unique_record_ids"] == checks["mismatch_unique_ids"] == checks["mismatch_unique_primary_keys"] == 2653
        assert checks["unique_source_frames"] == 2400
        assert all(v for k, v in checks.items() if k.endswith("_match") or k.startswith("mismatch_all_"))

        predictions: dict[str, np.ndarray] = {}
        correct: dict[str, np.ndarray] = {}
        for condition, old_name in HIST_FIELD.items():
            p = np.asarray([int(r[f"pred_{old_name}"]) for r in hist], dtype=np.int8)
            predictions[condition] = p
            correct[condition] = p == gold
        predictions["mismatch"] = np.asarray([int(r["prediction"]) for r in mismatch], dtype=np.int8)
        correct["mismatch"] = predictions["mismatch"] == gold

        all_data[arm] = {
            "ids": ids,
            "frames": frames,
            "gold": gold,
            "predictions": predictions,
            "correct": correct,
            "historical_wrong_prediction": np.asarray([int(r["pred_wrong"]) for r in hist], dtype=np.int8),
            "historical": hist,
            "mismatch_records": mismatch,
            "manifest": manifest,
        }
        input_audit["arms"][arm] = checks

        hist_canonical = f"./reports/runs/_meta/sufficiency_battery/20260625_201107/{arm}_examples.csv"
        for condition in ["source", "grey", "mask", "noise", "mirror"]:
            provenance.append({
                "arm": arm,
                "condition": condition,
                "artifact_path": hist_canonical,
                "SHA-256": EXPECTED_HIST_SHA[arm],
                "row_count": 2653,
                "ordered_ID_hash": checks["ordered_id_sha256"],
                "source_frame_count": 2400,
                "hardware_runtime_lineage": "NVIDIA RTX 3090; retained historical Stage-1 sufficiency-battery lineage",
                "runner_artifact_family": "scripts/sufficiency_battery.py; per-arm *_examples.csv (canonical assembled battery)",
                "status": "FINAL_RETAINED_HISTORICAL",
            })
        provenance.append({
            "arm": arm,
            "condition": "mismatch",
            "artifact_path": str(EXTERNAL_RUN / f"shards/stage1_wrong/{arm}/worker_0/records.jsonl"),
            "SHA-256": EXPECTED_MISMATCH_SHA[arm],
            "row_count": 2653,
            "ordered_ID_hash": checks["ordered_id_sha256"],
            "source_frame_count": 2400,
            "hardware_runtime_lineage": "NVIDIA RTX 3090; final v11 Option-B wrong-only production lineage; batch size 2",
            "runner_artifact_family": "frozen production_stage1_wrong_runner.py; output schema stage1_wrong.v1",
            "status": "FINAL_OPTION_B",
        })

    reference = all_data[ARMS[0]]
    cross_arm = {}
    for arm in ARMS:
        d = all_data[arm]
        cross_arm[arm] = {
            "ordered_IDs_match_plain": d["ids"] == reference["ids"],
            "gold_labels_match_plain": np.array_equal(d["gold"], reference["gold"]),
            "ordered_source_frames_match_plain": d["frames"] == reference["frames"],
        }
        assert all(cross_arm[arm].values())
    input_audit["cross_arm"] = cross_arm
    input_audit["overall_pass"] = True
    return all_data, provenance, input_audit


def bootstrap(all_data: dict[str, dict]):
    frame_ids = list(OrderedDict.fromkeys(all_data["plain"]["frames"]))
    frame_to_i = {frame: i for i, frame in enumerate(frame_ids)}
    row_frame = np.asarray([frame_to_i[x] for x in all_data["plain"]["frames"]], dtype=np.int16)
    frame_n = len(frame_ids)
    record_count_by_frame = np.bincount(row_frame, minlength=frame_n).astype(np.float64)

    linear_specs: list[dict] = []
    numerator_vectors: list[np.ndarray] = []
    point_values: list[float] = []
    n = 2653
    for arm in ARMS:
        for condition in CONDITIONS:
            values = all_data[arm]["correct"][condition].astype(np.float64)
            numerator_vectors.append(np.bincount(row_frame, weights=values, minlength=frame_n))
            point_values.append(float(values.mean()))
            linear_specs.append({"estimate": "accuracy", "arms": arm, "conditions": condition, "record_N": n, "frame_N": frame_n})
        for condition in PERTURBATIONS:
            values = all_data[arm]["correct"]["source"].astype(np.float64) - all_data[arm]["correct"][condition].astype(np.float64)
            numerator_vectors.append(np.bincount(row_frame, weights=values, minlength=frame_n))
            point_values.append(float(values.mean()))
            linear_specs.append({"estimate": "source_relative_accuracy_gap", "arms": arm, "conditions": f"source-minus-{condition}", "record_N": n, "frame_N": frame_n})
        values = all_data[arm]["correct"]["mirror"].astype(np.float64) - all_data[arm]["correct"]["mismatch"].astype(np.float64)
        numerator_vectors.append(np.bincount(row_frame, weights=values, minlength=frame_n))
        point_values.append(float(values.mean()))
        linear_specs.append({"estimate": "descriptive_accuracy_difference", "arms": arm, "conditions": "mirror-minus-mismatch", "record_N": n, "frame_N": frame_n})

    sg_pairs = [("plain_desc", "plain"), ("point_desc", "point")]
    for desc_arm, base_arm in sg_pairs:
        values = (
            all_data[desc_arm]["correct"]["source"].astype(float)
            - all_data[desc_arm]["correct"]["grey"].astype(float)
            - all_data[base_arm]["correct"]["source"].astype(float)
            + all_data[base_arm]["correct"]["grey"].astype(float)
        )
        numerator_vectors.append(np.bincount(row_frame, weights=values, minlength=frame_n))
        point_values.append(float(values.mean()))
        linear_specs.append({
            "estimate": "planned_sufficiency_gap_contrast",
            "arms": f"{desc_arm}-minus-{base_arm}",
            "conditions": "(source-minus-grey) contrast",
            "record_N": n,
            "frame_N": frame_n,
        })

    numerator_matrix = np.stack(numerator_vectors).astype(np.float64)

    def c2w_components(arm: str):
        src = all_data[arm]["correct"]["source"]
        grey = all_data[arm]["correct"]["grey"]
        cw = (src & ~grey).astype(float)
        den = src.astype(float)
        return (
            np.bincount(row_frame, weights=cw, minlength=frame_n),
            np.bincount(row_frame, weights=den, minlength=frame_n),
            float(cw.sum() / den.sum()),
        )

    c2w_specs = []
    for desc_arm, base_arm in sg_pairs:
        dn, dd, dp = c2w_components(desc_arm)
        bn, bd, bp = c2w_components(base_arm)
        c2w_specs.append({
            "estimate": "planned_grey_C_to_W_rate_contrast",
            "arms": f"{desc_arm}-minus-{base_arm}",
            "conditions": "grey; source-correct denominator",
            "record_N": n,
            "frame_N": frame_n,
            "desc_num": dn,
            "desc_den": dd,
            "base_num": bn,
            "base_den": bd,
            "point": dp - bp,
        })

    def one_run():
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        reps = np.empty((len(linear_specs), BOOTSTRAP_REPLICATES), dtype=np.float64)
        rate_reps = np.empty((len(c2w_specs), BOOTSTRAP_REPLICATES), dtype=np.float64)
        offset = 0
        chunk_size = 500
        while offset < BOOTSTRAP_REPLICATES:
            b = min(chunk_size, BOOTSTRAP_REPLICATES - offset)
            draws = rng.integers(0, frame_n, size=(b, frame_n), dtype=np.int16)
            counts = np.stack([np.bincount(row, minlength=frame_n) for row in draws]).astype(np.float64)
            denom = counts @ record_count_by_frame
            reps[:, offset:offset+b] = ((counts @ numerator_matrix.T) / denom[:, None]).T
            for j, spec in enumerate(c2w_specs):
                rate_reps[j, offset:offset+b] = (
                    (counts @ spec["desc_num"]) / (counts @ spec["desc_den"])
                    - (counts @ spec["base_num"]) / (counts @ spec["base_den"])
                )
            offset += b
        return reps, rate_reps

    first, first_rates = one_run()
    second, second_rates = one_run()
    reproducible = np.array_equal(first, second) and np.array_equal(first_rates, second_rates)
    assert reproducible

    rows = []
    for i, spec in enumerate(linear_specs):
        lo, hi = pct_ci(first[i])
        rows.append({
            **spec,
            "point_estimate": f"{point_values[i]:.12f}",
            "lower_95%": f"{lo:.12f}",
            "upper_95%": f"{hi:.12f}",
            "seed": BOOTSTRAP_SEED,
            "replicates": BOOTSTRAP_REPLICATES,
        })
    for i, spec in enumerate(c2w_specs):
        lo, hi = pct_ci(first_rates[i])
        rows.append({
            "estimate": spec["estimate"],
            "arms": spec["arms"],
            "conditions": spec["conditions"],
            "record_N": spec["record_N"],
            "frame_N": spec["frame_N"],
            "point_estimate": f"{spec['point']:.12f}",
            "lower_95%": f"{lo:.12f}",
            "upper_95%": f"{hi:.12f}",
            "seed": BOOTSTRAP_SEED,
            "replicates": BOOTSTRAP_REPLICATES,
        })
    audit = {
        "procedure": "source-frame cluster percentile bootstrap",
        "resampling_unit": "source_frame_id",
        "record_N": n,
        "frame_N": frame_n,
        "seed": BOOTSTRAP_SEED,
        "replicates": BOOTSTRAP_REPLICATES,
        "interval": "percentile 95%",
        "shared_frame_multiplicity": True,
        "exact_second_run_array_equality": reproducible,
        "pass": reproducible,
    }
    return rows, audit


def lookup_bootstrap(rows: list[dict], estimate: str, arms: str, conditions: str) -> dict:
    matches = [r for r in rows if r["estimate"] == estimate and r["arms"] == arms and r["conditions"] == conditions]
    assert len(matches) == 1, (estimate, arms, conditions, len(matches))
    return matches[0]


def make_results(all_data: dict[str, dict], bootstrap_rows: list[dict]):
    cell_rows = []
    gap_rows = []
    transition_rows = []
    identity_cells = []
    mismatch_mirror_rows = []
    for arm in ARMS:
        n = len(all_data[arm]["ids"])
        source_correct = all_data[arm]["correct"]["source"]
        for condition in CONDITIONS:
            correct = all_data[arm]["correct"][condition]
            b = lookup_bootstrap(bootstrap_rows, "accuracy", arm, condition)
            cell_rows.append({
                "arm": arm,
                "condition": condition,
                "N": n,
                "source_frames": 2400,
                "correct_count": int(correct.sum()),
                "wrong_count": int(n - correct.sum()),
                "accuracy": f"{correct.mean():.12f}",
                "lower_95%": b["lower_95%"],
                "upper_95%": b["upper_95%"],
            })
        source_acc = float(source_correct.mean())
        for condition in PERTURBATIONS:
            cond_correct = all_data[arm]["correct"][condition]
            cond_acc = float(cond_correct.mean())
            gap = source_acc - cond_acc
            b = lookup_bootstrap(bootstrap_rows, "source_relative_accuracy_gap", arm, f"source-minus-{condition}")
            gap_rows.append({
                "arm": arm,
                "condition": condition,
                "N": n,
                "source_frames": 2400,
                "source_accuracy": f"{source_acc:.12f}",
                "condition_accuracy": f"{cond_acc:.12f}",
                "source_minus_condition_gap": f"{gap:.12f}",
                "lower_95%": b["lower_95%"],
                "upper_95%": b["upper_95%"],
            })
            cc = int((source_correct & cond_correct).sum())
            cw = int((source_correct & ~cond_correct).sum())
            wc = int((~source_correct & cond_correct).sum())
            ww = int((~source_correct & ~cond_correct).sum())
            source_correct_n = int(source_correct.sum())
            source_wrong_n = n - source_correct_n
            identity_lhs_count = int(source_correct.sum() - cond_correct.sum())
            identity_rhs_count = cw - wc
            passed = identity_lhs_count == identity_rhs_count and cc + cw + wc + ww == n
            identity_cells.append({
                "arm": arm,
                "condition": condition,
                "N": n,
                "source_correct_minus_condition_correct": identity_lhs_count,
                "C_to_W_minus_W_to_C": identity_rhs_count,
                "transition_sum": cc + cw + wc + ww,
                "pass": passed,
            })
            transition_rows.append({
                "arm": arm,
                "condition": condition,
                "N": n,
                "source_correct": source_correct_n,
                "source_wrong": source_wrong_n,
                "C_to_C": cc,
                "C_to_W": cw,
                "W_to_C": wc,
                "W_to_W": ww,
                "C_to_W/source_correct": f"{cw/source_correct_n:.12f}",
                "C_to_W/N": f"{cw/n:.12f}",
                "W_to_C/source_wrong": f"{wc/source_wrong_n:.12f}",
                "W_to_C/N": f"{wc/n:.12f}",
                "churn_count": cw + wc,
                "churn/N": f"{(cw+wc)/n:.12f}",
                "net_movement_W_to_C_minus_C_to_W": wc - cw,
                "net_movement/N": f"{(wc-cw)/n:.12f}",
                "source_accuracy": f"{source_acc:.12f}",
                "condition_accuracy": f"{cond_acc:.12f}",
                "source_minus_condition_gap": f"{gap:.12f}",
                "identity_pass": passed,
            })
        b = lookup_bootstrap(bootstrap_rows, "descriptive_accuracy_difference", arm, "mirror-minus-mismatch")
        mismatch_mirror_rows.append({
            "arm": arm,
            "mirror_accuracy": f"{all_data[arm]['correct']['mirror'].mean():.12f}",
            "mismatch_accuracy": f"{all_data[arm]['correct']['mismatch'].mean():.12f}",
            "mirror_minus_mismatch": b["point_estimate"],
            "lower_95%": b["lower_95%"],
            "upper_95%": b["upper_95%"],
            "comparison_status": "DESCRIPTIVE_FROZEN",
        })
    identity_audit = {
        "identity": "accuracy_source - accuracy_condition = (C_to_W - W_to_C) / N",
        "expected_cells": 20,
        "passing_cells": sum(int(x["pass"]) for x in identity_cells),
        "all_pass": all(x["pass"] for x in identity_cells),
        "cells": identity_cells,
    }
    assert identity_audit["all_pass"] and identity_audit["passing_cells"] == 20
    return cell_rows, gap_rows, transition_rows, identity_audit, mismatch_mirror_rows


def planned_contrasts(all_data: dict[str, dict], bootstrap_rows: list[dict]) -> list[dict]:
    rows = []
    for desc_arm, base_arm in [("plain_desc", "plain"), ("point_desc", "point")]:
        desc_source = float(all_data[desc_arm]["correct"]["source"].mean())
        desc_grey = float(all_data[desc_arm]["correct"]["grey"].mean())
        base_source = float(all_data[base_arm]["correct"]["source"].mean())
        base_grey = float(all_data[base_arm]["correct"]["grey"].mean())
        b = lookup_bootstrap(bootstrap_rows, "planned_sufficiency_gap_contrast", f"{desc_arm}-minus-{base_arm}", "(source-minus-grey) contrast")
        rows.append({
            "contrast_family": "description_vs_no_description_sufficiency_gap",
            "description_arm": desc_arm,
            "no_description_arm": base_arm,
            "description_source_accuracy": f"{desc_source:.12f}",
            "description_grey_accuracy": f"{desc_grey:.12f}",
            "description_sufficiency_gap": f"{desc_source-desc_grey:.12f}",
            "no_description_source_accuracy": f"{base_source:.12f}",
            "no_description_grey_accuracy": f"{base_grey:.12f}",
            "no_description_sufficiency_gap": f"{base_source-base_grey:.12f}",
            "contrast_description_minus_no_description": b["point_estimate"],
            "lower_95%": b["lower_95%"],
            "upper_95%": b["upper_95%"],
            "denominator": "all N=2653 paired records; source-frame cluster bootstrap",
        })
        desc_src = all_data[desc_arm]["correct"]["source"]
        base_src = all_data[base_arm]["correct"]["source"]
        desc_rate = float((desc_src & ~all_data[desc_arm]["correct"]["grey"]).sum() / desc_src.sum())
        base_rate = float((base_src & ~all_data[base_arm]["correct"]["grey"]).sum() / base_src.sum())
        b = lookup_bootstrap(bootstrap_rows, "planned_grey_C_to_W_rate_contrast", f"{desc_arm}-minus-{base_arm}", "grey; source-correct denominator")
        rows.append({
            "contrast_family": "description_vs_no_description_grey_C_to_W_rate",
            "description_arm": desc_arm,
            "no_description_arm": base_arm,
            "description_source_accuracy": "",
            "description_grey_accuracy": f"{desc_rate:.12f}",
            "description_sufficiency_gap": "",
            "no_description_source_accuracy": "",
            "no_description_grey_accuracy": f"{base_rate:.12f}",
            "no_description_sufficiency_gap": "",
            "contrast_description_minus_no_description": b["point_estimate"],
            "lower_95%": b["lower_95%"],
            "upper_95%": b["upper_95%"],
            "denominator": "source-correct within arm; source-frame cluster bootstrap",
        })
    return rows


def mismatch_supersession(all_data: dict[str, dict]) -> list[dict]:
    rows = []
    for arm in ARMS:
        gold = all_data[arm]["gold"]
        historical_acc = float((all_data[arm]["historical_wrong_prediction"] == gold).mean())
        final_acc = float(all_data[arm]["correct"]["mismatch"].mean())
        rows.append({
            "arm": arm,
            "historical_mismatch_N": 2653,
            "historical_mismatch_accuracy": f"{historical_acc:.12f}",
            "historical_selector_map_description": "batch_size=2, shuffle=False; first image from previous batch with differing image_id, otherwise 256x256 RGB(128,128,128)",
            "historical_anomaly_status": "2 grey fallbacks; 9 same-orig_image_id targets; 2,642 source-frame-distinct targets",
            "final_Option_B_N": 2653,
            "final_Option_B_accuracy": f"{final_acc:.12f}",
            "absolute_change_final_minus_historical": f"{final_acc-historical_acc:.12f}",
            "final_mismatch_manifest_SHA": OPTION_B_SHA,
            "status": "SUPERSEDED -> FINAL",
        })
    return rows


def point_desc_anomaly(all_data: dict[str, dict]) -> dict:
    d = all_data["point_desc"]
    mask_pred = d["predictions"]["mask"]
    noise_pred = d["predictions"]["noise"]
    mask_correct = d["correct"]["mask"]
    noise_correct = d["correct"]["noise"]
    diff = mask_pred != noise_pred
    return {
        "N": 2653,
        "elementwise_prediction_identical": bool(not diff.any()),
        "prediction_difference_count": int(diff.sum()),
        "prediction_difference_percentage": float(100 * diff.mean()),
        "mask_correct_count": int(mask_correct.sum()),
        "noise_correct_count": int(noise_correct.sum()),
        "correct_record_overlap": int((mask_correct & noise_correct).sum()),
        "mask_only_correct": int((mask_correct & ~noise_correct).sum()),
        "noise_only_correct": int((~mask_correct & noise_correct).sum()),
        "equal_accuracy": bool(mask_correct.sum() == noise_correct.sum()),
        "interpretation": "equal aggregate accuracy is coincidental" if diff.any() else "prediction vectors are identical; requires provenance investigation",
    }


def record_level_rows(all_data: dict[str, dict]) -> list[dict]:
    rows = []
    for arm in ARMS:
        d = all_data[arm]
        for i, record_id in enumerate(d["ids"]):
            row = {
                "arm": arm,
                "record_id": record_id,
                "source_frame_id": d["frames"][i],
                "gold_label": int(d["gold"][i]),
            }
            for condition in CONDITIONS:
                row[f"pred_{condition}"] = int(d["predictions"][condition][i])
                row[f"correct_{condition}"] = int(d["correct"][condition][i])
            rows.append(row)
    return rows


def condition_ordering(cell_rows: list[dict]) -> list[dict]:
    rows = []
    for arm in ARMS:
        cells = [r for r in cell_rows if r["arm"] == arm]
        ordered = sorted(cells, key=lambda r: (-float(r["accuracy"]), CONDITIONS.index(r["condition"])))
        for rank, row in enumerate(ordered, start=1):
            rows.append({"arm": arm, "rank": rank, "condition": row["condition"], "accuracy": row["accuracy"], "correct_count": row["correct_count"]})
    return rows


def build_tables(cell_rows, gap_rows, transition_rows, contrast_rows, supersession_rows):
    tables = ROOT / "tables"
    cell_lookup = {(r["arm"], r["condition"]): r for r in cell_rows}

    table_a = []
    for arm in ARMS:
        row = {"arm": arm}
        for condition in CONDITIONS:
            c = cell_lookup[(arm, condition)]
            row[f"{condition}_accuracy"] = c["accuracy"]
            row[f"{condition}_N"] = c["N"]
            row[f"{condition}_lower_95%"] = c["lower_95%"]
            row[f"{condition}_upper_95%"] = c["upper_95%"]
        table_a.append(row)
    write_delimited(tables / "RQ1_TABLE_A_FINAL_ACCURACY_BATTERY.tsv", table_a)
    write_delimited(tables / "RQ1_TABLE_A_FINAL_ACCURACY_BATTERY.csv", table_a, ",")
    latex_rows = []
    for arm in ARMS:
        vals = [latex_escape(arm)]
        for condition in CONDITIONS:
            c = cell_lookup[(arm, condition)]
            vals.append(f"{float(c['accuracy']):.3f} [{float(c['lower_95%']):.3f}, {float(c['upper_95%']):.3f}]")
        latex_rows.append(vals)
    (tables / "RQ1_TABLE_A_FINAL_ACCURACY_BATTERY.tex").write_text(make_latex_table(
        ["Arm", "Source", "Grey", "Mismatch", "Mask", "Noise", "Mirror"], latex_rows, "lrrrrrr",
        "Final Stage-1 accuracy battery. Cells show accuracy and source-frame-cluster bootstrap 95\\% confidence intervals; $N=2{,}653$ in every cell.",
        "tab:rq1-final-accuracy"), encoding="utf-8")

    write_delimited(tables / "RQ1_TABLE_B_SOURCE_RELATIVE_GAPS.tsv", gap_rows)
    write_delimited(tables / "RQ1_TABLE_B_SOURCE_RELATIVE_GAPS.csv", gap_rows, ",")
    latex_rows = [[latex_escape(r["arm"]), latex_escape(r["condition"]), f"{float(r['source_accuracy']):.3f}", f"{float(r['condition_accuracy']):.3f}", f"{float(r['source_minus_condition_gap']):.3f}", f"[{float(r['lower_95%']):.3f}, {float(r['upper_95%']):.3f}]"] for r in gap_rows]
    (tables / "RQ1_TABLE_B_SOURCE_RELATIVE_GAPS.tex").write_text(make_latex_table(
        ["Arm", "Condition", "Source", "Condition", "Gap", "95\\% CI"], latex_rows, "llrrrr",
        "Source-relative Stage-1 accuracy gaps with source-frame-cluster bootstrap confidence intervals.",
        "tab:rq1-source-gaps"), encoding="utf-8")

    write_delimited(tables / "RQ1_TABLE_C_TRANSITION_DECOMPOSITION.tsv", transition_rows)
    write_delimited(tables / "RQ1_TABLE_C_TRANSITION_DECOMPOSITION.csv", transition_rows, ",")
    latex_rows = [[
        latex_escape(r["arm"]), latex_escape(r["condition"]), str(r["C_to_W"]), f"{float(r['C_to_W/source_correct']):.3f}",
        str(r["W_to_C"]), f"{float(r['W_to_C/source_wrong']):.3f}", f"{float(r['W_to_C/N']):.3f}",
        str(r["churn_count"]), str(r["net_movement_W_to_C_minus_C_to_W"]),
    ] for r in transition_rows]
    (tables / "RQ1_TABLE_C_TRANSITION_DECOMPOSITION.tex").write_text(make_latex_table(
        ["Arm", "Condition", "C$\\rightarrow$W", "/ source C", "W$\\rightarrow$C", "/ source W", "/ $N$", "Churn", "Net"],
        latex_rows, "llrrrrrrr", "Source-relative correctness transitions. Net is W$\\rightarrow$C minus C$\\rightarrow$W.",
        "tab:rq1-transitions"), encoding="utf-8")

    sg_rows = [r for r in contrast_rows if r["contrast_family"] == "description_vs_no_description_sufficiency_gap"]
    write_delimited(tables / "RQ1_TABLE_D_SUFFICIENCY_GAP_CONTRASTS.tsv", sg_rows)
    write_delimited(tables / "RQ1_TABLE_D_SUFFICIENCY_GAP_CONTRASTS.csv", sg_rows, ",")
    latex_rows = [[
        latex_escape(r["description_arm"]), latex_escape(r["no_description_arm"]),
        f"{float(r['description_source_accuracy']):.3f}", f"{float(r['description_grey_accuracy']):.3f}", f"{float(r['description_sufficiency_gap']):.3f}",
        f"{float(r['no_description_source_accuracy']):.3f}", f"{float(r['no_description_grey_accuracy']):.3f}", f"{float(r['no_description_sufficiency_gap']):.3f}",
        f"{float(r['contrast_description_minus_no_description']):.3f}", f"[{float(r['lower_95%']):.3f}, {float(r['upper_95%']):.3f}]",
    ] for r in sg_rows]
    (tables / "RQ1_TABLE_D_SUFFICIENCY_GAP_CONTRASTS.tex").write_text(make_latex_table(
        ["Desc arm", "Comparator", "Src$_d$", "Grey$_d$", "SG$_d$", "Src$_0$", "Grey$_0$", "SG$_0$", "$\\Delta$SG", "95\\% CI"],
        latex_rows, "llrrrrrrrr", "Planned description-versus-no-description Sufficiency Gap contrasts.",
        "tab:rq1-sg-contrasts"), encoding="utf-8")

    write_delimited(tables / "RQ1_TABLE_E_MISMATCH_SUPERSESSION.tsv", supersession_rows)
    write_delimited(tables / "RQ1_TABLE_E_MISMATCH_SUPERSESSION.csv", supersession_rows, ",")
    latex_rows = [[latex_escape(r["arm"]), f"{float(r['historical_mismatch_accuracy']):.3f}", f"{float(r['final_Option_B_accuracy']):.3f}", f"{float(r['absolute_change_final_minus_historical']):+.3f}", latex_escape(r["status"])] for r in supersession_rows]
    (tables / "RQ1_TABLE_E_MISMATCH_SUPERSESSION.tex").write_text(make_latex_table(
        ["Arm", "Historical", "Final Option-B", "Change", "Status"], latex_rows, "lrrrl",
        "Historical mismatch accuracy is retained only as superseded provenance; final RQ1 uses Option-B.",
        "tab:rq1-mismatch-supersession"), encoding="utf-8")


def main() -> int:
    for subdir in ["data", "audits", "tables", "provenance"]:
        (ROOT / subdir).mkdir(parents=True, exist_ok=True)
    all_data, provenance, alignment = load_inputs()
    write_delimited(ROOT / "provenance/RQ1_INPUT_PROVENANCE.tsv", provenance)
    write_json(ROOT / "audits/RQ1_ALIGNMENT_AUDIT.json", alignment)

    bootstrap_rows, bootstrap_audit = bootstrap(all_data)
    write_delimited(ROOT / "data/RQ1_BOOTSTRAP_RESULTS.tsv", bootstrap_rows)
    write_json(ROOT / "audits/RQ1_BOOTSTRAP_REPRODUCIBILITY_AUDIT.json", bootstrap_audit)

    cell_rows, gap_rows, transition_rows, identity_audit, mismatch_mirror = make_results(all_data, bootstrap_rows)
    contrast_rows = planned_contrasts(all_data, bootstrap_rows)
    supersession_rows = mismatch_supersession(all_data)
    ordering_rows = condition_ordering(cell_rows)
    record_rows = record_level_rows(all_data)

    write_delimited(ROOT / "data/RQ1_CELL_RESULTS.tsv", cell_rows)
    write_delimited(ROOT / "data/RQ1_GAP_RESULTS.tsv", gap_rows)
    write_delimited(ROOT / "data/RQ1_TRANSITION_RESULTS.tsv", transition_rows)
    write_delimited(ROOT / "data/RQ1_PLANNED_CONTRASTS.tsv", contrast_rows)
    write_delimited(ROOT / "data/RQ1_MISMATCH_VS_MIRROR.tsv", mismatch_mirror)
    write_delimited(ROOT / "data/RQ1_MISMATCH_SUPERSESSION_COMPARISON.tsv", supersession_rows)
    write_delimited(ROOT / "data/RQ1_CONDITION_ORDERING.tsv", ordering_rows)
    write_delimited(ROOT / "data/RQ1_FINAL_RECORD_RESULTS.tsv", record_rows)
    write_json(ROOT / "audits/RQ1_TRANSITION_IDENTITY_AUDIT.json", identity_audit)
    write_json(ROOT / "audits/RQ1_POINT_DESC_MASK_NOISE_ANOMALY.json", point_desc_anomaly(all_data))
    build_tables(cell_rows, gap_rows, transition_rows, contrast_rows, supersession_rows)

    print(json.dumps({
        "alignment_pass": alignment["overall_pass"],
        "bootstrap_reproducible": bootstrap_audit["pass"],
        "transition_identity_pass": identity_audit["all_pass"],
        "cell_results": cell_rows,
        "planned_contrasts": contrast_rows,
        "mismatch_supersession": supersession_rows,
        "point_desc_anomaly": point_desc_anomaly(all_data),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
