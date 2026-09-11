#!/usr/bin/env python3
"""Select and build the three illustrative records used by the main figures.

The thesis previously showed one record in three places. Three distinct records
are chosen here by fixed rules, none of which consults a model prediction to
pick the record for the methods or condition figures:

  Record M  the combined methods figure: source image, polygon image, question,
            choices, Stage 1 prediction, supplied gold answer, Stage 2 rationale
  Record C  the six image conditions
  Record R  the paired rationale comparison

Condition images are rebuilt from the frozen constructors. No inference is run
and no metric is recomputed; Record R's selection reads already-frozen Drift
values purely to pick a representative illustration.
"""

from __future__ import annotations

import csv
import glob
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = Path(".")
AUDIT_DIR = ROOT / "provenance" / "selected_final_audits"
FIGDIR = ROOT / "thesis/Figures/records"
OUT_METHODS = ROOT / "thesis/Figures/final/record_methods_example.tex"
OUT_CONDITIONS = ROOT / "thesis/Figures/final/record_six_conditions.tex"
OUT_PAIRED = ROOT / "thesis/Figures/final/rq2_paired_rationale_example.tex"
PROVENANCE = ROOT / "provenance/VISUAL_RECORD_PROVENANCE.json"

DATASET = CANONICAL / "data/final/vcr_val_10_pct.json"
EXPECTED_DATASET_SHA = "0288740b785021e667c12eb08ab2c222a65f80b2702d598a7ebb7b942b3a5fdc"
MISMATCH_MAP = ROOT / "frozen_execution/manifests/FINAL_OPTION_B_MISMATCH_MANIFEST.tsv"
STAGE1_PLAIN = CANONICAL / "reports/runs/plain/stage1/scratch_v1/outputs/stage1_preds_val.json"
SHARDS = Path(
    "data/external/execution/v11-final-2653-20260830-f8349fa0"
    "/incoming/shards/stage2/stage2_primary"
)
DRIFT = Path(
    "reports/frozen/rq2_final/data/RQ2_PRIMARY_DRIFT_PER_RECORD.tsv"
)
DRIFT_MEANS = Path(
    "reports/frozen/rq2_final/tables/RQ2_PRIMARY_DRIFT_RESULTS.tsv"
)
W2C_MATERIAL = Path(
    "reports/frozen/rq1_final/qualitative/RQ1_W2C_SELECTED_MATERIAL.tsv"
)

RECORD_M_INDEX = 2638          # already used for the methods illustration
CANVAS = (640, 360)
BORDER_RGB = (154, 160, 166)
LETTERS = "ABCD"
CONDITIONS = ("source", "grey", "mismatch", "mask", "noise", "mirror")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def latex(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
        "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(c, c) for c in " ".join(value.split()))


def panel(image: Image.Image) -> Image.Image:
    canvas = Image.new("RGB", CANVAS, "white")
    copy = image.convert("RGB")
    copy.thumbnail((CANVAS[0] - 4, CANVAS[1] - 4), Image.Resampling.LANCZOS)
    canvas.paste(copy, ((CANVAS[0] - copy.width) // 2, (CANVAS[1] - copy.height) // 2))
    return ImageOps.expand(canvas, border=1, fill=BORDER_RGB)


def build_conditions(index: int, record: dict, mismatch: dict, out: Path) -> dict:
    """Rebuild the six condition images for one record from the frozen rules."""
    out.mkdir(parents=True, exist_ok=True)
    plain_path = CANONICAL / "data/processed/vcr/val/images" / record["orig_image_id"]
    point_path = CANONICAL / "data/final/val/images" / record["image_id"]
    target_plain = CANONICAL / mismatch["plain_target_asset"]
    source = Image.open(plain_path).convert("RGB")
    array = np.array(source)

    images = {"source": source, "polygon": Image.open(point_path).convert("RGB")}
    images["grey"] = Image.new("RGB", (256, 256), (128, 128, 128))
    images["mismatch"] = Image.open(target_plain).convert("RGB")

    mask_rng = np.random.default_rng(index)
    masked = array.copy()
    masked[mask_rng.random(array.shape[:2]) < 0.5] = 0
    images["mask"] = Image.fromarray(masked)

    noise_rng = np.random.default_rng(1_000_000 + index)
    images["noise"] = Image.fromarray(
        np.clip(array.astype(np.int16) + noise_rng.normal(0, 80, array.shape).astype(np.int16), 0, 255).astype(np.uint8)
    )
    images["mirror"] = source.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    written = {}
    for name, image in images.items():
        path = out / f"{name}.png"
        panel(image).save(path, optimize=True)
        written[name] = {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
    written["_source_asset"] = {"path": str(plain_path), "sha256": sha256(plain_path)}
    written["_point_asset"] = {"path": str(point_path), "sha256": sha256(point_path)}
    written["_mismatch_target_asset"] = {"path": str(target_plain), "sha256": sha256(target_plain)}
    return written


def stage2_output(arm: str, condition: str, record_id: str) -> dict:
    for path in sorted(glob.glob(os.path.join(str(SHARDS), f"{arm}__{condition}", "worker_*", "records.jsonl"))):
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if record_id in line:
                row = json.loads(line)
                if row["record_id"] == record_id:
                    return row
    raise RuntimeError(f"no frozen output for {arm}/{condition}/{record_id}")


def main() -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    if sha256(DATASET) != EXPECTED_DATASET_SHA:
        raise SystemExit("evaluation population hash mismatch")
    records = json.loads(DATASET.read_text(encoding="utf-8"))
    mismatch = {int(r["source_dataset_index"]): r
                for r in csv.DictReader(MISMATCH_MAP.open(encoding="utf-8"), delimiter="\t")}
    w2c_ids = {r["record_id"] for r in csv.DictReader(W2C_MATERIAL.open(encoding="utf-8"), delimiter="\t")}
    stage1 = {r["image_id"]: r for r in
              (json.loads(l) for l in STAGE1_PLAIN.read_text(encoding="utf-8").splitlines() if l.strip())}

    # ---------------------------------------------------------------- Record M
    m_index = RECORD_M_INDEX
    m = records[m_index]

    # ---------------------------------------------------------------- Record C
    # First record in canonical order, excluding Record M and the W-to-C sample,
    # whose assets all exist. Model outputs play no part in this choice.
    c_index = None
    for index, record in enumerate(records):
        if index == m_index or record["image_id"] in w2c_ids:
            continue
        plain = CANONICAL / "data/processed/vcr/val/images" / record["orig_image_id"]
        point = CANONICAL / "data/final/val/images" / record["image_id"]
        target = CANONICAL / mismatch[index]["plain_target_asset"]
        if plain.is_file() and point.is_file() and target.is_file():
            c_index = index
            break
    if c_index is None:
        raise SystemExit("no eligible record for the six-condition figure")
    c = records[c_index]

    # ---------------------------------------------------------------- Record R
    # Most representative plain-arm record: minimises the summed distance of its
    # grey and mismatch Drift from the arm means, among records with all three
    # usable spans, excluding Records M and C and the W-to-C sample.
    means = {r["condition"]: float(r["mean_raw_drift"])
             for r in csv.DictReader(DRIFT_MEANS.open(encoding="utf-8"), delimiter="\t")
             if r["arm"] == "plain"}
    per_record: dict[str, dict[str, float]] = {}
    for row in csv.DictReader(DRIFT.open(encoding="utf-8"), delimiter="\t"):
        if row["arm"] != "plain" or row["condition"] not in ("grey", "mismatch"):
            continue
        per_record.setdefault(row["record_id"], {})[row["condition"]] = float(row["raw_drift"])
    index_of = {r["image_id"]: i for i, r in enumerate(records)}
    excluded = {m["image_id"], c["image_id"]} | w2c_ids
    candidates = [
        (abs(v["grey"] - means["grey"]) + abs(v["mismatch"] - means["mismatch"]), index_of[rid], rid)
        for rid, v in per_record.items()
        if rid not in excluded and {"grey", "mismatch"} <= set(v) and rid in index_of
    ]
    if not candidates:
        raise SystemExit("no eligible record for the paired rationale figure")
    _distance, r_index, r_id = min(candidates)
    r = records[r_index]

    assets = {
        "M": build_conditions(m_index, m, mismatch[m_index], FIGDIR / "M"),
        "C": build_conditions(c_index, c, mismatch[c_index], FIGDIR / "C"),
        "R": build_conditions(r_index, r, mismatch[r_index], FIGDIR / "R"),
    }

    # ------------------------------------------------- combined methods figure
    m_stage1 = stage1[m["image_id"]]
    m_stage2 = stage2_output("plain", "source", m["image_id"])
    choices = "\n".join(
        rf"\textbf{{{LETTERS[i]}.}} {latex(m['choices'][i])}\\" for i in range(4)
    )
    methods = f"""\\begin{{figure}}[ht]
\\centering
\\begin{{tabular}}{{cc}}
\\includegraphics[width=0.44\\textwidth]{{Figures/records/M/source.png}} &
\\includegraphics[width=0.44\\textwidth]{{Figures/records/M/polygon.png}} \\\\
\\small Source image, used by the plain arms &
\\small Polygon overlay, used by the point arms
\\end{{tabular}}

\\vspace{{5pt}}
\\begin{{minipage}}{{0.92\\textwidth}}
\\small
\\textbf{{Question:}} {latex(m['question'])}\\\\[2pt]
{choices}[2pt]
\\textbf{{Stage~1 predicted answer:}} {LETTERS[int(m_stage1['pred'])]}\\\\
\\textbf{{Gold answer supplied to Stage~2:}} {LETTERS[int(m['answer_label'])]}\\\\
\\textbf{{Stage~2 final rationale:}} {latex(m_stage2['tolerant_final'])}
\\end{{minipage}}
\\caption[One VCR record through both stages]{{One evaluation record in the two
image forms and through both stages. The plain arms receive the unmarked frame
and the point arms the polygon overlay of the same frame; the question, choices,
and answers are identical in both. Stage~1 predicts an answer, and in the primary
experiment Stage~2 receives the gold answer rather than Stage~1's output. The
displayed prediction and rationale come from the plain arm under the source
condition. The record is illustrative and was not selected for any result.
Original VCR wording and spacing are preserved.}}
\\label{{fig:record-plain-vs-point}}
\\end{{figure}}
"""
    OUT_METHODS.parent.mkdir(parents=True, exist_ok=True)
    OUT_METHODS.write_text(methods, encoding="utf-8")

    # ------------------------------------------------- six-condition figure
    conditions = f"""\\begin{{figure}}[ht]
\\centering
\\begin{{tabular}}{{ccc}}
\\includegraphics[width=0.30\\textwidth]{{Figures/records/C/source.png}} &
\\includegraphics[width=0.30\\textwidth]{{Figures/records/C/grey.png}} &
\\includegraphics[width=0.30\\textwidth]{{Figures/records/C/mismatch.png}} \\\\
\\small Source & \\small Grey & \\small Mismatch \\\\[4pt]
\\includegraphics[width=0.30\\textwidth]{{Figures/records/C/mask.png}} &
\\includegraphics[width=0.30\\textwidth]{{Figures/records/C/noise.png}} &
\\includegraphics[width=0.30\\textwidth]{{Figures/records/C/mirror.png}} \\\\
\\small Mask & \\small Noise & \\small Mirror
\\end{{tabular}}
\\caption[Six image conditions for one VCR record]{{The six image conditions for
one evaluation record, shown for a plain arm. The question, choices, supplied
answer, and any description stay fixed; only the image changes. Panels share one
canvas so their boxes are comparable and no image is stretched, which is why the
square grey image does not fill its panel. This record is different from the one
in Figure~\\ref{{fig:record-plain-vs-point}} and was selected without consulting
any model output.}}
\\label{{fig:six-image-conditions}}
\\end{{figure}}
"""
    OUT_CONDITIONS.write_text(conditions, encoding="utf-8")

    # ------------------------------------------------- paired rationale figure
    outputs = {cond: stage2_output("plain", cond, r_id) for cond in ("source", "grey", "mismatch")}
    drifts = {c: per_record[r_id][c] for c in ("grey", "mismatch")}
    supplied_index = outputs["source"]["supplied_answer_index"]
    r_choices = "\n".join(
        rf"\textbf{{{LETTERS[i]}.}} {latex(r['choices'][i])}\\" for i in range(4)
    )
    spans = []
    for cond, label in (("source", "Source"), ("grey", "Grey"), ("mismatch", "Mismatch")):
        row = outputs[cond]
        head = rf"\textbf{{{label}}}"
        if cond != "source":
            head += rf" (raw Drift on the final span, {drifts[cond]:.4f})"
        # The parsed span contents are reproduced verbatim; the output tags
        # themselves are replaced by field labels so the panel reads as text
        # rather than as a raw generation. The caption states that convention.
        spans.append(
            head + r"\\" + "\n"
            + rf"\textit{{Reasoning:}} {latex(row['tolerant_reasoning'])}\\" + "\n"
            + rf"\textit{{Final:}} {latex(row['tolerant_final'])}"
        )
    paired = f"""\\begin{{figure}}[ht]
\\centering
\\begin{{tabular}}{{ccc}}
\\includegraphics[width=0.30\\textwidth]{{Figures/records/R/source.png}} &
\\includegraphics[width=0.30\\textwidth]{{Figures/records/R/grey.png}} &
\\includegraphics[width=0.30\\textwidth]{{Figures/records/R/mismatch.png}} \\\\
\\small Source & \\small Grey & \\small Mismatch
\\end{{tabular}}

\\vspace{{4pt}}
\\begin{{minipage}}{{0.94\\textwidth}}
\\small
\\textbf{{Question:}} {latex(r['question'])}\\\\[2pt]
{r_choices}[2pt]
\\textbf{{Supplied gold answer:}} {LETTERS[supplied_index]}\\\\[3pt]
""" + "\n\n\\vspace{3pt}\n".join(spans) + f"""
\\end{{minipage}}
\\caption[Paired rationale example]{{One evaluation record from the
\\texttt{{plain}} arm under the source image, grey replacement, and mismatch
replacement. The question, choices, and supplied gold answer are identical in all
three cells; only the image changes. The parsed span contents are reproduced
verbatim and the output tags are omitted for display. Each displayed generation
contains the same one-sentence text in both fields, matching the rule that builds
the training target when the rationale has one sentence (Section~\\ref{{sec:setup-prompts}}). Raw
Drift on the final span was {drifts['grey']:.4f} under grey and {drifts['mismatch']:.4f} under mismatch, against
arm means of {means['grey']:.4f} and {means['mismatch']:.4f}. The record was chosen mechanically as the one
whose grey and mismatch Drift lie closest to those arm means, among records with
all three spans usable and excluding every record used elsewhere in the thesis.
It illustrates the comparison and carries no inferential weight.}}
\\label{{fig:rq2-paired-example}}
\\end{{figure}}
"""
    OUT_PAIRED.write_text(paired, encoding="utf-8")

    provenance = {
        "schema": "v11-visual-record-provenance-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "inference_rerun": False,
        "metric_recomputed": False,
        "records": {
            "M": {
                "role": "combined methods figure",
                "selection_rule": "the record already used for the methods illustration",
                "outcomes_used_for_selection": False,
                "dataset_index": m_index, "record_id": m["image_id"], "source_frame_id": m["orig_image_id"],
            },
            "C": {
                "role": "six image conditions",
                "selection_rule": (
                    "first record in canonical dataset order whose source, polygon, and mismatch "
                    "target assets all exist, excluding Record M and every W-to-C sample record"
                ),
                "outcomes_used_for_selection": False,
                "dataset_index": c_index, "record_id": c["image_id"], "source_frame_id": c["orig_image_id"],
            },
            "R": {
                "role": "paired rationale comparison",
                "selection_rule": (
                    "plain arm; usable source, grey, and mismatch final spans; excluding Records M "
                    "and C and every W-to-C sample record; minimising the summed absolute distance "
                    "of the record's grey and mismatch Drift from the plain-arm means; ties broken "
                    "by the lowest canonical dataset index"
                ),
                "outcomes_used_for_selection": False,
                "frozen_drift_used_for_display_selection_only": True,
                "dataset_index": r_index, "record_id": r["image_id"], "source_frame_id": r["orig_image_id"],
                "grey_drift": drifts["grey"], "mismatch_drift": drifts["mismatch"],
                "plain_arm_grey_mean": means["grey"], "plain_arm_mismatch_mean": means["mismatch"],
                "selection_distance": round(_distance, 6),
                "output_record_sha256": {c: outputs[c]["output_record_sha256"] for c in outputs},
            },
        },
        "assets": assets,
        "outputs": {
            "methods_figure": {"path": str(OUT_METHODS.relative_to(ROOT)), "sha256": sha256(OUT_METHODS)},
            "six_condition_figure": {"path": str(OUT_CONDITIONS.relative_to(ROOT)), "sha256": sha256(OUT_CONDITIONS)},
            "paired_rationale_figure": {"path": str(OUT_PAIRED.relative_to(ROOT)), "sha256": sha256(OUT_PAIRED)},
        },
    }
    PROVENANCE.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "M": {"index": m_index, "record": m["image_id"][:40]},
        "C": {"index": c_index, "record": c["image_id"][:40]},
        "R": {"index": r_index, "record": r["image_id"][:40],
              "grey": round(drifts["grey"], 4), "mismatch": round(drifts["mismatch"], 4)},
    }, indent=2))


if __name__ == "__main__":
    main()
