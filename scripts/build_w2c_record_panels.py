#!/usr/bin/env python3
"""Build the self-contained W-to-C appendix table from frozen material."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]
MATERIAL = Path(
    "reports/frozen/rq1_final/qualitative/"
    "RQ1_W2C_SELECTED_MATERIAL.tsv"
)
AUDIT = Path(
    "reports/frozen/rq1_final/qualitative/"
    "RQ1_W2C_QUALITATIVE_AUDIT.md"
)
DATASET = Path("./data/final/vcr_val_10_pct.json")
OUTPUT = ROOT / "thesis/Figures/final/w2c_record_panels.tex"
PANEL_DIR = ROOT / "thesis/Figures/w2c/panels"
INSPECTION = Path(
    "reports/frozen/rq1_final/qualitative/inspection_assets"
)
CANVAS = (640, 360)
BORDER_RGB = (154, 160, 166)
AUDIT_OUTPUT = ROOT / "provenance/selected_final_audits/COMPLETE_W2C_RECORD_PANEL_AUDIT.json"
EXPECTED_MATERIAL_SHA = "7f3f1e817d6836daa67833022dbb42ea5b44b1d68952005d1a4fe61a5f282883"
EXPECTED_AUDIT_SHA = "e8f5d0079d6faf4427fd051c982cb0d7df0fbc71db6fa3592915e6fd9880ae2a"
EXPECTED_DATASET_SHA = "0288740b785021e667c12eb08ab2c222a65f80b2702d598a7ebb7b942b3a5fdc"
LETTERS = "ABCD"

# Reader-facing text uses answer letters. This catches a zero-based reference in
# either number: "choice 2", "choices 2 and 3", "index 0", "indices 0 and 3",
# and the "choice index 3" form the prompt template uses.
NUMERIC_CHOICE_RE = re.compile(
    r"\b(?:choice|choices|option|options|index|indices|answer)\s+(?:index\s+)?[0-3]\b",
    re.I,
)

# The active implementation supplies `str(caption or "")[:256]` to both
# description arms at Stage 1 and Stage 2. The table must show that prefix.
DESCRIPTION_CHARACTER_LIMIT = 256

# Shorter reader-facing category names. The frozen category vocabulary is
# unchanged in meaning; only the printed label is shortened so it hyphenates
# cleanly in a narrow column.
CATEGORY_DISPLAY = {
    "perturbation introduces misleading/irrelevant content": "misleading or irrelevant content",
    "visible support for the correct answer remains/appears": "visible support for the correct answer",
    "unclear / cannot determine": "unclear",
    "scene/choice ambiguity": "scene or choice ambiguity",
}

# Observation replacements. Each entry records why the frozen wording could not
# stand and what replaced it. The frozen qualitative audit file is never edited;
# these overrides exist so that no printed observation cites information the
# model did not receive, and so that reader-facing text uses answer letters.
OBSERVATION_OVERRIDES = {
    1: (
        "answer letters instead of zero-based choice indices",
        "The question asks the purpose of a staircase; choice D gives the generic function of "
        "moving between levels, whereas the grey image contains no scene information.",
    ),
    2: (
        "answer letters instead of zero-based choice indices",
        "The replacement image shows a wedding group and no staircase, while choice D still "
        "states the generic function of a staircase.",
    ),
    7: (
        "answer letters instead of zero-based choice indices",
        "All four answer choices are paraphrases saying that person_2 leads or is in charge; "
        "noise moves the selected choice from D to the keyed choice A.",
    ),
    3: (
        "answer letters instead of zero-based choice indices",
        "Roughly half the image pixels are blacked out, but the central interaction remains "
        "partly visible; choices C and D both describe the woman not resisting.",
    ),
    5: (
        "answer letters instead of zero-based choice indices",
        "The source shows the labelled person looking down; the grey intervention removes that "
        "scene, and choices C and D offer nearby affective interpretations.",
    ),
    9: (
        "frozen wording cited description text beyond the supplied 256-character prefix",
        "The grey image removes the scene, and the supplied description reaches only person_0 "
        "and the start of the person_3 entry, so it states nothing about person_7. Neither the "
        "supplied text nor the image bears on the future action the question asks about.",
    ),
    13: (
        "frozen wording cited description text beyond the supplied 256-character prefix",
        "The grey image removes the scene, and the supplied description ends mid-word before it "
        "names person_8, so it describes only a crowded indoor arena. Nothing supplied indicates "
        "why the person is doubled over.",
    ),
    15: (
        "frozen wording cited description text beyond the supplied 256-character prefix, "
        "and said nothing about the answer choices the assigned category names",
        "Heavy noise leaves the foreground pair visible; the supplied description places the "
        "scene outdoors and describes person_1 in a suit, tie, and boutonniere. It is cut before "
        "the sentence that characterises the setting as a formal event. The keyed choice also "
        "says only that the groom is nearby rather than naming a location, as the other three "
        "choices do, so the case carries an answer-choice artefact as well.",
    ),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_sha(path: Path, expected: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"SHA-256 mismatch for {path}: {actual} != {expected}")


def latex(value: str) -> str:
    """Escape literal frozen text without changing its wording."""
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    text = " ".join(value.split())
    return "".join(replacements.get(character, character) for character in text)


def supplied_descriptions() -> dict[str, str]:
    """Return the exact text supplied to the description arms, keyed by record."""
    require_sha(DATASET, EXPECTED_DATASET_SHA)
    records = json.loads(DATASET.read_text(encoding="utf-8"))
    return {
        record["image_id"]: str(
            (record.get("image_descriptions") or {}).get("caption") or ""
        )[:DESCRIPTION_CHARACTER_LIMIT]
        for record in records
    }


def audit_rows() -> list[dict[str, str]]:
    rows = []
    for line in AUDIT.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|") or line.startswith("|---") or "Short factual" in line:
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != 10:
            continue
        rows.append(
            {
                "arm": cells[0],
                "condition": cells[1],
                "record_id": cells[2].strip("`"),
                "observation": cells[6],
                "category": cells[7],
            }
        )
    if len(rows) != 16:
        raise RuntimeError(f"Expected 16 audit rows, found {len(rows)}")
    return rows


def answer(row: dict[str, str], field: str) -> str:
    index = int(row[field])
    return f"{LETTERS[index]}: {latex(row[f'choice_{index}'])}"


def panel(path: Path) -> Image.Image:
    """Place one image on the shared canvas so both panels read at one size."""
    canvas = Image.new("RGB", CANVAS, "white")
    with Image.open(path) as opened:
        copy = opened.convert("RGB")
        copy.thumbnail((CANVAS[0] - 4, CANVAS[1] - 4), Image.Resampling.LANCZOS)
        canvas.paste(copy, ((CANVAS[0] - copy.width) // 2, (CANVAS[1] - copy.height) // 2))
    return ImageOps.expand(canvas, border=1, fill=BORDER_RGB)


def main() -> None:
    require_sha(MATERIAL, EXPECTED_MATERIAL_SHA)
    require_sha(AUDIT, EXPECTED_AUDIT_SHA)
    with MATERIAL.open(encoding="utf-8", newline="") as stream:
        material = list(csv.DictReader(stream, delimiter="\t"))
    observations = audit_rows()
    prefixes = supplied_descriptions()
    if len(material) != 16:
        raise RuntimeError(f"Expected 16 material rows, found {len(material)}")
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    audit_records: list[dict[str, object]] = []

    out: list[str] = []
    index_rows: list[str] = []
    for ordinal, (row, audit) in enumerate(zip(material, observations), 1):
        if (
            int(row["sample_ordinal"]) != ordinal
            or row["arm"] != audit["arm"]
            or row["condition"] != audit["condition"]
            or row["record_id"] != audit["record_id"]
        ):
            raise RuntimeError(f"Frozen material and audit do not align at E{ordinal}")

        source_asset = INSPECTION / f"{ordinal:02d}_{row['arm']}_{row['condition']}_source.png"
        intervention_asset = Path(row["intervention_asset_local"])
        if not source_asset.is_file() or not intervention_asset.is_file():
            raise RuntimeError(f"Missing image for E{ordinal}")
        if sha256(intervention_asset) != row["intervention_asset_sha256"]:
            raise RuntimeError(f"Frozen intervention asset hash mismatch for E{ordinal}")
        source_panel = PANEL_DIR / f"E{ordinal:02d}_source.png"
        intervention_panel = PANEL_DIR / f"E{ordinal:02d}_intervention.png"
        panel(source_asset).save(source_panel, optimize=True)
        panel(intervention_asset).save(intervention_panel, optimize=True)

        frozen_category = audit["category"]
        category = CATEGORY_DISPLAY.get(frozen_category, frozen_category)
        reason, replacement = OBSERVATION_OVERRIDES.get(ordinal, ("", ""))
        observation = replacement or audit["observation"]

        supplied = ""
        description_shown = row["description_supplied"].lower() in {"1", "true"}
        if description_shown:
            supplied = prefixes[row["record_id"]]
            if supplied != (row["description"] or "")[:DESCRIPTION_CHARACTER_LIMIT]:
                raise RuntimeError(f"Supplied prefix disagrees with frozen material at E{ordinal}")

        choices = "\n".join(
            rf"\textbf{{{LETTERS[index]}.}} {latex(row[f'choice_{index}'])}\\"
            for index in range(4)
        )
        header = (
            rf"\textbf{{E{ordinal}}} \quad \texttt{{{latex(row['arm'])}}} \quad "
            rf"{latex(row['condition'])} \quad {latex(category)}"
        )
        answers = (
            rf"\textbf{{Gold answer:}} {LETTERS[int(row['gold_label'])]} \quad "
            rf"\textbf{{Source prediction:}} {LETTERS[int(row['source_prediction'])]} \quad "
            rf"\textbf{{Intervention prediction:}} {LETTERS[int(row['perturbation_prediction'])]}"
        )
        body = [header + r"\\[3pt]"]
        body.append(rf"\textbf{{Question:}} {latex(row['question'])}\\[2pt]")
        body.append(choices + "[2pt]")
        if description_shown:
            body.append(rf"\textbf{{Description supplied to the model:}} {latex(supplied)}\\[2pt]")
        body.append(answers + r"\\[2pt]")
        body.append(rf"\textbf{{Observation:}} {latex(observation)}")

        index_rows.append(
            rf"E{ordinal} & \texttt{{{latex(row['arm'])}}} & {latex(row['condition'])} & "
            rf"Figure~\ref{{fig:w2c-e{ordinal:02d}}} \\"
        )

        out.append(r"\begin{figure}[htbp]")
        out.append(r"\centering")
        out.append(r"\begin{tabular}{cc}")
        out.append(
            rf"\includegraphics[width=0.44\textwidth]{{Figures/w2c/panels/E{ordinal:02d}_source.png}} &"
        )
        out.append(
            rf"\includegraphics[width=0.44\textwidth]{{Figures/w2c/panels/E{ordinal:02d}_intervention.png}} \\"
        )
        out.append(r"\small Source image & \small Intervention image")
        out.append(r"\end{tabular}")
        out.append("")
        out.append(r"\vspace{4pt}")
        out.append(r"\begin{minipage}{0.94\textwidth}")
        out.append(r"\small")
        out.append("\n".join(body))
        out.append(r"\end{minipage}")
        out.append(
            # Reading conventions are stated once in the section introduction,
            # so each panel carries only its own identity.
            rf"\caption[W-to-C record E{ordinal}]{{Record E{ordinal}: "
            rf"\texttt{{{latex(row['arm'])}}} under {latex(row['condition'])}.}}"
        )
        out.append(rf"\label{{fig:w2c-e{ordinal:02d}}}")
        out.append(r"\end{figure}")
        out.append("")

        audit_records.append({
            "example": f"E{ordinal}",
            "record_id": row["record_id"],
            "arm": row["arm"],
            "condition": row["condition"],
            "dataset_index": int(row["dataset_index"]),
            "gold_letter": LETTERS[int(row["gold_label"])],
            "source_letter": LETTERS[int(row["source_prediction"])],
            "intervention_letter": LETTERS[int(row["perturbation_prediction"])],
            "description_supplied": description_shown,
            "stored_description_characters": len(row["description"] or ""),
            "supplied_prefix_characters": len(supplied),
            "supplied_prefix_equals_active_caption_slice": (
                supplied == (row["description"] or "")[:DESCRIPTION_CHARACTER_LIMIT]
            ),
            "source_panel": str(source_panel.relative_to(ROOT)),
            "source_panel_sha256": sha256(source_panel),
            "intervention_panel": str(intervention_panel.relative_to(ROOT)),
            "intervention_panel_sha256": sha256(intervention_panel),
            "frozen_category": frozen_category,
            "printed_category": category,
            "frozen_observation": audit["observation"],
            "printed_observation": observation,
            "observation_override_reason": reason,
        })

    # An index ahead of the panels, so a reader can find one record without
    # paging through sixteen full-page floats, and so every panel is named.
    index_block = [
        r"\begin{center}",
        r"\small",
        r"\begin{tabular}{llll}",
        r"\toprule",
        r"Record & Arm & Condition & Panel \\",
        r"\midrule",
        *index_rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{center}",
        "",
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(index_block + out), encoding="utf-8")
    printed = "\n".join(index_block + out)

    checks = [
        {
            "check": "sixteen self-contained record panels",
            "status": "PASS" if len(audit_records) == 16 else "FAIL",
            "evidence": [r["example"] for r in audit_records],
        },
        {
            "check": "each panel carries both images",
            "status": "PASS" if printed.count("E01_source.png") + sum(
                printed.count(f"E{i:02d}_source.png") for i in range(2, 17)) == 16
            and sum(printed.count(f"E{i:02d}_intervention.png") for i in range(1, 17)) == 16 else "FAIL",
            "evidence": None,
        },
        {
            "check": "every displayed description equals the active caption[:256] prefix",
            "status": "PASS" if all(
                r["supplied_prefix_equals_active_caption_slice"]
                for r in audit_records if r["description_supplied"]
            ) else "FAIL",
            "evidence": None,
        },
        {
            "check": "non-description arms show no description field",
            "status": "PASS" if all(
                r["supplied_prefix_characters"] == 0
                for r in audit_records if not r["description_supplied"]
            ) else "FAIL",
            "evidence": None,
        },
        {
            "check": "answers are letters only, with no repeated answer text in the summary",
            "status": "PASS" if all(
                r["gold_letter"] in LETTERS and r["source_letter"] in LETTERS
                and r["intervention_letter"] in LETTERS for r in audit_records
            ) and "Gold answer:} A:" not in printed else "FAIL",
            "evidence": None,
        },
        {
            "check": "each answer choice begins on its own line",
            "status": "PASS" if printed.count(r"\textbf{A.}") == 16
            and printed.count(r"\textbf{D.}") == 16 and ";" not in "".join(
                line for line in printed.splitlines() if r"\textbf{A.}" in line) else "FAIL",
            "evidence": None,
        },
        {
            "check": "each record is one float, so no record block is split across pages",
            "status": "PASS" if printed.count(r"\begin{figure}[htbp]") == 16
            and printed.count(r"\end{figure}") == 16 else "FAIL",
            "evidence": None,
        },
        {
            "check": "each panel caption names only its own record",
            "status": "PASS" if printed.count(r"\caption[W-to-C record E") == 16
            and "first 256 characters" not in printed else "FAIL",
            "evidence": None,
        },
        {
            "check": "no printed observation makes a reader-facing numeric choice reference",
            "status": "PASS" if not any(
                NUMERIC_CHOICE_RE.search(r["printed_observation"]) for r in audit_records
            ) else "FAIL",
            "evidence": [
                r["example"] for r in audit_records
                if NUMERIC_CHOICE_RE.search(r["printed_observation"])
            ],
        },
        {
            "check": "no local type reduction below the manuscript classes",
            "status": "PASS" if not any(
                t in printed for t in (r"\scriptsize", r"\tiny", r"\resizebox", r"\scalebox")
            ) else "FAIL",
            "evidence": None,
        },
    ]

    negation = re.compile(
        r"\b(?:no|not|nothing|never|without|before it names|ends mid-word|reaches only|cut before)\b",
        re.I,
    )
    unsupported = []
    for record in audit_records:
        if not record["description_supplied"]:
            continue
        prefix = prefixes[record["record_id"]].lower()
        for sentence in re.split(r"(?<=[.;])\s+", record["printed_observation"]):
            if not re.search(r"\bdescription\b", sentence, re.I) or negation.search(sentence):
                continue
            for label in sorted(set(re.findall(r"person_\d+", sentence))):
                if label not in prefix:
                    unsupported.append(f"{record['example']}:{label}")
    checks.append({
        "check": "no observation attributes a person label to description text outside the prefix",
        "status": "PASS" if not unsupported else "FAIL",
        "evidence": unsupported,
    })

    audit_document = {
        "schema": "v11-complete-w2c-record-panel-audit-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "active_rule": 'str(record["image_descriptions"]["caption"] or "")[:256]',
        "character_limit": DESCRIPTION_CHARACTER_LIMIT,
        "limit_unit": "Python string characters, not bytes",
        "frozen_material": {"path": str(MATERIAL), "sha256": sha256(MATERIAL)},
        "frozen_qualitative_audit": {"path": str(AUDIT), "sha256": sha256(AUDIT)},
        "evaluation_population": {"path": str(DATASET), "sha256": sha256(DATASET)},
        "output": {"path": str(OUTPUT.relative_to(ROOT)), "sha256": sha256(OUTPUT)},
        "records": audit_records,
        "checks": checks,
    }
    audit_document["status"] = "PASS" if all(c["status"] == "PASS" for c in checks) else "FAIL"
    AUDIT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_OUTPUT.write_text(json.dumps(audit_document, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {OUTPUT} with 16 self-contained record panels")
    print(json.dumps({
        "status": audit_document["status"],
        "failed": [c["check"] for c in checks if c["status"] == "FAIL"],
    }, indent=2))
    if audit_document["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
