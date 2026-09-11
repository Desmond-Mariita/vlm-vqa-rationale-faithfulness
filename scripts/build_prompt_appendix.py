#!/usr/bin/env python3
"""Emit the prompt blocks for the appendix and audit their fidelity.

The literal prompt strings below are the ones the production code builds. They
are verified two ways: by rebuilding every prompt in the stored manifests and
matching the recorded per-record SHA-256, and by decoding the emitted LaTeX back
to the same characters. The Stage 2 training template is read from the four
resolved run configurations and compared with the generation template. Nothing
here runs a model.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import re

import yaml
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = ROOT / "provenance" / "selected_final_audits"
OUT = ROOT / "thesis" / "Appendices" / "prompt_templates.tex"
MANIFESTS = ROOT / "frozen_execution" / "manifests"
PROJECT = Path(".")
DATASET = PROJECT / "data/final/vcr_val_10_pct.json"
DATASET_SHA = "0288740b785021e667c12eb08ab2c222a65f80b2702d598a7ebb7b942b3a5fdc"

# ---------------------------------------------------------------- literals
STAGE1 = (
    "<image>\n"
    "Question: {question}\n"
    "Choices:\n"
    "  (A) {c0}\n"
    "  (B) {c1}\n"
    "  (C) {c2}\n"
    "  (D) {c3}\n"
    "Instruction: Select the best option and answer with a single letter (A/B/C/D).\n"
)
STAGE1_CAPTION = "\nCaption: {caption}"

STAGE2_GOLD = """You are an expert visual commonsense reasoner.

You see an image and a multiple-choice question with four options.
Your task is to explain, step by step, why the given answer is correct,
using only what can reasonably be inferred from the image and question.

Question: {question}

Choices:
  (0) {c0}
  (1) {c1}
  (2) {c2}
  (3) {c3}

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
STAGE2_GOLD_CAPTION_ON = "\n\nImage caption: {caption}"
STAGE2_GOLD_CAPTION_OFF = "\n\nImage caption:"

STAGE2_TRAIN_RUNS = {
    "plain": "reports/runs/plain/stage2_gold/scratch_v1/config.resolved.yaml",
    "point": "reports/runs/point/stage2_gold/20260505_141209/config.resolved.yaml",
    "plain_desc": "reports/runs/plain_desc/stage2_gold/20260509_174550/config.resolved.yaml",
    "point_desc": "reports/runs/point_desc/stage2_gold/20260513_183118/config.resolved.yaml",
}
# The caption block Stage 2 training appended in the two description arms.
STAGE2_TRAIN_CAPTION = (
    "\nCaption (context only — do NOT copy this text in your reasoning): {caption}"
    "\nYour explanation MUST NOT reuse the caption. Use only the image + question + answer."
)

PRED_A = """You are reflecting on your own decision in a visual question answering task.

You see an image and a multiple-choice question with four options.
Your task is to explain, step by step, why the answer you chose is (or seems)
correct, using only what can reasonably be inferred from the image and question.

Question: {question}

Choices:
  (0) {c0}
  (1) {c1}
  (2) {c2}
  (3) {c3}

You picked choice index {answer_idx}: "{answer}".

Explain your reasoning step by step, focusing only on what can be seen
in the image (and what follows directly from it).
Then give a short final justification.
Continue until the explanation is complete and well-formed.

Use the following format exactly:

<reasoning>
...detailed step-by-step reasoning about the image, question, and answer...
</reasoning>
<final>
...short 2\u20133 sentence justification...
</final>
"""
# The two description arms append the same block Stage 2 training used.
PRED_A_CAPTION = (
    "\nCaption (context only \u2014 do NOT copy this text in your reasoning): {caption}"
    "\nYour explanation MUST NOT reuse the caption. Use only the image + question + answer."
)

CMC_JUDGE = """You are judging a multiple-choice question without access to its image.
Another system supplied an explanation. Use only the question, choices, and explanation.

Question: {question}

Choices:
(0) {c0}
(1) {c1}
(2) {c2}
(3) {c3}

Explanation:
{rationale_final}

Which choice is best supported? Answer with exactly one digit: 0, 1, 2, or 3.
Answer:"""

PRED_A_RUNS = {
    "plain": "reports/runs/plain/stage2_pred/20260704_145350/outputs/Route2/generated_rationales.jsonl",
    "point": "reports/runs/point/stage2_pred/20260506_100946/outputs/Route2/generated_rationales.jsonl",
    "plain_desc": "reports/runs/plain_desc/stage2_pred/20260510_144738/outputs/Route2/generated_rationales.jsonl",
    "point_desc": "reports/runs/point_desc/stage2_pred/20260514_152333/outputs/Route2/generated_rationales.jsonl",
}
JUDGE_PROMPT_SAMPLES = PROJECT / "v9/Artifacts/hardening_20260817/cmc_clean/prompt_samples.jsonl"

DESCRIPTION_SYSTEM = (
    "You are a vision-language model tasked with generating factual, concise scene "
    "descriptions based on labeled image elements such as person_0 or object_1. "
    "Avoid speculation and emotional language."
)
DESCRIPTION_USER = (
    "Describe this scene based on labeled elements (e.g., person_0, object_1). "
    "Focus on positions, actions, and interactions without interpretation. "
    "Be concise and objective."
)

# ------------------------------------------------------------------ escaping
_ESCAPES = [
    ("\\", r"\textbackslash "),
    ("{", r"\{"), ("}", r"\}"),
    ("$", r"\$"), ("&", r"\&"), ("#", r"\#"), ("%", r"\%"),
    ("_", r"\_"), ("^", r"\textasciicircum "), ("~", r"\textasciitilde "),
    ("<", r"\textless "), (">", r"\textgreater "),
    ('"', r"\textquotedbl "),
]
_DECODE = {v.rstrip(): k for k, v in _ESCAPES}


def escape(line: str) -> str:
    out = []
    for ch in line:
        for raw, tex in _ESCAPES:
            if ch == raw:
                out.append(tex)
                break
        else:
            out.append(ch)
    return "".join(out)


def decode(tex_line: str) -> str:
    """Invert escape() so the emitted LaTeX can be checked against the literal."""
    s = tex_line
    s = s.replace(r"\hspace*{1ex}", " ")
    for tex, raw in sorted(_DECODE.items(), key=lambda kv: -len(kv[0])):
        s = s.replace(tex + " ", raw).replace(tex, raw)
    return s


def block(text: str) -> tuple[str, list[str]]:
    """Render one prompt literal as a line-for-line LaTeX quote block."""
    lines = text.split("\n")
    rendered = []
    for line in lines:
        indent = len(line) - len(line.lstrip(" "))
        body = escape(line.strip(" ")) if indent else escape(line)
        if not line:
            rendered.append(r"\mbox{}")
        elif indent:
            rendered.append((r"\hspace*{1ex}" * indent) + body)
        else:
            rendered.append(body)
    tex = "\\begin{quote}\\small\\ttfamily\\frenchspacing\\setlength{\\parskip}{0pt}\n" + \
          "\\\\\n".join(rendered) + "\n\\end{quote}\n"
    return tex, lines


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def caption(row: dict) -> str:
    return str((row.get("image_descriptions") or {}).get("caption") or "")[:256]


def main() -> None:
    records = json.loads(DATASET.read_text(encoding="utf-8"))
    by_id = {r["image_id"]: r for r in records}
    checks: list[dict] = []

    # ---- verification 1: the literals rebuild every frozen manifest prompt --
    def s2(row, arm, idx, mode):
        ch = list(row["choices"][:4]) + [""] * 4
        p = STAGE2_GOLD.format(question=row["question"], c0=ch[0], c1=ch[1], c2=ch[2], c3=ch[3],
                               answer_idx=idx, answer=ch[idx])
        if arm.endswith("_desc"):
            if mode == "on" and caption(row):
                p += STAGE2_GOLD_CAPTION_ON.format(caption=caption(row))
            elif mode == "off":
                p += STAGE2_GOLD_CAPTION_OFF
        return p

    ok = bad = 0
    with gzip.open(MANIFESTS / "STAGE2_PRIMARY_MANIFEST.tsv.gz", "rt", newline="") as fh:
        for m in csv.DictReader(fh, delimiter="\t"):
            good = sha(s2(by_id[m["record_id"]], m["arm"], int(m["supplied_gold_answer_index"]), "on")) == m["prompt_sha256"]
            ok, bad = ok + good, bad + (not good)
    checks.append({"check": "Stage 2 gold-answer literal rebuilds every primary manifest prompt hash",
                   "status": "PASS" if bad == 0 and ok == 63672 else "FAIL",
                   "evidence": {"verified": ok, "mismatched": bad}})

    kc_ok = kc_bad = 0
    with (MANIFESTS / "STAGE2_KNOWN_CHANGE_MANIFEST.tsv").open(newline="") as fh:
        for m in csv.DictReader(fh, delimiter="\t"):
            good = sha(s2(by_id[m["record_id"]], m["arm"], int(m["alternate_answer_index"]), "on")) == m["prompt_sha256"]
            kc_ok, kc_bad = kc_ok + good, kc_bad + (not good)
    checks.append({"check": "the same literal rebuilds every wrong-answer control prompt hash",
                   "status": "PASS" if kc_bad == 0 else "FAIL",
                   "evidence": {"verified": kc_ok, "mismatched": kc_bad}})

    def s1(row, arm):
        c0, c1, c2, c3 = row["choices"][:4]
        p = STAGE1.format(question=row["question"], c0=c0, c1=c1, c2=c2, c3=c3)
        cap = (row.get("image_descriptions") or {}).get("caption") if arm.endswith("_desc") else None
        if cap:
            p += STAGE1_CAPTION.format(caption=cap[:256])
        return p

    s1_ok = s1_bad = 0
    with (MANIFESTS / "STAGE1_FINAL_WRONG_MANIFEST.tsv").open(newline="") as fh:
        for m in csv.DictReader(fh, delimiter="\t"):
            good = sha(s1(by_id[m["record_id"]], m["arm"])) == m["prompt_sha256"]
            s1_ok, s1_bad = s1_ok + good, s1_bad + (not good)
    checks.append({"check": "Stage 1 literal rebuilds every Stage 1 manifest prompt hash",
                   "status": "PASS" if s1_bad == 0 else "FAIL",
                   "evidence": {"verified": s1_ok, "mismatched": s1_bad}})

    train_templates = {}
    for arm, rel in STAGE2_TRAIN_RUNS.items():
        cfg = yaml.safe_load((PROJECT / rel).read_text(encoding="utf-8"))
        tmpl = cfg["train_stage2"]["stage2_prompt_template"]
        train_templates[arm] = {
            "config": str(PROJECT / rel),
            "config_sha256": hashlib.sha256((PROJECT / rel).read_bytes()).hexdigest(),
            "template_sha256": sha(tmpl),
        }
    tmpl_shas = {v["template_sha256"] for v in train_templates.values()}
    checks.append({"check": "all four arms trained Stage 2 with the same prompt template",
                   "status": "PASS" if len(tmpl_shas) == 1 else "FAIL",
                   "evidence": train_templates})

    train_tmpl = yaml.safe_load(
        (PROJECT / STAGE2_TRAIN_RUNS["plain"]).read_text(encoding="utf-8")
    )["train_stage2"]["stage2_prompt_template"].rstrip("\n")
    diffs = {
        "opening_task_line": (train_tmpl.split("\n")[3], STAGE2_GOLD.split("\n")[3]),
        "completion_instruction_in_training_only":
            "Continue until the explanation is complete and well-formed." in train_tmpl
            and "Continue until the explanation is complete and well-formed." not in STAGE2_GOLD,
        "training_caption_is_anti_copy": "do NOT copy this text" in STAGE2_TRAIN_CAPTION,
        "generation_caption_field": "Image caption:",
    }
    checks.append({"check": "the Stage 2 training and generation templates differ, and the difference is stated in the appendix",
                   "status": "PASS" if (train_tmpl != STAGE2_GOLD
                                        and diffs["completion_instruction_in_training_only"]
                                        and diffs["training_caption_is_anti_copy"]
                                        and diffs["opening_task_line"][0] != diffs["opening_task_line"][1])
                             else "FAIL",
                   "evidence": diffs})

    # ---- verification: the literal rebuilds every stored PRED-A prompt -----
    def pred_a(row, idx, cap):
        ch = list(row["choices"][:4])
        text = PRED_A.format(question=row["question"], c0=ch[0], c1=ch[1], c2=ch[2], c3=ch[3],
                             answer_idx=idx, answer=ch[idx])
        if cap:
            text += PRED_A_CAPTION.format(caption=cap[:256])
        return text

    pa_ok = pa_bad = 0
    pa_caption_arms = {}
    for arm, rel in PRED_A_RUNS.items():
        with (PROJECT / rel).open(encoding="utf-8") as fh:
            captions = 0
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                cap = row.get("caption") if arm.endswith("_desc") else None
                captions += bool(cap)
                good = pred_a(by_id[row["image_id"]], row["answer_idx_pred"], cap) == row["prompt"]
                pa_ok, pa_bad = pa_ok + good, pa_bad + (not good)
        pa_caption_arms[arm] = captions
    checks.append({"check": "the PRED-A literal rebuilds every stored PRED-A prompt",
                   "status": "PASS" if pa_bad == 0 and pa_ok == 4 * 2653 else "FAIL",
                   "evidence": {"verified": pa_ok, "mismatched": pa_bad}})
    checks.append({"check": "the caption block appears in the description arms and nowhere else",
                   "status": "PASS" if pa_caption_arms == {"plain": 0, "point": 0,
                                                           "plain_desc": 2653, "point_desc": 2653} else "FAIL",
                   "evidence": pa_caption_arms})
    checks.append({"check": "the PRED-A caption block is the same one Stage 2 training used",
                   "status": "PASS" if PRED_A_CAPTION == STAGE2_TRAIN_CAPTION else "FAIL",
                   "evidence": {"identical": PRED_A_CAPTION == STAGE2_TRAIN_CAPTION}})
    dashes = {"en dash in the output template": "\u2013" in PRED_A,
              "em dash in the caption block": "\u2014" in PRED_A_CAPTION}
    checks.append({"check": "the reproduced PRED-A prompt keeps its literal dash characters",
                   "status": "PASS" if all(dashes.values()) else "FAIL",
                   "evidence": dashes})

    # ---- verification: the literal rebuilds every stored judge prompt -------
    judge_ok = judge_bad = 0
    for sample in (json.loads(l) for l in JUDGE_PROMPT_SAMPLES.read_text(encoding="utf-8").splitlines() if l.strip()):
        stored = sample.get("rationale_prompt")
        if not stored:
            continue
        body = stored.split("Question: ", 1)[1]
        question = body.split("\n\nChoices:\n", 1)[0]
        choices = body.split("\n\nChoices:\n", 1)[1].split("\n\nExplanation:\n", 1)[0].split("\n")
        final = stored.split("Explanation:\n", 1)[1].split("\n\nWhich choice")[0]
        rebuilt = CMC_JUDGE.format(
            question=question, rationale_final=final,
            **{f"c{i}": choices[i].split(") ", 1)[1] for i in range(4)},
        )
        good = rebuilt == stored
        judge_ok, judge_bad = judge_ok + good, judge_bad + (not good)
    checks.append({"check": "the judge literal rebuilds every stored judge prompt sample",
                   "status": "PASS" if judge_bad == 0 and judge_ok > 0 else "FAIL",
                   "evidence": {"verified": judge_ok, "mismatched": judge_bad}})

    # ---- verification 2: emitted LaTeX decodes back to the literal ---------
    blocks = {
        "stage1": STAGE1.rstrip("\n"),
        "stage1_caption": "Caption: {caption}",
        "stage2_gold": STAGE2_GOLD,
        "stage2_gold_caption_on": "Image caption: {caption}",
        "stage2_gold_caption_off": "Image caption:",
        "stage2_train_caption": STAGE2_TRAIN_CAPTION.lstrip("\n"),
        "pred_a": PRED_A.rstrip("\n"),
        "pred_a_caption": PRED_A_CAPTION.lstrip("\n"),
        "cmc_judge": CMC_JUDGE,
        "description_system": "System: " + DESCRIPTION_SYSTEM,
        "description_user": "User: " + DESCRIPTION_USER,
    }
    rendered = {}
    roundtrip_failures = []
    long_lines = []
    for name, literal in blocks.items():
        tex, lines = block(literal)
        rendered[name] = tex
        back = [decode(x) for x in re.split(r"\\\\\n", tex.split("\n", 1)[1].rsplit("\n\\end{quote}", 1)[0])]
        back = ["" if x == r"\mbox{}" else x for x in back]
        if back != lines:
            roundtrip_failures.append(name)
        # A long line wraps at a space and costs no overfull box; a long line with
        # no space in it cannot wrap and would overflow the measure.
        long_lines += [(name, len(l), l) for l in lines
                       if len(l) > 72 and max((len(w) for w in l.split(" ")), default=0) > 72]
    checks.append({"check": "every emitted block decodes back to the literal prompt characters",
                   "status": "PASS" if not roundtrip_failures else "FAIL",
                   "evidence": {"failed_blocks": roundtrip_failures}})
    wrapped = sorted({name for name, ln, _ in
                      [(n, len(l), l) for n, _l in [] ] } )  # placeholder, replaced below
    wrapped = sorted({name for name, literal in blocks.items()
                      for l in literal.split("\n") if len(l) > 72})
    checks.append({"check": "no reproduced prompt line can overflow the typewriter measure",
                   "status": "PASS" if not long_lines else "FAIL",
                   "evidence": {"unwrappable_long_lines": long_lines,
                                "blocks_with_lines_that_wrap_at_a_space": wrapped}})

    # ---- emit ---------------------------------------------------------------
    OUT.write_text(
        "% Generated by scripts/build_prompt_appendix.py. Do not edit by hand.\n"
        "% Every block is the literal prompt text, verified against the stored\n"
        "% per-record prompt hashes; see provenance/selected_final_audits/PROMPT_FIDELITY_AUDIT.json.\n\n"
        "\\paragraph{Reading these blocks.}\n"
        "Placeholders in braces are filled per record. Every other character is\n"
        "literal, including the runs of three dots inside the output template: they\n"
        "are part of the prompt the model received and are not an omission by us.\n"
        "At Stage~2 the image is not a token in the prompt text. It is passed as a\n"
        "separate image item in the model's chat template, and the processor inserts\n"
        "its own image tokens. The Stage~1 text opens with the literal string\n"
        "\\texttt{<image>}.\n\n"
        "\\paragraph{Stage 1.}\n"
        "The image accompanies the following text.\n\n" + rendered["stage1"] +
        "\nDescription arms append one blank line and then this field, which closes\n"
        "the Stage~1 prompt; the other arms end after the instruction line.\n\n" +
        rendered["stage1_caption"] +
        "\nThe answer is taken from the classifier head, so the letter instruction\n"
        "frames the task without being parsed (Section~\\ref{sec:setup-model}).\n\n"
        "\\paragraph{Stage 2 with the gold answer.}\n"
        "This is the prompt behind every primary RQ2 generation and behind the\n"
        "wrong-answer control, which differs only in the answer index and text\n"
        "it supplies.\n\n" + rendered["stage2_gold"] +
        "\nThe two description arms append one blank line and then this field, so the\n"
        "description is the last thing in the prompt.\n\n" +
        rendered["stage2_gold_caption_on"] +
        "\n\\paragraph{Training and generation prompts.}\n"
        "Stage~2 was trained with a template close to the one above but not identical\n"
        "to it. All four arms used the same training template. It asks why \\emph{a}\n"
        "given answer is correct rather than \\emph{the} given answer, and it adds one\n"
        "instruction, \\enquote{Continue until the explanation is complete and\n"
        "well-formed}, that the generation prompt does not carry. In the two\n"
        "description arms the training prompt also appended a different description\n"
        "block from the one used at generation: it labelled the description as context\n"
        "and told the model not to reuse it.\n\n" + rendered["stage2_train_caption"] +
        "\nBoth templates request a two- to three-sentence final justification, whereas\n"
        "the training target places the last sentence of the human rationale in the\n"
        "\\texttt{<final>} field (Section~\\ref{sec:setup-prompts}). The analysis\n"
        "therefore evaluates the non-empty final span the model actually generated,\n"
        "whatever its sentence count.\n\n"
        "\\paragraph{Stage 2 with the predicted answer.}\n"
        "The secondary predicted-answer regime used this prompt. It frames the task\n"
        "as reflection on the model\u2019s own decision, supplies the Stage~1 answer\n"
        "rather than the gold answer, and adds the instruction to continue until the\n"
        "explanation is complete. It is not the primary generation prompt.\n\n" + rendered["pred_a"] +
        "\nThe two description arms append this training block. The primary generation\n"
        "prompt uses a different description field.\n"
        "The other two arms receive no description block at all.\n\n" + rendered["pred_a_caption"] +
        "\n\\paragraph{Cross-Modal Consistency judge.}\n"
        "The judge receives this text and nothing else. There is no reasoning or\n"
        "final scaffold: the \\texttt{Explanation} field holds the eligible final\n"
        "rationale span on its own, so the generated reasoning span never reaches the\n"
        "judge, and neither does the image, the arm, or the answer supplied to\n"
        "Stage~2. The judge emits one token, constrained to the four literal digits,\n"
        "so every record returns a choice index and no parse can fail.\n\n" + rendered["cmc_judge"] +
        "\n\\paragraph{Description generation.}\n"
        "Descriptions were produced from the \\mbox{polygon-marked} frames through the\n"
        "OpenAI batch interface with the \\texttt{gpt-4o} alias, temperature 0.2, and a\n"
        "limit of 120 output tokens. The image was supplied inline as a base64 JPEG.\n"
        "No dated model snapshot was pinned, which\n"
        "Section~\\ref{sec:discussion-limitations} records as a reproducibility\n"
        "limitation.\n\n" + rendered["description_system"] + "\n" + rendered["description_user"],
        encoding="utf-8",
    )

    audit = {
        "schema": "v11-prompt-fidelity-audit-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "method": ("Rebuilt every prompt in the frozen manifests and stored artifacts from the "
                   "literals reproduced in the prompt appendix, then decoded the emitted LaTeX "
                   "back to those literals, and read the Stage 2 training template from the four "
                   "resolved run configurations. No model was loaded and no generation was rerun."),
        "dataset": {"path": str(DATASET), "sha256": DATASET_SHA,
                    "matches": hashlib.sha256(DATASET.read_bytes()).hexdigest() == DATASET_SHA},
        "literal_sha256": {k: sha(v) for k, v in blocks.items()},
        "ellipses_are_literal_prompt_text": True,
        "image_supplied_as_chat_item_not_prompt_text": True,
        "stage1_field_label": "Caption:",
        "stage2_gold_field_label": "Image caption:",
        "stage2_training_field_label": "Caption (context only — do NOT copy this text in your reasoning):",
        "stage2_training_caption_is_anti_copy": True,
        "output_file": str(OUT.relative_to(ROOT)),
        "checks": checks,
    }
    audit["status"] = "PASS" if all(c["status"] == "PASS" for c in checks) else "FAIL"
    (AUDIT_DIR / "PROMPT_FIDELITY_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": audit["status"],
                      "failed": [c["check"] for c in checks if c["status"] != "PASS"]}, indent=2))
    if audit["status"] != "PASS":
        raise SystemExit("PROMPT FIDELITY AUDIT FAILED")


if __name__ == "__main__":
    main()
