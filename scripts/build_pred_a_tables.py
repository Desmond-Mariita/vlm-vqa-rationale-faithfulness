#!/usr/bin/env python3
"""Rebuild the PRED-A tables from integer counts and audit their arithmetic.

The frozen provenance record stores conditional and pipeline CMC for two arms at
full precision and for the other two rounded to three decimals, so a fourth
printed decimal was not attainable from that file alone. The per-record judge
decisions behind those aggregates are stored, so this script recovers the
integer numerator and denominator for every arm and stratum, checks that the
recovered integers reproduce every frozen value, and derives the displayed rates
from those counts.

No judge is run and no aggregate is replaced by a recomputation of a different
quantity: this is a count over already stored per-record decisions.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = ROOT / "provenance" / "selected_final_audits"
TABLES = ROOT / "thesis" / "Tables" / "final"

FROZEN = ROOT / "frozen_execution" / "manifests" / "PREDICTED_REGIME_PROVENANCE.json"
JUDGE_RECORDS = Path(
    "./v9/Artifacts/hardening_20260817/cmc_clean/records_qwen.jsonl"
)
JUDGE_SUMMARY = Path(
    "./v9/Artifacts/hardening_20260817/cmc_clean/summary_qwen.json"
)
RATIONALES = {
    "plain": "reports/runs/plain/stage2_pred/20260704_145350/outputs/Route2/generated_rationales.jsonl",
    "point": "reports/runs/point/stage2_pred/20260506_100946/outputs/Route2/generated_rationales.jsonl",
    "plain_desc": "reports/runs/plain_desc/stage2_pred/20260510_144738/outputs/Route2/generated_rationales.jsonl",
    "point_desc": "reports/runs/point_desc/stage2_pred/20260514_152333/outputs/Route2/generated_rationales.jsonl",
}
PROJECT = Path(".")
ARMS = ["plain", "point", "plain_desc", "point_desc"]
TOTAL = 2653


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def four(numerator: int, denominator: int) -> str:
    return f"{numerator / denominator:.4f}"


def main() -> None:
    judge = defaultdict(dict)
    for row in jsonl(JUDGE_RECORDS):
        if row["setting"] == "pred":
            judge[row["arm"]][row["image_id"]] = row

    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))["arms"]
    lineage = {c["arm"]: c for c in json.loads(JUDGE_SUMMARY.read_text(encoding="utf-8"))["cells"]
               if c["setting"] == "pred"}

    counts: dict[str, dict] = {}
    checks: list[dict] = []
    for arm in ARMS:
        rows = jsonl(PROJECT / RATIONALES[arm])
        arm_judge = judge[arm]
        usable = [r for r in rows if arm_judge[r["image_id"]]["eligible_final"]]
        correct_n = sum(1 for r in rows if r["answer_idx_pred"] == r["answer_idx_gold"])
        judge_ok = sum(1 for r in usable if arm_judge[r["image_id"]]["rationale_correct"])

        strata = {}
        for label, want in (("correct", True), ("wrong", False)):
            sub = [r for r in rows if (r["answer_idx_pred"] == r["answer_idx_gold"]) is want]
            sub_usable = [r for r in sub if arm_judge[r["image_id"]]["eligible_final"]]
            sub_ok = sum(1 for r in sub_usable if arm_judge[r["image_id"]]["rationale_correct"])
            strata[label] = {"records": len(sub), "usable": len(sub_usable), "judge_correct": sub_ok}

        cell_lineage = dict(lineage[arm]["lineage"])
        counts[arm] = {
            "empty_final": cell_lineage.get("empty_final", 0),
            "ambiguous": cell_lineage.get("ambiguous_reparsed_from_reasoning", 0),
            "tolerant_recovered": cell_lineage.get("verified_reparsed_from_rationale_gen", 0),
            "records": len(rows),
            "usable_final_n": len(usable),
            "judge_correct_n": judge_ok,
            "stage1_correct_n": correct_n,
            "stage1_wrong_n": len(rows) - correct_n,
            "conditional_cmc_from_counts": judge_ok / len(usable),
            "pipeline_cmc_from_counts": judge_ok / len(rows),
            "strata": strata,
        }

        cell = lineage[arm]
        checks.append({
            "check": f"{arm}: recovered integers reproduce the full-precision lineage conditional CMC",
            "status": "PASS" if abs(judge_ok / len(usable) - cell["conditional_cmc"]) < 1e-12 else "FAIL",
            "evidence": {"counts": f"{judge_ok}/{len(usable)}", "lineage": cell["conditional_cmc"]},
        })
        checks.append({
            "check": f"{arm}: recovered integers reproduce the full-precision lineage pipeline CMC",
            "status": "PASS" if abs(judge_ok / len(rows) - cell["pipeline_cmc"]) < 1e-12 else "FAIL",
            "evidence": {"counts": f"{judge_ok}/{len(rows)}", "lineage": cell["pipeline_cmc"]},
        })
        checks.append({
            "check": f"{arm}: usable count matches the frozen manifest",
            "status": "PASS" if len(usable) == frozen[arm]["hardened_usable_final_n"] else "FAIL",
            "evidence": {"counts": len(usable), "frozen": frozen[arm]["hardened_usable_final_n"]},
        })
        checks.append({
            "check": f"{arm}: strata sum to the arm totals",
            "status": "PASS" if (
                strata["correct"]["records"] + strata["wrong"]["records"] == len(rows)
                and strata["correct"]["usable"] + strata["wrong"]["usable"] == len(usable)
                and strata["correct"]["judge_correct"] + strata["wrong"]["judge_correct"] == judge_ok
            ) else "FAIL",
            "evidence": strata,
        })
        for label in ("correct", "wrong"):
            s = strata[label]
            want = frozen[arm]["correct_wrong_cmc_split"][label]
            got_cond = round(s["judge_correct"] / s["usable"], 3)
            got_pipe = round(s["judge_correct"] / s["records"], 3)
            checks.append({
                "check": f"{arm}/{label}: counts reproduce the frozen three-decimal split",
                "status": "PASS" if got_cond == round(want["conditional"], 3) else "FAIL",
                "evidence": {"conditional_from_counts": got_cond, "frozen": want["conditional"],
                             "pipeline_from_counts": got_pipe, "frozen_pipeline": want["pipeline"],
                             "pipeline_agrees": got_pipe == round(want["pipeline"], 3)},
            })

    # ---------------------------------------------------------------- tables
    rows_compact = "\n".join(
        f"\\texttt{{{arm.replace('_', chr(92) + '_')}}} & {c['usable_final_n']:,} & "
        f"{c['usable_final_n'] / TOTAL * 100:.2f}\\% & {c['judge_correct_n']:,} & "
        f"{four(c['judge_correct_n'], c['usable_final_n'])} & "
        f"{four(c['judge_correct_n'], c['records'])} \\\\"
        for arm, c in ((a, counts[a]) for a in ARMS)
    )
    (TABLES / "pred_a_compact.tex").write_text(
        "\\begin{table}[ht]\n\\centering\n"
        "\\caption[Secondary predicted-answer regime]{Results of the PRED-A regime,\n"
        "which uses source images only. Stage~2 received the Stage~1 predictions\n"
        "saved at epoch~3. The RQ1 results use the retained epoch~2 predictions, so\n"
        "these figures describe the answers actually supplied to these generations.\n"
        "CMC here is conditioned on that supplied predicted answer. Answer recovered\n"
        "counts the records whose supplied answer the judge recovered. Conditional\n"
        "CMC divides that count by the usable final rationales; pipeline CMC divides\n"
        "it by all 2,653 records, so it also carries the coverage shown in the third\n"
        "column. Both rates are derived from the printed integers.}\n"
        "\\label{tab:pred-a-compact}\n"
        "\\begin{tabular}{lrrrrr}\n\\toprule\n"
        "Arm & Usable rationales & Coverage & Answer recovered & Conditional CMC & Pipeline CMC \\\\\n"
        "\\midrule\n" + rows_compact + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n",
        encoding="utf-8",
    )

    strata_lines = []
    for arm in ARMS:
        c = counts[arm]
        tex_arm = f"\\texttt{{{arm.replace('_', chr(92) + '_')}}}"
        for i, label in enumerate(("correct", "wrong")):
            s = c["strata"][label]
            strata_lines.append(
                f"{tex_arm if i == 0 else ''} & {label} & {s['records']:,} & {s['usable']:,} & "
                f"{s['judge_correct']:,} & {four(s['judge_correct'], s['usable'])} & "
                f"{four(s['judge_correct'], s['records'])} \\\\"
            )
        if arm != ARMS[-1]:
            strata_lines.append("\\addlinespace")
    rows_strata = "\n".join(strata_lines)
    rows_coverage = "\n".join(
        f"\\texttt{{{arm.replace('_', chr(92) + '_')}}} & {c['empty_final']:,} & "
        f"{c['ambiguous']:,} & {c['empty_final'] + c['ambiguous']:,} & "
        f"{c['tolerant_recovered']:,} & {c['usable_final_n']:,} \\\\"
        for arm, c in ((a, counts[a]) for a in ARMS)
    )
    (TABLES / "pred_a_coverage.tex").write_text(
        "\\begin{table}[ht]\n\\centering\n"
        "\\caption[Predicted-answer eligibility]{Where the 2,653 records went in the\n"
        "secondary predicted-answer regime. Empty final counts the records whose stored\n"
        "output held no final span; those records still carry reasoning text. Ambiguous\n"
        "counts the records whose only candidate final span came from text inside the\n"
        "reasoning field, which the rule against imputation does not admit, and which is\n"
        "why the ineligible total can exceed the empty finals. Ineligible is empty final\n"
        "plus ambiguous. For each arm, ineligible and\n"
        "usable sum to 2,653. Tolerant recovery counts the usable spans that needed the\n"
        "tolerant parser, so it is a subset of usable and is not added to it.}\n"
        "\\label{tab:app-pred-a-coverage}\n"
        "\\begin{tabular}{lrrrrr}\n\\toprule\n"
        "Arm & Empty final & Ambiguous & Ineligible & Tolerant recovery & Usable \\\\\n"
        "\\midrule\n" + rows_coverage + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n",
        encoding="utf-8",
    )

    for arm in ARMS:
        c = counts[arm]
        checks.append({
            "check": f"{arm}: eligibility categories account for every record",
            "status": "PASS" if c["empty_final"] + c["ambiguous"] + c["usable_final_n"] == TOTAL else "FAIL",
            "evidence": {"empty_final": c["empty_final"], "ambiguous": c["ambiguous"],
                         "usable": c["usable_final_n"], "total": TOTAL},
        })

    (TABLES / "pred_a_strata.tex").write_text(
        "\\begin{table}[ht]\n\\centering\n"
        "\\caption[Predicted-answer CMC by Stage~1 correctness]{CMC in the secondary\n"
        "predicted-answer regime, split by whether the supplied Stage~1 prediction was\n"
        "correct. Correct and wrong refer to the stored epoch~3 prediction set that\n"
        "conditioned these rationales; they are not the correct and wrong split of the\n"
        "epoch~2 RQ1 source predictions in Table~\\ref{tab:rq1-battery}. $N$ is the\n"
        "records in the stratum, usable the ones with a recoverable final span, and\n"
        "answer recovered the ones whose supplied answer the judge recovered. Conditional\n"
        "divides that count by usable within the stratum; pipeline divides it by $N$.}\n"
        "\\label{tab:app-pred-a-strata}\n"
        "\\begin{tabular}{llrrrrr}\n\\toprule\n"
        "Arm & Stage~1 & $N$ & Usable & Answer recovered & Conditional & Pipeline \\\\\n"
        "\\midrule\n" + rows_strata + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n",
        encoding="utf-8",
    )

    # Every integer the two tables print must be one this script recovered.
    printed = set()
    for name in ("pred_a_compact.tex", "pred_a_strata.tex"):
        body = (TABLES / name).read_text(encoding="utf-8").split("\\midrule", 1)[1]
        printed |= {int(v.replace(",", "")) for v in re.findall(r"(?<![.\d])(\d{1,3}(?:,\d{3})*)(?![.\d])", body)}
    recovered = {TOTAL}
    for c in counts.values():
        recovered |= {c["records"], c["usable_final_n"], c["judge_correct_n"],
                      c["stage1_correct_n"], c["stage1_wrong_n"]}
        for s in c["strata"].values():
            recovered |= {s["records"], s["usable"], s["judge_correct"]}
    unaccounted = sorted(printed - recovered)
    checks.append({
        "check": "every integer printed in the two PRED-A tables is a count this audit recovered",
        "status": "PASS" if not unaccounted else "FAIL",
        "evidence": {"unaccounted": unaccounted, "recovered": sorted(recovered)},
    })

    # A stable, timestamp-free record the value ledger can hash.
    prov = ROOT / "provenance" / "PRED_A_CMC_COUNTS.tsv"
    prov.parent.mkdir(parents=True, exist_ok=True)
    lines = ["arm\tstratum\trecords\tusable_final\tjudge_correct\tconditional_cmc\tpipeline_cmc"]
    for arm in ARMS:
        c = counts[arm]
        lines.append(f"{arm}\tall\t{c['records']}\t{c['usable_final_n']}\t{c['judge_correct_n']}\t"
                     f"{four(c['judge_correct_n'], c['usable_final_n'])}\t{four(c['judge_correct_n'], c['records'])}")
        for label in ("correct", "wrong"):
            s = c["strata"][label]
            lines.append(f"{arm}\tstage1_{label}\t{s['records']}\t{s['usable']}\t{s['judge_correct']}\t"
                         f"{four(s['judge_correct'], s['usable'])}\t{four(s['judge_correct'], s['records'])}")
    prov.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Read the tables back and check every printed rate against the integers
    # printed beside it, so a rate can never drift from its own numerator.
    printed_failures = []
    printed_rates = 0
    for name in ("pred_a_compact", "pred_a_strata"):
        for line in (TABLES / f"{name}.tex").read_text(encoding="utf-8").splitlines():
            if "&" not in line or "\\midrule" in line or "toprule" in line:
                continue
            cells = [c.strip() for c in line.replace("\\\\", "").split("&")]
            integers = [int(c.replace(",", "")) for c in cells
                        if re.fullmatch(r"[0-9,]+", c)]
            rates = [c for c in cells if re.fullmatch(r"0\.[0-9]{4}", c)]
            if len(rates) != 2 or len(integers) < 2:
                continue
            if name == "pred_a_compact":
                usable, judge_ok = integers[0], integers[1]
                denominators = (usable, TOTAL)
            else:
                records, usable, judge_ok = integers[0], integers[1], integers[2]
                denominators = (usable, records)
            for rate, denominator in zip(rates, denominators):
                printed_rates += 1
                if rate != four(judge_ok, denominator):
                    printed_failures.append({"table": name, "printed": rate,
                                             "from_integers": four(judge_ok, denominator),
                                             "line": line.strip()})
    checks.append({
        "check": "every printed rate equals the ratio of the integers printed beside it",
        "status": "PASS" if not printed_failures and printed_rates == 24 else "FAIL",
        "evidence": {"rates_checked": printed_rates, "mismatches": printed_failures},
    })

    audit = {
        "schema": "v11-pred-a-cmc-arithmetic-audit-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "method": ("Counted already stored per-record judge decisions. No judge inference, no "
                   "rescoring, and no model was loaded."),
        "per_record_judge_source": str(JUDGE_RECORDS),
        "full_precision_lineage_source": str(JUDGE_SUMMARY),
        "frozen_manifest": str(FROZEN.relative_to(ROOT)),
        "note": ("The frozen manifest stores plain and plain_desc at full precision and point and "
                 "point_desc rounded to three decimals. The recovered integers reproduce the "
                 "full-precision lineage values for all four arms, so the displayed four-decimal "
                 "rates are now derived from counts rather than from a rounded aggregate."),
        "counts": counts,
        "stable_counts_table": str(prov.relative_to(ROOT)),
        "checks": checks,
    }
    audit["status"] = "PASS" if all(c["status"] == "PASS" for c in checks) else "FAIL"
    (AUDIT_DIR / "PRED_A_CMC_ARITHMETIC_AUDIT.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": audit["status"],
                      "failed": [c["check"] for c in checks if c["status"] != "PASS"],
                      "counts": {a: (counts[a]["judge_correct_n"], counts[a]["usable_final_n"]) for a in ARMS}},
                     indent=2))
    if audit["status"] != "PASS":
        raise SystemExit("PRED-A CMC ARITHMETIC AUDIT FAILED")


if __name__ == "__main__":
    main()
