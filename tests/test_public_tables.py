from pathlib import Path
from scripts.build_public_tables import validate_tables, render_tables, read, ARMS


def test_all_active_tables_match_approved_bytes():
    checks = validate_tables()
    assert len(checks) == 12
    assert all(r["matches_approved_bytes"] for r in checks)


def test_pred_a_integer_rates_and_strata():
    rows = read("provenance/PRED_A_CMC_COUNTS.tsv")
    by = {(r["arm"], r["stratum"]): r for r in rows}
    for arm in ARMS:
        total = by[(arm, "all")]
        assert int(total["records"]) == 2653
        for key in ["records", "usable_final", "judge_correct"]:
            assert int(total[key]) == sum(int(by[(arm, s)][key]) for s in ["stage1_correct", "stage1_wrong"])
        assert f"{int(total['judge_correct']) / int(total['usable_final']):.4f}" == total["conditional_cmc"]
        assert f"{int(total['judge_correct']) / int(total['records']):.4f}" == total["pipeline_cmc"]


def test_frozen_stage1_counts_and_transition_identity():
    cells = read("reports/frozen/rq1_final/data/RQ1_CELL_RESULTS.tsv")
    assert len(cells) == 24
    for r in cells:
        assert int(r["correct_count"]) + int(r["wrong_count"]) == 2653
        assert abs(int(r["correct_count"]) / 2653 - float(r["accuracy"])) < 1e-11
    for r in read("reports/frozen/rq1_final/data/RQ1_TRANSITION_RESULTS.tsv"):
        assert sum(int(r[k]) for k in ["C_to_C", "C_to_W", "W_to_C", "W_to_W"]) == 2653
        assert int(r["churn_count"]) == int(r["C_to_W"]) + int(r["W_to_C"])
        assert int(r["net_movement_W_to_C_minus_C_to_W"]) == int(r["W_to_C"]) - int(r["C_to_W"])


def test_no_templates_contain_numeric_data_rows():
    for name, rendered in render_tables().items():
        template = (Path(__file__).resolve().parents[1] / "scripts/table_templates" / name).read_text()
        assert "@@ROW_" in template
        assert "@@ROW_" not in rendered
