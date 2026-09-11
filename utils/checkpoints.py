"""Single source of truth for canonical per-arm model checkpoints.

WHY THIS EXISTS (root-cause fix): checkpoint paths used to be hardcoded and
duplicated across several eval scripts. That let a real bug slip in: the
`plain_desc` REPLICATION Stage-1 (seed 4, 11% train,
`reports/runs/plain_desc/stage1/20260518_110519` -- the t2' run reported in
EXPERIMENTAL_FINDINGS S12 / Chapter 8 S8.2.1) was wired into the full-val
sufficiency (S14) and the regime battery as if it were the canonical
thesis-scale checkpoint, silently changing plain_desc's Sufficiency Gap from the
headline 0.221 to 0.195/0.199. Every script MUST import the canonical map from
here so two scripts can never disagree about which checkpoint an arm uses.

CANONICAL = the seed-42 thesis run that produced the headline Table 6.4 AND that
the canonical Stage-2 gold warm-started from (verified arm-by-arm against the
Stage-2 gold manifests' `stage1_ckpt`).
"""

from __future__ import annotations
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# arm -> canonical Stage-1 run dir (relative to PROJECT_ROOT)
CANONICAL_STAGE1 = {
    # plain repointed 2026-07-05 to the from-scratch retrain (scratch_v1, 3 epochs, seed 42,
    # no warm-start) — removes the ~5-vs-3-epoch asymmetry. The old warm-started run
    # (20260425_101006) is archived at reports/_archive/plain_pre_retrain_20260701/. See §21.
    "plain":      "reports/runs/plain/stage1/scratch_v1",
    "point":      "reports/runs/point/stage1/20260429_234801",
    "plain_desc": "reports/runs/plain_desc/stage1/20260507_203440",
    "point_desc": "reports/runs/point_desc/stage1/20260511_211533",
}

# arm -> canonical Stage-2 gold run dir (the one that warm-started from CANONICAL_STAGE1)
CANONICAL_STAGE2_GOLD = {
    # plain repointed 2026-07-05 to the from-scratch retrain Stage-2 gold (scratch_v1,
    # warm-started from the NEW scratch_v1 Stage-1). Old run 20260503_110600 archived. See §21.
    "plain":      "reports/runs/plain/stage2_gold/scratch_v1",
    "point":      "reports/runs/point/stage2_gold/20260505_141209",
    "plain_desc": "reports/runs/plain_desc/stage2_gold/20260509_174550",
    "point_desc": "reports/runs/point_desc/stage2_gold/20260513_183118",
}

# NON-canonical checkpoints that must NEVER be used as an arm's thesis-scale
# checkpoint. Kept here as an explicit denylist so the mistake is documented and
# greppable. These are legitimate experiments in their own right (do not delete).
NONCANONICAL_DO_NOT_USE_AS_ARM = {
    # plain_desc 2.2x replication (t2'): seed 4, 11% train. Reported as t2' only.
    "plain_desc_replication_t2prime": "reports/runs/plain_desc/stage1/20260518_110519",
}


def stage1_run(arm: str) -> Path:
    """Absolute path to the canonical Stage-1 run dir for `arm` (asserts it exists)."""
    if arm not in CANONICAL_STAGE1:
        raise KeyError(f"unknown arm {arm!r}; known: {sorted(CANONICAL_STAGE1)}")
    p = PROJECT_ROOT / CANONICAL_STAGE1[arm]
    if not (p / "checkpoints" / "model_best.pt").exists():
        raise FileNotFoundError(f"canonical Stage-1 ckpt missing for {arm}: {p}")
    return p


def stage1_ckpt(arm: str) -> Path:
    """Absolute path to the canonical Stage-1 model_best.pt for `arm`."""
    return stage1_run(arm) / "checkpoints" / "model_best.pt"


# Back-compat alias: scripts previously defined a local STAGE1_RUNS dict.
STAGE1_RUNS = CANONICAL_STAGE1
