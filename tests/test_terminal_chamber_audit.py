"""Synthetic report parser tests; no physical acceptance from fixtures."""

import pytest

from scripts.qualify_terminal_chambers import CAPS, CHAMBERS, FIELDS, audit, capture
from scripts.terminal_chamber_references import cases


def report():
    return dict(
        schema_version=1,
        signature=capture(),
        cases=8,
        trajectories=24,
        independent_programs=16,
        independent_ivp_segments=80,
        observations=120,
        horizon_s=0.2,
        validated=False,
        plant_referenced=False,
        rows=[
            dict(
                case=name,
                max_step_s=cap,
                observations=5,
                channel_errors=dict.fromkeys(FIELDS, 0.0),
                chamber_errors=dict.fromkeys(CHAMBERS, 0.0),
                max_ledger_errors=[0.0] * 4,
                max_balance_residuals=dict.fromkeys(
                    [
                        "mass_residual_kg",
                        "momentum_residual_kg_m_s",
                        "energy_residual_j",
                        "entropy_residual_j_k",
                    ],
                    0.0,
                ),
                observed_step_s=cap,
                stability_fraction=0.1,
                restart_exact=True,
            )
            for name, _ in cases()
            for cap in CAPS
        ],
    )


def test_complete_matrix():
    assert audit(report())["passed"]


@pytest.mark.parametrize(
    "defect",
    [
        "duplicate",
        "coverage",
        "nan",
        "source",
        "claim",
        "step",
        "restart",
        "pressure",
        "impulse",
        "entropy",
        "ledger_shape",
        "cfl",
        "bool",
        "balance_nan",
    ],
)
def test_invalid_reports(defect):
    r = report()
    row = r["rows"][0]
    if defect == "duplicate":
        r["rows"][1] = row.copy()
    elif defect == "coverage":
        r["rows"].pop()
    elif defect == "nan":
        row["chamber_errors"]["T"] = float("nan")
    elif defect == "source":
        r["signature"] = {}
    elif defect == "claim":
        r["validated"] = True
    elif defect == "step":
        row["observed_step_s"] = 0.0005
    elif defect == "restart":
        row["restart_exact"] = False
    elif defect == "pressure":
        row["chamber_errors"]["p"] = 1.0
    elif defect == "impulse":
        row["max_ledger_errors"][3] = 1.0
    elif defect == "entropy":
        row["max_ledger_errors"][2] = 1.0
    elif defect == "ledger_shape":
        row["max_ledger_errors"].pop()
    elif defect == "cfl":
        row["stability_fraction"] = 1.1
    elif defect == "bool":
        r["schema_version"] = True
    elif defect == "balance_nan":
        row["max_balance_residuals"]["energy_residual_j"] = float("nan")
    with pytest.raises(ValueError):
        audit(r)
