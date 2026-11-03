"""Synthetic evidence parser fixtures are not physical qualification results."""

import pytest

from scripts.open_channel_references import cases
from scripts.qualify_open_channel import CAPS, FIELDS, audit, capture


def report():
    return dict(
        schema_version=1,
        signature=capture(),
        cases=12,
        trajectories=36,
        independent_programs=24,
        independent_ivp_segments=120,
        observations=180,
        horizon_s=0.2,
        validated=False,
        plant_referenced=False,
        rows=[
            dict(
                case=name,
                max_step_s=cap,
                observations=5,
                errors=dict.fromkeys(FIELDS, 0.0),
                max_ledger_errors=[0.0] * 14,
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
                maximum_courant=0.1,
                restart_exact=True,
                final_mean_velocity_m_s=-1.0 if name == "opposed_reversal" else 0.0,
            )
            for name, _, _ in cases()
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
        "ledger_shape",
        "boundary_mass",
        "boundary_energy",
        "boundary_entropy",
        "cfl",
        "bool",
        "reverse",
        "balance_nan",
        "negative_error",
    ],
)
def test_corrupt_reports_rejected(defect):
    r = report()
    row = r["rows"][0]
    if defect == "duplicate":
        r["rows"][1] = row.copy()
    elif defect == "coverage":
        r["rows"].pop()
    elif defect == "nan":
        row["errors"]["pressures_pa"] = float("nan")
    elif defect == "source":
        r["signature"] = {}
    elif defect == "claim":
        r["plant_referenced"] = True
    elif defect == "step":
        row["observed_step_s"] = 0.0005
    elif defect == "restart":
        row["restart_exact"] = False
    elif defect == "ledger_shape":
        row["max_ledger_errors"].pop()
    elif defect == "boundary_mass":
        row["max_ledger_errors"][6] = 1.0
    elif defect == "boundary_energy":
        row["max_ledger_errors"][8] = 1.0
    elif defect == "boundary_entropy":
        row["max_ledger_errors"][13] = 1.0
    elif defect == "cfl":
        row["maximum_courant"] = 0.36
    elif defect == "bool":
        r["schema_version"] = True
    elif defect == "reverse":
        next(row for row in r["rows"] if row["case"] == "opposed_reversal")[
            "final_mean_velocity_m_s"
        ] = 1.0
    elif defect == "balance_nan":
        row["max_balance_residuals"]["mass_residual_kg"] = float("nan")
    elif defect == "negative_error":
        row["errors"]["temperatures_k"] = -1.0
    with pytest.raises(ValueError):
        audit(r)
