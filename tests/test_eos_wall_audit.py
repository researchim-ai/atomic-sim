"""Synthetic report parsing tests, not numerical validation evidence."""

import pytest

from scripts.eos_wall_references import cases
from scripts.qualify_eos_wall import CAPS, FIELDS, audit, capture


def report():
    return dict(
        schema_version=1,
        signature=capture(),
        cases=7,
        trajectories=21,
        independent_programs=14,
        independent_ivp_segments=70,
        observations=105,
        horizon_s=0.2,
        validated=False,
        plant_referenced=False,
        rows=[
            dict(
                case=name,
                max_step_s=cap,
                errors=dict.fromkeys(FIELDS, 0.0),
                wall_temperature_error_k=0.0,
                energy_residual_j=0.0,
                entropy_residual_j_k=0.0,
                observed_step_s=cap,
                stability_fraction=0.1,
                restart_exact=True,
                observations=5,
            )
            for name, _ in cases()
            for cap in CAPS
        ],
    )


def test_complete_synthetic_matrix():
    assert audit(report())["passed"]


@pytest.mark.parametrize(
    "defect",
    ["duplicate", "nan", "source", "claim", "step", "restart", "wall", "cfl", "bool", "coverage"],
)
def test_corrupt_reports(defect):
    r = report()
    row = r["rows"][0]
    if defect == "duplicate":
        r["rows"][1] = row.copy()
    elif defect == "nan":
        row["errors"]["masses_kg"] = float("nan")
    elif defect == "source":
        r["signature"] = {}
    elif defect == "claim":
        r["validated"] = True
    elif defect == "step":
        row["observed_step_s"] = 0.0005
    elif defect == "restart":
        row["restart_exact"] = False
    elif defect == "wall":
        row["wall_temperature_error_k"] = 1.0
    elif defect == "cfl":
        row["stability_fraction"] = 1.1
    elif defect == "bool":
        r["schema_version"] = True
    elif defect == "coverage":
        r["rows"].pop()
    with pytest.raises(ValueError):
        audit(r)
