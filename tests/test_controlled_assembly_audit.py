"""Synthetic audit matrices; these fixtures are not physical evidence."""

import pytest

from scripts.controlled_assembly_references import cases, initial_controls
from scripts.inertial_graph import digest
from scripts.qualify_controlled_assembly import CAPS, CHAMBERS, FIELDS, audit, capture


def report():
    return dict(
        schema_version=1,
        signature=capture(),
        cases=8,
        trajectories=24,
        independent_programs=16,
        independent_ivp_segments=80,
        observations=120,
        horizon_s=0.1,
        validated=False,
        plant_referenced=False,
        rows=[
            dict(
                case=name,
                max_step_s=cap,
                observations=5,
                channel_errors=dict.fromkeys(FIELDS, 0.0),
                chamber_errors=dict.fromkeys(CHAMBERS, 0.0),
                wall_error_k=0.0,
                max_wall_ledger_errors=[0.0] * 5,
                energy_residual_j=0.0,
                entropy_residual_j_k=0.0,
                observed_step_s=cap,
                stability_fraction=0.1,
                restart_exact=True,
                commands_committed=len(schedule),
                end_control_sha256=digest(schedule[-1][1] if schedule else initial_controls(spec)),
            )
            for name, spec, schedule in cases()
            for cap in CAPS
        ],
    )


def test_complete_synthetic_matrix():
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
        "wall",
        "ledger_shape",
        "ledger_energy",
        "ledger_entropy",
        "cfl",
        "bool",
        "balance_nan",
        "command_count",
        "command_mask",
    ],
)
def test_corrupt_report_rejected(defect):
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
        row["channel_errors"]["pressures_pa"] = 1.0
    elif defect == "wall":
        row["wall_error_k"] = 1.0
    elif defect == "ledger_shape":
        row["max_wall_ledger_errors"].pop()
    elif defect == "ledger_energy":
        row["max_wall_ledger_errors"][0] = 1.0
    elif defect == "ledger_entropy":
        row["max_wall_ledger_errors"][-1] = 1.0
    elif defect == "cfl":
        row["stability_fraction"] = 1.1
    elif defect == "bool":
        r["schema_version"] = True
    elif defect == "balance_nan":
        row["energy_residual_j"] = float("nan")
    elif defect == "command_count":
        next(row for row in r["rows"] if row["case"] == "online_command")["commands_committed"] = 0
    elif defect == "command_mask":
        next(row for row in r["rows"] if row["case"] == "online_command")["end_control_sha256"] = (
            "bad"
        )
    with pytest.raises(ValueError):
        audit(r)
