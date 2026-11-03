"""Adversarial report parser tests; synthetic fixtures are not physical evidence."""

from copy import deepcopy

import pytest

from scripts.audit_conservative_loop import FIELDS, audit, provenance
from scripts.conservative_eos_loop import signature
from scripts.conservative_loop_references import cases


@pytest.fixture
def report():
    rows = []
    for name, _, _ in cases():
        for cap in [0.002, 0.001, 0.0005]:
            rows.append(
                dict(
                    case=name,
                    max_step_s=cap,
                    observations=7,
                    restart_exact=True,
                    observed_internal_step_s=cap,
                    maximum_courant=0.1,
                    max_errors=dict.fromkeys(FIELDS, 0.0),
                    max_balance_residuals=dict.fromkeys(
                        [
                            "mass_residual_kg",
                            "momentum_residual_kg_m_s",
                            "energy_residual_j",
                            "entropy_residual_j_k",
                        ],
                        0.0,
                    ),
                    max_ledger_errors=[0.0] * 6,
                )
            )
    return dict(
        schema_version=1,
        case_count=9,
        trajectory_count=27,
        independent_ivps=18,
        observations=189,
        horizon_s=2.0,
        validated=False,
        plant_referenced=False,
        signature=signature(),
        provenance=provenance(),
        rows=rows,
    )


def test_audit_is_pure(report):
    before = deepcopy(report)
    assert audit(report)["status"] == "passed"
    assert report == before


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0, True, None, "0"])
def test_nonfinite_negative_and_non_numeric_metrics(report, value):
    report["rows"][0]["max_errors"]["masses_kg"] = value
    with pytest.raises(ValueError):
        audit(report)


@pytest.mark.parametrize(
    "mutation",
    [
        "signature",
        "provenance",
        "missing_row",
        "duplicate_row",
        "count",
        "validation",
        "restart",
        "cfl",
        "measured_step",
        "identical_mesh",
        "reference_error",
        "energy_ledger",
        "entropy_ledger",
        "missing_balance",
        "nan_balance",
        "nan_impulse",
        "ledger_shape",
        "bool_count",
        "unknown_case",
    ],
)
def test_rejects_corrupt_or_unqualified_report(report, mutation):
    row = report["rows"][0]
    if mutation == "signature":
        report["signature"] = {}
    elif mutation == "provenance":
        report["provenance"] = {}
    elif mutation == "missing_row":
        report["rows"].pop()
    elif mutation == "duplicate_row":
        report["rows"][1] = deepcopy(row)
    elif mutation == "count":
        report["observations"] = 188
    elif mutation == "validation":
        report["validated"] = True
    elif mutation == "restart":
        row["restart_exact"] = False
    elif mutation == "cfl":
        row["maximum_courant"] = 0.36
    elif mutation == "measured_step":
        row["observed_internal_step_s"] = 0.003
    elif mutation == "identical_mesh":
        row["observed_internal_step_s"] = 0.001
    elif mutation == "reference_error":
        row["max_errors"]["pressures_pa"] = 1.0
    elif mutation == "energy_ledger":
        row["max_ledger_errors"][0] = 1.0
    elif mutation == "entropy_ledger":
        row["max_ledger_errors"][2] = 1.0
    elif mutation == "missing_balance":
        row["max_balance_residuals"].pop("mass_residual_kg")
    elif mutation == "nan_balance":
        row["max_balance_residuals"]["mass_residual_kg"] = float("nan")
    elif mutation == "nan_impulse":
        row["max_ledger_errors"][5] = float("nan")
    elif mutation == "ledger_shape":
        row["max_ledger_errors"].pop()
    elif mutation == "bool_count":
        report["schema_version"] = True
    elif mutation == "unknown_case":
        row["case"] = "unknown"
    with pytest.raises(ValueError):
        audit(report)
