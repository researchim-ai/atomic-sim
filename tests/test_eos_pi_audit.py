"""Synthetic PI trace matrices exercise auditing, not physical validation."""

import pytest

from scripts.eos_pi_references import cases
from scripts.qualify_eos_pi import CAPS, audit, capture


def report():
    rows = []
    for name, spec in cases():
        p = spec["policy"]
        trace = [
            dict(
                time_s=(k + 1) * p["sample_period_s"],
                truth_k=300.0,
                filtered_k=300.0,
                integral_before_w=0.0,
                integral_w=0.0,
                output_before_w=0.0,
                output_w=0.0,
                integral_held=False,
            )
            for k in range(10)
        ]
        for cap in CAPS:
            rows.append(
                dict(
                    case=name,
                    max_step_s=cap,
                    observations=10,
                    errors=dict.fromkeys(
                        [
                            "temperature_k",
                            "filtered_k",
                            "integral_w",
                            "output_w",
                            "external_heat_j",
                        ],
                        0.0,
                    ),
                    energy_residual_j=0.0,
                    entropy_residual_j_k=0.0,
                    observed_step_s=cap,
                    stability_fraction=0.1,
                    restart_exact=True,
                    controller_trace=[r.copy() for r in trace],
                )
            )
    return dict(
        schema_version=1,
        signature=capture(),
        cases=8,
        trajectories=24,
        independent_programs=16,
        independent_ivp_segments=160,
        observations=240,
        horizon_s=0.1,
        validated=False,
        plant_referenced=False,
        rows=rows,
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
        "error",
        "trace_length",
        "trace_clock",
        "power",
        "slew",
        "held_integral",
        "continuity",
        "bool",
        "balance_nan",
    ],
)
def test_corrupt_report_rejected(defect):
    r = report()
    row = r["rows"][0]
    sample = row["controller_trace"][0]
    if defect == "duplicate":
        r["rows"][1] = row.copy()
    elif defect == "coverage":
        r["rows"].pop()
    elif defect == "nan":
        row["errors"]["filtered_k"] = float("nan")
    elif defect == "source":
        r["signature"] = {}
    elif defect == "claim":
        r["validated"] = True
    elif defect == "step":
        row["observed_step_s"] = 0.0005
    elif defect == "restart":
        row["restart_exact"] = False
    elif defect == "error":
        row["errors"]["output_w"] = 1.0
    elif defect == "trace_length":
        row["controller_trace"].pop()
    elif defect == "trace_clock":
        sample["time_s"] = 0.0
    elif defect == "power":
        sample["output_w"] = 1e9
    elif defect == "slew":
        target = next(row for row in r["rows"] if row["case"] == "slew")
        target["controller_trace"][0]["output_w"] = 1000.0
    elif defect == "held_integral":
        sample.update(integral_held=True, integral_w=1.0)
    elif defect == "continuity":
        sample["output_before_w"] = 1.0
    elif defect == "bool":
        r["schema_version"] = True
    elif defect == "balance_nan":
        row["entropy_residual_j_k"] = float("nan")
    with pytest.raises(ValueError):
        audit(r)
