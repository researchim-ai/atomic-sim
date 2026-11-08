"""Actual HEM phase trajectories at fixed geometry; shared-EOS numerical references."""

import json
from copy import deepcopy

import pytest

from atomic_model_hem_qualified.cases import matrix, program
from atomic_model_hem_qualified.model import QualifiedHEMNetwork
from atomic_model_hem_qualified.qualification import CRITERIA, errors
from atomic_model_hem_qualified.reference import solve


@pytest.mark.parametrize(
    "name,phase",
    [
        ("vapor_phase_exit", "vapor"),
        ("liquid_phase_exit", "liquid"),
        ("condensation_entry", "two_phase"),
    ],
)
def test_three_step_caps_across_phase_transition(name, phase, record_testsuite_property):
    request = dict(matrix())[name]
    calls = program(name)
    references = [solve(request, calls, method) for method in ("DOP853", "Radau")]
    reference = references[0]["observations"]
    for left, right in zip(reference, references[1]["observations"], strict=True):
        for key, value in errors(left, right).items():
            assert value <= CRITERIA[key], (name, "reference disagreement", key, value)
    measured = []
    for cap in (4e-5, 2e-5, 1e-5):
        refined = deepcopy(request)
        refined["model"]["max_step_s"] = cap
        model = QualifiedHEMNetwork(refined)
        initial = model.snapshot()
        trajectory = [initial]
        for call in calls:
            trajectory.append(model.advance(call["duration_s"]))
        assert any(cell["phase"] == phase for cell in trajectory[-1]["cells"]), name
        assert [cell["phase"] for cell in initial["cells"]] != [
            cell["phase"] for cell in trajectory[-1]["cells"]
        ], (name, "no phase transition")
        maxima = {key: 0.0 for key in errors(initial, reference[0])}
        for actual, expected in zip(trajectory, reference, strict=True):
            for key, value in errors(actual, expected).items():
                maxima[key] = max(maxima[key], value)
                if cap <= request["model"]["max_step_s"]:
                    assert value <= CRITERIA[key], (name, cap, key, value)
            ledger = actual["global_ledger"]
            assert abs(ledger["mass_residual_kg"]) <= CRITERIA["mass_balance_kg"]
            assert abs(ledger["energy_residual_j"]) <= CRITERIA["energy_balance_j"]
            assert min(cell["mass_kg"] for cell in actual["cells"]) > 0
        measured.append(maxima)
    # Existing qualification pressure floor; do not infer an order below it.
    for coarse, fine in zip(measured, measured[1:]):
        assert fine["pressure_error_pa"] <= max(
            1e-4, CRITERIA["temporal_refinement_ratio"] * coarse["pressure_error_pa"]
        ), (name, measured)

    record_testsuite_property(
        name,
        json.dumps(
            dict(
                step_caps_s=[4e-5, 2e-5, 1e-5],
                maximum_errors=measured,
                accepted_step_caps_s=[2e-5, 1e-5],
                reference_methods=["DOP853", "Radau"],
                shared_eos=True,
            )
        ),
    )
