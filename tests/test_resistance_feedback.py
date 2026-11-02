"""Analytic feedback, fixed commands, reverse flow and constitutive refusal."""

import json
import math
import sys
from copy import deepcopy

import pytest

from core.network import NetworkError
from scripts.solve_resistance_feedback import (
    ResistanceFeedbackError,
    main,
    solve_resistance_feedback,
)


def example(beta=0.01):
    steady = dict(
        format="atomic-thermal-hydraulic-steady",
        schema_version=1,
        hydraulic=dict(
            format="atomic-hydraulic-steady",
            schema_version=1,
            density_kg_m3=1000,
            gravity_m_s2=9.81,
            nodes=[
                dict(id="supply", elevation_m=0, pressure_pa=200000),
                dict(id="outlet", elevation_m=0, pressure_pa=199000),
            ],
            edges=[
                dict(
                    id="pipe",
                    **{"from": "supply", "to": "outlet"},
                    linear_pa_s_kg=1000,
                    quadratic_pa_s2_kg2=0,
                    pressure_gain_pa=0,
                )
            ],
        ),
        thermal=dict(
            medium="test-liquid",
            heat_capacity_j_kg_k=1000,
            enthalpy_reference_temperature_k=300,
            min_temperature_k=250,
            max_temperature_k=500,
            nodes={
                k: dict(
                    heat_power_w=0,
                    ambient_ua_w_k=0,
                    ambient_temperature_k=300,
                    supply_temperature_k=300,
                )
                for k in ("supply", "outlet")
            },
        ),
    )
    return dict(
        format="atomic-resistance-feedback-steady",
        schema_version=1,
        steady_input=steady,
        resistance_laws=dict(
            pipe=dict(
                kind="exponential_endpoint_mean_v1",
                reference_temperature_k=300,
                temperature_coefficient_per_k=beta,
                minimum_multiplier=0.1,
                maximum_multiplier=10,
            )
        ),
        initial_temperatures_k=dict(supply=320, outlet=330),
    )


@pytest.mark.parametrize("beta", [-0.01, 0.0, 0.01])
def test_analytic_pressure_driven_pipe_solves_feedback_and_preserves_input(beta):
    spec = example(beta)
    before = deepcopy(spec)
    result = solve_resistance_feedback(spec)
    point = result["solution"]
    assert spec == before
    assert point["temperatures_k"] == pytest.approx(dict(supply=300, outlet=300.001), abs=1e-9)
    assert point["hydraulic"]["flows_kg_s"]["pipe"] == pytest.approx(
        math.exp(-beta * 0.0005), abs=1e-10
    )
    assert result["metrics"]["constitutive_pressure_residual_pa"] <= 1e-6
    assert result["metrics"]["relative_resistance_error"] <= 1e-12
    assert not result["global_uniqueness_proven"] and not result["commands_changed"]


def test_hotter_supply_changes_flow_not_just_reported_resistance():
    spec = example()
    cold = solve_resistance_feedback(spec)
    spec["steady_input"]["thermal"]["nodes"]["supply"]["supply_temperature_k"] = 350
    hot = solve_resistance_feedback(spec)
    assert hot["solution"]["hydraulic"]["flows_kg_s"]["pipe"] / cold["solution"]["hydraulic"][
        "flows_kg_s"
    ]["pipe"] == pytest.approx(math.exp(-0.5), abs=1e-10)
    assert hot["solution"]["temperatures_k"]["outlet"] == pytest.approx(350.001, abs=1e-9)


def test_reverse_flow_uses_actual_supply_and_same_endpoint_law():
    spec = example()
    spec["steady_input"]["hydraulic"]["nodes"][1]["pressure_pa"] = 201000
    spec["steady_input"]["thermal"]["nodes"]["outlet"]["supply_temperature_k"] = 350
    result = solve_resistance_feedback(spec)
    assert result["solution"]["hydraulic"]["flows_kg_s"]["pipe"] == pytest.approx(
        -math.exp(-0.500005), abs=1e-10
    )
    assert result["solution"]["temperatures_k"]["supply"] == pytest.approx(350.001, abs=1e-9)


def test_quadratic_loss_stays_fixed_and_energy_is_conservative():
    spec = example()
    spec["steady_input"]["hydraulic"]["edges"][0]["quadratic_pa_s2_kg2"] = 100
    result = solve_resistance_feedback(spec)
    edge = result["solution"]["hydraulic"]["input"]["edges"][0]
    assert edge["quadratic_pa_s2_kg2"] == 100
    assert result["solution"]["residuals"]["maximum_global_energy_residual_w"] <= 1e-5


def test_initial_guesses_change_iterations_not_accepted_physics():
    spec = example()
    first = solve_resistance_feedback(spec)
    spec["initial_temperatures_k"] = dict(supply=400, outlet=400)
    second = solve_resistance_feedback(spec)
    assert first["solution"]["temperatures_k"] == pytest.approx(
        second["solution"]["temperatures_k"], abs=1e-9
    )
    assert first["solution"]["hydraulic"]["flows_kg_s"] == pytest.approx(
        second["solution"]["hydraulic"]["flows_kg_s"], abs=1e-10
    )


def test_law_domain_refuses_extrapolation_instead_of_clipping():
    spec = example()
    spec["resistance_laws"]["pipe"]["maximum_multiplier"] = 1.1
    with pytest.raises(ResistanceFeedbackError) as error:
        solve_resistance_feedback(spec)
    assert error.value.report["criterion"] == "resistance_law_domain"


def test_zero_exchange_pressure_reservoir_is_not_silently_retyped_as_junction():
    from scripts.resistance_feedback_references import cases
    from scripts.solve_thermal_graph import ThermalSolveError

    _, spec, _ = next(c for c in cases() if c[0] == "two_supplies_negative")
    mixer = next(n for n in spec["steady_input"]["hydraulic"]["nodes"] if n["id"] == "mix")
    mixer.pop("injection_kg_s")
    mixer["pressure_pa"] = 200000
    before = deepcopy(spec)
    with pytest.raises(ThermalSolveError) as error:
        solve_resistance_feedback(spec)
    assert error.value.report["criterion"] == "unresolved_external_flow"
    assert spec == before


def test_exponent_overflow_is_a_typed_law_refusal():
    spec = example()
    spec["resistance_laws"]["pipe"]["temperature_coefficient_per_k"] = 1e308
    with pytest.raises(ResistanceFeedbackError) as error:
        solve_resistance_feedback(spec)
    assert error.value.report["criterion"] == "resistance_law_domain"


def test_orientation_change_preserves_physics_and_law_multiplier():
    spec = example()
    first = solve_resistance_feedback(spec)
    edge = spec["steady_input"]["hydraulic"]["edges"][0]
    edge["from"], edge["to"] = edge["to"], edge["from"]
    second = solve_resistance_feedback(spec)
    assert second["solution"]["temperatures_k"] == pytest.approx(
        first["solution"]["temperatures_k"], abs=1e-9
    )
    assert second["multipliers"] == pytest.approx(first["multipliers"], abs=1e-12)
    assert second["solution"]["hydraulic"]["flows_kg_s"]["pipe"] == pytest.approx(
        -first["solution"]["hydraulic"]["flows_kg_s"]["pipe"], abs=1e-10
    )


def test_source_identity_change_prevents_acceptance(monkeypatch):
    from scripts import solve_resistance_feedback as module

    original = module.signature()
    count = 0

    def changed():
        nonlocal count
        count += 1
        result = deepcopy(original)
        if count > 1:
            result["python"] = "changed"
        return result

    monkeypatch.setattr(module, "signature", changed)
    with pytest.raises(ResistanceFeedbackError) as error:
        module.solve_resistance_feedback(example())
    assert error.value.report["criterion"] == "source_drift"


@pytest.mark.parametrize(
    "damage",
    [
        "schema",
        "format",
        "missing_law",
        "extra_law",
        "unknown_law",
        "unknown_field",
        "nan",
        "bounds",
        "guess",
        "guess_map",
    ],
)
def test_invalid_model_contract(damage):
    spec = example()
    law = spec["resistance_laws"]["pipe"]
    if damage == "schema":
        spec["schema_version"] = True
    elif damage == "format":
        spec["format"] = "water-viscosity"
    elif damage == "missing_law":
        spec["resistance_laws"] = {}
    elif damage == "extra_law":
        spec["resistance_laws"]["unknown"] = dict(law)
    elif damage == "unknown_law":
        law["kind"] = "auto-water"
    elif damage == "unknown_field":
        law["viscosity"] = 1.0
    elif damage == "nan":
        law["temperature_coefficient_per_k"] = float("nan")
    elif damage == "bounds":
        law["minimum_multiplier"] = 11
    elif damage == "guess":
        spec["initial_temperatures_k"]["supply"] = 1000
    elif damage == "guess_map":
        spec["initial_temperatures_k"].pop("outlet")
    with pytest.raises(NetworkError):
        solve_resistance_feedback(spec)


def test_iteration_limit_returns_no_accepted_point(tmp_path, monkeypatch):
    from scripts import solve_resistance_feedback as module

    budget = json.loads(module.BUDGET.read_text())
    budget["maximum_iterations"] = 1
    path = tmp_path / "budget.json"
    path.write_text(json.dumps(budget))
    monkeypatch.setattr(module, "BUDGET", path)
    with pytest.raises(ResistanceFeedbackError) as error:
        solve_resistance_feedback(example())
    assert error.value.report["criterion"] == "coupling_iteration_limit"


def test_energy_admission_is_rechecked_at_coupled_point(tmp_path, monkeypatch):
    from scripts import solve_resistance_feedback as module

    original = module.solve_thermal_graph

    def damaged(value):
        point = original(value)
        point["node_energy_residuals_w"]["outlet"] += 0.001
        return point

    monkeypatch.setattr(module, "solve_thermal_graph", damaged)
    budget = json.loads(module.BUDGET.read_text())
    budget["maximum_iterations"] = 2
    path = tmp_path / "budget.json"
    path.write_text(json.dumps(budget))
    monkeypatch.setattr(module, "BUDGET", path)
    spec = example(0.0)
    spec["initial_temperatures_k"] = dict(supply=300.0, outlet=300.001)
    with pytest.raises(ResistanceFeedbackError) as error:
        module.solve_resistance_feedback(spec)
    assert error.value.report["criterion"] == "coupling_iteration_limit"
    assert (
        error.value.report["details"]["last_metrics"]["constitutive_node_energy_residual_w"] > 1e-5
    )


def test_cli_success_and_refuse_overwrite(tmp_path, monkeypatch):
    source = tmp_path / "input.json"
    source.write_text(json.dumps(example()))
    out = tmp_path / "result"
    monkeypatch.setattr(sys, "argv", ["solve", str(source), "--output", str(out)])
    assert main() == 0 and (out / "solution.json").is_file()
    assert set(json.loads((out / "manifest.json").read_text())["files"]) == {
        "request.json",
        "solution.json",
    }
    with pytest.raises(FileExistsError):
        main()


def test_cli_refusal_has_diagnostics_not_solution(tmp_path, monkeypatch):
    spec = example()
    spec["resistance_laws"]["pipe"]["maximum_multiplier"] = 1.1
    source = tmp_path / "input.json"
    source.write_text(json.dumps(spec))
    out = tmp_path / "result"
    monkeypatch.setattr(sys, "argv", ["solve", str(source), "--output", str(out)])
    assert main() == 2 and not (out / "solution.json").exists()
    assert (out / "diagnostics.json").is_file()
