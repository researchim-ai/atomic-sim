"""Finite inventories, pure rates, interval admission and atomic restart."""

import json
import math
import sys
from copy import deepcopy

import pytest

from core.network import NetworkError
from scripts.thermal_graph_references import limiting_cases
from scripts.transient_thermal_graph import ThermalGraph, TransientGraphError, digest, main


def example(mode="zero_flow"):
    _, steady, _ = next(c for c in limiting_cases() if c[0] == mode)
    names = list(steady["thermal"]["nodes"])
    return dict(
        format="atomic-thermal-graph-transient",
        schema_version=1,
        steady_input=steady,
        volumes_m3={n: 1.0 for n in names},
        initial_temperatures_k={n: 310.0 for n in names},
        max_step_s=1.0,
    )


def test_cooling_matches_analytic_exponential_and_global_energy():
    spec = example()
    before = deepcopy(spec)
    model = ThermalGraph(spec)
    report = model.advance(120)
    expected = 300 + 10 * math.exp(-10 * 120 / 1e6)
    assert all(t == pytest.approx(expected, abs=1e-10) for t in report["temperatures_k"].values())
    assert abs(report["energy_residual_j"]) < 1e-4
    assert report["integrated_external_energy_j"] == pytest.approx(
        3 * 1e6 * (expected - 310), abs=1e-6
    )
    assert spec == before


def test_driven_loop_matches_analytic_approach_to_equilibrium():
    model = ThermalGraph(example("driven_cycle"))
    report = model.advance(120)
    expected = 300.9 + (310 - 300.9) * math.exp(-0.0012)
    assert list(report["temperatures_k"].values()) == pytest.approx([expected] * 3, abs=1e-9)


def test_closed_adiabatic_heated_loop_does_not_require_steady_temperature_gauge():
    spec = example("driven_cycle")
    for node in spec["steady_input"]["thermal"]["nodes"].values():
        node["ambient_ua_w_k"] = 0
    result = ThermalGraph(spec).advance(100)
    assert list(result["temperatures_k"].values()) == pytest.approx([310.0009] * 3, abs=1e-9)
    assert result["integrated_external_energy_j"] == pytest.approx(2700, abs=1e-6)


def test_rates_and_observations_are_pure_and_snapshots_are_detached():
    model = ThermalGraph(example())
    before = model.checkpoint()
    model.rates()
    observation = model.snapshot()
    observation["temperatures_k"]["a"] = 0
    checkpoint = model.checkpoint()
    checkpoint["payload"]["state"]["time_s"] = 999
    checkpoint["payload"]["signature"]["numpy"] = "caller mutation"
    assert model.checkpoint() == before


def test_split_restart_matches_continuous_trajectory_and_integrals():
    spec = example("driven_cycle")
    direct = ThermalGraph(spec)
    expected = direct.advance(120)
    split = ThermalGraph(spec)
    split.advance(37.5)
    checkpoint = json.loads(json.dumps(split.checkpoint()))
    restored = ThermalGraph(spec)
    restored.restore(checkpoint)
    actual = restored.advance(82.5)
    assert actual["time_s"] == 120
    assert actual["temperatures_k"] == pytest.approx(expected["temperatures_k"], abs=1e-9)
    assert actual["integrated_external_energy_j"] == pytest.approx(
        expected["integrated_external_energy_j"], abs=1e-4
    )


@pytest.mark.parametrize(
    "damage", ["digest", "source", "input", "temperature", "energy", "time", "increments"]
)
def test_restore_is_atomic_and_rejects_corrupt_or_inconsistent_checkpoint(damage):
    model = ThermalGraph(example())
    model.advance(2)
    before = model.checkpoint()
    bad = deepcopy(before)
    payload = bad["payload"]
    state = payload["state"]
    if damage == "digest":
        bad["sha256"] = "0" * 64
    elif damage == "source":
        payload["signature"]["numpy"] = "wrong"
    elif damage == "input":
        payload["input_sha256"] = "0" * 64
    elif damage == "temperature":
        state["temperatures_k"]["a"] += 1
    elif damage == "energy":
        state["integrated_external_energy_j"] += 1
    elif damage == "time":
        state["time_s"] = -1
    elif damage == "increments":
        state["temperature_increments_k"]["a"] += 1
    if damage != "digest":
        bad["sha256"] = digest(payload)
    with pytest.raises(NetworkError):
        model.restore(bad)
    assert model.checkpoint() == before


def test_domain_exit_rolls_back_the_entire_advance_not_just_last_internal_step():
    spec = example()
    spec["steady_input"]["thermal"]["max_temperature_k"] = 311
    for node in spec["steady_input"]["thermal"]["nodes"].values():
        node.update(heat_power_w=1e6, ambient_ua_w_k=0)
    model = ThermalGraph(spec)
    before = model.checkpoint()
    with pytest.raises(TransientGraphError):
        model.advance(10)
    assert model.checkpoint() == before


def test_unproved_interval_is_rejected_even_if_endpoint_might_be_admissible(monkeypatch):
    model = ThermalGraph(example())
    before = model.checkpoint()
    monkeypatch.setattr(model, "_certified", lambda value, step: False)
    with pytest.raises(TransientGraphError) as error:
        model.advance(1)
    assert error.value.report["criterion"] == "interval_domain"
    assert model.checkpoint() == before


def test_actual_interior_overshoot_is_not_hidden_by_admissible_final_endpoint():
    spec = example()
    hyd = spec["steady_input"]["hydraulic"]
    hyd["nodes"] = hyd["nodes"][:2]
    hyd["nodes"][1]["pressure_pa"] = 199000
    hyd["edges"] = hyd["edges"][:1]
    thermal = spec["steady_input"]["thermal"]
    thermal["nodes"] = {k: thermal["nodes"][k] for k in ("a", "b")}
    for node in thermal["nodes"].values():
        node["ambient_ua_w_k"] = 0
    thermal["nodes"]["b"]["heat_power_w"] = 190000
    spec["volumes_m3"] = {"a": 0.001, "b": 0.001}
    spec["initial_temperatures_k"] = {"a": 490.0, "b": 300.0}
    # Exact cascade: receiving temperature =490.001+(-190.001+190*t)*exp(-t).
    assert 490.001 + (-190.001 + 190 * 20) * math.exp(-20) < 500
    assert 490.001 + (-190.001 + 190 * 2) * math.exp(-2) > 500
    model = ThermalGraph(spec)
    before = model.checkpoint()
    with pytest.raises(TransientGraphError):
        model.advance(20)
    assert model.checkpoint() == before


def test_step_budget_failure_after_work_rolls_back_all_thermal_state():
    model = ThermalGraph(example())
    before = model.checkpoint()
    model.budget["maximum_internal_steps"] = 1
    with pytest.raises(TransientGraphError) as error:
        model.advance(3)
    assert error.value.report["criterion"] == "step_budget"
    assert model.checkpoint() == before


def test_energy_counter_failure_cannot_commit_temperature():
    model = ThermalGraph(example())
    before = model.checkpoint()
    model._transition(1.0)
    model._energy_scale *= 2
    with pytest.raises(TransientGraphError) as error:
        model.advance(1)
    assert error.value.report["criterion"] == "energy_residual"
    assert model.checkpoint() == before


def test_rate_argument_map_is_validated():
    model = ThermalGraph(example())
    with pytest.raises(NetworkError):
        model.rates({"a": 310})


def test_unrepresentable_time_increment_cannot_advance_thermal_state():
    model = ThermalGraph(example())
    model.advance(1)
    before = model.checkpoint()
    with pytest.raises(NetworkError, match="representable"):
        model.advance(1e-20)
    assert model.checkpoint() == before


def test_nonfinite_checkpoint_data_has_typed_rejection_and_no_mutation():
    model = ThermalGraph(example())
    before = model.checkpoint()
    bad = deepcopy(before)
    bad["payload"]["state"]["time_s"] = float("nan")
    with pytest.raises(NetworkError, match="Finite JSON"):
        model.restore(bad)
    assert model.checkpoint() == before


@pytest.mark.parametrize("change", ["volume", "missing", "initial", "max_step", "schema", "extra"])
def test_invalid_dynamic_contract(change):
    spec = example()
    if change == "volume":
        spec["volumes_m3"]["a"] = 0
    elif change == "missing":
        spec["initial_temperatures_k"].pop("a")
    elif change == "initial":
        spec["initial_temperatures_k"]["a"] = 600
    elif change == "max_step":
        spec["max_step_s"] = 11
    elif change == "schema":
        spec["schema_version"] = True
    elif change == "extra":
        spec["mass_projection"] = True
    with pytest.raises(NetworkError):
        ThermalGraph(spec)


def test_source_drift_rejects_advance_without_state_change(monkeypatch):
    model = ThermalGraph(example())
    before = model.checkpoint()
    monkeypatch.setattr("scripts.transient_thermal_graph.signature", lambda: {})
    with pytest.raises(TransientGraphError):
        model.advance(1)
    assert model.checkpoint() == before


def test_cli_checkpoint_resume(tmp_path, monkeypatch):
    source = tmp_path / "input.json"
    source.write_text(json.dumps(example()))
    first = tmp_path / "first"
    second = tmp_path / "second"
    monkeypatch.setattr(
        sys, "argv", ["run", str(source), "--duration-s", "2", "--output", str(first)]
    )
    assert main() == 0
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run",
            str(source),
            "--duration-s",
            "3",
            "--resume",
            str(first / "checkpoint.json"),
            "--output",
            str(second),
        ],
    )
    assert main() == 0
    assert json.loads((second / "report.json").read_text())["final"]["time_s"] == 5
    with pytest.raises(FileExistsError):
        main()
