"""Nonlinear feedback: analytic limits, interval bounds and atomic state changes."""

import json
import math
import sys
from copy import deepcopy

import numpy as np
import pytest

from core.network import NetworkError
from scripts.nonlinear_graph_references import bridge_spec, cycle_spec, pipe_spec
from scripts.nonlinear_thermal_graph import NonlinearGraphError, NonlinearThermalGraph, digest, main
from scripts.transient_thermal_graph import ThermalGraph


def test_zero_feedback_matches_affine_exponential_backend():
    spec = pipe_spec(0.0)
    legacy = dict(
        format="atomic-thermal-graph-transient",
        schema_version=1,
        steady_input=spec["steady_input"],
        volumes_m3=spec["volumes_m3"],
        initial_temperatures_k=spec["initial_temperatures_k"],
        max_step_s=spec["max_step_s"],
    )
    expected = ThermalGraph(legacy).advance(10)
    actual = NonlinearThermalGraph(spec).advance(10)
    assert actual["temperatures_k"] == pytest.approx(expected["temperatures_k"], abs=1e-8)
    assert actual["integrated_external_energy_j"] == pytest.approx(
        expected["integrated_external_energy_j"], abs=1e-4
    )


def test_uniform_adiabatic_cycle_matches_nonlinear_analytic_solution():
    spec = cycle_spec(0.01, True)
    actual = NonlinearThermalGraph(spec).advance(10)
    capacity = 1000 * 0.5 * 1000
    expected = 300 + math.log(math.exp(0.01 * 10) + 0.01 * 9 * 10 / capacity) / 0.01
    assert actual["temperatures_k"] == pytest.approx(
        {k: expected for k in ("a", "b", "c")}, abs=1e-9
    )
    assert actual["integrated_external_energy_j"] == pytest.approx(
        3 * capacity * (expected - 310), abs=1e-4
    )


def test_feedback_changes_flow_over_time_and_keeps_commands():
    spec = pipe_spec()
    before = deepcopy(spec)
    model = NonlinearThermalGraph(spec)
    initial = model.snapshot()
    final = model.advance(30)
    assert final["flows_kg_s"]["pipe"] < initial["flows_kg_s"]["pipe"]
    assert final["temperatures_k"]["a"] > initial["temperatures_k"]["a"]
    assert spec == before and abs(final["energy_residual_j"]) <= 1e-4


def test_reverse_flow_can_absorb_pressure_gain_work():
    spec = pipe_spec(reverse=True)
    spec["steady_input"]["hydraulic"]["nodes"][1]["pressure_pa"] = 203000
    spec["steady_input"]["hydraulic"]["edges"][0]["pressure_gain_pa"] = 1000
    model = NonlinearThermalGraph(spec)
    _, hyd, _ = model._evaluate(model.offset)
    assert hyd["powers_w"]["prescribed_gain"] < 0
    result = model.advance(5)
    assert result["flows_kg_s"]["pipe"] < 0 and abs(result["energy_residual_j"]) <= 1e-4


def test_bridge_reverses_flow_without_changing_pressure_commands():
    model = NonlinearThermalGraph(bridge_spec())
    initial = model.snapshot()
    final = model.advance(120)
    assert initial["flows_kg_s"]["bridge"] > 0 and final["flows_kg_s"]["bridge"] < 0
    assert final["pressures_pa"]["supply"] == initial["pressures_pa"]["supply"] == 200000
    assert final["pressures_pa"]["outlet"] == initial["pressures_pa"]["outlet"] == 190000


def test_idle_heating_has_no_spurious_mass_flow_or_thermal_gauge():
    spec = pipe_spec()
    spec["steady_input"]["hydraulic"]["nodes"][1]["pressure_pa"] = 200000
    for node in spec["steady_input"]["thermal"]["nodes"].values():
        node["heat_power_w"] = 1000
    result = NonlinearThermalGraph(spec).advance(10)
    assert result["temperatures_k"] == pytest.approx({"a": 310.02, "b": 320.02}, abs=1e-9)
    assert result["flows_kg_s"]["pipe"] == 0


def test_snapshot_rates_and_checkpoint_are_detached_and_pure():
    model = NonlinearThermalGraph(pipe_spec())
    before = model.checkpoint()
    model.rates()
    state = model.snapshot()
    state["flows_kg_s"]["pipe"] = 999
    exported = model.checkpoint()
    exported["payload"]["signature"]["python"] = "changed"
    assert model.checkpoint() == before


def test_restart_preserves_full_adaptive_solver_state_exactly():
    spec = pipe_spec()
    memory = NonlinearThermalGraph(spec)
    memory.advance(3.7)
    checkpoint = json.loads(json.dumps(memory.checkpoint()))
    restored = NonlinearThermalGraph(spec)
    assert restored.restore(checkpoint) == memory.snapshot()
    assert restored.advance(6.3) == memory.advance(6.3)


def test_split_path_stays_within_numerical_temperature_and_energy_budget():
    spec = bridge_spec()
    whole = NonlinearThermalGraph(spec).advance(20)
    split = NonlinearThermalGraph(spec)
    split.advance(7.3)
    actual = split.advance(12.7)
    assert actual["temperatures_k"] == pytest.approx(whole["temperatures_k"], abs=1e-6)
    assert actual["integrated_external_energy_j"] == pytest.approx(
        whole["integrated_external_energy_j"], abs=0.1
    )


@pytest.mark.parametrize(
    "damage",
    [
        "digest",
        "source",
        "input",
        "temperature",
        "increments",
        "energy",
        "scaled_energy",
        "flow",
        "pressure",
        "proposal",
        "time",
        "nan",
    ],
)
def test_bad_checkpoint_restore_cannot_mutate_state(damage):
    model = NonlinearThermalGraph(pipe_spec())
    model.advance(1)
    before = model.checkpoint()
    bad = deepcopy(before)
    p = bad["payload"]
    s = p["state"]
    if damage == "digest":
        bad["sha256"] = "0" * 64
    elif damage == "source":
        p["signature"]["python"] = "wrong"
    elif damage == "input":
        p["input_sha256"] = "0" * 64
    elif damage == "temperature":
        s["temperatures_k"]["a"] += 1
    elif damage == "increments":
        s["temperature_increments_k"]["a"] += 1
    elif damage == "energy":
        s["integrated_external_energy_j"] += 1
    elif damage == "scaled_energy":
        s["energy_integral_scaled_k"] += 1
    elif damage == "flow":
        s["flows_kg_s"]["pipe"] += 1
    elif damage == "pressure":
        s["pressures_pa"]["a"] += 1
    elif damage == "proposal":
        s["step_proposal_s"] = 99
    elif damage == "time":
        s["time_s"] = -1
    elif damage == "nan":
        s["time_s"] = float("nan")
    if damage not in ("digest", "nan"):
        bad["sha256"] = digest(p)
    with pytest.raises(NetworkError):
        model.restore(bad)
    assert model.checkpoint() == before


def test_attempt_limit_after_internal_work_rolls_back_whole_advance():
    model = NonlinearThermalGraph(pipe_spec())
    before = model.checkpoint()
    model.budget["maximum_attempts"] = 1
    with pytest.raises(NonlinearGraphError) as error:
        model.advance(10)
    assert error.value.report["criterion"] == "attempt_budget"
    assert model.checkpoint() == before


def test_energy_failure_cannot_commit_temperature_or_time(monkeypatch):
    model = NonlinearThermalGraph(pipe_spec())
    before = model.checkpoint()
    original = model._evaluate

    def damaged(t):
        rate, hyd, effective = original(t)
        rate[-1] += 1e-5
        return rate, hyd, effective

    monkeypatch.setattr(model, "_evaluate", damaged)
    with pytest.raises(NonlinearGraphError) as error:
        model.advance(1)
    assert error.value.report["criterion"] == "integrated_energy"
    assert model.checkpoint() == before


def test_observation_failure_is_validated_before_committing(monkeypatch):
    model = NonlinearThermalGraph(pipe_spec())
    before = model.checkpoint()
    original = model._observation

    def broken(state, time, proposal):
        if time > model._time:
            raise NetworkError("Injected observation failure")
        return original(state, time, proposal)

    monkeypatch.setattr(model, "_observation", broken)
    with pytest.raises(NetworkError, match="Injected"):
        model.advance(1)
    assert model.checkpoint() == before


def test_hidden_temperature_peak_is_not_accepted_by_its_cool_endpoint():
    spec = pipe_spec(0.0)
    spec["volumes_m3"] = {"a": 0.001, "b": 0.001}
    spec["initial_temperatures_k"] = {"a": 490.0, "b": 300.0}
    for node in spec["steady_input"]["thermal"]["nodes"].values():
        node["supply_temperature_k"] = 300
    spec["steady_input"]["thermal"]["nodes"]["b"]["heat_power_w"] = 190000
    assert 490.001 + (-190.001 + 190 * 20) * math.exp(-20) < 500
    assert 490.001 + (-190.001 + 190 * 2) * math.exp(-2) > 500
    model = NonlinearThermalGraph(spec)
    before = model.checkpoint()
    with pytest.raises(NonlinearGraphError):
        model.advance(20)
    assert model.checkpoint() == before


def test_law_domain_exit_refuses_whole_duration_without_clipping():
    spec = pipe_spec()
    spec["initial_temperatures_k"] = {"a": 300.0, "b": 300.0}
    spec["resistance_laws"]["pipe"]["maximum_multiplier"] = 1.1
    spec["volumes_m3"] = {"a": 0.05, "b": 0.05}
    model = NonlinearThermalGraph(spec)
    before = model.checkpoint()
    with pytest.raises(NonlinearGraphError):
        model.advance(120)
    assert model.checkpoint() == before


@pytest.mark.parametrize("quadratic", [0.0, 50.0])
@pytest.mark.parametrize("injection", [-0.5, 0.0, 0.5])
def test_local_hydraulic_and_rate_bounds_contain_sampled_states(injection, quadratic):
    spec = bridge_spec()
    spec["steady_input"]["hydraulic"]["nodes"][1]["injection_kg_s"] = injection
    for edge in spec["steady_input"]["hydraulic"]["edges"]:
        edge["quadratic_pa_s2_kg2"] = quadratic
    model = NonlinearThermalGraph(spec)
    t = model.offset.copy()
    _, hyd, effective = model._evaluate(t)
    lo, hi, lower, upper = model._box(t, hyd, effective)
    bound, _, _ = model._flow_bound(t, lo, hi, hyd, effective)
    rng = np.random.default_rng(1907)
    for _ in range(12):
        sample = lo + (hi - lo) * rng.uniform(size=len(t))
        rate, actual, _ = model._evaluate(sample)
        assert np.linalg.norm(list(actual["flows_kg_s"].values())) <= bound * (1 + 1e-12)
        assert np.all(rate[:-1] >= lower - 1e-10) and np.all(rate[:-1] <= upper + 1e-10)


def test_tight_pressure_margin_can_be_conservatively_refused():
    spec = bridge_spec()
    point = NonlinearThermalGraph(spec).snapshot()
    spec["steady_input"]["hydraulic"]["nodes"][1]["max_pressure_pa"] = (
        point["pressures_pa"]["a"] + 1e-9
    )
    model = NonlinearThermalGraph(spec)
    before = model.checkpoint()
    with pytest.raises(NonlinearGraphError) as error:
        model.advance(1)
    assert error.value.report["criterion"] == "interval_box"
    assert model.checkpoint() == before


@pytest.mark.parametrize(
    "damage",
    [
        "schema",
        "base_format",
        "base_schema",
        "volume",
        "volume_map",
        "temperature_map",
        "cp",
        "ua",
        "supply",
        "max_step",
        "laws",
        "nan",
    ],
)
def test_invalid_input_contract(damage):
    spec = pipe_spec()
    if damage == "schema":
        spec["schema_version"] = True
    elif damage == "base_format":
        spec["steady_input"]["format"] = "general-eos"
    elif damage == "base_schema":
        spec["steady_input"]["schema_version"] = True
    elif damage == "volume":
        spec["volumes_m3"]["a"] = 0
    elif damage == "volume_map":
        spec["volumes_m3"].pop("a")
    elif damage == "temperature_map":
        spec["initial_temperatures_k"].pop("a")
    elif damage == "cp":
        spec["steady_input"]["thermal"]["heat_capacity_j_kg_k"] = 0
    elif damage == "ua":
        spec["steady_input"]["thermal"]["nodes"]["a"]["ambient_ua_w_k"] = -1
    elif damage == "supply":
        spec["steady_input"]["thermal"]["nodes"]["b"].pop("supply_temperature_k")
    elif damage == "max_step":
        spec["max_step_s"] = 3
    elif damage == "laws":
        spec["resistance_laws"] = {}
    elif damage == "nan":
        spec["resistance_laws"]["pipe"]["temperature_coefficient_per_k"] = float("nan")
    with pytest.raises(NetworkError):
        NonlinearThermalGraph(spec)


def test_rate_map_and_time_increment_have_explicit_validation():
    model = NonlinearThermalGraph(pipe_spec())
    with pytest.raises(NetworkError):
        model.rates({"a": 300})
    model.advance(1)
    before = model.checkpoint()
    with pytest.raises(NetworkError):
        model.advance(1e-30)
    assert model.checkpoint() == before


def test_source_drift_prevents_advance(monkeypatch):
    model = NonlinearThermalGraph(pipe_spec())
    before = model.checkpoint()
    monkeypatch.setattr("scripts.nonlinear_thermal_graph.signature", lambda: {})
    with pytest.raises(NonlinearGraphError) as error:
        model.advance(1)
    assert error.value.report["criterion"] == "source_drift" and model.checkpoint() == before


def test_cli_resume_and_refuse_overwrite(tmp_path, monkeypatch):
    source = tmp_path / "input.json"
    source.write_text(json.dumps(pipe_spec()))
    first = tmp_path / "first"
    second = tmp_path / "second"
    monkeypatch.setattr(
        sys, "argv", ["run", str(source), "--duration-s", "1", "--output", str(first)]
    )
    assert main() == 0
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run",
            str(source),
            "--duration-s",
            "1",
            "--resume",
            str(first / "checkpoint.json"),
            "--output",
            str(second),
        ],
    )
    assert main() == 0 and json.loads((second / "report.json").read_text())["final"]["time_s"] == 2
    with pytest.raises(FileExistsError):
        main()


def test_cli_failure_never_creates_checkpoint(tmp_path, monkeypatch):
    spec = pipe_spec()
    spec["volumes_m3"]["a"] = 0
    source = tmp_path / "input.json"
    source.write_text(json.dumps(spec))
    output = tmp_path / "result"
    monkeypatch.setattr(
        sys, "argv", ["run", str(source), "--duration-s", "1", "--output", str(output)]
    )
    assert main() == 2 and not (output / "checkpoint.json").exists()
