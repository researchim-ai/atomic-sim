"""Dynamic steam inventories, port conservation, pure algebra and boundary rollback."""

import json
from copy import deepcopy
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from core.network import NetworkError
from core.status import DomainError
from physics.power.steam_cycle import SteamPowerCycle, balanced_steam_cycle, inventory
from sdk.steam_cycle_experiment import prepare, restore, run_steam_cycle_experiment


@pytest.fixture
def spec():
    return json.loads(Path("scenarios/ai05_dynamic_steam_cycle.json").read_text())


@pytest.mark.parametrize("method", ["Radau", "BDF"])
def test_balanced_setpoint_without_warmup(spec, method):
    model = balanced_steam_cycle(spec["model"])
    initial = model.observation()
    rates = model.initialization_report()
    assert max(abs(v) for v in rates["mass_rate_kg_s"].values()) < 1e-9
    assert max(abs(v) for v in rates["internal_energy_rate_w"].values()) < 1e-4
    row = model.advance(2, method=method)
    for name in ("steam_generator", "condenser"):
        assert row["vessels"][name]["pressure_pa"] == pytest.approx(
            initial["vessels"][name]["pressure_pa"], abs=0.001
        )
    assert row["omega_rad_s"] == pytest.approx(300, abs=1e-8)
    assert abs(row["energy_residual_j"]) < 1e-5
    assert row["feed_pump_power_w"] > 0 and row["electric_power_w"] > 0


def test_heater_changes_pressure_and_admission_not_external_mass(spec):
    model = prepare(spec)[1]
    before = model.observation()
    model.set_controls(
        dict(spec["controls"], heater_power_w=spec["controls"]["heater_power_w"] * 1.2)
    )
    after = model.advance(1)
    assert (
        after["vessels"]["steam_generator"]["pressure_pa"]
        > before["vessels"]["steam_generator"]["pressure_pa"]
    )
    assert after["steam_mass_flow_kg_s"] > before["steam_mass_flow_kg_s"]
    assert abs(after["mass_residual_kg"]) < 1e-10
    assert abs(after["energy_residual_j"]) < 1e-5


def test_internal_ports_and_feed_work_cancel_once(spec):
    model = prepare(spec)[1]
    row = model.observation()
    ports = row["internal_enthalpy_ports_w"]
    assert ports["hot_to_turbine"] - ports["turbine_to_condenser"] == pytest.approx(
        row["steam_power_w"], abs=1e-7
    )
    assert ports["feed_to_hot"] - ports["condenser_to_feed"] == pytest.approx(
        row["feed_pump_power_w"], abs=1e-7
    )
    assert abs(row["power_residual_w"]) < 1e-5
    # Work retained as pressure rise and feed heating is not deposited a second time.
    model.advance(0.5)
    row = model.observation()
    energies = row["energy_integrals_j"]
    total = (
        energies["heater"]
        + energies["feed_pump"]
        - sum(energies[k] for k in ("electric", "generator_heat", "bearing_heat", "sink_heat"))
    )
    assert row["stored_energy_change_j"] == pytest.approx(total, abs=1e-5)


def test_inventory_closure_and_no_shared_mutable_saturation_state(spec):
    model = prepare(spec)[1]
    mass, energy = model._origins[0]
    a = inventory(spec["model"]["steam_generator"], mass, energy)
    with pytest.raises(FrozenInstanceError):
        a["liquid"].h_j_kg = -1
    a["liquid"] = None
    b = inventory(spec["model"]["steam_generator"], mass, energy)
    assert b["liquid"].h_j_kg > 0
    assert b["pressure_pa"] == pytest.approx(6e6, abs=0.001)
    assert abs(b["energy_closure_j"]) < 0.01
    assert abs(b["volume_closure_m3"]) < 1e-9


def test_cold_inventory_resolves_submillijoule_energy_changes(spec):
    model = prepare(spec)[1]
    mass, energy = model._origins[1]
    pressures = []
    for delta in (-0.01, -0.001, 0, 0.001, 0.01):
        point = inventory(spec["model"]["condenser"], mass, energy + delta)
        assert abs(point["energy_closure_j"]) < 1e-5
        pressures.append(point["pressure_pa"])
    assert pressures == sorted(set(pressures))


@pytest.mark.parametrize("index,rate_coefficient", [(4, -0.1), (5, -0.4)])
def test_wall_residual_preserves_sub_ulp_temperature_deviations(spec, index, rate_coefficient):
    model = prepare(spec)[1]
    initial = model._state.copy()
    changed = initial.copy()
    changed[index] = 1e-14
    before = model.derivatives(0.0, initial)
    after = model.derivatives(0.0, changed)
    assert after[index] - before[index] == pytest.approx(
        rate_coefficient * 1e-14, rel=1e-9, abs=1e-25
    )
    assert abs(sum((after[0], after[2]))) < 1e-12


def test_rhs_jacobian_and_observation_do_not_mutate_state(spec):
    model = prepare(spec)[1]
    saved = model.snapshot()
    y = model._state.copy()
    y[1] += 100
    rates = model.derivatives(0, y)
    jac = model.jacobian(0, y)
    assert np.isfinite(jac).all()
    assert np.max(abs(jac[:, 7:])) == 0
    direction = np.array([0.0001, 10, -0.0001, -10, 0.001, -0.001, 10, *([0] * 6)])
    finite = (model.derivatives(0, y + direction) - model.derivatives(0, y - direction)) / 2
    assert np.max(abs(jac @ direction - finite) / (1 + abs(finite))) < 1e-5
    assert abs(rates[0] + rates[2]) < 1e-12
    assert model.snapshot() == saved


def test_stationary_initial_pressure_at_margin_does_not_terminate(spec):
    spec["model"]["steam_generator"]["min_pressure_pa"] = 6e6
    model = prepare(spec)[1]
    result = model.advance_until_limit(0.2)
    assert result["boundary"] is None


def test_speed_limit_prefix_and_rejected_full_step_rollback(spec):
    spec["model"]["turbine"]["max_omega_rad_s"] = 300.01
    model = prepare(spec)[1]
    model.set_controls(dict(spec["controls"], generator_connected=False))
    saved = model.snapshot()
    with pytest.raises(DomainError):
        model.advance(1)
    assert model.snapshot() == saved
    result = model.advance_until_limit(1)
    assert result["boundary"]["criterion"] == "shaft.speed_high"
    assert abs(result["boundary"]["residual"]) <= result["boundary"]["root_tolerance"]
    assert 0 < model.time_s < 1
    assert result["observation"]["omega_rad_s"] == pytest.approx(300.01, abs=1e-8)
    assert SteamPowerCycle.from_snapshot(model.snapshot()).snapshot() == model.snapshot()


def test_phase_margin_boundary_is_not_clipped(spec):
    spec["model"]["steam_generator"]["min_quality"] = 0.1 - 1e-7
    model = prepare(spec)[1]
    model.set_controls(dict(spec["controls"], feed_mass_flow_kg_s=0.2))
    result = model.advance_until_limit(1)
    assert result["boundary"]["criterion"] == "steam_generator.quality_low"
    assert abs(result["boundary"]["residual"]) <= 1e-10
    assert abs(result["observation"]["mass_residual_kg"]) < 1e-10


def test_boundary_restart_requires_actual_returning_vector_field(spec):
    spec["model"]["steam_generator"]["max_pressure_pa"] = 6e6 + 20
    model = prepare(spec)[1]
    model.set_controls(
        dict(spec["controls"], heater_power_w=spec["controls"]["heater_power_w"] * 1.2)
    )
    stopped = model.advance_until_limit(2)
    assert stopped["boundary"]["criterion"] == "steam_generator.pressure_high"
    model.set_controls(spec["controls"])
    saved = model.snapshot()
    again = model.advance_until_limit(0.0001)
    assert again["boundary"]["criterion"] == "steam_generator.pressure_high"
    assert model.snapshot() == saved
    model.set_controls(
        dict(
            spec["controls"],
            heater_power_w=0,
            feed_mass_flow_kg_s=2 * spec["controls"]["feed_mass_flow_kg_s"],
        )
    )
    recovered = model.advance_until_limit(0.0001)
    assert recovered["boundary"] is None
    assert recovered["observation"]["vessels"]["steam_generator"]["pressure_pa"] < 6e6 + 20


def test_controls_and_failed_acceptance_are_atomic(spec, monkeypatch):
    model = prepare(spec)[1]
    saved = model.snapshot()

    def reject(*args, **kwargs):
        raise NetworkError("synthetic acceptance failure")

    monkeypatch.setattr(model, "observation", reject)
    with pytest.raises(NetworkError):
        model.set_controls(dict(spec["controls"], generator_connected=False))
    assert model.snapshot() == saved
    with pytest.raises(NetworkError):
        model.advance_until_limit(0.1)
    assert model.snapshot() == saved


@pytest.mark.parametrize("defect", ["energy", "mass", "schema", "history", "shape", "bool", "time"])
def test_invalid_checkpoints_rejected(spec, defect):
    model = prepare(spec)[1]
    model.advance(0.1)
    raw = model.snapshot()
    if defect == "energy":
        raw["state"][1] += 10
    elif defect == "mass":
        raw["state"][0] += 0.01
    elif defect == "schema":
        raw["schema_version"] = True
    elif defect == "history":
        raw["controls"]["generator_connected"] = False
    elif defect == "shape":
        raw["state"].pop()
    elif defect == "bool":
        raw["state"][0] = True
    else:
        raw["time_s"] = -1
    with pytest.raises((NetworkError, DomainError)):
        SteamPowerCycle.from_snapshot(raw)


def test_restart_and_command_prefix(spec, tmp_path):
    spec.update(
        duration_s=1,
        step_s=0.3,
        timeline=[dict(t=0.37, controls=dict(spec["controls"], heater_power_w=280000))],
    )
    prefix = deepcopy(spec)
    prefix["duration_s"] = 0.6
    run_steam_cycle_experiment(prefix, tmp_path / "prefix")
    checkpoint = json.loads((tmp_path / "prefix/checkpoint.json").read_text())
    full = run_steam_cycle_experiment(spec, tmp_path / "full")
    resumed = run_steam_cycle_experiment(spec, tmp_path / "resumed", checkpoint=checkpoint)
    assert full["numerical_acceptance"] and resumed["numerical_acceptance"]
    a = json.loads((tmp_path / "full/checkpoint.json").read_text())["state"]
    b = json.loads((tmp_path / "resumed/checkpoint.json").read_text())["state"]
    assert a == pytest.approx(b, abs=1e-4)
    spec["timeline"][0]["controls"]["valve_opening"] = 0.5
    with pytest.raises(NetworkError, match="prefix"):
        restore(spec, checkpoint)


def test_domain_exit_is_distinct_from_failed_job(spec, tmp_path):
    spec["model"]["turbine"]["max_omega_rad_s"] = 300.01
    spec.update(
        duration_s=1,
        timeline=[dict(t=0, controls=dict(spec["controls"], generator_connected=False))],
    )
    report = run_steam_cycle_experiment(spec, tmp_path / "bounded")
    assert report["status"] == "domain_exit"
    assert report["accepted_prefix"] and not report["numerical_acceptance"]
    assert report["boundary"]["criterion"] == "shaft.speed_high"


def test_failed_output_interval_restores_commands(spec, tmp_path, monkeypatch):
    spec.update(
        duration_s=1,
        step_s=1,
        timeline=[dict(t=0.3, controls=dict(spec["controls"], valve_opening=0.8))],
    )
    original = SteamPowerCycle.advance_until_limit

    def fail_after_command(self, dt, **kwargs):
        if self.events:
            raise NetworkError("synthetic failure after command")
        return original(self, dt, **kwargs)

    monkeypatch.setattr(SteamPowerCycle, "advance_until_limit", fail_after_command)
    report = run_steam_cycle_experiment(spec, tmp_path / "failed")
    assert report["status"] == "failed" and report["completed_time_s"] == 0
    assert report["events"] == []
    assert json.loads((tmp_path / "failed/solver_trace.json").read_text()) == []
    assert report["solver_statistics"]["available"] is False
    assert (
        json.loads((tmp_path / "failed/checkpoint.json").read_text()) == prepare(spec)[1].snapshot()
    )


def test_deadline_returns_consistent_prefix(spec, tmp_path):
    report = run_steam_cycle_experiment(spec, tmp_path / "deadline", wall_limit_s=1e-15)
    assert report["status"] == "failed" and report["completed_time_s"] == 0
