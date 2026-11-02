"""Reciprocal primary/steam coupling, physical limits and whole-job rollback."""

import json
from pathlib import Path

import numpy as np
import pytest

from core.network import NetworkError
from core.status import DomainError
from physics.power.primary_steam import PrimarySteamCycle
from sdk.primary_steam_experiment import prepare, restore, run_primary_steam_experiment


@pytest.fixture
def spec():
    return json.loads(Path("scenarios/ai05_primary_steam.json").read_text())


@pytest.mark.parametrize("method", ["Radau", "BDF"])
def test_monolithic_ports_stocks_and_restart(spec, method):
    model = prepare(spec)[1]
    initial = model.initialization_report()
    assert abs(initial["primary_flow_rate_kg_s2"]) < 1e-7
    assert abs(initial["primary_shaft_acceleration_rad_s2"]) < 1e-7
    assert initial["primary_fluid_temperature_rates_k_s"][1] < 0
    result = model.advance_until_limit(0.2, method=method, max_step_s=0.02, rtol=1e-9)
    assert result["boundary"] is None
    row = result["observation"]
    assert row["heat_transfer_energy_j"] > 0
    assert row["steam_cycle"]["energy_integrals_j"]["heater"] == 0
    ports = row["thermal_ports"]
    assert ports["primary_into_w"] + ports["steam_into_w"] == 0
    assert ports["primary_into_j"] + ports["steam_into_j"] == 0
    assert abs(row["energy_residual_j"]) < 1e-4
    assert abs(row["power_residual_w"]) < 1e-6
    assert abs(row["primary"]["total_energy_residual_j"]) < 1e-4
    assert abs(row["steam_cycle"]["energy_residual_j"]) < 1e-4
    restored = PrimarySteamCycle.from_snapshot(model.snapshot())
    assert restored.snapshot() == model.snapshot()
    model.advance(0.1, method=method)
    restored.advance(0.1, method=method)
    assert restored.observation() == model.observation()


def test_heat_source_has_wall_storage_before_fluid_feedback(spec):
    base, hot = prepare(spec)[1], prepare(spec)[1]
    a = base.derivatives(0, base._state)
    hot.set_controls(dict(hot.boundary_controls, primary_wall_heater_power_w=5e5))
    b = hot.derivatives(0, hot._state)
    assert np.array_equal(a[1:4], b[1:4])
    assert b[4] > a[4]
    assert hot.observation()["heat_transfer_w"] == base.observation()["heat_transfer_w"]
    x, y = base.advance(0.5), hot.advance(0.5)
    assert y["heat_transfer_w"] > x["heat_transfer_w"]
    assert (
        y["primary"]["external_ambient_heater_heat_j"]
        > x["primary"]["external_ambient_heater_heat_j"]
    )


def test_motor_trip_changes_flow_and_shaft_in_same_system(spec):
    model = prepare(spec)[1]
    before = model.observation()
    model.set_controls(dict(model.boundary_controls, primary_motor_torque_nm=0))
    after = model.advance(0.2)
    assert after["primary"]["omega_rad_s"] < before["primary"]["omega_rad_s"]
    assert after["primary"]["loop"]["mass_flow_kg_s"] < before["primary"]["loop"]["mass_flow_kg_s"]
    assert abs(after["energy_residual_j"]) < 1e-4


def pressure_limited(spec):
    original = prepare(spec)[1]
    cfg = original.configuration
    pressures = [
        cfg["coupling"]["max_primary_pressure_pa"] - value
        for name, value, _, _ in original._margins(original._state)
        if name.startswith("primary.") and name.endswith(".pressure_high")
    ]
    cfg["coupling"]["max_primary_pressure_pa"] = max(pressures) + 10
    model = PrimarySteamCycle(**cfg, boundary_controls=original.boundary_controls)
    model.set_controls(
        dict(
            model.boundary_controls,
            primary_motor_torque_nm=2 * model.boundary_controls["primary_motor_torque_nm"],
        )
    )
    return model


def test_primary_pressure_root_hold_return_and_strict_rollback(spec):
    model = pressure_limited(spec)
    saved = model.snapshot()
    with pytest.raises(DomainError):
        model.advance(0.02, max_step_s=0.001)
    assert model.snapshot() == saved
    result = model.advance_until_limit(0.02, max_step_s=0.001)
    boundary = result["boundary"]
    assert boundary["criterion"].startswith("primary.")
    assert boundary["criterion"].endswith(".pressure_high")
    assert 0 < model.time_s < 0.02
    assert abs(boundary["residual"]) <= boundary["root_tolerance"]
    snapshot = model.snapshot()
    held = model.advance_until_limit(0.001)
    assert held["solver"]["backend"] == "boundary_hold"
    assert model.snapshot() == snapshot
    restored = PrimarySteamCycle.from_snapshot(snapshot)
    restored.set_controls(dict(restored.boundary_controls, primary_motor_torque_nm=0))
    assert restored.advance_until_limit(0.001, max_step_s=0.0001)["boundary"] is None


@pytest.mark.parametrize(
    "defect", ["heat_integral", "control_chain", "source", "shape", "initial_time"]
)
def test_checkpoint_rejects_inconsistent_context(spec, defect):
    model = prepare(spec)[1]
    model.set_controls(dict(model.boundary_controls, primary_motor_torque_nm=0))
    model.advance(0.1)
    payload = model.snapshot()
    if defect == "heat_integral":
        payload["state"][-1] += 1
    elif defect == "control_chain":
        payload["events"][0]["before"]["primary_motor_torque_nm"] += 1
    elif defect == "source":
        payload["configuration_sha256"] = "0" * 64
    elif defect == "shape":
        payload["state"].pop()
    else:
        payload["time_s"] = 0
    with pytest.raises((NetworkError, DomainError)):
        PrimarySteamCycle.from_snapshot(payload)


@pytest.mark.parametrize(
    "key,value",
    [
        ("ua_w_k", 0),
        ("minimum_subcooling_k", -1),
        ("min_primary_pressure_pa", 17e6),
        ("max_primary_pressure_pa", 17e6),
        ("min_primary_temperature_k", 270),
        ("steam_generator_segment", "missing"),
    ],
)
def test_explicit_primary_water_scope(spec, key, value):
    spec["coupling"][key] = value
    with pytest.raises((NetworkError, DomainError)):
        prepare(spec)


def test_primary_subcooling_guard_rejects_hot_initial_liquid(spec):
    temperatures = spec["primary"]["loop"]["initialization"]["fluid_temperature_k"]
    for name in temperatures:
        temperatures[name] = 620
    with pytest.raises(DomainError, match="subcooling"):
        prepare(spec)


def test_sdk_internal_control_splits_and_continuation(spec, tmp_path):
    spec.update(duration_s=0.2, step_s=0.1)
    spec["timeline"] = [dict(t=0.13, controls=dict(spec["controls"], primary_motor_torque_nm=0))]
    folder = tmp_path / "first"
    result = run_primary_steam_experiment(spec, folder)
    assert result["status"] == "completed"
    assert result["solver_statistics"]["segments"] == 3
    snapshot = json.loads((folder / "checkpoint.json").read_text())
    spec["duration_s"] = 0.3
    restored = restore(spec, snapshot)[1]
    report = run_primary_steam_experiment(spec, tmp_path / "continued", checkpoint=snapshot)
    assert report["completed_time_s"] == 0.3
    restored.advance(0.1, **spec["solver"])
    last = json.loads((tmp_path / "continued/trajectory.json").read_text())[-1]
    assert last["energy_residual_j"] == pytest.approx(
        restored.observation()["energy_residual_j"], abs=1e-7
    )


def test_entire_output_interval_rolls_back_heat_and_commands(spec, tmp_path, monkeypatch):
    spec.update(duration_s=0.2, step_s=0.1)
    spec["timeline"] = [dict(t=0.04, controls=dict(spec["controls"], primary_motor_torque_nm=0))]
    original = PrimarySteamCycle.advance_until_limit
    calls = 0

    def broken(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        result = original(self, *args, **kwargs)
        if calls == 2:
            raise RuntimeError("injected failure after port energy commit")
        return result

    monkeypatch.setattr(PrimarySteamCycle, "advance_until_limit", broken)
    folder = tmp_path / "failed"
    report = run_primary_steam_experiment(spec, folder)
    assert report["status"] == "failed" and report["completed_time_s"] == 0
    assert report["events"] == []
    assert json.loads((folder / "solver_trace.json").read_text()) == []
    checkpoint = json.loads((folder / "checkpoint.json").read_text())
    assert checkpoint == prepare(spec)[1].snapshot()
    assert checkpoint["state"][-1] == 0


@pytest.mark.parametrize("method", ["bad", True])
def test_invalid_solver_is_transactional(spec, method):
    model = prepare(spec)[1]
    snapshot = model.snapshot()
    with pytest.raises(NetworkError):
        model.advance_until_limit(0.1, method=method)
    assert model.snapshot() == snapshot
