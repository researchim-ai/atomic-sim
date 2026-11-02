"""Steam power, positive-speed rotor limits and independent time integration."""

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from core.network import NetworkError
from core.status import ModelLimitError
from physics.power.steam_turbine_train import SteamTurbineTrain
from sdk.turbine_experiment import prepare, restore, run_turbine_experiment


@pytest.fixture
def config():
    return dict(
        inlet_pressure_pa=6e6,
        outlet_pressure_pa=8000,
        isentropic_efficiency=0.85,
        inertia_kg_m2=5000,
        bearing_drag_nm_s_rad=1,
        generator_efficiency=0.97,
        min_omega_rad_s=100,
        max_omega_rad_s=400,
        energy_tolerance_j=0.001,
    )


@pytest.fixture
def boundary():
    return dict(mass_flow_kg_s=10, generator_conductance_nm_s_rad=80, generator_connected=True)


def test_thermodynamic_work_and_entropy(config, boundary):
    model = SteamTurbineTrain(config, boundary, 300)
    thermo = model.thermodynamics
    assert thermo["work_j_kg"] > 0
    assert 0 < thermo["isentropic_quality"] < thermo["outlet_quality"] < 1
    assert thermo["outlet_entropy_j_kg_k"] > thermo["inlet_entropy_j_kg_k"]
    assert thermo["work_j_kg"] == pytest.approx(
        config["isentropic_efficiency"]
        * (thermo["inlet_enthalpy_j_kg"] - thermo["isentropic_outlet_enthalpy_j_kg"]),
        abs=1e-7,
    )
    row = model.advance(1)
    assert abs(row["fluid_power_residual_w"]) < 1e-5
    assert abs(row["energy_residual_j"]) < 1e-5
    assert row["steam_work_j"] == pytest.approx(
        row["electric_energy_j"]
        + row["generator_heat_j"]
        + row["bearing_heat_j"]
        + row["rotor_energy_j"]
        - 0.5 * 5000 * 300**2,
        abs=1e-5,
    )


@pytest.mark.parametrize("efficiency", [0.5, 1])
def test_generator_efficiency_changes_electricity_not_mechanical_drain(
    config, boundary, efficiency
):
    reference = SteamTurbineTrain(config, boundary, 300).advance(2)
    config["generator_efficiency"] = efficiency
    row = SteamTurbineTrain(config, boundary, 300).advance(2)
    assert row["omega_rad_s"] == reference["omega_rad_s"]
    assert row["electric_energy_j"] + row["generator_heat_j"] == pytest.approx(
        reference["electric_energy_j"] + reference["generator_heat_j"]
    )
    assert row["electric_energy_j"] == pytest.approx(
        efficiency * (row["electric_energy_j"] + row["generator_heat_j"])
    )


@pytest.mark.parametrize("conductance", [0, 1e-8, 80])
def test_propagator_against_independent_dop853(config, boundary, conductance):
    config["bearing_drag_nm_s_rad"] = 0
    boundary.update(mass_flow_kg_s=0.1, generator_conductance_nm_s_rad=conductance)
    model = SteamTurbineTrain(config, boundary, 300)
    power = boundary["mass_flow_kg_s"] * model.thermodynamics["work_j_kg"]
    inertia = config["inertia_kg_m2"]
    eta = config["generator_efficiency"]

    def rhs(_t, y):
        mechanical = conductance * y[0] ** 2
        return [
            (power - mechanical) / (inertia * y[0]),
            power,
            eta * mechanical,
            (1 - eta) * mechanical,
        ]

    reference = solve_ivp(
        rhs, (0, 1), [300, 0, 0, 0], method="DOP853", rtol=1e-12, atol=1e-10, max_step=0.01
    ).y[:, -1]
    row = model.advance(1)
    assert row["omega_rad_s"] == pytest.approx(reference[0], abs=1e-10)
    assert [
        row["steam_work_j"],
        row["electric_energy_j"],
        row["generator_heat_j"],
    ] == pytest.approx(reference[1:], abs=1e-6)


def test_zero_dissipation_uniform_energy_acceleration(config, boundary):
    config["bearing_drag_nm_s_rad"] = 0
    boundary["generator_connected"] = False
    model = SteamTurbineTrain(config, boundary, 300)
    before = model.observation()
    row = model.advance(2)
    assert row["rotor_energy_j"] == pytest.approx(
        before["rotor_energy_j"] + 2 * row["steam_power_w"]
    )
    assert row["electric_energy_j"] == row["generator_heat_j"] == row["bearing_heat_j"] == 0


def test_upper_boundary_exact_time_and_rejected_step_is_atomic(config, boundary):
    config["bearing_drag_nm_s_rad"] = 0
    boundary["generator_connected"] = False
    model = SteamTurbineTrain(config, boundary, 300)
    snap = model.snapshot()
    power = model.observation()["steam_power_w"]
    expected = 0.5 * config["inertia_kg_m2"] * (400**2 - 300**2) / power
    with pytest.raises(ModelLimitError) as failure:
        model.advance(100)
    assert failure.value.evaluation_time_s == pytest.approx(expected, abs=1e-12)
    assert model.snapshot() == snap
    result = model.advance_until_limit(100)
    assert result["boundary"]["limit"] == "upper"
    assert model.time_s == pytest.approx(expected, abs=1e-12)
    assert model.observation()["omega_rad_s"] == 400
    assert abs(model.observation()["energy_residual_j"]) < 1e-5
    assert SteamTurbineTrain.from_snapshot(model.snapshot()).snapshot() == model.snapshot()


def test_trip_lower_boundary_and_recovery(config, boundary):
    boundary["mass_flow_kg_s"] = 0
    model = SteamTurbineTrain(config, boundary, 300)
    decay = 2 * (80 + 1) / 5000
    expected = np.log(300**2 / 100**2) / decay
    result = model.advance_until_limit(1000)
    assert result["boundary"]["limit"] == "lower"
    assert model.time_s == pytest.approx(expected, abs=1e-10)
    assert model.observation()["omega_rad_s"] == 100
    model.set_controls(dict(boundary, mass_flow_kg_s=10, generator_connected=False))
    assert model.advance(1)["omega_rad_s"] > 100


def test_steady_operating_point(config, boundary):
    probe = SteamTurbineTrain(config, boundary, 300)
    power = probe.observation()["steam_power_w"]
    boundary["generator_conductance_nm_s_rad"] = power / 300**2 - config["bearing_drag_nm_s_rad"]
    model = SteamTurbineTrain(config, boundary, 300)
    row = model.advance(100)
    assert row["omega_rad_s"] == pytest.approx(300, abs=1e-12)
    assert abs(row["energy_residual_j"]) < 0.001


def test_checkpoint_restart_and_subdivision(config, boundary):
    model = SteamTurbineTrain(config, boundary, 300)
    model.advance(1)
    model.set_controls(dict(boundary, generator_connected=False))
    restored = SteamTurbineTrain.from_snapshot(model.snapshot())
    assert restored.snapshot() == model.snapshot()
    assert restored.advance(2) == model.advance(2)
    single = SteamTurbineTrain(config, boundary, 300)
    subdivided = SteamTurbineTrain(config, boundary, 300)
    a = single.advance(2)
    for _ in range(20):
        b = subdivided.advance(0.1)
    assert a["omega_rad_s"] == pytest.approx(b["omega_rad_s"], abs=1e-10)
    assert a["electric_energy_j"] == pytest.approx(b["electric_energy_j"], abs=1e-5)


@pytest.mark.parametrize("defect", ["partial", "bool", "negative", "overflow"])
def test_controls_atomic_validation(config, boundary, defect):
    model = SteamTurbineTrain(config, boundary, 300)
    before = model.snapshot()
    candidate = deepcopy(boundary)
    if defect == "partial":
        candidate.pop("generator_connected")
    elif defect == "bool":
        candidate["generator_connected"] = 1
    elif defect == "negative":
        candidate["mass_flow_kg_s"] = -1
    else:
        candidate["generator_conductance_nm_s_rad"] = 1e308
    with pytest.raises(NetworkError):
        model.set_controls(candidate)
    assert model.snapshot() == before


@pytest.mark.parametrize("defect", ["energy", "schema", "history", "state", "time"])
def test_corrupt_snapshots_rejected(config, boundary, defect):
    model = SteamTurbineTrain(config, boundary, 300)
    model.advance(1)
    snap = model.snapshot()
    if defect == "energy":
        snap["state"][2] += 10
    elif defect == "schema":
        snap["schema_version"] = True
    elif defect == "history":
        snap["controls"]["generator_connected"] = False
    elif defect == "state":
        snap["state"][0] = True
    else:
        snap["time_s"] = -1
    with pytest.raises(NetworkError):
        SteamTurbineTrain.from_snapshot(snap)


def test_acceptance_failure_keeps_state(config, boundary, monkeypatch):
    model = SteamTurbineTrain(config, boundary, 300)
    before = model.snapshot()

    def fail(*args):
        raise NetworkError("synthetic acceptance failure")

    monkeypatch.setattr(model, "_observe", fail)
    with pytest.raises(NetworkError):
        model.advance(1)
    assert model.snapshot() == before


def turbine_spec():
    return json.loads(Path("scenarios/ai05_turbine_load_rejection.json").read_text())


def test_large_trip_request_still_reports_physical_limit(config, boundary):
    boundary["mass_flow_kg_s"] = 0
    model = SteamTurbineTrain(config, boundary, 300)
    with pytest.raises(ModelLimitError):
        model.advance(1e6)
    assert model.time_s == 0


def test_sdk_distinguishes_domain_exit_from_failure(tmp_path):
    spec = turbine_spec()
    spec["duration_s"] = 120
    report = run_turbine_experiment(spec, tmp_path / "bounded")
    assert report["status"] == "domain_exit"
    assert report["accepted_prefix"] and not report["numerical_acceptance"]
    assert 5.3 < report["completed_time_s"] < 120
    assert report["boundary"]["omega_rad_s"] == 400
    snapshot = json.loads((tmp_path / "bounded/checkpoint.json").read_text())
    assert SteamTurbineTrain.from_snapshot(snapshot).time_s == report["completed_time_s"]


def test_sdk_prefix_restart_and_invalid_history(tmp_path):
    spec = turbine_spec()
    prefix = deepcopy(spec)
    prefix["duration_s"] = 7
    run_turbine_experiment(prefix, tmp_path / "prefix")
    checkpoint = json.loads((tmp_path / "prefix/checkpoint.json").read_text())
    resumed = run_turbine_experiment(spec, tmp_path / "resumed", checkpoint=checkpoint)
    full = run_turbine_experiment(spec, tmp_path / "full")
    assert resumed["numerical_acceptance"] and full["numerical_acceptance"]
    a = json.loads((tmp_path / "resumed/checkpoint.json").read_text())["state"]
    b = json.loads((tmp_path / "full/checkpoint.json").read_text())["state"]
    assert a == pytest.approx(b, abs=0.0001)
    spec["timeline"][0]["controls"]["mass_flow_kg_s"] = 9
    with pytest.raises(NetworkError, match="prefix"):
        restore(spec, checkpoint)


def test_sdk_failed_interval_rolls_back_controls(tmp_path, monkeypatch):
    spec = turbine_spec()
    spec.update(duration_s=1, step_s=1)
    spec["timeline"][0]["t"] = 0.3
    original = SteamTurbineTrain.advance_until_limit

    def failed(self, dt):
        if self.events:
            raise NetworkError("synthetic failure after command")
        return original(self, dt)

    monkeypatch.setattr(SteamTurbineTrain, "advance_until_limit", failed)
    report = run_turbine_experiment(spec, tmp_path / "failed")
    assert report["status"] == "failed"
    assert not report["accepted_prefix"]
    assert report["completed_time_s"] == 0
    assert report["events"] == []
    assert (
        json.loads((tmp_path / "failed/checkpoint.json").read_text()) == prepare(spec)[1].snapshot()
    )


def test_sdk_deadline_keeps_initial_checkpoint(tmp_path):
    report = run_turbine_experiment(turbine_spec(), tmp_path / "deadline", wall_limit_s=1e-15)
    assert report["status"] == "failed"
    assert report["completed_time_s"] == 0
