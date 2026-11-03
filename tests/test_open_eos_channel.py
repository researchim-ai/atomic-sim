"""Open-system conservation, signed numerical boundaries and restart."""

import json
from copy import deepcopy

import numpy as np
import pytest

from core.network import NetworkError
from scripts.inertial_graph import InertialGraphError, digest
from scripts.open_channel_references import IndependentOpen, cases, request
from scripts.open_eos_channel import BOUNDARY_LEDGERS, OpenEOSChannel


@pytest.mark.parametrize("velocity", [-5.0, 0.0, 5.0])
def test_uniform_solution_and_each_boundary_transport(velocity):
    r = request(velocity=velocity)
    e = OpenEOSChannel(r)
    initial = e.snapshot()["state"]
    state = e.advance(0.1)["state"]
    eos = e.system.eos
    flow = e.system.area * e.system.rho0[0] * velocity
    energy = flow * (eos.cp * 300 + eos.e0 + 0.5 * velocity**2)
    momentum = e.system.area * (e.system.rho0[0] * velocity**2 + 200000.0)
    entropy = flow * e.system.s0[0]
    for field in ["masses_kg", "temperatures_k", "pressures_pa", "velocities_m_s"]:
        np.testing.assert_array_equal(list(state[field].values()), list(initial[field].values()))
    for side, sign in [("left", 1.0), ("right", -1.0)]:
        values = state["boundary_transport"][side]
        for name, expected in zip(BOUNDARY_LEDGERS, [flow, momentum, energy, entropy]):
            assert values[name] == pytest.approx(sign * 0.1 * expected, abs=1e-8)
        assert state["boundary_flows_into_channel_kg_s"][side] == pytest.approx(
            sign * flow, abs=1e-12
        )
    assert abs(state["energy_residual_j"]) < 1e-8


@pytest.mark.parametrize("shift", [1e5, 1e8])
def test_open_hydrostatic_rest(shift):
    r = request(height=10, shift=shift)
    e = OpenEOSChannel(r)
    e.advance(0.1)
    s = e.snapshot()["state"]
    assert max(map(abs, s["velocities_m_s"].values())) < 1e-10
    np.testing.assert_allclose(
        list(s["pressures_pa"].values()), r["initial"]["pressures_pa"], rtol=0, atol=1e-7
    )
    assert max(map(abs, s["boundary_flows_into_channel_kg_s"].values())) < 1e-10


@pytest.mark.parametrize("name,spec,schedule", list(cases()))
def test_independent_absolute_energy_reference(name, spec, schedule):
    e = OpenEOSChannel(spec, max_step_s=0.0005)
    e.advance(0.02)
    s = e.snapshot()["state"]
    for method in ["DOP853", "Radau"]:
        ref = IndependentOpen(spec).integrate([0.02], [], method)[0]
        for field, tol in [
            ("masses_kg", 1e-7),
            ("velocities_m_s", 1e-5),
            ("temperatures_k", 1e-5),
            ("pressures_pa", 0.01),
        ]:
            np.testing.assert_allclose(list(s[field].values()), ref[field], rtol=0, atol=tol)


def test_pressure_bias_reverses_initial_positive_flow():
    e = OpenEOSChannel(request(velocity=2, bias=-10000), max_step_s=0.0005)
    assert e.snapshot()["state"]["mean_velocity_m_s"] > 0
    e.advance(0.2)
    state = e.snapshot()["state"]
    assert state["mean_velocity_m_s"] < 0
    assert abs(state["mass_residual_kg"]) < 1e-9
    assert abs(state["energy_residual_j"]) < 1e-5


def test_pressure_bias_orientation_symmetry():
    states = []
    for bias in [1000, -1000]:
        e = OpenEOSChannel(request(bias=bias))
        e.advance(0.1)
        states.append(e.snapshot()["state"])
    for field in ["masses_kg", "temperatures_k", "pressures_pa", "velocities_m_s"]:
        a = list(states[0][field].values())
        b = list(states[1][field].values())[::-1]
        np.testing.assert_allclose(
            a, -np.array(b) if field == "velocities_m_s" else b, rtol=0, atol=1e-8
        )


def test_full_numerical_entropy_flux_includes_rusanov_diffusion():
    spec = request()
    spec["boundaries"]["left"]["temperature_k"] = 330.0
    e = OpenEOSChannel(spec, max_step_s=0.0001)
    s = e.system
    _, point = s.evaluate(e._state, e._heaters)
    rL, T, p, u, phi = s.baths[0]
    rR = s.rho0[0]
    sL = s.eos.cv * np.log(T / s.eos.Tref) - s.eos.R * np.log(rL / s.eos.rref)
    expected = (
        -0.5
        * s.area
        * max(np.sqrt(s.eos.gamma * s.eos.R * T), np.sqrt(s.eos.gamma * s.eos.R * 300))
        * (rR * s.s0[0] - rL * sL)
    )
    assert point["entropy_face_flux"][0] == pytest.approx(expected, abs=1e-10)
    assert abs(expected) > 1e-6


def test_heater_command_json_restart_and_residual():
    spec = request(bias=1000, heat=500)
    e = OpenEOSChannel(spec)
    e.advance(0.02)
    powers = {f"c{i}": 500.0 if i == 6 else 0.0 for i in range(8)}
    e.step(powers, 0.02, expected_revision=e.revision, command_id="heater.move")
    cp = json.loads(json.dumps(e.checkpoint()))
    other = OpenEOSChannel(spec)
    other.restore(cp)
    y = np.array(cp["payload"]["coordinates"])
    rate = e.system.evaluate(y, e._heaters)[0]
    np.testing.assert_array_equal(e.residual(y, rate), 0)
    assert e.checkpoint() == cp
    e.advance(0.02)
    other.advance(0.02)
    assert e.checkpoint() == other.checkpoint()


@pytest.mark.parametrize(
    "defect", ["coordinate", "layout", "snapshot", "receipt", "digest", "identity", "clock"]
)
def test_corrupt_checkpoints_are_atomic(defect):
    e = OpenEOSChannel(request())
    e.step(e._heaters, 0.01, expected_revision=0, command_id="one")
    before = e.checkpoint()
    bad = deepcopy(before)
    if defect == "coordinate":
        bad["payload"]["coordinates"][0] = -1e9
    elif defect == "layout":
        bad["payload"]["coordinates"].pop()
    elif defect == "snapshot":
        bad["payload"]["snapshot"]["state"]["pressures_pa"]["c0"] += 1
    elif defect == "receipt":
        bad["payload"]["snapshot"]["history"][0]["command_id"] = "bad id"
    elif defect == "identity":
        bad["payload"]["input_sha256"] = "bad"
    elif defect == "clock":
        bad["payload"]["snapshot"]["revision"] = True
    bad["sha256"] = digest(bad["payload"]) if defect != "digest" else "bad"
    with pytest.raises((NetworkError, InertialGraphError)):
        e.restore(bad)
    assert e.checkpoint() == before


@pytest.mark.parametrize(
    "defect", ["pressure", "temperature", "velocity", "elevation", "unknown", "missing", "format"]
)
def test_invalid_bath_rejected(defect):
    spec = request()
    b = spec["boundaries"]["left"]
    if defect == "pressure":
        b["pressure_pa"] = -1
    elif defect == "temperature":
        b["temperature_k"] = 1
    elif defect == "velocity":
        b["velocity_m_s"] = 1000
    elif defect == "elevation":
        b["elevation_m"] = float("nan")
    elif defect == "unknown":
        b["unknown"] = 0
    elif defect == "missing":
        spec["boundaries"].pop("right")
    elif defect == "format":
        spec["format"] = "wrong"
    with pytest.raises((NetworkError, InertialGraphError)):
        OpenEOSChannel(spec)


@pytest.mark.parametrize("defect", ["mass", "energy", "entropy"])
def test_boundary_ledger_defects_rejected_with_rollback(monkeypatch, defect):
    e = OpenEOSChannel(request(bias=1000))
    before = e.checkpoint()
    cls = type(e.system)
    original = cls.evaluate
    index = {"mass": -8, "energy": -6, "entropy": -5}[defect]

    def broken(self, y, heaters):
        rate, point = original(self, y, heaters)
        rate[index] += 100.0
        return rate, point

    monkeypatch.setattr(cls, "evaluate", broken)
    with pytest.raises(InertialGraphError):
        e.advance(0.01)
    monkeypatch.setattr(cls, "evaluate", original)
    assert e.checkpoint() == before


@pytest.mark.parametrize("operation", ["stale", "duplicate", "mask", "duration", "source"])
def test_failed_commands_and_source_drift_are_atomic(monkeypatch, operation):
    import scripts.open_eos_channel as module

    e = OpenEOSChannel(request())
    e.step(e._heaters, 0.01, expected_revision=0, command_id="one")
    before = e.checkpoint()
    original = module.signature
    with pytest.raises((NetworkError, InertialGraphError)):
        if operation == "stale":
            e.step(e._heaters, 0.01, expected_revision=0, command_id="two")
        elif operation == "duplicate":
            e.step(e._heaters, 0.01, expected_revision=1, command_id="one")
        elif operation == "mask":
            e.step({}, 0.01, expected_revision=1, command_id="two")
        elif operation == "duration":
            e.advance(3)
        else:
            monkeypatch.setattr(module, "signature", lambda: {"drift": True})
            e.advance(0.01)
    monkeypatch.setattr(module, "signature", original)
    assert e.checkpoint() == before


def test_physical_energy_offset_is_transported_with_boundary_mass():
    outputs = []
    for offset in [0.0, 1e8]:
        spec = request(bias=1000)
        spec["fluid"]["energy_offset_j_kg"] = offset
        e = OpenEOSChannel(spec)
        e.advance(0.02)
        outputs.append(e.snapshot()["state"])
    for field in ["masses_kg", "velocities_m_s", "temperatures_k", "pressures_pa"]:
        np.testing.assert_array_equal(
            list(outputs[0][field].values()), list(outputs[1][field].values())
        )
    for side in ["left", "right"]:
        a = outputs[0]["boundary_transport"][side]
        b = outputs[1]["boundary_transport"][side]
        assert b["boundary_energy_j"] - a["boundary_energy_j"] == pytest.approx(
            1e8 * a["boundary_mass_kg"], abs=1e-6
        )
