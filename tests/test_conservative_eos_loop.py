"""Conservation, analytic solutions, transactions and spatial refinement."""

import json
from copy import deepcopy

import numpy as np
import pytest

from core.network import NetworkError
from scripts.conservative_eos_loop import ConservativeEOSLoop
from scripts.conservative_loop_references import IndependentLoop, cases, request
from scripts.inertial_graph import InertialGraphError, digest


@pytest.mark.parametrize("name,spec,schedule", list(cases()))
def test_independent_absolute_energy_reference(name, spec, schedule):
    engine = ConservativeEOSLoop(spec)
    engine.advance(0.1)
    actual = engine.snapshot()["state"]
    reference = IndependentLoop(spec).integrate(0.1, [], "DOP853", [0.1])[-1]
    for key, tolerance in [
        ("masses_kg", 1e-7),
        ("velocities_m_s", 1e-5),
        ("temperatures_k", 1e-5),
        ("pressures_pa", 0.01),
    ]:
        np.testing.assert_allclose(
            list(actual[key].values()), reference[key], rtol=0, atol=tolerance
        )
    assert abs(actual["mass_residual_kg"]) < 1e-9
    assert abs(actual["energy_residual_j"]) < 1e-5
    assert abs(actual["entropy_residual_j_k"]) < 2e-6


@pytest.mark.parametrize("shift", [1e5, 1e8])
def test_isothermal_hydrostatic_rest(shift):
    spec = request(height=10, gravity=9.81, shift=shift)
    engine = ConservativeEOSLoop(spec)
    engine.advance(0.2)
    s = engine.snapshot()["state"]
    assert max(map(abs, s["velocities_m_s"].values())) < 1e-9
    np.testing.assert_allclose(
        list(s["pressures_pa"].values()), spec["initial"]["pressures_pa"], rtol=0, atol=1e-7
    )


def test_uniform_drag_analytic_energy_partition():
    engine = ConservativeEOSLoop(request(velocity=10, drag=0.4, export=0.25))
    initial = engine.snapshot()["state"]
    engine.advance(0.2)
    state = engine.snapshot()["state"]
    np.testing.assert_allclose(
        list(state["velocities_m_s"].values()), 10 * np.exp(-0.4 * 0.2), rtol=0, atol=1e-8
    )
    loss = initial["kinetic_energy_j"] * (1 - np.exp(-2 * 0.4 * 0.2))
    assert state["exported_drag_heat_j"] == pytest.approx(0.25 * loss, abs=1e-7)
    assert state["internal_energy_change_j"] == pytest.approx(0.75 * loss, abs=1e-7)


def test_uniform_heat_analytic():
    spec = request()
    spec["heat"]["power_w"] = [10.0] * 8
    engine = ConservativeEOSLoop(spec)
    mass = engine.snapshot()["state"]["masses_kg"]["c0"]
    engine.advance(0.2)
    state = engine.snapshot()["state"]
    np.testing.assert_allclose(
        list(state["temperatures_k"].values()), 300 + 2 / (mass * 1000), rtol=0, atol=1e-10
    )
    assert state["external_heat_j"] == pytest.approx(16, abs=1e-10)
    assert max(map(abs, state["velocities_m_s"].values())) < 1e-10


def test_checkpoint_json_restart_and_command():
    spec = request(height=10, gravity=9.81, heat=500)
    original = ConservativeEOSLoop(spec)
    original.advance(0.05)
    powers = {f"c{i}": (500.0 if i == 6 else 0.0) for i in range(8)}
    original.step(powers, 0.05, expected_revision=original.revision, command_id="heater.move")
    checkpoint = json.loads(json.dumps(original.checkpoint()))
    restored = ConservativeEOSLoop(spec)
    restored.restore(checkpoint)
    original.advance(0.05)
    restored.advance(0.05)
    assert original.checkpoint() == restored.checkpoint()


@pytest.mark.parametrize(
    "failure", ["stale", "duplicate", "mask", "duration", "coordinate", "snapshot"]
)
def test_failed_operations_are_atomic(failure):
    engine = ConservativeEOSLoop(request())
    powers = engine.snapshot()["active_heat_power_w"]
    engine.step(powers, 0.01, expected_revision=0, command_id="one")
    before = engine.checkpoint()
    with pytest.raises((NetworkError, InertialGraphError, ValueError)):
        if failure == "stale":
            engine.step(powers, 0.01, expected_revision=0, command_id="two")
        elif failure == "duplicate":
            engine.step(powers, 0.01, expected_revision=engine.revision, command_id="one")
        elif failure == "mask":
            engine.step({}, 0.01, expected_revision=engine.revision, command_id="two")
        elif failure == "duration":
            engine.advance(11)
        else:
            bad = deepcopy(before)
            if failure == "coordinate":
                bad["payload"]["coordinates"][0] = -1e10
            else:
                bad["payload"]["snapshot"]["state"]["pressures_pa"]["c0"] += 1
            bad["sha256"] = digest(bad["payload"])
            engine.restore(bad)
    assert engine.checkpoint() == before


def test_first_order_contact_spatial_refinement():
    errors = []
    for n in [16, 32, 64]:
        engine = ConservativeEOSLoop(request(n, velocity=20, wave=0.01), max_step_s=0.0002)
        engine.advance(0.05)
        actual = np.array(list(engine.snapshot()["state"]["densities_kg_m3"].values()))
        x = (np.arange(n) + 0.5) * 100 / n
        exact = 2.5 * (1 + 0.01 * np.sinc(1 / n) * np.sin(2 * np.pi * (x - 20 * 0.05) / 100))
        errors.append(np.mean(abs(actual - exact)))
    assert errors[0] / errors[1] >= 1.5
    assert errors[1] / errors[2] >= 1.5


@pytest.mark.parametrize(
    "field,value",
    [
        ("cells", True),
        ("cells", 3),
        ("cells", 129),
        ("length_m", 0),
        ("area_m2", -1),
        ("gravity_m_s2", -1),
        ("elevations_m", [0.0]),
    ],
)
def test_invalid_geometry(field, value):
    spec = request()
    spec["geometry"][field] = value
    with pytest.raises(NetworkError):
        ConservativeEOSLoop(spec)


def test_residual_is_pure_and_source_drift_blocks_mutation(monkeypatch):
    import scripts.conservative_eos_loop as module

    engine = ConservativeEOSLoop(request(heat=500))
    before = engine.checkpoint()
    y = np.array(before["payload"]["coordinates"])
    rate = engine.system.evaluate(y, engine.snapshot()["active_heat_power_w"])[0]
    np.testing.assert_array_equal(engine.residual(y, rate), 0)
    assert engine.checkpoint() == before
    original = module.signature
    monkeypatch.setattr(module, "signature", lambda: {"changed": True})
    with pytest.raises(InertialGraphError):
        engine.advance(0.1)
    monkeypatch.setattr(module, "signature", original)
    assert engine.checkpoint() == before


def test_rhs_work_counts_all_integration_evaluations(monkeypatch):
    from scripts.conservative_eos_loop import ConservativeLoopSystem

    engine = ConservativeEOSLoop(request(), max_step_s=0.002)
    calls = []
    original = ConservativeLoopSystem.evaluate

    def counted(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(ConservativeLoopSystem, "evaluate", counted)
    result = engine.advance(0.004)
    assert result["work"]["substeps"] == 2
    assert result["work"]["rhs"] == 6
    # One additional RHS forms the final observation; none are hidden per stage.
    assert len(calls) == result["work"]["rhs"] + 1
