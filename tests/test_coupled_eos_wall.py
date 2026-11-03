"""Finite-capacity thermal ports on two conservative inertial EOS loops."""

import json
from copy import deepcopy

import numpy as np
import pytest
from scipy.linalg import expm

from core.network import NetworkError
from scripts.coupled_eos_wall import CoupledEOSWall
from scripts.eos_wall_references import IndependentWall, request
from scripts.inertial_graph import InertialGraphError, digest


@pytest.mark.parametrize("cap", [0.001, 0.0005, 0.00025])
def test_uniform_three_inventory_analytic(cap):
    e = CoupledEOSWall(request(uniform=True), max_step_s=cap)
    C = [loop.system.m0[0] * loop.system.eos.cv for loop in e.loops]
    wall = e.capacity[0]
    a, b = e.ua[0]
    matrix = np.array(
        [[-a / C[0], 0, a / C[0]], [0, -b / C[1], b / C[1]], [a / wall, b / wall, -(a + b) / wall]]
    )
    expected = expm(matrix * 0.2) @ np.array([350.0, 280.0, 300.0])
    e.advance(0.2)
    state = e.snapshot()
    for i in range(2):
        np.testing.assert_allclose(
            list(state["loops"][i]["temperatures_k"].values()), expected[i], rtol=0, atol=1e-6
        )
        assert max(map(abs, state["loops"][i]["velocities_m_s"].values())) < 1e-9
    np.testing.assert_allclose(state["wall_temperatures_k"], expected[2], rtol=0, atol=1e-6)
    assert abs(state["energy_residual_j"]) < 1e-7
    assert abs(state["entropy_residual_j_k"]) < 2e-6
    assert state["generated_entropy_j_k"] > 0
    assert state["external_heat_j"] == pytest.approx(0, abs=1e-8)


@pytest.mark.parametrize("options", [{}, dict(gravity=True), dict(stiff=True), dict(ambient=True)])
def test_independent_absolute_energy_reference(options):
    spec = request(**options)
    e = CoupledEOSWall(spec, max_step_s=0.0005)
    e.advance(0.1)
    actual = e.snapshot()
    for method in ["DOP853", "Radau"]:
        expected = IndependentWall(spec).integrate([0.1], method)[0]
        np.testing.assert_allclose(
            actual["wall_temperatures_k"], expected["wall_temperatures_k"], rtol=0, atol=1e-5
        )
        for i in range(2):
            for field, tol in [
                ("temperatures_k", 1e-5),
                ("pressures_pa", 0.02),
                ("masses_kg", 1e-7),
                ("velocities_m_s", 1e-5),
            ]:
                np.testing.assert_allclose(
                    list(actual["loops"][i][field].values()),
                    expected["loops"][i][field],
                    rtol=0,
                    atol=tol,
                )


def test_ambient_is_only_external_heat():
    e = CoupledEOSWall(request(ambient=True))
    e.advance(0.1)
    s = e.snapshot()
    assert s["external_heat_j"] == pytest.approx(
        s["wall_ledgers"]["wall_external_heat_j"], abs=1e-8
    )
    assert s["external_heat_j"] < 0
    for loop in s["loops"]:
        assert abs(loop["mass_residual_kg"]) < 1e-10
        assert abs(loop["momentum_residual_kg_m_s"]) < 1e-8


def test_exact_json_restart_and_pure_residual():
    spec = request(gravity=True)
    e = CoupledEOSWall(spec)
    e.advance(0.02)
    cp = json.loads(json.dumps(e.checkpoint()))
    restored = CoupledEOSWall(spec)
    restored.restore(cp)
    y = np.array(cp["payload"]["coordinates"])
    rate = e.evaluate(y)[0]
    np.testing.assert_array_equal(e.residual(y, rate), 0)
    assert e.checkpoint() == cp
    e.advance(0.03)
    restored.advance(0.03)
    assert e.checkpoint() == restored.checkpoint()


@pytest.mark.parametrize(
    "kind", ["coordinate", "snapshot", "identity", "digest", "layout", "clock"]
)
def test_checkpoint_corruption_rolls_back(kind):
    e = CoupledEOSWall(request())
    e.advance(0.01)
    cp = e.checkpoint()
    bad = deepcopy(cp)
    if kind == "coordinate":
        bad["payload"]["coordinates"][e.wall_start] = -1e10
    elif kind == "snapshot":
        bad["payload"]["snapshot"]["wall_temperatures_k"][0] += 1
    elif kind == "identity":
        bad["payload"]["input_sha256"] = "wrong"
    elif kind == "layout":
        bad["payload"]["coordinates"].pop()
    elif kind == "clock":
        bad["payload"]["snapshot"]["revision"] = True
    bad["sha256"] = digest(bad["payload"]) if kind != "digest" else "wrong"
    with pytest.raises((NetworkError, InertialGraphError)):
        e.restore(bad)
    assert e.checkpoint() == cp


@pytest.mark.parametrize(
    "kind",
    [
        "port",
        "bool_port",
        "negative_ua",
        "capacity",
        "empty",
        "unknown",
        "loops",
        "wall_temperature",
        "ambient_ua",
    ],
)
def test_invalid_assembly(kind):
    spec = request()
    w = spec["walls"][0]
    if kind == "port":
        w["cells"][0] = 999
    elif kind == "bool_port":
        w["cells"][0] = True
    elif kind == "negative_ua":
        w["conductance_w_k"][0] = -1
    elif kind == "capacity":
        w["capacity_j_k"] = 0
    elif kind == "empty":
        spec["walls"] = []
    elif kind == "unknown":
        w["unknown"] = 1
    elif kind == "loops":
        spec["loops"].pop()
    elif kind == "wall_temperature":
        w["initial_temperature_k"] = 1
    elif kind == "ambient_ua":
        w["ambient_ua_w_k"] = -1
    with pytest.raises(NetworkError):
        CoupledEOSWall(spec)


def test_failed_stage_is_atomic(monkeypatch):
    e = CoupledEOSWall(request())
    before = e.checkpoint()
    original = CoupledEOSWall.evaluate

    def fail(self, y):
        if np.any(y):
            raise InertialGraphError("injected", "Stage failure")
        return original(self, y)

    monkeypatch.setattr(CoupledEOSWall, "evaluate", fail)
    with pytest.raises(InertialGraphError):
        e.advance(0.1)
    monkeypatch.setattr(CoupledEOSWall, "evaluate", original)
    assert e.checkpoint() == before


def test_mismatched_checkpoint_step_and_source_drift(monkeypatch):
    import scripts.coupled_eos_wall as module

    e = CoupledEOSWall(request())
    before = e.checkpoint()
    with pytest.raises(NetworkError):
        CoupledEOSWall(request(), max_step_s=0.0005).restore(before)
    original = module.signature
    monkeypatch.setattr(module, "signature", lambda: {"changed": True})
    with pytest.raises(InertialGraphError):
        e.advance(0.1)
    monkeypatch.setattr(module, "signature", original)
    assert e.checkpoint() == before


@pytest.mark.parametrize("duration", [0, -1, 3, float("nan"), True])
def test_invalid_duration_is_atomic(duration):
    e = CoupledEOSWall(request())
    before = e.checkpoint()
    with pytest.raises(NetworkError):
        e.advance(duration)
    assert e.checkpoint() == before


@pytest.mark.parametrize("defect", ["energy", "entropy"])
def test_deliberately_broken_ledger_is_rejected(monkeypatch, defect):
    e = CoupledEOSWall(request())
    before = e.checkpoint()
    original = CoupledEOSWall.evaluate

    def broken(self, y):
        rate, ratio = original(self, y)
        if defect == "energy":
            rate[self.wall_start] += 1.0
        else:
            rate[-1] -= 1000.0
        return rate, ratio

    monkeypatch.setattr(CoupledEOSWall, "evaluate", broken)
    with pytest.raises(InertialGraphError):
        e.advance(0.01)
    monkeypatch.setattr(CoupledEOSWall, "evaluate", original)
    assert e.checkpoint() == before


def test_distinct_caloric_fluids_exchange_energy_conservatively():
    spec = request()
    spec["loops"][1]["fluid"].update(
        cv_j_kg_k=800.0, gamma=1.2, pressure_shift_pa=500000.0, energy_offset_j_kg=30000.0
    )
    e = CoupledEOSWall(spec, max_step_s=0.0005)
    e.advance(0.05)
    actual = e.snapshot()
    reference = IndependentWall(spec).integrate([0.05], "DOP853")[0]
    np.testing.assert_allclose(
        actual["wall_temperatures_k"], reference["wall_temperatures_k"], rtol=0, atol=1e-5
    )
    assert abs(actual["energy_residual_j"]) < 1e-7


def test_stiff_wall_and_repeated_ports_limit_actual_step():
    spec = request()
    for loop, T in zip(spec["loops"], [300.001, 299.999]):
        loop["initial"]["temperatures_k"] = [T] * 8
    for wall in spec["walls"]:
        wall["cells"] = [1, 6]
        wall["conductance_w_k"] = [1e7, 1e7]
    e = CoupledEOSWall(spec)
    result = e.advance(0.00001)
    assert result["work"]["maximum_internal_step_s"] < 0.000002
    assert result["work"]["maximum_stability_fraction"] <= 1
    assert result["work"]["substeps"] > 1
    assert abs(result["energy_residual_j"]) < 1e-7


def test_stiff_large_contrast_requires_accuracy_refinement():
    spec = request()
    for wall in spec["walls"]:
        wall["cells"] = [1, 6]
        wall["conductance_w_k"] = [1e7, 1e7]
    coarse = CoupledEOSWall(spec)
    before = coarse.checkpoint()
    with pytest.raises(InertialGraphError):
        coarse.advance(0.00001)
    assert coarse.checkpoint() == before
    refined = CoupledEOSWall(spec, max_step_s=1e-8)
    state = refined.advance(0.00001)
    assert abs(state["entropy_residual_j_k"]) < 2e-6
