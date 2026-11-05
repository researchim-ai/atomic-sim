"""Finite end-volume fluid/wall coupling, local ledgers and transactional replay."""

import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import numpy as np
import pytest

from atomic_model_hem import eos
from atomic_model_hem_system import model as module
from atomic_model_hem_system.cases import baseline, matrix, programs
from atomic_model_hem_system.model import FiniteHEMSystem, isochoric_capacity
from atomic_model_workspace.common import digest


def test_uniform_closed_rest_has_zero_mass_energy_flux_and_no_stock_drift():
    m = FiniteHEMSystem(baseline())
    start = m.snapshot()
    r = m.advance(0.002)
    for a, b in zip(start["cells"], r["cells"]):
        assert a == b
    assert all(w["energy_change_j"] == 0 for w in r["walls"])
    face = r["local_ledgers"]
    assert face["integrated_face_mass_kg"][0] == face["integrated_face_mass_kg"][-1] == 0
    assert face["integrated_face_energy_j"][0] == face["integrated_face_energy_j"][-1] == 0
    assert r["global_ledger"]["endwall_impulse_kg_m_s"] == 0


def test_face_flux_and_wall_exchange_cancel_in_total_rhs():
    spec = dict(matrix())["wall_driven_evaporation"]
    m = FiniteHEMSystem(spec)
    rhs, data = m.evaluate(m._y, m._controls)
    flux = data["shared_flux"]
    assert flux.shape == (3, m.n + 1)
    assert np.array_equal(flux[[0, 2], :][:, [0, -1]], np.zeros((2, 2)))
    total_energy = (
        sum(rhs[2 * m.n : 3 * m.n])
        + m.eref * sum(rhs[: m.n])
        + np.dot(m.capacity, rhs[m.wall_slice])
    )
    external = sum(m._controls["fluid"]) + sum(data["wall_external_w"])
    assert total_energy == pytest.approx(external, abs=1e-8)
    assert sum(rhs[m.n : 2 * m.n]) == pytest.approx(m.area * (flux[1, 0] - flux[1, -1]), abs=1e-8)
    assert np.all(data["wall_to_fluid_w"] > 0)
    assert rhs[m.production_index] > 0


def test_nonperiodic_terminal_exchange_changes_real_terminal_masses():
    m = FiniteHEMSystem(dict(matrix())["pressure_transfer"])
    start = m.snapshot()
    r = m.advance(0.002)
    assert r["cells"][0]["mass_kg"] < start["cells"][0]["mass_kg"]
    assert r["cells"][-1]["mass_kg"] > start["cells"][-1]["mass_kg"]
    assert abs(r["global_ledger"]["mass_residual_kg"]) < 1e-10
    assert max(map(abs, r["local_ledgers"]["mass_residual_kg"])) < 1e-10


def test_unequal_terminal_volumes_and_actual_cfl_limit():
    spec = dict(matrix())["unequal_volumes"]
    spec["channel"]["max_step_s"] = 0.001
    m = FiniteHEMSystem(spec)
    r = m.advance(0.002)
    assert r["cells"][0]["volume_m3"] != r["cells"][-1]["volume_m3"]
    assert r["maximum_step_s"] <= m.cfl * min(m.volumes) / m.area / m.wave


def test_reflection_changes_momentum_by_wall_impulse_without_external_energy():
    spec = dict(matrix())["reflected_momentum"]
    for p in spec["channel"]["initial"]:
        p["velocity_m_s"] = 1.0
    for p in spec["terminals"].values():
        p["initial"]["velocity_m_s"] = 1.0
    m = FiniteHEMSystem(spec)
    start = m.snapshot()
    r = m.advance(0.002)
    delta = sum(c["momentum_kg_m_s"] for c in r["cells"]) - sum(
        c["momentum_kg_m_s"] for c in start["cells"]
    )
    assert abs(delta) > 1
    assert delta == pytest.approx(r["global_ledger"]["endwall_impulse_kg_m_s"], abs=1e-8)
    assert abs(r["global_ledger"]["energy_residual_j"]) < 1e-5


def test_wall_inventory_supplies_actual_evaporation_without_external_heat():
    spec = dict(matrix())["wall_driven_evaporation"]
    m = FiniteHEMSystem(spec)
    start = m.snapshot()
    r = m.advance(0.008)
    assert any(c["phase"] == "vapor" for c in r["cells"])
    fluid_delta = sum(
        c["total_energy_j"] - s["total_energy_j"] for c, s in zip(r["cells"], start["cells"])
    )
    wall_delta = sum(w["energy_change_j"] for w in r["walls"])
    assert fluid_delta > 0 and wall_delta < 0
    assert fluid_delta + wall_delta == pytest.approx(0, abs=1e-6)
    assert (
        r["global_ledger"]["external_fluid_heat_j"]
        == r["global_ledger"]["external_wall_heat_j"]
        == 0
    )
    assert r["global_ledger"]["total_entropy_change_j_k"] > 0
    assert r["global_ledger"]["thermal_entropy_production_j_k"] > 0
    assert max(abs(w["energy_residual_j"]) for w in r["walls"]) < 1e-6


def test_ambient_and_wall_power_are_in_external_energy_once():
    m = FiniteHEMSystem(dict(matrix())["wall_heater_ambient"])
    r = m.advance(0.002)
    assert r["global_ledger"]["external_wall_heat_j"] < 4 * 1000 * 0.002
    assert r["global_ledger"]["external_fluid_heat_j"] == 0
    assert abs(r["global_ledger"]["energy_residual_j"]) < 1e-6


@pytest.mark.parametrize("kind", ["liquid", "vapor", "mixture"])
def test_isochoric_capacity_positive_and_single_phase_matches_if97(kind):
    s = baseline(kind)
    m = FiniteHEMSystem(s)
    state = m.states0[0]
    cv = isochoric_capacity(state)
    assert np.isfinite(cv) and cv > 0
    if state.region != 4:
        from physics.materials.iapws import properties_pt

        assert cv == properties_pt(state.pressure_pa, state.temperature_k).cv_j_kg_k
    else:
        assert 1 / cv == pytest.approx(eos.derivatives(state)["temperature_wrt_energy"], rel=2e-5)


def test_local_and_wall_balance_damage_refused_without_mutation():
    m = FiniteHEMSystem(baseline())
    before = m.checkpoint()
    for index in [
        0,
        m.n,
        2 * m.n,
        m.wall_slice.start,
        m.wall_exchange_slice.start,
        m.wall_external_slice.start,
    ]:
        candidate = m._y.copy()
        candidate[index] += 0.001
        with pytest.raises(ValueError):
            m._observe(candidate, 0.0, m._controls, None, 0)
    assert m.checkpoint() == before


def test_equal_outer_transport_offsets_cannot_hide_in_global_balance():
    m = FiniteHEMSystem(baseline())
    y = m._y.copy()
    faces = y[m.face_slice].reshape(3, m.n + 1)
    faces[0, :] = 0.01
    with pytest.raises(ValueError, match="Closed endwall"):
        m._observe(y, 0.0, m._controls, None, 0)


def test_pure_residual_units_and_detached_controls():
    m = FiniteHEMSystem(baseline())
    before = m.checkpoint()
    rhs = m.evaluate(m._y, m._controls)[0]
    assert np.max(abs(m.residual(m._y, rhs))) == 0
    assert m.checkpoint() == before
    assert m.contract()["ledger_units"]["face_momentum"] == "kg m/s"
    control = m.control()
    control["walls"]["w0"] = 100.0
    assert m.control()["walls"]["w0"] == 0


@pytest.mark.parametrize("cut", [0.001, 0.003])
def test_exact_json_restart_preserves_wall_frame_and_interfaces(cut):
    spec = dict(matrix())["wall_heater_ambient"]
    a = FiniteHEMSystem(spec)
    a.advance(cut)
    cp = json.loads(json.dumps(a.checkpoint()))
    b = FiniteHEMSystem(spec)
    assert b.restore(cp) == a.snapshot()
    assert a.advance(0.002) == b.advance(0.002)


@pytest.mark.parametrize(
    "field", ["clock", "coordinates", "wall", "face", "controller", "calls", "source"]
)
def test_rehashed_checkpoint_corruption_rejected(field):
    m = FiniteHEMSystem(baseline())
    m.step(m.control(), 0.001, expected_revision=0, controller_state={"tick": 1})
    before = m.checkpoint()
    cp = deepcopy(before)
    p = cp["payload"]
    if field == "clock":
        p["observation"]["time_s"] += 1
    elif field == "coordinates":
        p["coordinates"][m.wall_slice.start] += 0.001
    elif field == "wall":
        p["observation"]["walls"][0]["temperature_k"] += 1
    elif field == "face":
        p["observation"]["local_ledgers"]["integrated_face_mass_kg"][1] += 0.1
    elif field == "controller":
        p["observation"]["controller_state"]["tick"] = 2
    elif field == "calls":
        p["calls"][0]["duration_s"] += 0.001
    else:
        p["source"] = {"changed": True}
    cp["sha256"] = digest(p)
    with pytest.raises(ValueError):
        m.restore(cp)
    assert m.checkpoint() == before


def test_late_stage_failure_rolls_back_fluid_wall_and_controller(monkeypatch):
    m = FiniteHEMSystem(baseline())
    before = m.checkpoint()
    original = m.evaluate
    calls = []

    def fail(*a):
        calls.append(1)
        if len(calls) == 8:
            raise ValueError("late stage failure")
        return original(*a)

    monkeypatch.setattr(m, "evaluate", fail)
    control = m.control()
    control["walls"]["w0"] = 1000.0
    with pytest.raises(ValueError):
        m.step(control, 0.002, expected_revision=0, controller_state={"tick": 1})
    assert len(calls) == 8 and m.checkpoint() == before


def test_missing_heat_exchange_on_fluid_side_rejected(monkeypatch):
    m = FiniteHEMSystem(dict(matrix())["wall_driven_evaporation"])
    before = m.checkpoint()
    original = m._evaluate_point

    def broken(*a):
        rhs, data = original(*a)
        rhs[2 * m.n + m.ports[0]] -= data["wall_to_fluid_w"][0]
        return rhs, data

    monkeypatch.setattr(m, "_evaluate_point", broken)
    with pytest.raises(ValueError, match="ledger"):
        m.advance(0.001)
    assert m.checkpoint() == before


def test_late_source_drift_cannot_publish_state(monkeypatch):
    m = FiniteHEMSystem(baseline())
    before = m.checkpoint()
    original = module.signature
    calls = []

    def changed():
        calls.append(1)
        return original() if len(calls) == 1 else {"changed": True}

    monkeypatch.setattr(module, "signature", changed)
    with pytest.raises(ValueError, match="source changed"):
        m.advance(0.001)
    monkeypatch.setattr(module, "signature", original)
    assert m.checkpoint() == before


def test_step_budget_and_stale_revision_are_atomic(monkeypatch):
    m = FiniteHEMSystem(baseline())
    before = m.checkpoint()
    with pytest.raises(ValueError):
        m.step(m.control(), 0.001, expected_revision=True, controller_state={})
    monkeypatch.setattr(module, "MAX_STEPS", 1)
    with pytest.raises(ValueError):
        m.advance(0.002)
    assert m.checkpoint() == before


def test_concurrent_command_revision_only_commits_once():
    m = FiniteHEMSystem(baseline())
    control = m.control()

    def step(i):
        try:
            return m.step(control, 0.001, expected_revision=0, controller_state={"owner": i})[
                "revision"
            ]
        except ValueError:
            return "stale"

    with ThreadPoolExecutor(max_workers=2) as pool:
        result = list(pool.map(step, [1, 2]))
    assert sorted(map(str, result)) == ["1", "stale"]


@pytest.mark.parametrize(
    "bad",
    [
        "area",
        "volume",
        "capacity",
        "conductance",
        "wall_id",
        "port",
        "temperature",
        "schema",
        "fluid_mask",
    ],
)
def test_invalid_geometry_wall_or_controls_refused(bad):
    spec = baseline()
    if bad == "area":
        spec["channel"]["geometry"]["area_m2"] = 0
    elif bad == "volume":
        spec["terminals"]["left"]["volume_m3"] = 0
    elif bad == "capacity":
        spec["walls"][0]["capacity_j_k"] = 0
    elif bad == "conductance":
        spec["walls"][0]["conductance_w_k"] = -1
    elif bad == "wall_id":
        spec["walls"][1]["id"] = spec["walls"][0]["id"]
    elif bad == "port":
        spec["walls"][0]["cell"] = "missing"
    elif bad == "temperature":
        spec["walls"][0]["initial_temperature_k"] = 901.0
    elif bad == "schema":
        spec["schema_version"] = True
    else:
        m = FiniteHEMSystem(spec)
        control = m.control()
        control["fluid"].pop("left")
        with pytest.raises(ValueError):
            m.step(control, 0.001, expected_revision=0, controller_state={})
        return
    with pytest.raises(ValueError):
        FiniteHEMSystem(spec)


def test_online_program_has_joint_controller_and_physics_state():
    _, spec, program = next(c for c in programs() if c[0] == "online_walls_power")
    from atomic_model_hem_system.workflow import apply

    m = FiniteHEMSystem(spec)
    for call in program:
        result = apply(m, call)
    assert result["controller_state"] == {"tick": 2}
    assert result["global_ledger"]["external_wall_heat_j"] == pytest.approx(96.0, abs=1e-7)
    assert result["controls"]["walls"] == {name: 0.0 for name in m.wall_ids}
