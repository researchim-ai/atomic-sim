"""Shared-flux HEM conservation, phase transport, acoustic refinement and transactions."""

import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import numpy as np
import pytest

from atomic_model_hem import eos
from atomic_model_hem_flow import model as module
from atomic_model_hem_flow import workflow
from atomic_model_hem_flow.cases import acoustic, matrix, uniform
from atomic_model_hem_flow.model import PeriodicHEMChannel, sound_bound
from atomic_model_workspace.common import atomic_json, digest, read


@pytest.mark.parametrize(
    "kind,velocity", [("mixture", 0.0), ("mixture", 1.0), ("liquid", 1.0), ("vapor", -1.0)]
)
def test_uniform_state_is_preserved(kind, velocity):
    m = PeriodicHEMChannel(uniform(kind=kind, velocity=velocity))
    initial = m.snapshot()
    r = m.advance(0.002)
    assert np.max(abs(m._y)) == 0
    for a, b in zip(initial["cells"], r["cells"]):
        assert a == b
    assert abs(r["ledgers"]["mass_residual_kg"]) < 1e-12
    assert abs(r["ledgers"]["energy_residual_j"]) < 1e-8


def test_each_face_shared_once_and_total_rhs_has_only_heat_sources():
    m = PeriodicHEMChannel(acoustic(4, 100.0))
    heaters = np.array([10.0, 20.0, 30.0, 40.0])
    derivative, faces = m.evaluate(m._y, heaters)
    assert faces.shape == (3, 4)
    rates = derivative[:12].reshape(3, 4)
    assert rates.sum(axis=1) == pytest.approx([0.0, 0.0, 100.0], abs=1e-8)
    expected = -m.area * (faces - np.roll(faces, 1, axis=1))
    expected[2] += heaters
    assert np.array_equal(rates, expected)


def test_mass_momentum_and_total_energy_are_separate_invariants():
    m = PeriodicHEMChannel(acoustic(4, 100.0))
    r = m.advance(0.008)
    ledger = r["ledgers"]
    assert abs(ledger["mass_residual_kg"]) < 1e-9
    assert abs(ledger["momentum_residual_kg_m_s"]) < 1e-8
    assert abs(ledger["energy_residual_j"]) < 1e-5
    assert ledger["numerical_entropy_production_j_k"] >= -1e-5
    assert max(abs(c["velocity_m_s"]) for c in r["cells"]) > 0


def test_actual_mass_transport_for_quality_contact_and_pressure_diagnostic():
    spec = dict(matrix())["isobaric_quality_contact"]
    m = PeriodicHEMChannel(spec)
    initial = m.snapshot()
    r = m.advance(0.008)
    assert any(
        abs(a["mass_kg"] - b["mass_kg"]) > 1e-5 for a, b in zip(initial["cells"], r["cells"])
    )
    assert max(abs(c["pressure_pa"] - 1e6) for c in r["cells"]) < 0.05
    assert abs(r["ledgers"]["mass_residual_kg"]) < 1e-9


def test_uniform_evaporation_and_condensation_keep_total_energy():
    spec = dict(matrix())["uniform_evaporation"]
    m = PeriodicHEMChannel(spec)
    first = m.advance(0.018)
    assert all(c["phase"] == "vapor" for c in first["cells"])
    assert first["ledgers"]["external_heat_j"] == pytest.approx(4 * 20000 * 0.018, abs=1e-8)
    second = m.step([-20000.0] * 4, 0.018, expected_revision=1, controller_state={"tick": 1})
    assert all(c["phase"] == "two_phase" for c in second["cells"])
    assert all(c["quality"] == pytest.approx(0.999, abs=1e-9) for c in second["cells"])
    assert abs(second["ledgers"]["external_heat_j"]) < 1e-7
    assert abs(second["ledgers"]["energy_residual_j"]) < 1e-5
    assert "diagnostic" in second["entropy_acceptance"]


@pytest.mark.parametrize("x", [0.0, 0.000001, 0.5, 0.999999, 1.0])
def test_phase_endpoint_characteristic_bound_is_finite_without_quality_clipping(x):
    state = eos.px(1e6, x)
    value = sound_bound(state)
    assert np.isfinite(value) and value > 0
    if 1e-5 < x < 1 - 1e-5:
        assert value**2 == pytest.approx(
            eos.derivatives(state)["equilibrium_sound_speed_squared_m2_s2"], rel=2e-6
        )


def test_pure_residual_and_detached_observation():
    m = PeriodicHEMChannel(acoustic(4, 100.0))
    before = m.checkpoint()
    rate = m.evaluate(m._y, m._heat)[0]
    assert np.max(abs(m.residual(m._y, rate))) == 0
    assert m.checkpoint() == before
    snapshot = m.snapshot()
    snapshot["cells"][0]["mass_kg"] = 0
    assert m.snapshot()["cells"][0]["mass_kg"] > 0


@pytest.mark.parametrize("cut", [0.001, 0.003, 0.008])
def test_json_restart_exact_for_real_transport(cut):
    spec = acoustic(4, 100.0)
    m = PeriodicHEMChannel(spec)
    m.advance(cut)
    cp = json.loads(json.dumps(m.checkpoint()))
    other = PeriodicHEMChannel(spec)
    assert other.restore(cp) == m.snapshot()
    assert m.advance(0.003) == other.advance(0.003)


@pytest.mark.parametrize("field", ["clock", "coordinates", "controller", "calls", "cell"])
def test_rehashed_checkpoint_mutation_rejected(field):
    m = PeriodicHEMChannel(acoustic(4, 100.0))
    m.step([0.0] * 4, 0.002, expected_revision=0, controller_state={"tick": 1})
    before = m.checkpoint()
    cp = deepcopy(before)
    p = cp["payload"]
    if field == "clock":
        p["observation"]["time_s"] += 0.001
    elif field == "coordinates":
        p["coordinates"][0] += 0.001
    elif field == "controller":
        p["observation"]["controller_state"]["tick"] = 2
    elif field == "calls":
        p["calls"][0]["duration_s"] += 0.001
    else:
        p["observation"]["cells"][0]["pressure_pa"] += 1
    cp["sha256"] = digest(p)
    with pytest.raises(ValueError):
        m.restore(cp)
    assert m.checkpoint() == before


def test_late_flux_failure_rolls_back_controller_and_all_inventories(monkeypatch):
    m = PeriodicHEMChannel(acoustic(4, 100.0))
    before = m.checkpoint()
    original = m.evaluate
    calls = []

    def broken(*a):
        calls.append(1)
        if len(calls) == 8:
            raise ValueError("late stage failure")
        return original(*a)

    monkeypatch.setattr(m, "evaluate", broken)
    with pytest.raises(ValueError):
        m.step([100.0] * 4, 0.002, expected_revision=0, controller_state={"tick": 1})
    assert len(calls) == 8 and m.checkpoint() == before


def test_broken_energy_flux_or_heat_journal_is_not_silently_accepted(monkeypatch):
    m = PeriodicHEMChannel(uniform())
    before = m.checkpoint()
    original = m.evaluate

    def broken(*a):
        rhs, faces = original(*a)
        rhs[2 * m.n] += 1000.0
        return rhs, faces

    monkeypatch.setattr(m, "evaluate", broken)
    with pytest.raises(ValueError, match="ledger"):
        m.advance(0.001)
    assert m.checkpoint() == before


def test_wave_domain_failure_rolls_back_without_flux_fallback():
    spec = uniform(kind="liquid")
    spec["wave_speed_bound_m_s"] = 100.0
    with pytest.raises(ValueError, match="wave"):
        PeriodicHEMChannel(spec)


def test_low_cfl_is_enforced_and_actual_step_reported():
    spec = uniform()
    spec["cfl"] = 0.01
    spec["max_step_s"] = 0.001
    m = PeriodicHEMChannel(spec)
    r = m.advance(0.001)
    assert r["maximum_step_s"] <= 0.01 * m.dx / m.wave
    assert r["steps"] > 10


@pytest.mark.parametrize("bad", ["cells", "cfl", "wave", "heaters", "velocity", "schema"])
def test_invalid_inputs_refused(bad):
    spec = uniform()
    if bad == "cells":
        spec["geometry"]["cells"] = True
    elif bad == "cfl":
        spec["cfl"] = 0.5
    elif bad == "wave":
        spec["wave_speed_bound_m_s"] = float("nan")
    elif bad == "heaters":
        spec["heat_power_w"] = [0.0]
    elif bad == "velocity":
        spec["initial"][0]["velocity_m_s"] = 1000.0
    else:
        spec["schema_version"] = True
    with pytest.raises(ValueError):
        PeriodicHEMChannel(spec)


def test_call_step_budget_and_tiny_clock_refuse_transactionally(monkeypatch):
    m = PeriodicHEMChannel(uniform())
    m.advance(0.001)
    before = m.checkpoint()
    with pytest.raises(ValueError):
        m.advance(1e-300)
    with pytest.raises(ValueError):
        m.step([0.0] * 4, 0.001, expected_revision=0, controller_state={})
    monkeypatch.setattr(module, "MAX_STEPS", 1)
    with pytest.raises(ValueError, match="budget"):
        m.advance(0.002)
    assert m.checkpoint() == before


def test_concurrent_revision_only_one_command_commits():
    m = PeriodicHEMChannel(uniform())

    def step(i):
        try:
            return m.step([0.0] * 4, 0.001, expected_revision=0, controller_state={"owner": i})[
                "revision"
            ]
        except ValueError:
            return "stale"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(step, [1, 2]))
    assert sorted(map(str, results)) == ["1", "stale"]


def test_saved_demo_replays_and_rejects_modified_result(tmp_path):
    folder = tmp_path / "demo"
    result = workflow.demo(folder)
    assert workflow.replay(folder) == result
    saved = read(folder / "result.json")
    saved["observations"][0]["cells"][0]["pressure_pa"] *= 2
    atomic_json(folder / "result.json", saved)
    with pytest.raises(ValueError, match="replay"):
        workflow.replay(folder)


def test_spatial_acoustic_cell_average_error_decreases():
    result = workflow.spatial()
    errors = [r["pressure_l1_error_pa"] for r in result]
    assert errors[0] > errors[1] > errors[2]


def test_adiabatic_entropy_guard_rejects_decrease_and_rolls_back(monkeypatch):
    m = PeriodicHEMChannel(uniform())
    before = m.checkpoint()
    original = m._observe
    calls = []

    def decreasing(*args):
        observation = original(*args)
        calls.append(1)
        observation["ledgers"]["entropy_change_j_k"] -= 0.001 * len(calls)
        return observation

    monkeypatch.setattr(m, "_observe", decreasing)
    with pytest.raises(ValueError, match="entropy decreased"):
        m.advance(0.001)
    monkeypatch.setattr(m, "_observe", original)
    assert m.checkpoint() == before


def test_late_source_change_cannot_publish_a_step(monkeypatch):
    m = PeriodicHEMChannel(uniform())
    before = m.checkpoint()
    original = module.signature
    calls = []

    def signature():
        calls.append(1)
        return original() if len(calls) == 1 else {"changed": True}

    monkeypatch.setattr(module, "signature", signature)
    with pytest.raises(ValueError, match="source changed"):
        m.advance(0.001)
    monkeypatch.setattr(module, "signature", original)
    assert m.checkpoint() == before


def test_malformed_initial_chart_and_nonfinite_control_refused():
    spec = uniform()
    spec["initial"][0] = None
    with pytest.raises(ValueError, match="object"):
        PeriodicHEMChannel(spec)
    m = PeriodicHEMChannel(uniform())
    before = m.checkpoint()
    with pytest.raises(ValueError):
        m.step([0.0] * 4, 0.001, expected_revision=0, controller_state={"bad": float("nan")})
    assert m.checkpoint() == before


def test_mirror_symmetry_of_shared_momentum_flux():
    spec = acoustic(4, 100.0)
    mirror = deepcopy(spec)
    mirror["initial"] = list(reversed(mirror["initial"]))
    for p in mirror["initial"]:
        p["velocity_m_s"] = -p["velocity_m_s"]
    a = PeriodicHEMChannel(spec).advance(0.002)
    b = PeriodicHEMChannel(mirror).advance(0.002)
    for left, right in zip(a["cells"], reversed(b["cells"])):
        assert left["mass_kg"] == pytest.approx(right["mass_kg"], abs=1e-10)
        assert left["momentum_kg_m_s"] == pytest.approx(-right["momentum_kg_m_s"], abs=1e-10)
        assert left["total_energy_j"] == pytest.approx(right["total_energy_j"], abs=1e-6)


def test_near_phase_boundary_mach_guard_uses_smaller_branch_speed():
    state = eos.px(1e6, 1e-6)
    lower, upper = module.characteristic_speeds(state)
    assert upper > lower
    spec = uniform()
    for p in spec["initial"]:
        p["quality"] = 1e-6
        p["velocity_m_s"] = 0.3 * lower
    with pytest.raises(ValueError, match="Mach"):
        PeriodicHEMChannel(spec)
