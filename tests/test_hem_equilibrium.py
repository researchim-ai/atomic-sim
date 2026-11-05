"""Conserved IF97 states, equilibrium derivatives and rigid phase-change inventories."""

import json
from copy import deepcopy

import pytest
from scipy.optimize import brentq

from atomic_model_hem import eos
from atomic_model_hem.vessel import RigidEquilibriumVessel
from physics.materials import iapws


def spec(segment=20000.0):
    return dict(
        schema_version=1,
        volume_m3=1.0,
        initial=dict(kind="px", pressure_pa=1e6, quality=0.5),
        initial_power_w=20000.0,
        maximum_energy_segment_j_kg=segment,
    )


@pytest.mark.parametrize("p", [1e5, 2e5, 1e6, 5e6, 1e7])
@pytest.mark.parametrize("x", [0.0, 0.001, 0.5, 0.999, 1.0])
def test_two_phase_conserved_roundtrip_and_volume_fraction(p, x):
    s = eos.px(p, x)
    recovered = eos.rho_u(
        s.density_kg_m3,
        s.internal_energy_j_kg,
        pressure_seed_pa=p,
        temperature_seed_k=s.temperature_k,
    )
    assert recovered.pressure_pa == pytest.approx(p, rel=1e-8)
    assert recovered.quality == pytest.approx(x, abs=1e-10)
    assert recovered.density_kg_m3 == pytest.approx(s.density_kg_m3, rel=eos.RHO_RTOL)
    assert eos.void_fraction(s) >= x
    assert eos.void_fraction(s) <= 1 + 1e-14
    assert "sound_speed_m_s" not in s.as_dict()


@pytest.mark.parametrize("p,t", [(3e6, 300.0), (3e6, 500.0), (1e6, 600.0), (2e5, 700.0)])
def test_single_phase_inverse_and_enthalpy_identity(p, t):
    s = eos.pt(p, t)
    r = eos.rho_u(s.density_kg_m3, s.internal_energy_j_kg, pressure_seed_pa=p, temperature_seed_k=t)
    assert r.pressure_pa == pytest.approx(p, rel=1e-7)
    assert r.enthalpy_j_kg == pytest.approx(
        r.internal_energy_j_kg + r.pressure_pa / r.density_kg_m3, rel=1e-12
    )
    assert r.quality is None


@pytest.mark.parametrize("p,t", [(3e6, 350.0), (3e6, 500.0), (1e6, 600.0), (2e5, 700.0)])
def test_single_phase_acoustic_derivative_matches_fundamental_if97(p, t):
    s = eos.pt(p, t)
    d = eos.derivatives(s)
    expected = iapws.properties_pt(p, t).sound_speed_m_s ** 2
    assert d["equilibrium_sound_speed_squared_m2_s2"] == pytest.approx(expected, rel=2e-6)


@pytest.mark.parametrize("p,x", [(2e5, 0.01), (1e6, 0.5), (5e6, 0.9)])
def test_mixture_acoustic_derivative_against_independent_isentropic_chart_root(p, x):
    s = eos.px(p, x)
    d = eos.derivatives(s)

    def pressure_for_density(rho):
        v = 1 / rho

        def residual(pressure):
            f, g = iapws.saturation_properties(pressure)
            q = (v - f.v_m3_kg) / (g.v_m3_kg - f.v_m3_kg)
            return float((1 - q) * f.s_j_kg_k + q * g.s_j_kg_k - s.entropy_j_kg_k)

        return brentq(residual, p * 0.99, p * 1.01, xtol=1e-5)

    delta = s.density_kg_m3 * 1e-5
    c2 = (
        pressure_for_density(s.density_kg_m3 + delta)
        - pressure_for_density(s.density_kg_m3 - delta)
    ) / (2 * delta)
    assert d["equilibrium_sound_speed_squared_m2_s2"] == pytest.approx(c2, rel=2e-4)


@pytest.mark.parametrize("x", [0.0, 1.0, 1e-7, 1 - 1e-7])
def test_phase_boundary_derivatives_refused(x):
    with pytest.raises(ValueError, match="phase boundary"):
        eos.derivatives(eos.px(1e6, x))


@pytest.mark.parametrize("p,t", [(1e4, 400.0), (2e7, 600.0), (1e6, 299.0), (1e6, 1000.0)])
def test_unsupported_states_are_refused(p, t):
    with pytest.raises(ValueError):
        eos.pt(p, t)


def test_saturation_pt_and_invalid_quality_not_guessed_or_clipped():
    ts = iapws.saturation_temperature_k(1.0)
    with pytest.raises(ValueError, match="explicit quality"):
        eos.pt(1e6, float(ts))
    for x in [-1e-12, 1 + 1e-12, True, float("nan")]:
        with pytest.raises(ValueError):
            eos.px(1e6, x)


def test_heating_and_cooling_phase_events_exact_inventory_and_independent_temperature_root():
    m = RigidEquilibriumVessel(spec())
    initial = m.snapshot()
    r = m.advance(600.0)
    assert r["thermodynamic_state"]["phase"] == "vapor"
    assert len(r["events"]) == 1 and r["events"][0]["branch"] == "vapor"
    rho = initial["thermodynamic_state"]["density_kg_m3"]

    def volume_gap(t):
        p = iapws.saturation_pressure_mpa(t) * 1e6
        return float(iapws.saturation_properties(p)[1].rho_kg_m3 - rho)

    t = brentq(volume_gap, 400.0, 550.0, xtol=1e-10)
    p = iapws.saturation_pressure_mpa(t) * 1e6
    u = iapws.saturation_properties(p)[1].u_j_kg
    expected = (
        initial["mass_kg"] * (u - initial["thermodynamic_state"]["internal_energy_j_kg"]) / 20000.0
    )
    assert r["events"][0]["time_s"] == pytest.approx(expected, abs=1e-6)
    cooled = m.step(-20000.0, 600.0, expected_revision=1, controller_state={"tick": 1})
    assert cooled["thermodynamic_state"]["quality"] == pytest.approx(0.5, abs=1e-10)
    assert len(cooled["events"]) == 2
    assert cooled["external_heat_j"] == 0 and cooled["internal_energy_change_j"] == 0
    assert cooled["mass_kg"] == initial["mass_kg"]
    assert abs(cooled["eos_mass_residual_kg"]) < 1e-8
    assert abs(cooled["eos_energy_residual_j"]) < 0.1


def test_sampling_refinement_and_partitioned_heat_agree():
    results = []
    for segment in [20000.0, 10000.0, 5000.0]:
        m = RigidEquilibriumVessel(spec(segment))
        r = m.advance(600.0)
        results.append(r)
    for r in results[1:]:
        assert r["thermodynamic_state"]["pressure_pa"] == pytest.approx(
            results[0]["thermodynamic_state"]["pressure_pa"], rel=1e-9
        )
        assert r["events"][0]["time_s"] == results[0]["events"][0]["time_s"]
    assert len(results[0]["samples"]) < len(results[1]["samples"]) < len(results[2]["samples"])
    m = RigidEquilibriumVessel(spec())
    m.advance(300.0)
    r = m.advance(300.0)
    assert r["thermodynamic_state"]["temperature_k"] == pytest.approx(
        results[0]["thermodynamic_state"]["temperature_k"], abs=1e-6
    )


def test_checkpoint_full_replay_preserves_controller_and_phase_history():
    m = RigidEquilibriumVessel(spec())
    m.advance(500.0)
    cp = json.loads(json.dumps(m.checkpoint()))
    other = RigidEquilibriumVessel(spec())
    assert other.restore(cp) == m.snapshot()
    a = m.step(-20000.0, 300.0, expected_revision=1, controller_state={"state": 0.1})
    b = other.step(-20000.0, 300.0, expected_revision=1, controller_state={"state": 0.1})
    assert a == b


@pytest.mark.parametrize("field", ["clock", "events", "controller", "calls"])
def test_rehashed_tampered_checkpoint_refused(field):
    m = RigidEquilibriumVessel(spec())
    m.step(20000.0, 500.0, expected_revision=0, controller_state={"tick": 1})
    before = m.checkpoint()
    cp = deepcopy(before)
    p = cp["payload"]
    if field == "clock":
        p["observation"]["time_s"] += 1
    elif field == "events":
        p["observation"]["events"][0]["time_s"] += 1
    elif field == "controller":
        p["observation"]["controller_state"]["tick"] = 2
    else:
        p["calls"][0]["duration_s"] += 1
    from atomic_model_workspace.common import digest

    cp["sha256"] = digest(p)
    with pytest.raises(ValueError):
        m.restore(cp)
    assert m.checkpoint() == before


def test_late_eos_failure_rolls_back_whole_heat_and_control(monkeypatch):
    m = RigidEquilibriumVessel(spec())
    before = m.checkpoint()
    original = eos.rho_u
    calls = []

    def fail(*a, **kw):
        calls.append(1)
        if len(calls) == 5:
            raise ValueError("Injected inverse failure")
        return original(*a, **kw)

    monkeypatch.setattr(eos, "rho_u", fail)
    with pytest.raises(ValueError):
        m.step(20000.0, 300.0, expected_revision=0, controller_state={"tick": 1})
    assert len(calls) == 5 and m.checkpoint() == before


def test_stale_revision_and_oversized_calls_refused_without_changes():
    m = RigidEquilibriumVessel(spec())
    before = m.checkpoint()
    with pytest.raises(ValueError):
        m.step(20000.0, 1.0, expected_revision=True, controller_state={})
    with pytest.raises(ValueError):
        m.advance(100000.0)
    with pytest.raises(ValueError):
        m.step(1e8, 1000.0, expected_revision=0, controller_state={})
    assert m.checkpoint() == before


def test_actual_domain_exit_after_evaporation_rolls_back_all_phase_events():
    m = RigidEquilibriumVessel(spec())
    before = m.checkpoint()
    with pytest.raises(ValueError, match="domain"):
        m.advance(1000.0)
    assert m.checkpoint() == before


def test_exact_saturation_boundary_checkpoint_can_continue_both_directions():
    m = RigidEquilibriumVessel(spec())
    boundary = next(b for b in m._boundaries if b["branch"] == "vapor")
    m.advance(boundary["heat_j"] / 20000.0)
    cp = m.checkpoint()
    other = RigidEquilibriumVessel(spec())
    other.restore(cp)
    assert m.advance(1.0) == other.advance(1.0)
    cooled = RigidEquilibriumVessel(spec())
    cooled.restore(cp)
    assert (
        cooled.step(-20000.0, 1.0, expected_revision=1, controller_state={})["thermodynamic_state"][
            "phase"
        ]
        == "two_phase"
    )
