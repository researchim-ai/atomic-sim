"""Pressure, signed momentum, shared donor flux and transactional state."""

import json
from dataclasses import replace

import numpy as np
import pytest

from core.status import DomainError
from physics.hydraulics.dynamic_channel import DynamicConfig, DynamicWallChannel


def config(**changes):
    base = DynamicConfig(
        1, 0.002, 7e6, 7e6, 500.0, 500.0, 7e6, 500.0, 0.001, 1.0, 10.0, 8000.0, 1000.0, 0.0
    )
    return replace(base, **changes)


@pytest.mark.parametrize("sign", [-1.0, 1.0])
@pytest.mark.parametrize("method", ["BDF", "Radau"])
def test_signed_coastdown_matches_exact_quadratic_drag(sign, method):
    c = DynamicWallChannel(config(initial_flow_kg_s=sign * 2.0))
    rho = c.network.volumes["0"].thermo.rho_kg_m3
    k = c.config.loss_coefficient / (2 * rho * c.config.area_m2 * c.config.junction_length_m)
    expected = sign * 2 / (1 + k * 2 * 2.0)
    c.enable_continuation(2.0)
    row = c.advance(2.0, rtol=1e-9, max_step_s=0.05, method=method)
    assert row["inlet_mass_flow_kg_s"] == pytest.approx(expected, abs=2e-7)
    assert row["outlet_mass_flow_kg_s"] == pytest.approx(expected, abs=2e-7)
    assert row["min_pressure_pa"] == pytest.approx(7e6, abs=0.01)
    assert abs(row["mass_residual_kg"]) < 1e-10
    assert abs(row["energy_residual_j"]) < 0.001


def test_zero_loss_uniform_motion_is_stationary():
    c = DynamicWallChannel(config(cells=3, loss_coefficient=0.0, initial_flow_kg_s=-2.0))
    assert c.derivatives(0.0, c._state) == pytest.approx(np.zeros_like(c._state), abs=1e-8)


@pytest.mark.parametrize("flow", [-2.0, 2.0])
def test_internal_mass_and_heat_fluxes_cancel_with_reverse_donor(flow):
    c = DynamicWallChannel(
        config(
            cells=2,
            initial_flow_kg_s=flow,
            heat_w=3000.0,
            right_temperature_k=490.0,
            initial_wall_temperature_k=510.0,
        )
    )
    # Different fluid temperatures exercise internal enthalpy donor switching.
    v = c.network.volumes["1"]
    v.set_liquid(7e6, 495.0)
    y = c._state.copy()
    y[2:4] = [v.mass_kg, v.internal_energy_j]
    rates = c.rates(0.0, y)
    d = rates["derivative"]
    assert sum(d[:4:2]) == pytest.approx(rates["boundary_mass_kg_s"], abs=1e-12)
    assert sum(d[1:4:2]) + rates["wall_energy_rate_w"] == pytest.approx(
        rates["boundary_energy_w"], abs=1e-8
    )
    donor = c.network.volumes["0" if flow > 0 else "1"].thermo.h_j_kg
    q0 = 500.0 * (510.0 - c.network.volumes["0"].temperature_k)
    left_h = (
        c.network.boundaries["left"].enthalpy_j_kg
        if flow > 0
        else c.network.volumes["0"].thermo.h_j_kg
    )
    assert d[1] == pytest.approx(flow * left_h - flow * donor + q0, abs=1e-7)


def test_pressure_drives_momentum_and_checkpoint_continues():
    c = DynamicWallChannel(config(left_pressure_pa=7.02e6, initial_pressure_pa=7.01e6))
    d = c.derivatives(0.0, c._state)
    assert d[2:4] == pytest.approx([10.0, 10.0], abs=1e-8)
    row = c.advance(0.05, max_step_s=0.01)
    assert row["inlet_mass_flow_kg_s"] > 0
    assert row["max_pressure_pa"] != 7.01e6
    payload = json.loads(json.dumps(c.snapshot()))
    other = DynamicWallChannel.from_snapshot(payload)
    assert other.snapshot() == payload
    a = c.advance(0.01, rtol=1e-8)
    b = other.advance(0.01, rtol=1e-8)
    assert a["inlet_mass_flow_kg_s"] == pytest.approx(b["inlet_mass_flow_kg_s"], abs=1e-7)
    assert a["min_pressure_pa"] == pytest.approx(b["min_pressure_pa"], abs=0.02)


def test_failed_rhs_restores_inventory_time_and_network(monkeypatch):
    c = DynamicWallChannel(config(left_pressure_pa=7.02e6))
    saved = c.snapshot()
    original = c.derivatives

    def fail(t, y):
        result = original(t, y)
        if t > 0.001:
            raise DomainError("injected")
        return result

    monkeypatch.setattr(c, "derivatives", fail)
    with pytest.raises(DomainError, match="injected"):
        c.advance(0.1)
    assert c.snapshot() == saved
    assert c.network.total_mass() == pytest.approx(c.initial_mass_kg, abs=1e-15)
    assert c._continuation is None


@pytest.mark.parametrize(
    "changes",
    [
        dict(cells=True),
        dict(area_m2=0.0),
        dict(heat_w=True),
        dict(initial_temperature_k=600.0),
        dict(loss_coefficient=-1.0),
    ],
)
def test_invalid_geometry_or_steam_is_rejected(changes):
    with pytest.raises((ValueError, DomainError)):
        config(**changes)


@pytest.mark.parametrize(
    "key,value", [("schema_version", True), ("time_s", -1.0), ("state", [1.0, 2.0])]
)
def test_corrupt_checkpoint_rejected(key, value):
    p = DynamicWallChannel(config()).snapshot()
    p[key] = value
    with pytest.raises(DomainError):
        DynamicWallChannel.from_snapshot(p)


def test_rhs_is_independent_of_previous_thermodynamic_probes():
    c = DynamicWallChannel(config())
    y = c._state.copy()
    expected = c.derivatives(0.0, y).copy()
    probe = y.copy()
    probe[1] += 1.0
    c.derivatives(0.0, probe)
    assert np.array_equal(c.derivatives(0.0, y), expected)


def test_flow_reverses_against_initial_motion_without_clamping():
    c = DynamicWallChannel(
        config(
            left_pressure_pa=7.02e6,
            initial_pressure_pa=7.01e6,
            initial_flow_kg_s=-1.0,
            right_temperature_k=495.0,
        )
    )
    assert c.observation()["outlet_mass_flow_kg_s"] < 0
    row = c.advance(0.5, max_step_s=0.01)
    assert row["inlet_mass_flow_kg_s"] > 0 and row["outlet_mass_flow_kg_s"] > 0
    assert abs(row["energy_residual_j"]) < 0.01


def test_high_mach_is_rejected_without_silent_flow_limit():
    with pytest.raises(DomainError, match="Mach"):
        DynamicWallChannel(config(initial_flow_kg_s=200.0))


@pytest.mark.parametrize("pressure,temperature", [(7e6, 500.0), (15.5e6, 580.0)])
def test_mach_guard_uses_if97_isentropic_sound_speed(pressure, temperature):
    from physics.hydraulics.dynamic_channel import _sound_speed
    from physics.materials.iapws import properties_pt

    assert _sound_speed(pressure, temperature) == pytest.approx(
        properties_pt(pressure, temperature).sound_speed_m_s, rel=1e-12
    )
