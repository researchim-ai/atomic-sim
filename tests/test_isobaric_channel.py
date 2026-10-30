"""Открытые балансы, аналитическое испарение, производные и рестарт канала."""

import json

import numpy as np
import pytest

from core.continuation import ImplicitContinuation
from core.status import DomainError, SolverError
from physics.hydraulics.isobaric import ChannelConfig, IsobaricChannel, isobaric_state
from physics.materials.iapws import properties_pt, saturation_properties


def config(**changes):
    values = dict(
        cells=4,
        total_volume_m3=0.004,
        pressure_pa=7e6,
        inlet_enthalpy_j_kg=1.1e6,
        inlet_mass_flow_kg_s=1.0,
        heat_w=3e5,
    )
    values.update(changes)
    return ChannelConfig(**values)


@pytest.mark.parametrize(
    "pressure,temperature", [(3e6, 400.0), (7e6, 540.0), (0.5e6, 600.0), (10e6, 700.0)]
)
def test_density_derivative_matches_independent_ph_finite_difference(pressure, temperature):
    target = properties_pt(pressure, temperature)
    delta = 10.0
    actual = isobaric_state(pressure, target.h_j_kg)
    difference = (
        isobaric_state(pressure, target.h_j_kg + delta).density
        - isobaric_state(pressure, target.h_j_kg - delta).density
    ) / (2 * delta)
    assert actual.density_dh == pytest.approx(difference, rel=3e-6)


def test_two_phase_derivative_and_void_are_not_mixture_cp_approximations():
    liquid, vapor = saturation_properties(7e6)
    x = 0.2
    h = liquid.h_j_kg + x * (vapor.h_j_kg - liquid.h_j_kg)
    state = isobaric_state(7e6, h)
    volume = (1 - x) * liquid.v_m3_kg + x * vapor.v_m3_kg
    assert state.quality == pytest.approx(x)
    assert state.void == pytest.approx(x * vapor.v_m3_kg / volume)
    assert state.density_dh == pytest.approx(
        -(vapor.v_m3_kg - liquid.v_m3_kg) / (vapor.h_j_kg - liquid.h_j_kg) / volume**2
    )


def test_cell_and_global_mass_energy_rates_close_during_heating():
    channel = IsobaricChannel(config())
    state = np.r_[np.linspace(1.15e6, 1.55e6, 4), 0.0, 0.0]
    rates = channel.rates(0.0, state)
    assert np.sum(rates["dm"]) == pytest.approx(rates["mass_boundary_kg_s"], abs=1e-12)
    assert np.sum(rates["du"]) == pytest.approx(rates["energy_boundary_w"], abs=1e-8)
    assert rates["flows"][-1] != pytest.approx(rates["flows"][0])


@pytest.mark.parametrize("method", ["BDF", "Radau"])
def test_boiling_reservoir_matches_analytic_expansion_and_energy(method):
    liquid, vapor = saturation_properties(7e6)
    h0 = liquid.h_j_kg + 0.1 * (vapor.h_j_kg - liquid.h_j_kg)
    c = config(
        cells=1,
        total_volume_m3=0.01,
        inlet_mass_flow_kg_s=0.0,
        heat_w=20000.0,
        initial_enthalpy_j_kg=(h0,),
    )
    channel = IsobaricChannel(c)
    channel.enable_continuation(2.0)
    for _ in range(4):
        row = channel.advance(0.5, rtol=1e-9, method=method, max_step_s=0.1)
    b = (vapor.v_m3_kg - liquid.v_m3_kg) / (vapor.h_j_kg - liquid.h_j_kg)
    v0 = 1 / isobaric_state(7e6, h0).density
    expected_volume = v0 * np.exp(c.heat_w * b / c.total_volume_m3 * 2.0)
    expected_h = h0 + (expected_volume - v0) / b
    assert row["outlet_enthalpy_j_kg"] == pytest.approx(expected_h, rel=2e-8)
    assert row["stored_mass_kg"] == pytest.approx(c.total_volume_m3 / expected_volume, rel=2e-8)
    assert abs(row["mass_residual_kg"]) < 1e-7
    assert abs(row["energy_residual_j"]) < 0.1


def test_steady_profile_and_outlet_have_analytic_energy_gain():
    c = config()
    profile = np.array(
        [
            c.inlet_enthalpy_j_kg + (i + 1) * c.heat_w / c.cells / c.inlet_mass_flow_kg_s
            for i in range(c.cells)
        ]
    )
    channel = IsobaricChannel(config(initial_enthalpy_j_kg=tuple(profile)))
    rates = channel.rates(0.0, channel._state)
    assert rates["dh"] == pytest.approx(np.zeros(4), abs=1e-8)
    row = channel.advance(1.0, rtol=1e-8)
    assert row["outlet_enthalpy_j_kg"] == pytest.approx(
        c.inlet_enthalpy_j_kg + c.heat_w / c.inlet_mass_flow_kg_s, abs=0.001
    )
    assert row["outlet_quality"] > 0


def test_json_checkpoint_restores_and_continues_same_prescribed_drives():
    original = IsobaricChannel(config(flow_decay_s=600.0, heat_decay_s=1200.0))
    original.advance(0.5, rtol=1e-8)
    payload = json.loads(json.dumps(original.snapshot()))
    restored = IsobaricChannel.from_snapshot(payload)
    assert restored.snapshot() == payload
    a = original.advance(0.25, rtol=1e-8)
    b = restored.advance(0.25, rtol=1e-8)
    assert a == b


def test_invalid_method_rolls_back_state_and_history():
    channel = IsobaricChannel(config())
    channel.enable_continuation(2.0)
    before = channel.snapshot()
    with pytest.raises(SolverError):
        channel.advance(1.0, method="invalid")
    assert channel.snapshot() == before
    assert channel._continuation is None


@pytest.mark.parametrize("method", ["BDF", "Radau"])
def test_output_bounds_are_exact_and_rhs_does_not_cross_future_command_time(method):
    active_bound = [0.1]

    def rhs(t, y):
        assert t <= active_bound[0]
        return -0.03 * y

    session = ImplicitContinuation(
        rhs,
        [1.0, 1e6],
        0.0,
        2.0,
        rtol=1e-9,
        atol=1e-10,
        max_step=0.1,
        method=method,
        initial_target=0.1,
    )
    engine = session.solver
    for end in (0.1, 0.3, 0.5, 1.0, 1.1, 2.0):
        active_bound[0] = end
        result = session.advance_to(end)
        assert session.solver is engine
        assert session.time_s == end
        assert result.t[-1] == end
        assert result.final == pytest.approx(np.array([1.0, 1e6]) * np.exp(-0.03 * end), rel=2e-8)


@pytest.mark.parametrize(
    "changes",
    [{"cells": True}, {"cells": 0}, {"heat_w": -1.0}, {"pressure_pa": 20e6}, {"flow_decay_s": 0.0}],
)
def test_bad_input_is_refused(changes):
    with pytest.raises((ValueError, DomainError)):
        config(**changes)


def test_enthalpy_reference_is_not_required_to_be_positive():
    cold = properties_pt(1000.0, 273.151)
    assert cold.h_j_kg < 0.0
    c = IsobaricChannel(
        config(
            cells=1,
            total_volume_m3=0.001,
            pressure_pa=1000.0,
            inlet_enthalpy_j_kg=float(cold.h_j_kg),
            heat_w=0.0,
        )
    )
    row = c.advance(0.01)
    assert row["outlet_enthalpy_j_kg"] == pytest.approx(cold.h_j_kg, abs=1e-9)


@pytest.mark.parametrize(
    "key,value",
    [("schema_version", True), ("model_version", "other"), ("state", [float("nan")] * 6)],
)
def test_invalid_checkpoint_is_rejected(key, value):
    snapshot = IsobaricChannel(config()).snapshot()
    snapshot[key] = value
    with pytest.raises(DomainError):
        IsobaricChannel.from_snapshot(snapshot)
