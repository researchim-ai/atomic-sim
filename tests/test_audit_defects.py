"""Регрессии дефектов аудита 27 октября 2026."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from components.separator import SteamDrum
from core.solvers import IntegrationResult
from core.status import DomainError
from physics.hydraulics.two_phase_ext import TwoFluidVolume
from physics.materials.iapws import (
    properties_ph,
    properties_pt,
    properties_rho_u,
    saturation_properties,
)
from physics.neutronics.transport import sn_s2_slab
from plants.vver import VVERPlant
from validation.ekf import ekf_track_amplitude


def test_two_fluid_adiabatic_transfer_conserves_energy():
    vessel = TwoFluidVolume(2.0, 0.5, 5.0e6, 1.0e6)
    energy0 = vessel.total_energy_j
    moved = vessel.transfer(1.0, 1.0, 1.0e5, 2.0e6)
    assert moved == pytest.approx(1.0)
    assert vessel.total_energy_j == pytest.approx(energy0)
    heated = TwoFluidVolume(2.0, 0.5, 5.0e6, 1.0e6)
    before = heated.total_energy_j
    latent = 1.0 * (2.0e6 - 1.0e5)
    heated.transfer(1.0, 1.0, 1.0e5, 2.0e6, interface_heat_w=latent)
    assert heated.total_energy_j - before == pytest.approx(latent)


def test_region3_and_region5_round_trip():
    for pressure, temperature in ((25.5837018e6, 650.0), (0.5e6, 1500.0)):
        state = properties_pt(pressure, temperature)
        restored = properties_ph(state.p_pa, state.h_j_kg)
        assert restored.region == state.region
        assert restored.T_k == pytest.approx(state.T_k, abs=1e-3)
        assert restored.p_pa == pytest.approx(state.p_pa, rel=1e-6)
        assert restored.h_j_kg == pytest.approx(state.h_j_kg, rel=1e-6)
        inverted = properties_rho_u(state.rho_kg_m3, state.u_j_kg, state.p_pa, state.T_k)
        assert inverted.region == state.region
        assert inverted.rho_kg_m3 == pytest.approx(state.rho_kg_m3, rel=1e-5)
        assert inverted.u_j_kg == pytest.approx(state.u_j_kg, rel=1e-5)
    with pytest.raises(DomainError):
        properties_rho_u(1.0e-8, 1.0e9, 1.0e5, 300.0, allow_projection=False)


def test_reflective_s2_is_flat_source_over_sigma():
    sigma = 0.1
    source = 1.0
    phi = sn_s2_slab(1.0, 20, sigma, source, left="reflective", right="reflective")
    assert phi == pytest.approx(np.full(20, source / sigma), rel=1e-6, abs=1e-6)
    vacuum = sn_s2_slab(1.0, 20, 0.5, 1.0)
    assert float(np.max(vacuum)) < source / 0.5


def _energy_rates(plant: VVERPlant) -> tuple[float, float]:
    dot = plant.derivatives(plant.time_s, plant.pack())
    stored = sum(plant.account_rates_w(dot).values())
    supplied = sum(plant.boundary_terms_w().values())
    return stored, supplied


def _assert_closed(plant: VVERPlant) -> None:
    stored, supplied = _energy_rates(plant)
    assert stored == pytest.approx(supplied, rel=0.0, abs=1.0)


def test_fuel_heat_is_interface_flux_and_plant_energy_closes():
    plant = VVERPlant()
    state = plant.pack()
    state[10] += 40.0
    dot = plant.derivatives(0.0, state)
    coolant = plant.network.volumes["core"].temperature_k
    interface = plant.ua_fuel * (plant.fuel_temperature - coolant)
    fission = plant.kinetics.power_w(plant.kinetics_state[0])
    total = plant.decay.total_power_w(fission, plant.decay_state)
    assert plant.network.volumes["core"].heat_w == pytest.approx(interface)
    assert interface != pytest.approx(total, rel=1e-4)
    assert plant.fuel_capacity * dot[10] + interface == pytest.approx(total)
    _assert_closed(plant)


def test_energy_balance_on_scram_trip_coastdown_and_feed():
    nominal = VVERPlant()
    _assert_closed(nominal)
    nominal.scram_latched = True
    _assert_closed(nominal)
    nominal.scram_latched = False
    nominal.throttle = 0.0
    _assert_closed(nominal)
    coast = VVERPlant()
    coast.motor_on = False
    omega0 = coast.omega
    coast.advance(0.3, rtol=1e-4, atol=1e-2)
    assert coast.omega < omega0 - 0.5
    assert coast.network.junctions["pump"].mass_flow_kg_s > 0.5 * coast.mass_flow_nom
    _assert_closed(coast)
    fed = VVERPlant()
    fed.condenser_mass_ref_kg -= 4000.0
    feed0 = fed.network.junctions["feed"].mass_flow_kg_s
    fed.advance(0.3, rtol=1e-4, atol=1e-2)
    assert fed.network.junctions["feed"].mass_flow_kg_s > feed0 + 50.0
    _assert_closed(fed)


def test_snapshot_keeps_command_and_rejected_step_rolls_back(monkeypatch):
    plant = VVERPlant()
    plant.rod_command = 0.123
    plant.throttle = 0.8
    plant.power_protection.timer_s = 0.4
    saved = plant.snapshot()
    plant.rod_command = 0.0
    plant.restore(saved)
    assert plant.rod_command == pytest.approx(0.123)
    assert plant.observation()["rod_insertion"] == pytest.approx(plant.rod)

    plant.rod_command = 0.123
    plant.throttle = 0.8

    def fail(_rhs, y0, _span, **_kwargs):
        plant.rod_command = -1.0
        plant.throttle = 0.0
        plant.power_protection.timer_s = 9.0
        final = np.array(y0, dtype=float).copy()
        final[10] = 100.0
        return IntegrationResult(
            t=np.array([0.0, 0.1]),
            y=np.column_stack([y0, final]),
            n_steps=1,
            message="probe",
            success=True,
        )

    monkeypatch.setattr("core.solvers.integrate", fail)
    with pytest.raises(DomainError):
        plant.advance(0.1)
    assert plant.rod_command == pytest.approx(0.123)
    assert plant.throttle == pytest.approx(0.8)
    assert plant.power_protection.timer_s == pytest.approx(0.4)
    assert plant.fuel_temperature == pytest.approx(plant.t_fuel_ref)


def test_power_protection_latches_scram_from_channels():
    plant = VVERPlant()
    plant.power_protection.setpoint = 0.0
    plant.power_protection.delay_s = 0.0
    observed = plant.advance(0.05)
    assert plant.power_protection.latched
    assert observed["scram_latched"]


def test_ekf_estimates_precursors_without_reading_truth():
    source = inspect.getsource(ekf_track_amplitude)
    assert "truth[index][1:]" not in source
    tracked = ekf_track_amplitude(0.0003, 0.02, 80, 0.05, 11)
    assert tracked["rmse"] < tracked["measurement_rmse"]
    assert tracked["holdout_rmse"] < tracked["measurement_rmse"]
    assert tracked["coverage"] > 0.9
    assert tracked["precursor_holdout_rmse"] < tracked["precursor_measurement_rmse"]


def test_drum_energy_is_initialized():
    drum = SteamDrum(2.0, 20.0, 6.4e6, 5000.0, 200.0)
    liquid, vapor = saturation_properties(6.4e6)
    expected = 5000.0 * liquid.u_j_kg + 200.0 * vapor.u_j_kg
    assert drum.accounted_energy_j == pytest.approx(expected)
