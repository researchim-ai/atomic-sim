"""Эталоны оставшихся режимов: области 3 и 5, транспорт, типы реакторов, CRAM, UQ."""

import math

import numpy as np
import pytest
from scipy.linalg import expm

from components.grid import electrical_power_w, speed_after_constant_torques, synchronous_balance
from components.separator import SteamDrum
from coupling.openmc_adapter import OpenMCAdapter
from physics.depletion.cram import cram16
from physics.hydraulics.two_phase_ext import (
    TwoFluidVolume,
    choked_mass_flow,
    critical_mass_flux_ideal_gas,
    drift_void_fraction,
    four_quadrant_head_pa,
    hem_void_matches_drift_limit,
    interfacial_drag_pair,
    pump_shaft_power_w,
    terminal_relative_velocity,
)
from physics.materials.iapws import properties_pt, void_fraction
from physics.neutronics.batch import integrate_batch_euler, torch_batch_derivative
from physics.neutronics.point_kinetics import PointKinetics, batch_derivatives
from physics.neutronics.transport import (
    analytical_rectangle_k,
    diffusion_k_nd,
    sn_infinite_medium,
    sn_particle_balance,
)
from plants.reactor_types import BWRPlant, HTGRBrayton, MSRPrecursors, SFRLoop
from plants.registry import build_plant
from validation.ekf import ekf_track_amplitude
from validation.uq import finite_difference_sensitivity, monte_carlo_amplitude


def test_region3_and_region5_reference_points():
    state = properties_pt(25.5837018e6, 650.0)
    assert state.region == 3
    assert state.rho_kg_m3 == pytest.approx(500.0, rel=1e-6)
    assert state.h_j_kg / 1000.0 == pytest.approx(1863.43019, rel=1e-6)
    vapor = properties_pt(0.5e6, 1500.0)
    assert vapor.region == 5
    assert vapor.v_m3_kg == pytest.approx(1.38455090, rel=1e-7)
    assert vapor.h_j_kg / 1000.0 == pytest.approx(5219.76855, rel=1e-7)
    assert properties_pt(30.0e6, 2000.0).sound_speed_m_s == pytest.approx(1067.36948, rel=1e-6)


def test_multigroup_dimensions_and_transport_balance():
    diffusion = 0.012
    sigma_a = 0.2
    nu_sigma_f = 0.25
    lengths = (1.2, 1.2)
    expected = analytical_rectangle_k(diffusion, sigma_a, nu_sigma_f, lengths)
    coarse, _phi, ok = diffusion_k_nd((8, 8), lengths, diffusion, sigma_a, nu_sigma_f)
    fine, _phi2, ok2 = diffusion_k_nd((16, 16), lengths, diffusion, sigma_a, nu_sigma_f)
    assert ok and ok2
    assert abs(fine - expected) < abs(coarse - expected)
    cube = (1.0, 1.0, 1.0)
    expected3 = analytical_rectangle_k(diffusion, sigma_a, nu_sigma_f, cube)
    k3, _flux, ok3 = diffusion_k_nd((6, 6, 6), cube, diffusion, sigma_a, nu_sigma_f)
    assert ok3
    assert k3 == pytest.approx(expected3, rel=0.05)
    assert sn_infinite_medium(0.2, 1.0) == pytest.approx(2.5)
    absorption, leakage, source = sn_particle_balance(1.0, 20, 0.5, 1.0)
    assert absorption + leakage == pytest.approx(source, rel=1e-9)


def test_drift_two_fluid_choke_and_pump_quadrants():
    rho_l, rho_v = 800.0, 20.0
    quality = 0.1
    hem = void_fraction(quality, rho_l, rho_v)
    assert hem_void_matches_drift_limit(quality, rho_l, rho_v) == pytest.approx(hem, rel=1e-6)
    drifted = drift_void_fraction(quality, rho_l, rho_v, 1000.0, distribution=1.2, drift_velocity_m_s=0.3)
    assert drifted < hem
    vessel = TwoFluidVolume(10.0, 1.0, 1.0e7, 2.0e6)
    mass0 = vessel.total_mass_kg
    moved = vessel.transfer(0.5, 1.0, 1.0e6, 2.5e6)
    assert moved == pytest.approx(0.5)
    assert vessel.total_mass_kg == pytest.approx(mass0)
    assert vessel.total_energy_j - (1.0e7 + 2.0e6) == pytest.approx(0.5 * (2.5e6 - 1.0e6))
    empty = TwoFluidVolume(0.1, 0.0, 1.0, 0.0)
    limited = empty.transfer(5.0, 1.0, 1.0, 2.0)
    assert limited == pytest.approx(0.1)
    assert empty.exchange_limited
    assert empty.liquid_mass_kg == pytest.approx(0.0)
    positive, negative = interfacial_drag_pair(1.5, 2.0)
    assert positive == pytest.approx(-negative)
    assert terminal_relative_velocity(100.0, 9.81, 4.0) == pytest.approx(math.sqrt(100.0 * 9.81 / 4.0))
    flux = critical_mass_flux_ideal_gas(1.0e6, 300.0, 1.4, 287.0)
    assert flux > 0.0
    choked = choked_mass_flow(1.0e6, 1.0e5, 0.5, 20.0, 0.1)
    still = choked_mass_flow(1.0e6, 1.0e4, 0.5, 20.0, 0.1)
    assert choked == pytest.approx(still) == pytest.approx(20.0)
    forward = four_quadrant_head_pa(1.0, 1.0, 1.2, -0.2, -0.1, 1.0e5)
    reverse = four_quadrant_head_pa(1.0, -1.0, 1.2, -0.2, -0.1, 1.0e5)
    assert forward != reverse
    assert pump_shaft_power_w(10.0, forward, 1000.0) == pytest.approx(10.0 * forward / 1000.0)


def test_cram_matches_expm_on_decay_chain():
    matrix = np.array([[-0.3, 0.0, 0.0], [0.3, -0.05, 0.0], [0.0, 0.05, 0.0]])
    initial = np.array([1.0, 0.0, 0.0])
    exact = expm(matrix * 4.0) @ initial
    numerical = cram16(matrix, initial, 4.0)
    assert numerical == pytest.approx(exact, rel=1e-6, abs=1e-8)


def test_reactor_types_have_their_own_balances():
    bwr = build_plant("bwr")
    assert isinstance(bwr, BWRPlant)
    split = bwr.separation()
    assert split["steam_kg_s"] + split["recirculation_kg_s"] == pytest.approx(bwr.core_flow_kg_s)
    assert split["heat_w"] == pytest.approx(bwr.fission_power_w, rel=1e-9)
    assert bwr.void_coefficient < 0.0
    sfr = build_plant("sfr")
    assert isinstance(sfr, SFRLoop)
    assert sfr.steady_temperature() == pytest.approx(sfr.sink_temperature_k + sfr.fission_power_w / sfr.ua_w_k)
    assert sfr.density() == pytest.approx(1016.0 - 0.235 * sfr.temperature_k)
    htgr = build_plant("htgr")
    assert isinstance(htgr, HTGRBrayton)
    assert htgr.cooler_power_w() + htgr.turbine_power_w == pytest.approx(htgr.reactor_power_w + htgr.compressor_power_w)
    assert htgr.net_electric_w() == pytest.approx(htgr.turbine_power_w - htgr.compressor_power_w)
    assert htgr.helium_density() > 0.0
    msr = build_plant("msr")
    assert isinstance(msr, MSRPrecursors)
    msr.core_precursors = 2.0
    msr.loop_precursors = 2.0
    d_core, d_loop = msr.derivatives(0.0)
    assert d_core + d_loop == pytest.approx(-msr.decay_constant * 4.0)
    before = msr.core_precursors + msr.loop_precursors
    msr.decay_constant = 0.0
    msr.advance(2.0, 0.0)
    assert msr.core_precursors + msr.loop_precursors == pytest.approx(before, rel=1e-8)


def test_drum_generator_uq_ekf_batch_and_openmc_adapter():
    drum = SteamDrum(2.0, 20.0, 6.4e6, 5000.0, 200.0)
    assert drum.level_m() > 0.0
    torque = synchronous_balance(1.0e6, 157.0)
    assert electrical_power_w(torque, 157.0) == pytest.approx(1.0e6)
    assert speed_after_constant_torques(157.0, 1000.0, 0.0, 10.0, 2.0) == pytest.approx(157.0 - 0.02)
    report = monte_carlo_amplitude(-0.01, 0.001, 5.0e-5, 1.0, 2000, 3)
    assert report["p05"] < report["mean"] < report["p95"]
    analytical = 1.0 * 5.0e-5 / (-0.01) ** 2
    assert finite_difference_sensitivity(-0.01, 5.0e-5, 1.0) == pytest.approx(analytical, rel=1e-4)
    tracked = ekf_track_amplitude(0.0003, 0.02, 80, 0.05, 11)
    assert tracked["rmse"] < tracked["measurement_rmse"]
    assert tracked["holdout_rmse"] < tracked["measurement_rmse"]
    assert tracked["coverage"] > 0.9
    kinetics = PointKinetics.thermal_u235()
    rho = np.array([0.0, 0.0004])
    batch = integrate_batch_euler(rho, 0.01, 20, kinetics)
    scalar = []
    for value in rho:
        amplitude = np.array([1.0])
        precursors = kinetics.equilibrium_precursors(1.0)[None, :]
        step = 0.01 / 20
        for _ in range(20):
            dn, dc_dt = batch_derivatives(amplitude, precursors, np.array([value]), kinetics)
            amplitude = amplitude + step * dn
            precursors = precursors + step * dc_dt
        scalar.append(amplitude[0])
    assert batch == pytest.approx(scalar, rel=1e-12)
    dn_torch, _dc = torch_batch_derivative(rho, np.ones(2), np.vstack([kinetics.equilibrium_precursors(1.0), kinetics.equilibrium_precursors(1.0)]), kinetics)
    dn_numpy, _dc2 = batch_derivatives(np.ones(2), np.vstack([kinetics.equilibrium_precursors(1.0), kinetics.equilibrium_precursors(1.0)]), rho, kinetics)
    assert dn_torch == pytest.approx(dn_numpy, rel=1e-12)
    adapter = OpenMCAdapter()
    result = adapter.run_mgxs({"groups": 2})
    assert result["applicability"] in ("in_domain", "unsupported")
