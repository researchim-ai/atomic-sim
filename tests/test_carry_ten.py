"""Горло сопла, одногрупповой inhour, отверстия и цилиндрическая оболочка."""

import math

import numpy as np
import pytest

from components.equipment import (
    equivalent_series_orifice_area_m2,
    orifice_pressure_drop,
    series_orifice_drop_pa,
)
from physics.hydraulics.two_phase_ext import (
    acoustic_transit_s,
    critical_mass_flux_ideal_gas,
    ideal_gas_sound_speed_m_s,
    throat_mass_flux_ideal_gas,
)
from physics.materials.coolants import (
    helium_cp_j_kg_k,
    helium_cv_j_kg_k,
    helium_density_kg_m3,
    helium_specific_gas_constant_j_kg_k,
)
from physics.neutronics.diffusion import generation_time_s, uniform_multigroup_delayed_step
from physics.neutronics.point_kinetics import (
    PointKinetics,
    inhour_root,
    one_group_inhour_root,
    prompt_jump_ratio,
    reactivity_in_dollars,
)
from physics.thermal.fuel_rod import (
    cylindrical_shell_heat_per_m,
    cylindrical_shell_temperature_k,
    log_mean_radius_m,
)


def test_throat_flux_matches_sound_and_helium():
    pressure, temperature, gamma, gas_r = 1.0e6, 300.0, 1.4, 287.0
    throat = throat_mass_flux_ideal_gas(pressure, temperature, gamma, gas_r)
    assert throat == pytest.approx(
        critical_mass_flux_ideal_gas(pressure, temperature, gamma, gas_r), rel=1.0e-12
    )
    sound = ideal_gas_sound_speed_m_s(gamma, gas_r, temperature)
    assert acoustic_transit_s(12.0, sound) * sound == pytest.approx(12.0)
    helium_t, helium_p = 600.0, 2.0e6
    gamma_he = helium_cp_j_kg_k(helium_t) / helium_cv_j_kg_k(helium_t)
    density = helium_density_kg_m3(helium_p, helium_t)
    assert ideal_gas_sound_speed_m_s(
        gamma_he, helium_specific_gas_constant_j_kg_k(), helium_t
    ) == pytest.approx(math.sqrt(gamma_he * helium_p / density))


def test_one_group_period_dollars_and_generation_time():
    beta, decay, generation, reactivity = 0.0065, 0.08, 1.0e-5, 0.001
    omega = one_group_inhour_root(reactivity, generation, beta, decay)
    kinetics = PointKinetics(np.array([beta]), np.array([decay]), generation)
    assert omega == pytest.approx(inhour_root(kinetics, reactivity), rel=1.0e-8)
    negative = one_group_inhour_root(-reactivity, generation, beta, decay)
    assert negative == pytest.approx(inhour_root(kinetics, -reactivity), rel=1.0e-6)
    assert negative < 0.0
    residual = reactivity - omega * generation - omega * beta / (omega + decay)
    assert residual == pytest.approx(0.0, abs=1.0e-12)
    state = kinetics.initial_state(1.0)
    early = kinetics.reference_step(state, reactivity, 1.0)
    late = kinetics.reference_step(state, reactivity, 2.0)
    assert late[0] / early[0] == pytest.approx(math.exp(omega), rel=1.0e-8)
    dollars = reactivity_in_dollars(reactivity, beta)
    assert prompt_jump_ratio(beta, reactivity) == pytest.approx(1.0 / (1.0 - dollars))

    speed, nu_sigma, absorption = 2200.0, 1.0, 1.0
    assert generation_time_s(speed, nu_sigma) == pytest.approx(1.0 / (speed * nu_sigma))
    groups = np.array([0.002, 0.0045])
    decays = np.array([0.08, 0.3])
    flux = 3.0
    precursors = groups * nu_sigma * flux / decays
    held = uniform_multigroup_delayed_step(
        flux, precursors, absorption, nu_sigma, speed, groups, decays, 0.2
    )
    assert held[0] == pytest.approx(flux, rel=1.0e-10)


def test_series_orifices_and_cylindrical_shell():
    density = 800.0
    flow = 1.5
    areas = [0.01, 0.02]
    drop = series_orifice_drop_pa(flow, [(0.62, area) for area in areas], density)
    equivalent = equivalent_series_orifice_area_m2(areas)
    assert drop == pytest.approx(
        orifice_pressure_drop(flow, 0.62, equivalent, density), rel=1.0e-12
    )
    assert orifice_pressure_drop(flow, 0.62, areas[0], density) + orifice_pressure_drop(
        flow, 0.62, areas[1], density
    ) == pytest.approx(drop)

    inner, outer = 0.004, 0.0048
    conductivity, hot, cold = 15.0, 700.0, 600.0
    heat = cylindrical_shell_heat_per_m(conductivity, hot, cold, inner, outer)
    by_logarithm = 2.0 * math.pi * conductivity * (hot - cold) / math.log(outer / inner)
    assert heat == pytest.approx(by_logarithm, rel=1.0e-12)
    assert log_mean_radius_m(inner, outer) == pytest.approx(
        (outer - inner) / math.log(outer / inner)
    )
    geometric = math.sqrt(inner * outer)
    assert cylindrical_shell_temperature_k(geometric, inner, outer, hot, cold) == pytest.approx(
        0.5 * (hot + cold)
    )
    thin_outer = inner * 1.001
    mean = 0.5 * (inner + thin_outer)
    thin = cylindrical_shell_heat_per_m(conductivity, hot, cold, inner, thin_outer)
    arithmetic = conductivity * 2.0 * math.pi * mean * (hot - cold) / (thin_outer - inner)
    assert thin == pytest.approx(arithmetic, rel=1.0e-3)
