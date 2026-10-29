"""E03–E04: тождество энтальпии, k(T) твэла, обратный клапан и отключение сети."""

import numpy as np
import pytest

from components.equipment import check_valve_mass_flow, orifice_mass_flow
from components.grid import grid_electrical_torque_nm, speed_derivative, synchronous_balance
from physics.materials.iapws import properties_pt
from physics.materials.solids import uo2_conductivity_w_m_k
from physics.thermal.fuel_rod import FuelRod


def _rod() -> FuelRod:
    return FuelRod(
        r_fuel_m=0.004,
        r_clad_inner_m=0.0041,
        r_clad_outer_m=0.00475,
        n_fuel=8,
        n_clad=2,
        length_m=1.0,
        n_rods=1,
        rho_fuel=10000.0,
        cp_fuel=300.0,
        rho_clad=6500.0,
        cp_clad=300.0,
        k_fuel=3.0,
        k_clad=18.0,
        h_gap=1.0e5,
        h_coolant=2.0e4,
    )


def test_enthalpy_pressure_derivative_matches_volume_identity():
    pressure = 3.0e6
    temperature = 450.0
    step_p = 5.0e3
    step_t = 0.05
    enthalpy_slope = (
        properties_pt(pressure + step_p, temperature).h_j_kg
        - properties_pt(pressure - step_p, temperature).h_j_kg
    ) / (2.0 * step_p)
    state = properties_pt(pressure, temperature)
    volume_slope = (
        properties_pt(pressure, temperature + step_t).v_m3_kg
        - properties_pt(pressure, temperature - step_t).v_m3_kg
    ) / (2.0 * step_t)
    assert enthalpy_slope == pytest.approx(state.v_m3_kg - temperature * volume_slope, rel=1.0e-3)


def test_variable_conductivity_matches_constant_k_and_raises_hot_center():
    rod = _rod()
    power = 5.0e7
    surface = 600.0
    constant = rod.steady_fuel(power, surface)
    iterated = rod.steady_fuel_kt(power, surface, lambda _temperature: rod.k_fuel)
    assert iterated == pytest.approx(constant, rel=1.0e-8)
    surface_conductivity = uo2_conductivity_w_m_k(surface)
    reference = rod.steady_fuel_kt(power, surface, lambda _temperature: surface_conductivity)
    variable = rod.steady_fuel_kt(power, surface, uo2_conductivity_w_m_k)
    assert variable[0] > reference[0]
    cold = rod.capacitance_at(
        np.full(rod.n_nodes, 500.0),
        lambda temperature: 200.0 + 0.1 * temperature,
        lambda temperature: 300.0,
    )
    hot = rod.capacitance_at(
        np.full(rod.n_nodes, 1200.0),
        lambda temperature: 200.0 + 0.1 * temperature,
        lambda temperature: 300.0,
    )
    assert hot[0] > cold[0]


def test_check_valve_blocks_reverse_and_open_grid_drops_electrical_torque():
    forward = check_valve_mass_flow(0.62, 0.01, 800.0, 1.2e6, 1.0e6)
    assert forward == pytest.approx(orifice_mass_flow(0.62, 0.01, 800.0, 0.2e6))
    assert check_valve_mass_flow(0.62, 0.01, 800.0, 1.0e6, 1.2e6) == 0.0
    omega = 157.0
    mechanical = 1.0e6
    connected = grid_electrical_torque_nm(True, mechanical, omega)
    assert connected == pytest.approx(synchronous_balance(mechanical, omega))
    assert speed_derivative(2000.0, connected, connected) == pytest.approx(0.0)
    assert grid_electrical_torque_nm(False, mechanical, omega) == 0.0
    assert speed_derivative(2000.0, connected, 0.0) > 0.0
