"""Компенсатор, мощность насоса и приток бора через границу."""

import pytest

from components.equipment import hydraulic_power_w, torque_matching_hydraulic_power_nm
from components.pressurizer import Pressurizer
from core.status import DomainError
from physics.hydraulics.network import BoundaryState, FlowNetwork, Junction, Volume
from physics.hydraulics.two_phase_ext import four_quadrant_head_pa, pump_shaft_power_w
from physics.materials.iapws import properties_pt, saturation_properties


def test_pressurizer_heater_and_spray_keep_separate_balances():
    vessel = Pressurizer(10.0, 10.0e6, 0.4)
    pressure_0 = vessel.vessel.state().p_pa
    idle_mass, idle_energy = vessel.derivatives(0.0, 0.0, 0.0, 0.0, 0.0)
    assert idle_mass == pytest.approx(0.0)
    assert idle_energy == pytest.approx(0.0)
    _mass, heated = vessel.derivatives(2.0e6, 0.0, 0.0, 0.0, 0.0)
    assert heated == pytest.approx(2.0e6)
    liquid, _vapor = saturation_properties(pressure_0)
    spray_mass, spray_energy = vessel.derivatives(0.0, 2.0, liquid.h_j_kg - 1.0e4, 0.0, 0.0)
    assert spray_mass == pytest.approx(2.0)
    assert spray_energy == pytest.approx(2.0 * (liquid.h_j_kg - 1.0e4))
    vessel.vessel.energy_j += 5.0e6
    assert vessel.vessel.state().p_pa > pressure_0
    with pytest.raises(DomainError):
        vessel.derivatives(-1.0, 0.0, 0.0, 0.0, 0.0)
    with pytest.raises(DomainError):
        vessel.derivatives(0.0, 1.0, liquid.h_j_kg + 1.0e4, 0.0, 0.0)


def test_forward_pump_power_matches_shaft_and_boron_enters_only_from_the_boundary():
    density = 750.0
    flow = 20.0
    head = 1.5e5
    omega = 150.0
    torque = torque_matching_hydraulic_power_nm(flow, head, density, omega)
    assert hydraulic_power_w(flow, head, density) == pytest.approx(torque * omega)
    reverse_head = four_quadrant_head_pa(-1.0, -0.5, 1.0, 0.2, 0.5, 1.0e5)
    assert pump_shaft_power_w(-10.0, reverse_head, density) == pytest.approx(-10.0 * reverse_head / density)

    volume = Volume("core", 1.0, 0.0)
    volume.set_liquid(5.0e6, 500.0, boron_ppm=0.0)
    water = properties_pt(5.0e6, 500.0)
    network = FlowNetwork(
        {"core": volume},
        {
            "inlet": Junction("inlet", "boundary:in", "core", 0.05, 1.0, 0.0, mass_flow_kg_s=10.0),
            "outlet": Junction("outlet", "core", "boundary:out", 0.05, 1.0, 0.0, mass_flow_kg_s=10.0),
        },
    )
    network.boundaries["boundary:in"] = BoundaryState(5.0e6, water.h_j_kg, water.rho_kg_m3, 0.0, 500.0e-6)
    network.boundaries["boundary:out"] = BoundaryState(5.0e6, water.h_j_kg, water.rho_kg_m3, 0.0, 0.0)
    derivative = network.derivatives(network.pack())
    assert derivative[2] == pytest.approx(10.0 * 500.0e-6)
    assert derivative[0] == pytest.approx(0.0, abs=1.0e-8)
