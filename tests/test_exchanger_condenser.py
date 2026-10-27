"""Стенка теплообменника и конденсатор с давлением из запасов."""

import pytest

from components.condenser import DynamicCondenser
from components.heat_exchanger import HeatExchangerWall
from physics.materials.iapws import properties_pt


def test_wall_relaxes_to_the_conductance_weighted_temperature():
    wall = HeatExchangerWall(2.0e5, 1000.0, 3000.0, 500.0)
    equilibrium = wall.equilibrium_temperature_k(600.0, 400.0)
    assert equilibrium == pytest.approx((1000.0 * 600.0 + 3000.0 * 400.0) / 4000.0)
    wall.temperature_k = equilibrium
    to_wall, to_secondary = wall.heat_flows_w(600.0, 400.0)
    assert to_wall == pytest.approx(to_secondary)
    assert wall.derivative_k_s(600.0, 400.0) == pytest.approx(0.0, abs=1.0e-12)
    wall.temperature_k = 500.0
    wall.advance(600.0, 400.0, 1.0e6)
    assert wall.temperature_k == pytest.approx(equilibrium, abs=1.0e-6)


def test_dynamic_condenser_pressure_follows_inventory():
    original = properties_pt(1.0e5, 320.0)
    condenser = DynamicCondenser(1.0, 1.0e5, 320.0)
    recovered = condenser.state()
    assert recovered.p_pa == pytest.approx(1.0e5, rel=1.0e-6)
    assert recovered.T_k == pytest.approx(320.0, abs=1.0e-3)
    mass_rate, energy_rate = condenser.derivatives(0.0, 0.0, 0.0, 0.0, -1.0e5)
    assert mass_rate == pytest.approx(0.0)
    assert energy_rate == pytest.approx(-1.0e5)
    condenser.energy_j += energy_rate
    cooled = condenser.state()
    assert cooled.T_k < original.T_k
    assert cooled.p_pa != pytest.approx(original.p_pa, rel=1.0e-8)

    two_phase = DynamicCondenser.from_quality(2.0, 0.5e6, 0.2)
    boiling = two_phase.state()
    assert boiling.p_pa == pytest.approx(0.5e6, rel=1.0e-4)
    assert 0.0 < boiling.quality < 1.0
