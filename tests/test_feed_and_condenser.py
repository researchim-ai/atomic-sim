"""Кривая питания и двухфазная оболочка конденсатора ВВЭР."""

import pytest

from core.status import DomainError
from physics.hydraulics.network import FlowNetwork, Junction, Volume
from plants.vver import VVERPlant


def test_feed_curve_matches_nominal_and_shuts_without_head():
    plant = VVERPlant()
    feed = plant.network.junctions["feed"]
    assert plant._feed_curve_kg_s(feed) == pytest.approx(plant.m_steam_nom, rel=1.0e-8)
    feed.pump_pressure_pa = 0.0
    assert plant._feed_curve_kg_s(feed) == 0.0
    state = plant.condenser.state()
    assert state.p_pa == pytest.approx(plant.p_condenser, rel=1.0e-8)
    assert 1.0e-4 < state.quality < 1.0 - 1.0e-4


def test_dump_holds_sg_pressure_without_making_electricity():
    plant = VVERPlant()
    plant.throttle = 0.0
    electric_0 = plant.electrical_power_w()
    for _ in range(12):
        plant.advance(0.25, rtol=1e-4, atol=1e-2)
    pressure = plant.network.volumes["sg_secondary"].pressure_pa
    assert plant.p_dump_open_pa < pressure < plant.p_dump_open_pa + 2.0 * plant.dump_band_pa
    assert plant.network.junctions["dump"].mass_flow_kg_s > 0.2 * plant.m_steam_nom
    assert plant.network.junctions["steam"].mass_flow_kg_s < 0.1 * plant.m_steam_nom
    assert plant.electrical_power_w() < 0.1 * electric_0
    assert 1.0e-4 < plant.condenser.state().quality < 1.0 - 1.0e-4


def test_check_valve_does_not_reverse_a_pump():
    left = Volume("left", 1.0, 0.0)
    right = Volume("right", 1.0, 0.0)
    left.set_liquid(2.0e6, 400.0)
    right.set_liquid(1.0e6, 400.0)
    junction = Junction(
        "feed",
        "right",
        "left",
        0.01,
        1.0,
        1.0,
        mass_flow_kg_s=-0.2,
        pump_pressure_pa=1.0e5,
        check_valve=True,
        flow_time_constant_s=0.1,
    )
    network = FlowNetwork({"left": left, "right": right}, {"feed": junction})
    derivative = network.derivatives(network.pack())
    assert float(derivative[-1]) > 0.0
    plain = Junction(
        "pump",
        "right",
        "left",
        0.01,
        1.0,
        1.0,
        mass_flow_kg_s=-0.2,
        pump_pressure_pa=1.0e5,
    )
    blocked = FlowNetwork({"left": left, "right": right}, {"pump": plain})
    with pytest.raises(DomainError):
        blocked.derivatives(blocked.pack())
