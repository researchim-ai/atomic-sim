"""Тепловые запасы петли и останов насоса. В ВВЭР эта сборка не входит."""

from physics.hydraulics.circulation import ThermalCirculation


def test_pump_trip_keeps_buoyant_flow_and_moves_stored_energy():
    loop = ThermalCirculation()
    hot_0 = loop.network.volumes["hot_bot"].temperature_k
    flow_0 = loop.network.junctions["bottom"].mass_flow_kg_s
    omega_0 = loop.omega
    loop.motor_on = False
    report = loop.advance(2.0)
    assert report["omega"] < 0.7 * omega_0
    assert 0.3 * flow_0 < report["mass_flow_kg_s"] < 0.95 * flow_0
    assert abs(report["hot_temperature_k"] - hot_0) > 0.02
    scale = max(1.0, abs(loop.heater_w))
    assert abs(loop.last_energy_residual_j) <= 1.0e-3 * scale
    for volume in loop.network.volumes.values():
        assert volume.thermo.quality < 1.0e-4
