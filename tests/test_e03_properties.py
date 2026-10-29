"""E03: du/dT совпадает с cp, а p,T не задаёт двухфазную смесь."""

import pytest

from physics.materials.coolants import sodium_cp_j_kg_k, sodium_internal_energy_j_kg
from physics.materials.iapws import properties_pt, saturation_pressure_mpa, saturation_temperature_k
from physics.materials.solids import (
    graphite_internal_energy_j_kg,
    graphite_specific_heat_j_kg_k,
    uo2_internal_energy_j_kg,
    uo2_specific_heat_j_kg_k,
    zircaloy_internal_energy_j_kg,
    zircaloy_specific_heat_j_kg_k,
)


def _matches_cp(energy, heat_capacity, temperature_k: float) -> None:
    step = 0.05
    slope = (energy(temperature_k + step) - energy(temperature_k - step)) / (2.0 * step)
    assert slope == pytest.approx(heat_capacity(temperature_k), rel=1.0e-6)


def test_solid_and_sodium_energy_derivatives_match_heat_capacity():
    _matches_cp(uo2_internal_energy_j_kg, uo2_specific_heat_j_kg_k, 1000.0)
    _matches_cp(zircaloy_internal_energy_j_kg, zircaloy_specific_heat_j_kg_k, 800.0)
    _matches_cp(graphite_internal_energy_j_kg, graphite_specific_heat_j_kg_k, 900.0)
    _matches_cp(sodium_internal_energy_j_kg, sodium_cp_j_kg_k, 700.0)
    assert uo2_internal_energy_j_kg(298.15) == pytest.approx(0.0)


def test_if97_cp_is_the_isobaric_enthalpy_derivative_and_pt_is_single_phase():
    for pressure, temperature in ((3.0e6, 400.0), (0.5e6, 500.0)):
        state = properties_pt(pressure, temperature)
        step = 0.05
        enthalpy_slope = (
            properties_pt(pressure, temperature + step).h_j_kg
            - properties_pt(pressure, temperature - step).h_j_kg
        ) / (2.0 * step)
        assert enthalpy_slope == pytest.approx(state.cp_j_kg_k, rel=1.0e-5)
        assert state.quality in (0.0, 1.0)
    saturation = saturation_temperature_k(3.0)
    saturation_pressure = saturation_pressure_mpa(saturation) * 1.0e6
    liquid = properties_pt(saturation_pressure + 2.0e3, saturation)
    vapor = properties_pt(saturation_pressure - 2.0e3, saturation)
    assert liquid.region == 1 and liquid.quality == 0.0
    assert vapor.region == 2 and vapor.quality == 1.0
