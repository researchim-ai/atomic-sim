"""Производные EOS проверяются независимо разностями прямой формулировки."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from physics.materials import iapws


@pytest.mark.parametrize(
    "pressure,temperature", [(3.0, 300.0), (15.7, 563.15), (16.0, 593.15), (80.0, 600.0)]
)
def test_region1_jacobian_matches_independent_centered_differences(pressure, temperature):
    raw = iapws._region1_raw(temperature, pressure)
    dp, dt = 0.001, 0.001
    p1 = iapws._region1_raw(temperature, pressure + dp)
    p0 = iapws._region1_raw(temperature, pressure - dp)
    t1 = iapws._region1_raw(temperature + dt, pressure)
    t0 = iapws._region1_raw(temperature - dt, pressure)
    for value in ("v", "u"):
        assert raw["d" + value + "_dp_mpa"] == pytest.approx(
            (p1[value] - p0[value]) / (2 * dp), rel=2e-6
        )
        assert raw["d" + value + "_dt"] == pytest.approx(
            (t1[value] - t0[value]) / (2 * dt), rel=2e-6
        )


@pytest.mark.parametrize("pressure,temperature", [(3e6, 400.0), (15.7e6, 563.15), (16e6, 593.15)])
def test_local_newton_matches_general_solver_and_converges_without_it(
    monkeypatch, pressure, temperature
):
    target = iapws.properties_pt(pressure, temperature)
    general = iapws._invert_single_phase(
        target.v_m3_kg, target.u_j_kg, pressure * 0.99, temperature + 0.1
    )

    def unexpected(*args, **kwargs):
        raise AssertionError("General root should not be needed for this liquid case")

    monkeypatch.setattr(iapws, "_invert_single_phase", unexpected)
    local = iapws.properties_rho_u(
        target.rho_kg_m3, target.u_j_kg, pressure * 0.99, temperature + 0.1
    )
    assert local.p_pa == pytest.approx(general.p_pa, rel=2e-10)
    assert local.T_k == pytest.approx(general.T_k, abs=1e-8)
    assert local.u_j_kg == pytest.approx(target.u_j_kg, rel=1e-12)


def test_local_solver_declines_other_regions_and_saturation_cache_is_exact():
    assert iapws._invert_liquid_local(1.0, 2000.0, 0.5, 600.0) is None
    iapws.saturation_properties.cache_clear()
    first = iapws.saturation_properties(6.4e6)
    again = iapws.saturation_properties(6.4e6)
    shifted = iapws.saturation_properties(np.nextafter(6.4e6, np.inf))
    assert again is first
    assert shifted is not first
    assert iapws.saturation_properties.cache_info().hits == 1
    assert iapws.saturation_properties.cache_info().maxsize == 256
    with pytest.raises(FrozenInstanceError):
        first[0].T_k = 0.0
