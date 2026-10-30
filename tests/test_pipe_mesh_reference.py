"""Independent checks of the steady reference and the implemented operator."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.integrate import quad

from physics.materials.iapws import properties_ph, properties_pt
from scripts.verify_pipe_mesh import configuration, continuous_reference, discrete_steady


def test_continuous_reference_reduces_to_constant_density_pressure_drop(monkeypatch):
    import scripts.verify_pipe_mesh as benchmark

    rho = 800.0
    monkeypatch.setattr(
        benchmark, "properties_ph", lambda p, h: SimpleNamespace(region=1, rho_kg_m3=rho)
    )
    c = configuration(4, 3000.0)
    flow, solution, _ = continuous_reference(c)
    expected = c.area_m2 * np.sqrt(2 * rho * (c.left_pressure_pa - c.right_pressure_pa) / 30.0)
    assert flow == pytest.approx(expected, rel=1e-11)
    x = np.linspace(0, 1, 9)
    assert solution.sol(x)[0] == pytest.approx(
        c.left_pressure_pa - (c.left_pressure_pa - c.right_pressure_pa) * x, abs=1e-6
    )


def test_unheated_reference_agrees_with_independent_pressure_quadrature():
    c = configuration(4, 0.0)
    h = properties_pt(c.left_pressure_pa, c.left_temperature_k).h_j_kg
    integral, _ = quad(
        lambda p: properties_ph(p, h).rho_kg_m3,
        c.right_pressure_pa,
        c.left_pressure_pa,
        epsabs=1e-5,
    )
    exact = c.area_m2 * np.sqrt(2 * integral / 30.0)
    flow, _, _ = continuous_reference(c)
    assert flow == pytest.approx(exact, rel=1e-10)


def test_constructed_equilibrium_satisfies_pressure_momentum_and_wall_operator():
    flow, states, residual = discrete_steady(configuration(4, 3000.0))
    assert flow > 0 and len(states) == 4
    assert residual["acceleration_kg_s2"] < 1e-5
    assert residual["fluid_energy_rate_w"] < 0.001
    assert residual["mass_rate_kg_s"] < 1e-10
    assert residual["wall_temperature_rate_k_s"] < 1e-7
