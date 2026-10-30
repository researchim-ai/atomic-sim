"""EOS не должен вносить шум на масштабе разностного якобиана BDF."""

import numpy as np
import pytest

from core.solvers import integrate
from physics.hydraulics.network import Volume
from physics.materials.iapws import (
    properties_ph,
    properties_pt,
    properties_rho_u,
    saturation_properties,
)


@pytest.mark.parametrize(
    "pressure,temperature", [(15.7e6, 563.15), (16.0e6, 593.15), (0.5e6, 600.0)]
)
def test_single_phase_inverse_is_precise_for_different_initial_guesses(pressure, temperature):
    target = properties_pt(pressure, temperature)
    for factor in (0.99, 1.0, 1.01):
        actual = properties_rho_u(
            target.rho_kg_m3, target.u_j_kg, pressure * factor, temperature + 0.1
        )
        assert actual.p_pa == pytest.approx(pressure, rel=2e-10)
        assert actual.T_k == pytest.approx(temperature, abs=1e-8)
        assert actual.v_m3_kg == pytest.approx(target.v_m3_kg, rel=5e-12)
        assert actual.u_j_kg == pytest.approx(target.u_j_kg, rel=5e-12)


@pytest.mark.parametrize("pressure,quality", [(8000.0, 0.05), (6.4e6, 0.12), (15.7e6, 0.35)])
def test_two_phase_inverse_does_not_return_a_nearby_pressure_guess(pressure, quality):
    liquid, vapor = saturation_properties(pressure)
    target = properties_ph(pressure, (1 - quality) * liquid.h_j_kg + quality * vapor.h_j_kg)
    for factor in (1.0 - 1e-7, 1.0 + 1e-7, 1.01):
        actual = properties_rho_u(target.rho_kg_m3, target.u_j_kg, pressure * factor, target.T_k)
        assert actual.p_pa == pytest.approx(pressure, rel=2e-10)
        assert actual.v_m3_kg == pytest.approx(target.v_m3_kg, rel=5e-12)
        assert actual.quality == pytest.approx(quality, abs=1e-10)


def test_property_reuse_requires_exact_inputs_and_preserves_small_perturbations(monkeypatch):
    from physics.hydraulics import network

    original = network.properties_rho_u
    calls = []

    def count(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(network, "properties_rho_u", count)
    volume = Volume("test", 1.0, 0.0)
    volume.set_liquid(15.7e6, 563.15)
    first = volume.sync()
    assert volume.sync() is first
    assert len(calls) == 1
    volume.mass_kg *= 1 + 1e-9
    perturbed = volume.sync()
    assert len(calls) == 2
    assert perturbed.p_pa != first.p_pa
    assert perturbed.rho_kg_m3 == pytest.approx(volume.mass_kg / volume.volume_m3, rel=5e-12)


def test_solver_statistics_separate_function_evaluations_from_steps():
    solved = integrate(lambda t, y: -100 * y, np.array([1.0]), (0.0, 0.1))
    assert solved.n_accepted_steps == len(solved.t) - 1
    assert solved.n_rhs_evaluations >= solved.n_accepted_steps
    assert solved.n_jacobian_evaluations > 0
    assert solved.n_lu_decompositions > 0
    sampled = integrate(lambda t, y: -y, np.array([1.0]), (0.0, 1.0), t_eval=np.linspace(0, 1, 3))
    assert sampled.n_accepted_steps is None
