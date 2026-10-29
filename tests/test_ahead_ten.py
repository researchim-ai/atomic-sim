"""Прямоток, дрейф, баклинг двух групп, штраф срыва и стационар РБМК."""

import numpy as np
import pytest

from agent_envs.pwr_env import step_reward
from components.heat_exchanger import counterflow_effectiveness, parallelflow_effectiveness
from core.status import DomainError
from coupling.cosim import hold_pair_crossing
from data.catalog import bundled_redistributable_ids, source_by_id
from physics.hydraulics.two_phase_ext import drift_void_fraction, quality_from_drift_void
from physics.materials.iapws import latent_heat_j_kg
from physics.neutronics.diffusion import two_group_buckling_k, two_group_infinite_k
from plants.rbmk import RBMKChannel


def test_parallel_flow_is_below_counterflow_and_drift_inverts():
    assert parallelflow_effectiveness(1.5, 0.0) == pytest.approx(
        counterflow_effectiveness(1.5, 0.0)
    )
    assert parallelflow_effectiveness(1.5, 0.6) < counterflow_effectiveness(1.5, 0.6)
    assert parallelflow_effectiveness(0.0, 0.4) == pytest.approx(0.0)
    with pytest.raises(DomainError):
        parallelflow_effectiveness(-0.1, 0.2)

    quality = 0.18
    rho_l, rho_v, flux = 760.0, 18.0, 800.0
    void = drift_void_fraction(
        quality, rho_l, rho_v, flux, distribution=1.15, drift_velocity_m_s=0.25
    )
    assert quality_from_drift_void(
        void, rho_l, rho_v, flux, distribution=1.15, drift_velocity_m_s=0.25
    ) == pytest.approx(quality)
    assert latent_heat_j_kg(1.0e6) > 0.0


def test_buckling_lowers_two_group_k_and_rbmk_temperatures_settle():
    sigma = np.array([0.2, 0.8])
    nu = np.array([0.15, 1.1])
    chi = np.array([1.0, 0.0])
    diffusion = np.array([0.02, 0.004])
    bare = two_group_buckling_k(sigma, 0.25, nu, chi, diffusion, 0.0)
    leaky = two_group_buckling_k(sigma, 0.25, nu, chi, diffusion, 12.0)
    assert bare == pytest.approx(two_group_infinite_k(sigma, 0.25, nu, chi))
    assert leaky < bare

    channel = RBMKChannel()
    channel.steady_temperatures()
    fuel_rate, graphite_rate, _alpha, _rho = channel.derivatives()
    assert fuel_rate == pytest.approx(0.0, abs=1.0e-8)
    assert graphite_rate == pytest.approx(0.0, abs=1.0e-8)


def test_crossing_converges_failure_reward_and_catalog():
    reference = hold_pair_crossing(1.0, 0.0, 0.6, 0.01, 2.0)
    fine = hold_pair_crossing(1.0, 0.0, 0.6, 0.05, 2.0)
    coarse = hold_pair_crossing(1.0, 0.0, 0.6, 0.25, 2.0)
    assert reference is not None and fine is not None and coarse is not None
    assert abs(fine - reference) < abs(coarse - reference)

    assert step_reward(1.0, failed=False) == pytest.approx(0.0)
    assert step_reward(1.4, failed=False) == pytest.approx(-0.4)
    assert step_reward(2.0, failed=True) < step_reward(2.0, failed=False)

    bundled = bundled_redistributable_ids()
    assert bundled == ["project-open-two-group"]
    assert source_by_id("ans-5.1")["redistribution_allowed"] is False
    assert source_by_id("endf-b-nndc")["bundled"] is False
    assert source_by_id("plant-measurements")["bundled"] is False
