"""Регрессии дефектов, найденных аудитом готовности."""

import numpy as np
import pytest
import torch
from gymnasium.utils.env_checker import check_env

from envs.gym_reactor import ReactorEnv, ReactorEnvContinuous
from reactor.neutronics_1d import Neutronics1D
from reactor.reactor_configs import VVER_1000
from reactor.simulator_1d import ReactorSimulator1D


def test_1d_power_is_percent_of_nominal():
    simulator = ReactorSimulator1D(num_nodes=20)
    state = simulator.step(dt=0.05)
    assert 1.0 < state["power"] < 500.0
    assert state["thermal_power_mw"] == pytest.approx(
        state["power"] / 100.0 * VVER_1000.thermal_power, rel=1e-6
    )


def test_burnup_step_updates_composition():
    simulator = ReactorSimulator1D(num_nodes=8)
    before = simulator.fuel.N_U235.mean().item()
    simulator.burnup_step(1.0)
    assert simulator.fuel.N_U235.mean().item() < before
    simulator.step(dt=0.01)


def test_neutron_speed_follows_configuration():
    neutronics = Neutronics1D(config=VVER_1000, num_nodes=8)
    assert neutronics.v.item() == VVER_1000.neutron_speed
    assert neutronics.v.item() != 200.0


def test_noisy_legacy_env_stays_inside_observation_space():
    env = ReactorEnv(obs_noise=0.01)
    check_env(env, skip_render_check=True)


def test_continuous_action_is_proportional():
    env = ReactorEnvContinuous()
    env.reset(seed=1)
    before = env.sim.control.rod_positions.mean().item()
    env.step(np.array([-0.05], dtype=np.float32))
    after = env.sim.control.rod_positions.mean().item()
    assert after < before
    assert before - after < 0.01


def test_step_latches_protection_like_run():
    from reactor import ReactorSimulator

    simulator = ReactorSimulator()
    simulator.thermal.T_fuel = torch.tensor(1300.0, dtype=simulator.dtype)
    state = simulator.step()
    assert simulator.scram_latched
    assert state["avg_rod_position"] == 0.0
