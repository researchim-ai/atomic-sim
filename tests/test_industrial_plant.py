"""Интегральные проверки установки и среды агента."""

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from agent_envs.pwr_env import PWRPlantEnv
from core.status import DomainError
from plants.vver import VVERPlant
from scenarios.replay import restart_matches
from sdk.api import create_session, step_session
from validation.estimation import kalman_track


def test_vver_steady_residual_and_short_hold():
    plant = VVERPlant()
    derivative = plant.derivatives(0.0, plant.pack())
    scale = np.maximum(1.0, np.abs(plant.pack()))
    assert np.max(np.abs(derivative) / scale) < 1e-3
    before = plant.observation()
    after = plant.advance(0.5, rtol=1e-4, atol=1e-2)
    assert after["mass_flow_kg_s"] == pytest.approx(before["mass_flow_kg_s"], rel=1e-4)
    assert after["thermal_power_w"] == pytest.approx(before["thermal_power_w"], rel=1e-4)
    assert after["sg_pressure_pa"] == pytest.approx(before["sg_pressure_pa"], rel=1e-4)
    assert after["electrical_power_w"] > 0.8e9


def test_load_rejection_raises_sg_pressure_and_drops_electric_power():
    plant = VVERPlant()
    plant.network.refresh_properties()
    plant.throttle = 0.0
    plant._apply_heat_and_actuators()
    assert plant.network.junctions["steam"].flow_target_kg_s == 0.0
    mild = VVERPlant()
    initial_pressure = mild.network.volumes["sg_secondary"].pressure_pa
    initial_electric = mild.electrical_power_w()
    mild.throttle = 0.9
    final = mild.advance(0.2, rtol=1e-5, atol=1e-4)
    assert final["sg_pressure_pa"] > initial_pressure
    assert final["electrical_power_w"] < initial_electric
    assert final["steam_flow_kg_s"] < mild.m_steam_nom
    tripped = VVERPlant()
    tripped.throttle = 0.0
    with pytest.raises(DomainError):
        tripped.advance(1.0, rtol=1e-4, atol=1e-2)


def test_scram_inserts_rods_with_finite_speed():
    plant = VVERPlant()
    plant.scram_latched = True
    plant.advance(0.4, rtol=1e-4, atol=1e-2)
    assert plant.rod < plant.rod_ref
    assert plant.rod > 0.0


def test_restart_matches_continuous_run():
    assert restart_matches(VVERPlant, step_s=0.25, splits=2, atol=1.0, rtol=1e-5)


def test_session_api_and_agent_env_checker():
    session = create_session("vver")
    observed = step_session(session, 0.2, {"rod_command": -0.2})
    assert observed["rod_insertion"] <= session.rod_ref
    env = PWRPlantEnv()
    check_env(env, skip_render_check=True)
    env2 = PWRPlantEnv(sensor_noise=0.01)
    obs, _info = env2.reset(seed=3)
    assert env2.observation_space.contains(obs)
    action = np.array([0.05], dtype=np.float32)
    obs, _reward, terminated, truncated, info = env2.step(action)
    assert env2.observation_space.contains(obs)
    assert terminated is False
    assert "constraint_power_error" in info
    assert truncated is False


def test_kalman_beats_raw_measurement_on_synthetic_series():
    rng = np.random.default_rng(0)
    truth = 600.0 + np.cumsum(rng.normal(0.0, 0.3, size=400))
    report = kalman_track(truth, measurement_std=2.0, process_std=0.3, seed=7)
    assert report["rmse"] < report["measurement_rmse"]
    assert report["coverage"] > 0.85
