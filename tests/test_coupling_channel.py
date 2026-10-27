"""Осевая связка, канал, DAE-ранг и границы среды агента."""

import numpy as np
import pytest

from agent_envs.pwr_env import PWRPlantEnv
from core.assembly import jacobian_rank
from core.lumped import LumpedEnergy
from core.status import DomainError, SolverError
from coupling.axial import steady_axial_power
from coupling.maps import ExternalSolverAdapter, conserve_integral
from physics.hydraulics.channel import channel_energy_closure, heated_water_channel


def test_axial_power_is_conserved_and_doppler_moves_k():
    common = dict(
        height_m=1.0,
        n_cells=16,
        power_w=1.0e6,
        diffusion_m=0.012,
        sigma_a_ref_per_m=1.0,
        nu_sigma_f_per_m=1.05,
        temperature_ref_k=900.0,
        coolant_temperature_k=580.0,
        linear_conductance_w_k_m=4000.0,
    )
    plain = steady_axial_power(doppler_per_sqrt_k=0.0, **common)
    shifted = steady_axial_power(doppler_per_sqrt_k=0.004, **common)
    assert plain["power_w"].sum() == pytest.approx(common["power_w"])
    assert shifted["power_w"].sum() == pytest.approx(common["power_w"])
    assert plain["temperature_k"].max() > common["coolant_temperature_k"]
    assert shifted["k"] != pytest.approx(plain["k"], rel=1e-6)
    widths = np.array([0.5, 0.5])
    coarse = conserve_integral(plain["power_w"] / plain["mesh"].dx, plain["mesh"].dx, widths)
    assert np.sum(coarse * widths) == pytest.approx(common["power_w"])


def test_heated_channel_closes_energy():
    channel = heated_water_channel(3, 0.05, 15.7e6, 560.0, 12.0, 1.0e5)
    stored, supplied = channel_energy_closure(channel)
    assert stored == pytest.approx(supplied, abs=1.0)
    assert stored == pytest.approx(3.0e5, rel=1.0e-6)


def test_lumped_energy_residual_and_rank():
    body = LumpedEnergy(10.0, 2.0, 300.0, 40.0)
    state = body.initialize({})
    assert state[0] == pytest.approx(320.0)
    assert body.residual(0.0, state, np.zeros(1), {})[0] == pytest.approx(0.0)
    assert body.residual(0.0, state + 1.0, np.zeros(1), {})[0] == pytest.approx(0.2)
    payload = body.serialize()
    body.restore(payload)
    assert body.heat_w == pytest.approx(40.0)
    assert jacobian_rank(np.eye(2)) == 2
    assert jacobian_rank(np.array([[1.0, 2.0], [2.0, 4.0]])) == 1
    assert jacobian_rank(np.zeros((2, 2))) == 0


def test_agent_clips_sensors_and_does_not_hide_bugs():
    env = PWRPlantEnv(sensor_noise=0.0)
    env.reset(seed=1)
    env.power_sensor.value = 10.0
    env.power_sensor.tau_s = 1.0e9
    observed = env._observe()
    assert observed[0] == pytest.approx(1.5)
    assert env.sensor_saturated
    assert env.observation_space.contains(observed)

    def bug(_self, _dt, rtol=1e-4, atol=1e-2):
        raise ValueError("программная ошибка")

    env.plant.advance = bug.__get__(env.plant, type(env.plant))
    with pytest.raises(ValueError):
        env.step(np.array([0.0], dtype=np.float32))

    boiling = PWRPlantEnv(sensor_noise=0.0)
    boiling.reset(seed=2)

    def out(_self, _dt, rtol=1e-4, atol=1e-2):
        raise DomainError("кипение")

    boiling.plant.advance = out.__get__(boiling.plant, type(boiling.plant))
    _obs, _reward, terminated, truncated, info = boiling.step(np.array([0.0], dtype=np.float32))
    assert terminated is False
    assert truncated is True
    assert info["error_type"] == "DomainError"
    assert info["out_of_domain"] is True

    def fail(_payload):
        raise SolverError("внешний отказ")

    report = ExternalSolverAdapter(fail, "test").exchange({})
    assert report["applicability"] == "unsupported"

    def crash(_payload):
        raise KeyError("поле")

    with pytest.raises(KeyError):
        ExternalSolverAdapter(crash, "test").exchange({})
