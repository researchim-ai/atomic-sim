"""Среда агента поверх проверенной установки ВВЭР.

Непрерывное действие — команда скорости стержня в долях предела, без скрытого
квантования в три направления. Наблюдение снимается датчиком, а не вектором состояния.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from core.status import DomainError, SolverError, classify_model_error
from instrumentation.channels import Sensor
from plants.vver import VVERPlant
from sdk.api import apply_commands

FAILURE_REWARD = -10.0


def step_reward(power_fraction: float, failed: bool) -> float:
    """Срыв шага хуже отклонения мощности вплоть до десяти номиналов.

    На номинале награда 0. Срыв даёт -10, а не 0: иначе отказ численно выгоднее точного слежения.
    """
    if failed:
        return FAILURE_REWARD
    return -abs(float(power_fraction) - 1.0)


class PWRPlantEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self, action_dt_s: float = 0.2, sensor_tau_s: float = 0.5, sensor_noise: float = 0.0
    ):
        super().__init__()
        self.action_dt_s = action_dt_s
        self.sensor_tau_s = sensor_tau_s
        self.sensor_noise = sensor_noise
        self.plant = VVERPlant()
        self._steps = 0
        self.sensor_saturated = False
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([0.0, 400.0, 10.0, 0.0], dtype=np.float32),
            high=np.array([1.5, 2000.0, 20.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self._bind_sensors(nominal=True)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.plant = VVERPlant()
        self._steps = 0
        self._bind_sensors(nominal=True)
        obs = self._observe()
        return obs, {
            "applicability": "in_domain",
            "out_of_domain": False,
            "sensor_saturated": self.sensor_saturated,
        }

    def step(self, action):
        command = float(np.asarray(action, dtype=float).reshape(-1)[0])
        apply_commands(self.plant, {"rod_command": max(-1.0, min(1.0, command))})
        info = {"out_of_domain": False, "applicability": "in_domain", "constraint_power_error": 0.0}
        try:
            observation = self.plant.advance(self.action_dt_s, rtol=1e-4, atol=1e-2)
        except (DomainError, SolverError) as exc:
            info["out_of_domain"] = True
            info["applicability"] = exc.applicability.value if isinstance(exc, DomainError) else "solver_error"
            info["error_type"] = type(exc).__name__
            info["error_class"] = classify_model_error(exc)
            info["message"] = str(exc)
            info["sensor_saturated"] = self.sensor_saturated
            obs = self._observe()
            return obs, step_reward(1.0, failed=True), False, True, info
        self._steps += 1
        fraction = observation["thermal_power_w"] / self.plant.fission_power_nom
        info["constraint_power_error"] = abs(fraction - 1.0)
        info["scram_latched"] = observation["scram_latched"]
        reward = step_reward(fraction, failed=False)
        obs = self._observe()
        info["sensor_saturated"] = self.sensor_saturated
        terminated = False
        truncated = self._steps >= 50
        return obs, reward, terminated, truncated, info

    def _bind_sensors(self, nominal: bool) -> None:
        noise = self.sensor_noise
        tau = self.sensor_tau_s
        if nominal:
            self.power_sensor = Sensor(tau, value=1.0, noise_std=noise)
            self.fuel_sensor = Sensor(tau, value=self.plant.t_fuel_ref, noise_std=noise)
            self.pressure_sensor = Sensor(tau, value=self.plant.p_primary / 1.0e6, noise_std=noise)
            self.rod_sensor = Sensor(tau, value=self.plant.rod, noise_std=0.0)
        else:
            self.power_sensor = Sensor(tau, noise_std=noise)
            self.fuel_sensor = Sensor(tau, noise_std=noise)
            self.pressure_sensor = Sensor(tau, noise_std=noise)
            self.rod_sensor = Sensor(tau, noise_std=0.0)

    def _observe(self) -> np.ndarray:
        fraction = (
            self.plant.decay.total_power_w(
                self.plant.kinetics.power_w(self.plant.kinetics_state[0]),
                self.plant.decay_state,
            )
            / self.plant.fission_power_nom
        )
        pressure_mpa = self.plant.network.volumes["pressurizer"].pressure_pa / 1.0e6
        rng = self.np_random
        raw = np.array(
            [
                self.power_sensor.step(fraction, self.action_dt_s, rng),
                self.fuel_sensor.step(self.plant.fuel_temperature, self.action_dt_s, rng),
                self.pressure_sensor.step(pressure_mpa, self.action_dt_s, rng),
                self.rod_sensor.step(self.plant.rod, self.action_dt_s, rng),
            ],
            dtype=np.float64,
        )
        low = self.observation_space.low.astype(np.float64)
        high = self.observation_space.high.astype(np.float64)
        clipped = np.clip(raw, low, high)
        self.sensor_saturated = bool(np.any(np.abs(clipped - raw) > 1.0e-8))
        return clipped.astype(np.float32)
