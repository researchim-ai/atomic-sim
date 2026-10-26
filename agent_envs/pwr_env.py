"""Среда агента поверх проверенной установки ВВЭР.

Непрерывное действие — команда скорости стержня в долях предела, без скрытого
квантования в три направления. Наблюдение снимается датчиком, а не вектором состояния.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from instrumentation.channels import Sensor
from plants.vver import VVERPlant


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
        self.power_sensor = Sensor(sensor_tau_s, value=1.0, noise_std=sensor_noise)
        noise_margin = 1.0 + 8.0 * sensor_noise
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([0.0, 400.0, 10.0, 0.0], dtype=np.float32),
            high=np.array([1.5 * noise_margin, 2000.0, 20.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self._steps = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.plant = VVERPlant()
        self.power_sensor = Sensor(self.sensor_tau_s, value=1.0, noise_std=self.sensor_noise)
        self._steps = 0
        obs = self._observe()
        return obs, {"applicability": "in_domain", "out_of_domain": False}

    def step(self, action):
        command = float(np.asarray(action, dtype=float).reshape(-1)[0])
        self.plant.rod_command = max(-1.0, min(1.0, command))
        info = {"out_of_domain": False, "applicability": "in_domain", "constraint_power_error": 0.0}
        try:
            observation = self.plant.advance(self.action_dt_s, rtol=1e-4, atol=1e-2)
        except Exception as exc:
            info["out_of_domain"] = True
            info["applicability"] = "out_of_domain"
            info["message"] = str(exc)
            obs = self._observe()
            return obs, 0.0, False, True, info
        self._steps += 1
        fraction = observation["thermal_power_w"] / self.plant.fission_power_nom
        info["constraint_power_error"] = abs(fraction - 1.0)
        info["scram_latched"] = observation["scram_latched"]
        reward = -info["constraint_power_error"]
        obs = self._observe()
        terminated = False
        truncated = self._steps >= 50
        return obs, reward, terminated, truncated, info

    def _observe(self) -> np.ndarray:
        fraction = (
            self.plant.decay.total_power_w(
                self.plant.kinetics.power_w(self.plant.kinetics_state[0]),
                self.plant.decay_state,
            )
            / self.plant.fission_power_nom
        )
        measured = self.power_sensor.step(fraction, self.action_dt_s, self.np_random)
        obs = np.array(
            [
                measured,
                self.plant.fuel_temperature,
                self.plant.network.volumes["pressurizer"].pressure_pa / 1.0e6,
                self.plant.rod,
            ],
            dtype=np.float32,
        )
        return obs
