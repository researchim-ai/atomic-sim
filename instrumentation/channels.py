"""Датчик первого порядка и привод с ограничением скорости."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.status import ModelPassport, ModelStatus

SENSOR = ModelPassport(
    model_id="sensor-lag",
    version="2.0.0",
    status=ModelStatus.VERIFIED,
    phenomenon="Инерционный измерительный канал",
    approximations=("один полюс", "шум добавляется после инерции"),
    applicability="Постоянная времени больше нуля; отказ замораживает последнее значение",
    data_sources=("Стандартная модель измерительного лага dy/dt = (u-y)/tau",),
)


@dataclass
class Sensor:
    tau_s: float
    value: float = 0.0
    bias: float = 0.0
    noise_std: float = 0.0
    failed: bool = False
    quality: str = "good"

    def step(self, truth: float, dt_s: float, rng: np.random.Generator) -> float:
        if self.failed:
            self.quality = "bad"
            return self.value
        decay = 1.0 - np.exp(-dt_s / self.tau_s)
        self.value += decay * ((truth + self.bias) - self.value)
        measured = self.value
        if self.noise_std > 0.0:
            measured = self.value + float(rng.normal(0.0, self.noise_std))
        self.quality = "good"
        return measured


@dataclass
class Actuator:
    position: float
    speed_limit: float
    minimum: float = 0.0
    maximum: float = 1.0
    failed: bool = False

    def step(self, demand: float, dt_s: float) -> float:
        if self.failed:
            return self.position
        delta = max(-self.speed_limit * dt_s, min(self.speed_limit * dt_s, demand - self.position))
        self.position = min(self.maximum, max(self.minimum, self.position + delta))
        return self.position
