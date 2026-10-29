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
    applicability="Постоянная времени больше нуля; залипание держит последнее число, обрыв числа не даёт",
    data_sources=("Стандартная модель измерительного лага dy/dt = (u-y)/tau",),
)


def step_lag(initial: float, target: float, tau_s: float, time_s: float) -> float:
    """Точный лаг на ступенчатом входе: y = y_∞ + (y0 − y_∞) exp(−t/τ)."""
    if tau_s <= 0.0 or time_s < 0.0:
        raise ValueError("Постоянная времени ступени положительна, время неотрицательно")
    return target + (initial - target) * np.exp(-time_s / tau_s)


def rate_limited_position(
    position: float,
    demand: float,
    speed_limit: float,
    dt_s: float,
    minimum: float,
    maximum: float,
) -> float:
    """Привод идёт к команде не быстрее предела и остаётся в своих упорах."""
    if dt_s < 0.0 or speed_limit < 0.0 or minimum > maximum:
        raise ValueError("Скорость и шаг привода неотрицательны, упор сверху не ниже упора снизу")
    delta = max(-speed_limit * dt_s, min(speed_limit * dt_s, demand - position))
    return min(maximum, max(minimum, position + delta))


def ramp_lag(initial: float, rate_per_s: float, tau_s: float, time_s: float) -> float:
    """Точный лаг на линейном входе: y = y0 + r t - r tau (1 - exp(-t/tau))."""
    if tau_s <= 0.0 or time_s < 0.0:
        raise ValueError("Постоянная времени рампы положительна, время неотрицательно")
    return initial + rate_per_s * time_s - rate_per_s * tau_s * (1.0 - np.exp(-time_s / tau_s))


@dataclass
class Sensor:
    tau_s: float
    value: float = 0.0
    bias: float = 0.0
    noise_std: float = 0.0
    failed: bool = False
    fault: str = ""
    quality: str = "good"
    quantum: float = 0.0
    drift_per_s: float = 0.0
    sample_period_s: float = 0.0
    held: float = 0.0
    sample_elapsed_s: float = 0.0

    def __post_init__(self) -> None:
        self.held = self.value
        if self.fault not in ("", "stuck", "open"):
            raise ValueError("Неисправность датчика задаётся как stuck или open")
        if self.failed and self.fault == "open":
            raise ValueError("Обрыв и залипание задаются раздельно")
        if self.failed:
            self.fault = "stuck"

    def step(self, truth: float, dt_s: float, rng: np.random.Generator) -> float:
        if self.fault == "open":
            self.quality = "bad"
            return float("nan")
        if self.fault == "stuck":
            self.quality = "bad"
            return self.held
        if self.sample_period_s > 0.0:
            self.sample_elapsed_s += dt_s
            if self.sample_elapsed_s + 1.0e-15 < self.sample_period_s:
                self.quality = "held"
                return self.held
            dt_s = self.sample_elapsed_s
            self.sample_elapsed_s = 0.0
        self.bias += self.drift_per_s * dt_s
        decay = 1.0 - np.exp(-dt_s / self.tau_s)
        self.value += decay * ((truth + self.bias) - self.value)
        measured = self.value
        if self.noise_std > 0.0:
            measured = self.value + float(rng.normal(0.0, self.noise_std))
        if self.quantum > 0.0:
            measured = float(self.quantum * np.round(measured / self.quantum))
        self.quality = "good"
        self.held = float(measured)
        return self.held


_MEASUREMENT_UNITS = {
    "power": "W",
    "temperature": "K",
    "pressure": "Pa",
    "flow": "kg/s",
    "level": "m",
}


@dataclass
class Measurement:
    """Канал с лагом датчика, качеством и отметкой времени. Вид измерения задаёт единицу."""

    kind: str
    sensor: Sensor
    time_s: float = 0.0

    def read(self, truth: float, dt_s: float, rng: np.random.Generator) -> dict:
        if self.kind not in _MEASUREMENT_UNITS:
            raise ValueError("Канал измерения %s не заявлен" % self.kind)
        self.time_s += dt_s
        value = self.sensor.step(truth, dt_s, rng)
        return {
            "value": value,
            "unit": _MEASUREMENT_UNITS[self.kind],
            "quality": self.sensor.quality,
            "fault": self.sensor.fault,
            "time_s": self.time_s,
        }


@dataclass
class DelayedActuator:
    """Команда доходит через чистое запаздывание, затем ограничивается скоростью."""

    position: float
    speed_limit: float
    delay_s: float
    minimum: float = 0.0
    maximum: float = 1.0
    _time_s: float = 0.0
    _held_demand: float = 0.0
    _queue: list = None

    def __post_init__(self) -> None:
        self._held_demand = self.position
        self._queue = []

    def step(self, demand: float, dt_s: float) -> float:
        if dt_s < 0.0 or self.delay_s < 0.0 or self.speed_limit < 0.0:
            raise ValueError("Запаздывание, шаг и скорость привода неотрицательны")
        issued_at = self._time_s
        self._time_s += dt_s
        self._queue.append((issued_at + self.delay_s, float(demand)))
        pending = []
        for release_s, value in self._queue:
            if release_s <= self._time_s + 1.0e-15:
                self._held_demand = value
            else:
                pending.append((release_s, value))
        self._queue = pending
        self.position = rate_limited_position(
            self.position, self._held_demand, self.speed_limit, dt_s, self.minimum, self.maximum
        )
        return self.position


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
        self.position = rate_limited_position(
            self.position, demand, self.speed_limit, dt_s, self.minimum, self.maximum
        )
        return self.position
