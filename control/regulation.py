"""ПИД с ограничением интеграла и защита 2 из 3."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from core.status import ModelPassport, ModelStatus

PID_PASSPORT = ModelPassport(
    model_id="pid-antiwindup",
    version="2.0.0",
    status=ModelStatus.VERIFIED,
    phenomenon="ПИД-регулятор с условным интегрированием",
    approximations=("непрерывное время", "насыщение выхода известно вызывающему коду"),
    applicability="Один контур, выход в заданных пределах",
    data_sources=("Стандартная форма ПИД с anti-windup через останов интеграла",),
)

PROTECTION_PASSPORT = ModelPassport(
    model_id="protection-2oo3",
    version="2.0.0",
    status=ModelStatus.VERIFIED,
    phenomenon="Логика защиты два из трёх с выдержкой и защёлкой",
    approximations=("уставки задаёт паспорт модели, это не эксплуатационные уставки блока",),
    applicability="Три независимых канала одного параметра",
    data_sources=("Логическая схема, заданная в этом модуле",),
)


@dataclass
class PID:
    kp: float
    ki: float
    kd: float
    integral: float = 0.0
    previous_error: float = 0.0
    output_limit: float = 1.0

    def reset_to_output(self, error: float, output: float) -> None:
        """Безударное включение: интеграл подстраивается под текущий выход."""
        self.integral = (output - self.kp * error) / self.ki if self.ki != 0.0 else 0.0
        self.previous_error = error

    def step(self, error: float, dt_s: float) -> float:
        derivative = 0.0 if dt_s <= 0.0 else (error - self.previous_error) / dt_s
        proposal = self.kp * error + self.ki * (self.integral + error * dt_s) + self.kd * derivative
        saturated = abs(proposal) > self.output_limit
        pushing = proposal * error > 0.0
        if not (saturated and pushing):
            self.integral += error * dt_s
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        output = max(-self.output_limit, min(self.output_limit, output))
        self.previous_error = error
        return output


@dataclass
class ProtectionLogic:
    setpoint: float
    delay_s: float
    timer_s: float = 0.0
    latched: bool = False
    direction: str = "high"

    def channel_trip(self, value: float) -> bool:
        if self.direction == "high":
            return value >= self.setpoint
        return value <= self.setpoint

    def step(self, channels: Sequence[float], dt_s: float, reset: bool = False) -> bool:
        if reset and not any(self.channel_trip(value) for value in channels):
            self.latched = False
            self.timer_s = 0.0
        votes = sum(self.channel_trip(value) for value in channels)
        if votes >= 2:
            self.timer_s += dt_s
        else:
            self.timer_s = 0.0
        if self.timer_s >= self.delay_s:
            self.latched = True
        return self.latched


@dataclass
class ProtectionEvent:
    demand: bool
    actuator_moved: bool
    note: str = ""
    log: list = field(default_factory=list)
