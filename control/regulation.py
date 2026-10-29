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


def pid_integrator_output(error: float, ki: float, time_s: float) -> float:
    """Выход чистого интегратора без насыщения: ki · ошибка · t."""
    if time_s < 0.0:
        raise ValueError("Время интегратора отрицательно")
    return ki * error * time_s


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

    def engage(self, error: float, manual_output: float) -> float:
        """Безударный переход с ручного выхода. Первый автоматический шаг продолжает это значение."""
        if self.ki == 0.0:
            raise ValueError("Безударное включение требует ненулевого интегрального коэффициента")
        held = max(-self.output_limit, min(self.output_limit, float(manual_output)))
        self.reset_to_output(error, held)
        return held


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
            if self.timer_s >= self.delay_s:
                self.latched = True
        else:
            self.timer_s = 0.0
        return self.latched


def protection_step_good_channels(
    logic: ProtectionLogic, values: Sequence[float], qualities: Sequence[str], dt_s: float
) -> bool:
    """Голос имеет только канал качества good. Один исправный канал не даёт 2 из 3."""
    if len(values) != len(qualities):
        raise ValueError("Число значений и качеств каналов не совпадает")
    voting = [value for value, quality in zip(values, qualities) if quality == "good"]
    return logic.step(voting, dt_s)


def _trip_regions(t0: float, t1: float, v0: float, v1: float, setpoint: float, direction: str):
    def active(value: float) -> bool:
        if direction == "high":
            return value >= setpoint
        if direction == "low":
            return value <= setpoint
        raise ValueError("Направление защиты задаётся как high или low")

    start_active = active(v0)
    end_active = active(v1)
    if start_active == end_active or v1 == v0:
        return [(t0, t1, start_active)]
    fraction = (setpoint - v0) / (v1 - v0)
    crossing = t0 + fraction * (t1 - t0)
    if end_active:
        return [(t0, crossing, False), (crossing, t1, True)]
    return [(t0, crossing, True), (crossing, t1, False)]


def _latch_time(
    logic: ProtectionLogic, value_at, t0: float, t1: float, output_step_s: float, direction: str
):
    if output_step_s <= 0.0:
        raise ValueError("Шаг вывода защиты должен быть положительным")
    clock = t0
    while clock < t1 - 1.0e-15:
        nxt = min(t1, clock + output_step_s)
        for start, stop, active in _trip_regions(
            clock, nxt, float(value_at(clock)), float(value_at(nxt)), logic.setpoint, direction
        ):
            span = stop - start
            if span <= 0.0:
                continue
            sample = float(value_at(0.5 * (start + stop)))
            channels = [sample, sample, sample]
            if active:
                remaining = logic.delay_s - logic.timer_s
                if span + 1.0e-12 >= remaining:
                    logic.timer_s = logic.delay_s
                    logic.latched = True
                    return start + remaining
            logic.step(channels, span)
            if logic.latched:
                return stop
        clock = nxt
    return None


def high_trip_time(logic: ProtectionLogic, value_at, t0: float, t1: float, output_step_s: float):
    """Время защёлки 2 из 3 по линейному сигналу выше уставки внутри шага вывода."""
    return _latch_time(logic, value_at, t0, t1, output_step_s, "high")


def directed_trip_time(
    logic: ProtectionLogic, value_at, t0: float, t1: float, output_step_s: float
):
    """То же для направления logic.direction. Таймер копит только время в зоне срабатывания."""
    return _latch_time(logic, value_at, t0, t1, output_step_s, logic.direction)


@dataclass
class ProtectionEvent:
    demand: bool
    actuator_moved: bool
    note: str = ""
    log: list = field(default_factory=list)


def _smooth_worth(insertion: float) -> float:
    position = min(1.0, max(0.0, insertion))
    return position * position * (3.0 - 2.0 * position)


@dataclass
class HysteresisSwitch:
    """Включается на верхней границе и держится, пока значение не опустится ниже нижней."""

    engage_at: float
    release_at: float
    engaged: bool = False

    def __post_init__(self) -> None:
        if self.release_at >= self.engage_at:
            raise ValueError("Отпускание гистерезиса должно быть ниже включения")

    def step(self, value: float) -> bool:
        if self.engaged:
            if value <= self.release_at:
                self.engaged = False
        elif value >= self.engage_at:
            self.engaged = True
        return self.engaged


@dataclass
class RodBank:
    name: str
    insertion: float
    speed_per_s: float
    worth: float
    reference: float


class RodBanks:
    """Несколько групп. SCRAM вводит все группы и не уступает ручной команде."""

    def __init__(self, banks: list):
        self.banks = list(banks)

    def step(self, command: float, scram: bool, dt_s: float) -> None:
        if dt_s < 0.0:
            raise ValueError("Шаг группы стержней отрицателен")
        for bank in self.banks:
            speed = bank.speed_per_s if scram else command * bank.speed_per_s
            bank.insertion = min(1.0, max(0.0, bank.insertion + speed * dt_s))

    def reactivity(self) -> float:
        total = 0.0
        for bank in self.banks:
            total -= bank.worth * (_smooth_worth(bank.insertion) - _smooth_worth(bank.reference))
        return total
