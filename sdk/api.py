"""Headless API одного физического ядра ВВЭР."""

from __future__ import annotations

import json
import math

import numpy as np

from core.status import Applicability, DomainError
from plants.registry import build_plant
from plants.vver import PlantSnapshot, VVERPlant

_COMMANDS = {
    "throttle": (0.0, 1.0),
    "rod_command": (-1.0, 1.0),
    "feed_bias_kg_s": (-1.0e5, 1.0e5),
    "boron_ppm": (0.0, 2.0e4),
    "sink_temperature_k": (273.15, 373.15),
}
_FLAGS = ("scram", "motor_off", "motor_on")


def create_session(kind: str = "vver") -> VVERPlant:
    plant = build_plant(kind)
    if not isinstance(plant, VVERPlant):
        raise DomainError(
            "Динамическая сессия определена для паспорта ВВЭР; тип %s к ней не подменяется" % kind,
            Applicability.UNSUPPORTED,
        )
    return plant


def _finite_command(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Команда %s должна быть числом" % name)
    number = float(value)
    low, high = _COMMANDS[name]
    if not math.isfinite(number) or number < low or number > high:
        raise ValueError("Команда %s вне диапазона [%s, %s]" % (name, low, high))
    return number


def validate_commands(commands: dict | None = None) -> dict:
    """Проверить команды без изменения физического состояния."""
    if commands is not None and not isinstance(commands, dict):
        raise ValueError("Команды должны быть объектом")
    commands = dict(commands or {})
    unknown = sorted(set(commands) - set(_COMMANDS) - set(_FLAGS))
    if unknown:
        raise ValueError("Неизвестные команды: %s" % ", ".join(unknown))
    for name in _COMMANDS:
        if name in commands:
            commands[name] = _finite_command(name, commands[name])
    for name in _FLAGS:
        if name in commands and not isinstance(commands[name], bool):
            raise ValueError("Флаг %s должен быть bool" % name)
    if commands.get("motor_off") and commands.get("motor_on"):
        raise ValueError("Нельзя одновременно включить и отключить двигатель насоса")
    return commands


def apply_commands(plant: VVERPlant, commands: dict | None = None) -> VVERPlant:
    """Единая проверенная точка команд SDK, сценария, датасета, среды и пульта."""
    commands = validate_commands(commands)
    if "throttle" in commands:
        plant.throttle = commands["throttle"]
    if "rod_command" in commands:
        plant.rod_command = commands["rod_command"]
    if "feed_bias_kg_s" in commands:
        plant.feed_bias_kg_s = commands["feed_bias_kg_s"]
    if "boron_ppm" in commands:
        plant.set_boron_ppm(commands["boron_ppm"])
    if "sink_temperature_k" in commands:
        plant.set_sink_temperature_k(commands["sink_temperature_k"])
    if commands.get("scram"):
        plant.scram_latched = True
    if commands.get("motor_off"):
        plant.motor_on = False
    if commands.get("motor_on"):
        plant.motor_on = True
    return plant


def step_session(plant: VVERPlant, dt_s: float, commands: dict | None = None) -> dict:
    if isinstance(dt_s, bool) or not isinstance(dt_s, (int, float)):
        raise ValueError("Шаг сессии должен быть числом")
    if not math.isfinite(float(dt_s)) or float(dt_s) <= 0.0 or float(dt_s) > 5.0:
        raise ValueError("Шаг сессии должен быть в (0, 5] с; длинный горизонт собирается из шагов")
    apply_commands(plant, commands)
    return plant.advance(float(dt_s), rtol=1e-4, atol=1e-2)


SNAPSHOT_SCHEMA = 2


def export_snapshot(plant: VVERPlant) -> dict:
    snap = plant.snapshot()
    return {
        "schema_version": SNAPSHOT_SCHEMA,
        "model_id": plant.passport.model_id,
        "model_version": plant.passport.version,
        "time_s": snap.time_s,
        "state": np.asarray(snap.state, dtype=float).tolist(),
        "scram_latched": snap.scram_latched,
        "throttle": snap.throttle,
        "motor_on": snap.motor_on,
        "rod_command": snap.rod_command,
        "protection_timer_s": snap.protection_timer_s,
        "protection_latched": snap.protection_latched,
        "feed_bias_kg_s": snap.feed_bias_kg_s,
        "protection_setpoint_w": snap.protection_setpoint_w,
        "protection_delay_s": snap.protection_delay_s,
        "sink_temperature_k": snap.sink_temperature_k,
    }


def restore_snapshot(plant: VVERPlant, payload: dict) -> None:
    """Проверить снимок на изолированной сессии до изменения целевого объекта."""
    if not isinstance(payload, dict):
        raise DomainError("Снимок должен быть объектом")
    if payload.get("schema_version") != SNAPSHOT_SCHEMA:
        raise DomainError("Схема снимка не совпадает с текущей сессией")
    if payload.get("model_id") != plant.passport.model_id:
        raise DomainError("Снимок относится к другой модели и не восстанавливается")
    if payload.get("model_version") != plant.passport.version:
        raise DomainError("Версия паспорта снимка не совпадает с текущей сессией")
    if "sink_temperature_k" not in payload:
        raise DomainError("Снимок без температуры стока не восстанавливается")
    sink = payload["sink_temperature_k"]
    if (
        isinstance(sink, bool)
        or not isinstance(sink, (int, float))
        or not math.isfinite(float(sink))
    ):
        raise DomainError("Температура стока в снимке должна быть конечным числом")
    for name in ("scram_latched", "motor_on", "protection_latched"):
        if type(payload.get(name)) is not bool:
            raise DomainError("Флаг снимка %s должен быть bool" % name)
    scalars = {}
    for name in (
        "time_s",
        "throttle",
        "rod_command",
        "protection_timer_s",
        "feed_bias_kg_s",
        "protection_setpoint_w",
        "protection_delay_s",
        "sink_temperature_k",
    ):
        value = payload.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise DomainError("Поле снимка %s должно быть числом" % name)
        if not math.isfinite(float(value)):
            raise DomainError("Поле снимка %s должно быть конечным" % name)
        scalars[name] = float(value)
    for name in ("time_s", "protection_timer_s", "protection_delay_s"):
        if scalars[name] < 0.0:
            raise DomainError("Поле снимка %s отрицательно" % name)
    if scalars["protection_setpoint_w"] <= 0.0:
        raise DomainError("Уставка защиты должна быть положительной")
    try:
        state = np.asarray(payload.get("state"), dtype=float)
    except (TypeError, ValueError) as exc:
        raise DomainError("Вектор снимка должен быть числовым") from exc
    if state.shape != plant.pack().shape or not np.all(np.isfinite(state)):
        raise DomainError("Форма или значения вектора снимка недопустимы")
    if np.any(state[:10] < 0.0) or not 0.0 <= state[12] <= 1.0:
        raise DomainError("Отрицательный источник/предшественники или стержень вне упоров")
    candidate = create_session("vver")
    try:
        apply_commands(
            candidate,
            {
                key: scalars[key]
                for key in (
                    "throttle",
                    "rod_command",
                    "feed_bias_kg_s",
                    "sink_temperature_k",
                )
            },
        )
    except ValueError as exc:
        raise DomainError(str(exc)) from exc
    snapshot = PlantSnapshot(
        scalars["time_s"],
        state.copy(),
        payload["scram_latched"],
        scalars["throttle"],
        payload["motor_on"],
        scalars["rod_command"],
        scalars["protection_timer_s"],
        payload["protection_latched"],
        scalars["feed_bias_kg_s"],
        scalars["protection_setpoint_w"],
        scalars["protection_delay_s"],
        scalars["sink_temperature_k"],
    )
    candidate.restore(snapshot)
    candidate._reject_if_out_of_domain()
    plant.restore(snapshot)


def snapshot_json(plant: VVERPlant) -> str:
    return json.dumps(export_snapshot(plant), separators=(",", ":"))
