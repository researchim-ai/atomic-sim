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


def apply_commands(plant: VVERPlant, commands: dict | None = None) -> VVERPlant:
    """Единственная точка команд для SDK, сценария, датасета, среды и пульта.

    Неизвестный ключ, нечисло и выход за диапазон — ошибка входа, не физический режим.
    """
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
    if "throttle" in commands:
        plant.throttle = commands["throttle"]
    if "rod_command" in commands:
        plant.rod_command = commands["rod_command"]
    if "feed_bias_kg_s" in commands:
        plant.feed_bias_kg_s = commands["feed_bias_kg_s"]
    if "boron_ppm" in commands:
        plant.set_boron_ppm(commands["boron_ppm"])
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


def export_snapshot(plant: VVERPlant) -> dict:
    snap = plant.snapshot()
    return {
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
    }


def restore_snapshot(plant: VVERPlant, payload: dict) -> None:
    if payload.get("model_id") != plant.passport.model_id:
        raise DomainError("Снимок относится к другой модели и не восстанавливается")
    if payload.get("model_version") != plant.passport.version:
        raise DomainError("Версия паспорта снимка не совпадает с текущей сессией")
    plant.restore(
        PlantSnapshot(
            float(payload["time_s"]),
            np.asarray(payload["state"], dtype=float),
            bool(payload["scram_latched"]),
            float(payload["throttle"]),
            bool(payload["motor_on"]),
            float(payload["rod_command"]),
            float(payload["protection_timer_s"]),
            bool(payload["protection_latched"]),
            float(payload["feed_bias_kg_s"]),
        )
    )


def snapshot_json(plant: VVERPlant) -> str:
    return json.dumps(export_snapshot(plant), separators=(",", ":"))
