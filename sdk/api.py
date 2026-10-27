"""Headless API одного физического ядра ВВЭР."""

from __future__ import annotations

import json

import numpy as np

from core.status import Applicability, DomainError
from plants.registry import build_plant
from plants.vver import PlantSnapshot, VVERPlant


def create_session(kind: str = "vver") -> VVERPlant:
    plant = build_plant(kind)
    if not isinstance(plant, VVERPlant):
        raise DomainError(
            "Динамическая сессия определена для паспорта ВВЭР; тип %s к ней не подменяется" % kind,
            Applicability.UNSUPPORTED,
        )
    return plant


def apply_commands(plant: VVERPlant, commands: dict | None = None) -> VVERPlant:
    """Единственная точка команд для SDK, сценария, датасета, среды и пульта."""
    commands = commands or {}
    if "throttle" in commands:
        plant.throttle = float(commands["throttle"])
    if "rod_command" in commands:
        plant.rod_command = float(commands["rod_command"])
    if "feed_bias_kg_s" in commands:
        plant.feed_bias_kg_s = float(commands["feed_bias_kg_s"])
    if "boron_ppm" in commands:
        plant.set_boron_ppm(float(commands["boron_ppm"]))
    if commands.get("scram"):
        plant.scram_latched = True
    if commands.get("motor_off"):
        plant.motor_on = False
    if commands.get("motor_on"):
        plant.motor_on = True
    return plant


def step_session(plant: VVERPlant, dt_s: float, commands: dict | None = None) -> dict:
    apply_commands(plant, commands)
    return plant.advance(dt_s, rtol=1e-4, atol=1e-2)


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
