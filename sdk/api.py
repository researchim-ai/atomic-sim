"""Headless API одного физического ядра."""

from __future__ import annotations

from __future__ import annotations

from plants.registry import build_plant
from plants.vver import VVERPlant


def create_session(kind: str = "vver") -> VVERPlant:
    plant = build_plant(kind)
    if not isinstance(plant, VVERPlant):
        raise TypeError("Сессия динамики пока определена для ВВЭР/PWR")
    return plant


def step_session(plant: VVERPlant, dt_s: float, commands: dict | None = None) -> dict:
    commands = commands or {}
    if "throttle" in commands:
        plant.throttle = float(commands["throttle"])
    if "rod_command" in commands:
        plant.rod_command = float(commands["rod_command"])
    if commands.get("scram"):
        plant.scram_latched = True
    if commands.get("motor_off"):
        plant.motor_on = False
    return plant.advance(dt_s, rtol=1e-4, atol=1e-2)
