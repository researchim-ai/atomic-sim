"""Снимок и повторный прогон. Совпадение проверяется допуском теста, не «на глаз»."""

from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np

from plants.vver import PlantSnapshot, VVERPlant


def run_actions(plant: VVERPlant, actions: List[Dict], step_s: float) -> np.ndarray:
    rows = []
    for action in actions:
        if "throttle" in action:
            plant.throttle = float(action["throttle"])
        if "rod_command" in action:
            plant.rod_command = float(action["rod_command"])
        if action.get("motor_off"):
            plant.motor_on = False
        if action.get("scram"):
            plant.scram_latched = True
        obs = plant.advance(step_s, rtol=1e-4, atol=1e-2)
        rows.append([obs["thermal_power_w"], obs["sg_pressure_pa"], obs["mass_flow_kg_s"], obs["rod_insertion"]])
    return np.array(rows)


def replay_matches(factory: Callable[[], VVERPlant], actions: List[Dict], step_s: float, atol: float, rtol: float) -> bool:
    first = run_actions(factory(), actions, step_s)
    second = run_actions(factory(), actions, step_s)
    return np.allclose(first, second, atol=atol, rtol=rtol)


def restart_matches(plant_factory: Callable[[], VVERPlant], step_s: float, splits: int, atol: float, rtol: float) -> bool:
    continuous = plant_factory()
    whole = []
    for _ in range(splits):
        obs = continuous.advance(step_s, rtol=1e-4, atol=1e-2)
        whole.append(obs["thermal_power_w"])
    partial = plant_factory()
    for _ in range(splits // 2):
        partial.advance(step_s, rtol=1e-4, atol=1e-2)
    snapshot: PlantSnapshot = partial.snapshot()
    restored = plant_factory()
    restored.restore(snapshot)
    restarted = []
    # Первая половина берётся из непрерывного прогона до снимка: снимок уже её содержит.
    # Сравниваем только продолжение.
    tail = []
    for _ in range(splits - splits // 2):
        obs = restored.advance(step_s, rtol=1e-4, atol=1e-2)
        tail.append(obs["thermal_power_w"])
    return np.allclose(np.array(whole[splits // 2 :]), np.array(tail), atol=atol, rtol=rtol)
