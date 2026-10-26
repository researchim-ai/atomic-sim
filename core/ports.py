"""Порты компонентов. Поток на интерфейсе входит в соседние балансы с противоположным знаком."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class HydraulicPort:
    pressure_pa: float
    mass_flow_kg_s: float
    enthalpy_j_kg: float
    boron_fraction: float = 0.0


@dataclass
class ThermalPort:
    temperature_k: float
    heat_flow_w: float
    area_m2: float


@dataclass
class MechanicalPort:
    torque_nm: float
    omega_rad_s: float


@dataclass
class SignalPort:
    value: float
    unit: str
    quality: str
    time_s: float
    available: bool = True


@dataclass
class NeutronCurrentPort:
    group: int
    current_per_m2_s: float
    area_m2: float


def hydraulic_exchange(
    port_a: HydraulicPort, port_b: HydraulicPort, tol: float = 1e-9
) -> Optional[str]:
    """Проверка, что общий расход согласован по знаку и величине."""
    if abs(port_a.mass_flow_kg_s + port_b.mass_flow_kg_s) > tol:
        return "Массовый расход на общем гидравлическом интерфейсе не антисимметричен"
    return None
