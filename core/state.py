"""Контракт состояния. Поля не смешивают физические запасы, алгебру и датчики."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from core.status import Applicability


@dataclass
class PhysicalState:
    """Истинные физические поля и запасы. Единицы СИ."""

    time_s: float = 0.0
    masses_kg: Dict[str, float] = field(default_factory=dict)
    internal_energy_j: Dict[str, float] = field(default_factory=dict)
    neutron_density: float = 1.0
    precursors: Dict[str, float] = field(default_factory=dict)
    temperatures_k: Dict[str, float] = field(default_factory=dict)
    boron_kg: Dict[str, float] = field(default_factory=dict)


@dataclass
class AlgebraicState:
    """Давления, расходы и другие величины, определяемые ограничениями формулировки."""

    pressure_pa: Dict[str, float] = field(default_factory=dict)
    mass_flow_kg_s: Dict[str, float] = field(default_factory=dict)
    quality: Dict[str, float] = field(default_factory=dict)


@dataclass
class SensorState:
    value: float
    unit: str
    quality: str = "good"
    time_s: float = 0.0
    available: bool = True


@dataclass
class ActuatorState:
    position: float
    speed: float
    demand: float
    failed: bool = False
    at_limit: bool = False


@dataclass
class ControlState:
    """Интеграторы регуляторов, дискретные режимы и защёлки защит."""

    integrals: Dict[str, float] = field(default_factory=dict)
    mode: str = "manual"
    scram_demand: bool = False
    scram_latched: bool = False
    trip_timer_s: float = 0.0


@dataclass
class Observation:
    """Только то, что разрешено видеть агенту."""

    values: Dict[str, float]
    units: Dict[str, str]
    quality: Dict[str, str]
    time_s: float


@dataclass
class Diagnostics:
    residual_mass_kg: float = 0.0
    residual_energy_j: float = 0.0
    applicability: Applicability = Applicability.IN_DOMAIN
    message: str = ""
    limiter_applied: bool = False
    iterations: int = 0
    converged: bool = True
