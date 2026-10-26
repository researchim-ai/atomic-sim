"""Диагностика невязок и применимости."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from core.status import Applicability


@dataclass
class ComponentDiagnostics:
    name: str
    applicability: Applicability = Applicability.IN_DOMAIN
    message: str = ""
    residuals: Dict[str, float] = field(default_factory=dict)
    limiter_applied: bool = False


def mass_residual(mass_t: float, mass_0: float, net_in_kg: float) -> float:
    """R_M = M(t) - M(0) - интеграл притока."""
    return mass_t - mass_0 - net_in_kg


def energy_residual(energy_t: float, energy_0: float, net_in_j: float) -> float:
    """R_E при зафиксированном составе энергии: внутренняя энергия контрольных объёмов."""
    return energy_t - energy_0 - net_in_j
