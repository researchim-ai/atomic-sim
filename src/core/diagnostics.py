"""Диагностика невязок и применимости."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict

from core.status import Applicability, classify_model_error


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


@dataclass(frozen=True)
class FailureReport:
    component: str
    constraint: str
    residual: float
    reproduction: str
    error_class: str


def failure_report(
    component: str, constraint: str, residual: float, reproduction: str, exc: BaseException
) -> FailureReport:
    """Отказ называет компонент, ограничение, невязку и способ повторить расчёт."""
    if not str(component).strip() or not str(constraint).strip() or not str(reproduction).strip():
        raise ValueError("Отчёт отказа требует компонент, ограничение и способ повтора")
    if not math.isfinite(residual):
        raise ValueError("Невязка в отчёте отказа должна быть конечной")
    return FailureReport(
        str(component),
        str(constraint),
        float(residual),
        str(reproduction),
        classify_model_error(exc),
    )
