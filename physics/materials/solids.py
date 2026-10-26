"""Твёрдые материалы активной зоны. Корреляции не подменяют свойства воды."""

from __future__ import annotations

from core.status import DomainError

# Fink, J.K. Thermophysical properties of uranium dioxide. J. Nucl. Mater. 279 (2000).
# t = T/1000. Диапазон исходной рекомендации примерно 298–3120 K.


def uo2_conductivity_w_m_k(temperature_k: float) -> float:
    if temperature_k < 298.0 or temperature_k > 3000.0:
        raise DomainError("Теплопроводность UO2 вне 298–3000 K")
    t = temperature_k / 1000.0
    phonon = 100.0 / (7.5408 + 17.692 * t + 3.6142 * t * t)
    electronic = 6400.0 * np_exp(-16.35 / t) / t**2.5
    return phonon + electronic


def uo2_specific_heat_j_kg_k(temperature_k: float) -> float:
    """Удельная теплоёмкость UO2, упрощённая полиномиальная ветвь Fink для проверки баланса."""
    if temperature_k < 298.0 or temperature_k > 3000.0:
        raise DomainError("Теплоёмкость UO2 вне 298–3000 K")
    t = temperature_k / 1000.0
    # Дж/(кг·К): перевод мольной оценки 270 Дж/(кг·К) при 300 K к росту с температурой.
    return 193.0 + 325.0 * t - 30.0 * t * t


def zircaloy_conductivity_w_m_k(temperature_k: float) -> float:
    """Представительная линейная ветвь для сплава циркония, не паспорт конкретного твэла."""
    if temperature_k < 300.0 or temperature_k > 1500.0:
        raise DomainError("Теплопроводность циркониевой оболочки вне 300–1500 K")
    return 7.51 + 2.09e-2 * temperature_k - 1.45e-5 * temperature_k**2 + 7.67e-9 * temperature_k**3


def zircaloy_specific_heat_j_kg_k(temperature_k: float) -> float:
    if temperature_k < 300.0 or temperature_k > 1500.0:
        raise DomainError("Теплоёмкость циркониевой оболочки вне 300–1500 K")
    return 255.0 + 0.16 * temperature_k


def graphite_conductivity_w_m_k(temperature_k: float) -> float:
    if temperature_k < 300.0 or temperature_k > 1800.0:
        raise DomainError("Теплопроводность графита вне 300–1800 K")
    return 80.0 * (300.0 / temperature_k) ** 0.5


def graphite_specific_heat_j_kg_k(temperature_k: float) -> float:
    if temperature_k < 300.0 or temperature_k > 1800.0:
        raise DomainError("Теплоёмкость графита вне 300–1800 K")
    return 700.0 + 1.2 * (temperature_k - 300.0)


def np_exp(value: float) -> float:
    import math

    return math.exp(value)
