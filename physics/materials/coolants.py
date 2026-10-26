"""Теплоносители, отличные от воды. Водяные корреляции сюда не подставляются."""

from __future__ import annotations

from core.status import DomainError

R_UNIVERSAL = 8.314462618  # Дж/(моль·К)


def _check(name: str, temperature_k: float, low: float, high: float) -> None:
    if temperature_k < low or temperature_k > high:
        raise DomainError("%s вне диапазона %.0f–%.0f K" % (name, low, high))


def sodium_density_kg_m3(temperature_k: float) -> float:
    """Линейная плотность жидкого натрия, 400–1000 K: 1016 - 0.235*T."""
    _check("Натрий", temperature_k, 400.0, 1000.0)
    return 1016.0 - 0.235 * temperature_k


def sodium_cp_j_kg_k(temperature_k: float) -> float:
    _check("Натрий", temperature_k, 400.0, 1000.0)
    return 1436.0 - 0.58 * (temperature_k - 371.0) + 4.6e-4 * (temperature_k - 371.0) ** 2


def lead_density_kg_m3(temperature_k: float) -> float:
    """Жидкий свинец, 600–1200 K."""
    _check("Свинец", temperature_k, 600.0, 1200.0)
    return 11441.0 - 1.2795 * temperature_k


def helium_density_kg_m3(pressure_pa: float, temperature_k: float) -> float:
    """Идеальный газ. Не применяется к воде и не использует IF97."""
    _check("Гелий", temperature_k, 250.0, 1500.0)
    if pressure_pa <= 0.0 or pressure_pa > 10.0e6:
        raise DomainError("Гелий: давление вне 0–10 МПа этой модели")
    molar_mass = 0.004002602
    return pressure_pa * molar_mass / (R_UNIVERSAL * temperature_k)


def helium_cp_j_kg_k(temperature_k: float) -> float:
    _check("Гелий", temperature_k, 250.0, 1500.0)
    return 5193.0


def flibe_density_kg_m3(temperature_k: float) -> float:
    """Соль FLiBe, представительная линейная плотность, 800–1100 K."""
    _check("FLiBe", temperature_k, 800.0, 1100.0)
    return 2413.0 - 0.488 * temperature_k
