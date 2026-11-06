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


def sodium_internal_energy_j_kg(temperature_k: float, reference_k: float = 400.0) -> float:
    """Интеграл sodium_cp. Опорная точка 400 K лежит внутри диапазона плотности."""
    _check("Натрий", temperature_k, 400.0, 1000.0)
    _check("Натрий", reference_k, 400.0, 1000.0)

    def primitive(temperature: float) -> float:
        offset = temperature - 371.0
        return 1436.0 * offset - 0.29 * offset**2 + 4.6e-4 * offset**3 / 3.0

    return primitive(temperature_k) - primitive(reference_k)


def lead_specific_heat_j_kg_k(temperature_k: float) -> float:
    """Контрольная линейная ветвь проекта на 600–1200 K, не справочник конкретного контура."""
    _check("Свинец", temperature_k, 600.0, 1200.0)
    return 160.0 - 0.02 * (temperature_k - 600.0)


def lead_internal_energy_j_kg(temperature_k: float, reference_k: float = 600.0) -> float:
    _check("Свинец", temperature_k, 600.0, 1200.0)
    _check("Свинец", reference_k, 600.0, 1200.0)

    def primitive(temperature: float) -> float:
        offset = temperature - 600.0
        return 160.0 * offset - 0.01 * offset**2

    return primitive(temperature_k) - primitive(reference_k)


def lead_density_kg_m3(temperature_k: float) -> float:
    """Жидкий свинец, 600–1200 K."""
    _check("Свинец", temperature_k, 600.0, 1200.0)
    return 11441.0 - 1.2795 * temperature_k


def helium_specific_gas_constant_j_kg_k() -> float:
    """R/M для гелия этой модели."""
    return R_UNIVERSAL / 0.004002602


def helium_density_kg_m3(pressure_pa: float, temperature_k: float) -> float:
    """Идеальный газ. Не применяется к воде и не использует IF97."""
    _check("Гелий", temperature_k, 250.0, 1500.0)
    if pressure_pa <= 0.0 or pressure_pa > 10.0e6:
        raise DomainError("Гелий: давление вне 0–10 МПа этой модели")
    return pressure_pa / (helium_specific_gas_constant_j_kg_k() * temperature_k)


def helium_cp_j_kg_k(temperature_k: float) -> float:
    _check("Гелий", temperature_k, 250.0, 1500.0)
    return 5193.0


def helium_cv_j_kg_k(temperature_k: float) -> float:
    """cv = cp − R/M. Теплоёмкость этой модели не зависит от температуры."""
    cv = helium_cp_j_kg_k(temperature_k) - helium_specific_gas_constant_j_kg_k()
    if cv <= 0.0:
        raise DomainError("cv гелия этой модели не положительна")
    return cv


def helium_internal_energy_j_kg(temperature_k: float, reference_k: float = 300.0) -> float:
    _check("Гелий", reference_k, 250.0, 1500.0)
    return helium_cv_j_kg_k(temperature_k) * (temperature_k - reference_k)


def helium_enthalpy_j_kg(temperature_k: float, reference_k: float = 300.0) -> float:
    _check("Гелий", reference_k, 250.0, 1500.0)
    return helium_cp_j_kg_k(temperature_k) * (temperature_k - reference_k)


def sodium_frozen_cp_gap_j_kg(start_k: float, end_k: float) -> float:
    """Энергия по переменной cp минус шаг с cp, замороженной в начале интервала."""
    actual = sodium_internal_energy_j_kg(end_k) - sodium_internal_energy_j_kg(start_k)
    frozen = sodium_cp_j_kg_k(start_k) * (end_k - start_k)
    return actual - frozen


def flibe_density_kg_m3(temperature_k: float) -> float:
    """Соль FLiBe, представительная линейная плотность, 800–1100 K."""
    _check("FLiBe", temperature_k, 800.0, 1100.0)
    return 2413.0 - 0.488 * temperature_k
