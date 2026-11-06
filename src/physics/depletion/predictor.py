"""Один нуклид при постоянном микросечении. Поток на шаге предиктора-корректора усредняется.

Это не библиотека выгорания топлива и не запуск OpenMC.
"""

from __future__ import annotations

import math

from core.status import DomainError


def number_density_exponential(
    number_density_0: float, microscopic_m2: float, flux_per_m2_s: float, dt_s: float
) -> float:
    if min(number_density_0, microscopic_m2, flux_per_m2_s, dt_s) < 0.0:
        raise DomainError("Выгорание одного нуклида задано при отрицательном входе")
    return number_density_0 * math.exp(-microscopic_m2 * flux_per_m2_s * dt_s)


def predictor_corrector_density(
    number_density_0: float,
    microscopic_m2: float,
    flux_start_per_m2_s: float,
    flux_end_per_m2_s: float,
    dt_s: float,
) -> float:
    """Корректор использует среднее потоков концов шага. При постоянном потоке совпадает с экспонентой."""
    if flux_end_per_m2_s < 0.0:
        raise DomainError("Конечный поток выгорания отрицателен")
    mean_flux = 0.5 * (flux_start_per_m2_s + flux_end_per_m2_s)
    return number_density_exponential(number_density_0, microscopic_m2, mean_flux, dt_s)


def constant_power_step(
    number_density_0: float, microscopic_m2: float, flux_start_per_m2_s: float, dt_s: float
) -> tuple:
    """Постоянная мощность: поток растёт как 1/N. Возвращает предиктор, корректор и прямую N(t).

    Прямая — точное решение dN/dt = -σ φ0 N0. Корректор берёт поток конца по предсказанной плотности.
    """
    if number_density_0 <= 0.0 or microscopic_m2 <= 0.0 or flux_start_per_m2_s < 0.0 or dt_s < 0.0:
        raise DomainError("Постоянная мощность задана при неположительном запасе или сечении")
    removal = microscopic_m2 * flux_start_per_m2_s * number_density_0
    linear = number_density_0 - removal * dt_s
    if linear <= 0.0:
        raise DomainError("Постоянная мощность исчерпала нуклид на этом шаге")
    predicted = number_density_exponential(
        number_density_0, microscopic_m2, flux_start_per_m2_s, dt_s
    )
    flux_end = flux_start_per_m2_s * number_density_0 / predicted
    corrected = predictor_corrector_density(
        number_density_0, microscopic_m2, flux_start_per_m2_s, flux_end, dt_s
    )
    return predicted, corrected, linear
