"""Чувствительность и распространение неопределённости без закрытых бенчмарков."""

from __future__ import annotations

import numpy as np

from core.assembly import jacobian_rank
from core.status import DomainError


def critical_root(function, low: float, high: float) -> float:
    """Корень скалярной невязки на отрезке. Вне отрезка корень не ищется."""
    from scipy.optimize import brentq

    if high <= low:
        raise DomainError("Отрезок поиска корня должен иметь положительную длину")
    left = float(function(low))
    right = float(function(high))
    if left == 0.0:
        return float(low)
    if right == 0.0:
        return float(high)
    if left * right > 0.0:
        raise DomainError("Корень не заключён между границами поиска")
    return float(brentq(function, low, high))


def critical_boron_ppm(reactivity_of_ppm, low_ppm: float, high_ppm: float) -> float:
    """Концентрация бора, на которой реактивность равна нулю."""
    if low_ppm < 0.0 or high_ppm < 0.0:
        raise DomainError("Поиск критического бора начинается с неотрицательной концентрации")
    return critical_root(reactivity_of_ppm, low_ppm, high_ppm)


def normal_central_interval(mean: float, std: float, coverage: float = 0.9) -> tuple:
    """Симметричный интервал нормального закона. Покрытие задаётся до выборки."""
    from scipy.special import ndtri

    if std < 0.0 or not 0.0 < coverage < 1.0:
        raise DomainError("Интервал задан при отрицательном разбросе или покрытии вне (0, 1)")
    if std == 0.0:
        return float(mean), float(mean)
    tail = 0.5 * (1.0 - coverage)
    half = float(ndtri(1.0 - tail))
    return float(mean - half * std), float(mean + half * std)


def coverage_across_seeds(
    mean: float, std: float, coverage: float, samples: int, seeds: np.ndarray
) -> np.ndarray:
    """Покрытие одного и того же интервала на нескольких независимых seed."""
    if samples < 1:
        raise DomainError("Число выборок покрытия должно быть положительным")
    seed_values = np.asarray(seeds, dtype=int)
    if seed_values.size == 0 or len(set(seed_values.tolist())) != seed_values.size:
        raise DomainError("Seed покрытия должны быть непустыми и неповторяющимися")
    low, high = normal_central_interval(mean, std, coverage)
    covered = []
    for seed in seed_values:
        draws = np.random.default_rng(int(seed)).normal(mean, std, size=samples)
        covered.append(sample_coverage(draws, low, high))
    return np.asarray(covered, dtype=float)


def sample_coverage(samples: np.ndarray, low: float, high: float) -> float:
    values = np.asarray(samples, dtype=float)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise DomainError("Выборка покрытия пуста или не конечна")
    return float(np.mean((values >= low) & (values <= high)))


class FrozenLinear:
    """Прямая, замороженная до проверочной выборки. Повторная подгонка запрещена."""

    def __init__(self) -> None:
        self.slope = 0.0
        self.intercept = 0.0
        self.frozen = False

    def fit(self, abscissa: np.ndarray, ordinate: np.ndarray) -> tuple:
        if self.frozen:
            raise DomainError("Коэффициенты уже заморожены и на проверочной выборке не подгоняются")
        x_values = np.asarray(abscissa, dtype=float)
        y_values = np.asarray(ordinate, dtype=float)
        if x_values.shape != y_values.shape or x_values.size < 2:
            raise DomainError("Подгонка прямой требует не меньше двух согласованных точек")
        design = np.vstack([x_values, np.ones(x_values.size)]).T
        self.slope, self.intercept = (
            float(value) for value in np.linalg.lstsq(design, y_values, rcond=None)[0]
        )
        self.frozen = True
        return self.slope, self.intercept

    def predict(self, abscissa: np.ndarray) -> np.ndarray:
        if not self.frozen:
            raise DomainError("Прямая ещё не заморожена")
        return self.slope * np.asarray(abscissa, dtype=float) + self.intercept

    def holdout_abs_error(self, abscissa: np.ndarray, ordinate: np.ndarray) -> float:
        slope = self.slope
        intercept = self.intercept
        error = float(np.max(np.abs(self.predict(abscissa) - np.asarray(ordinate, dtype=float))))
        if self.slope != slope or self.intercept != intercept:
            raise DomainError("Проверочная выборка изменила замороженные коэффициенты")
        return error


def equilibrium_amplitude(reactivity: float, generation_time_s: float, source: float) -> float:
    """Подкритическая амплитуда n = -S * Lambda / rho."""
    return -source * generation_time_s / reactivity


def monte_carlo_amplitude(
    reactivity_mean: float,
    reactivity_std: float,
    generation_time_s: float,
    source: float,
    samples: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    values = reactivity_mean + reactivity_std * rng.standard_normal(samples)
    values = values[values < -1.0e-6]
    amplitudes = equilibrium_amplitude(values, generation_time_s, source)
    return {
        "mean": float(np.mean(amplitudes)),
        "p05": float(np.quantile(amplitudes, 0.05)),
        "p95": float(np.quantile(amplitudes, 0.95)),
        "samples": int(amplitudes.size),
    }


def finite_difference_sensitivity(
    reactivity: float, generation_time_s: float, source: float, step: float = 1.0e-5
) -> float:
    high = equilibrium_amplitude(reactivity + step, generation_time_s, source)
    low = equilibrium_amplitude(reactivity - step, generation_time_s, source)
    return (high - low) / (2.0 * step)


def observability_rank(state_matrix: np.ndarray, output_matrix: np.ndarray) -> int:
    """Ранг матрицы наблюдаемости [C; CA; ...; CA^{n-1}]."""
    state = np.asarray(state_matrix, dtype=float)
    output = np.asarray(output_matrix, dtype=float)
    if state.ndim != 2 or state.shape[0] != state.shape[1] or state.shape[0] == 0:
        raise DomainError("Матрица состояния для наблюдаемости должна быть квадратной")
    if output.ndim != 2 or output.shape[1] != state.shape[0]:
        raise DomainError("Выход наблюдаемости не согласован с состоянием")
    rows = [output]
    block = output
    for _ in range(state.shape[0] - 1):
        block = block @ state
        rows.append(block)
    stacked = np.vstack(rows)
    # Степени жёсткой матрицы различаются на порядки. Ранг считается после нормировки строк и столбцов.
    row_scale = np.linalg.norm(stacked, axis=1)
    row_scale[row_scale == 0.0] = 1.0
    balanced = stacked / row_scale[:, None]
    column_scale = np.linalg.norm(balanced, axis=0)
    column_scale[column_scale == 0.0] = 1.0
    return jacobian_rank(balanced / column_scale)


def require_observable(state_matrix: np.ndarray, output_matrix: np.ndarray) -> int:
    rank = observability_rank(state_matrix, output_matrix)
    size = int(np.asarray(state_matrix).shape[0])
    if rank < size:
        raise DomainError("Выход не наблюдает всё состояние: ранг %s из %s" % (rank, size))
    return rank


def linear_normal_moments(
    mean: np.ndarray, covariance: np.ndarray, coefficients: np.ndarray
) -> tuple:
    """Среднее и дисперсия a·x для нормального x. Ковариация должна быть положительно полуопределена."""
    center = np.asarray(mean, dtype=float)
    cov = np.asarray(covariance, dtype=float)
    weights = np.asarray(coefficients, dtype=float)
    if center.ndim != 1 or weights.shape != center.shape or cov.shape != (center.size, center.size):
        raise DomainError("Моменты заданы для одного вектора и его ковариации")
    symmetric = 0.5 * (cov + cov.T)
    if np.any(np.linalg.eigvalsh(symmetric) < -1.0e-10):
        raise DomainError("Ковариация не положительно полуопределена")
    expected = float(weights @ center)
    variance = float(weights @ symmetric @ weights)
    return expected, variance


def require_identifiable(sensitivity: np.ndarray) -> int:
    """Столбцы — параметры, строки — выходы. Ранг меньше числа параметров означает, что они не разделяются.

    Строки и столбцы нормируются: иначе крупная производная прячет независимое измерение другой единицы.
    """
    array = np.atleast_2d(np.asarray(sensitivity, dtype=float))
    if array.shape[1] == 0 or not np.all(np.isfinite(array)):
        raise DomainError("Матрица чувствительности пуста или не конечна")
    row_scale = np.linalg.norm(array, axis=1)
    row_scale[row_scale == 0.0] = 1.0
    balanced = array / row_scale[:, None]
    column_scale = np.linalg.norm(balanced, axis=0)
    if np.any(column_scale == 0.0):
        raise DomainError("Один из параметров не входит ни в один выход")
    rank = jacobian_rank(balanced / column_scale)
    if rank < array.shape[1]:
        raise DomainError(
            "Параметры не разделяются по этим выходам: ранг %s из %s" % (rank, array.shape[1])
        )
    return rank
