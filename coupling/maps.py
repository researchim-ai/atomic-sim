"""Консервативное отображение мощности между сетками и внешний решатель."""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np

from core.status import Applicability, DomainError


def conserve_integral(
    source: np.ndarray, source_widths: np.ndarray, target_widths: np.ndarray
) -> np.ndarray:
    """Кусочно-постоянное отображение с сохранением интеграла по длине."""
    if (
        source.size != source_widths.size
        or abs(float(np.sum(source_widths) - np.sum(target_widths))) > 1e-8
    ):
        raise DomainError("Сетки источника и приёмника должны покрывать одну длину")
    src_edges = np.concatenate([[0.0], np.cumsum(source_widths)])
    tgt_edges = np.concatenate([[0.0], np.cumsum(target_widths)])
    total = source * source_widths
    # Плотность источника.
    values = np.zeros(target_widths.size)
    for i, (left, right) in enumerate(zip(tgt_edges[:-1], tgt_edges[1:])):
        integral = 0.0
        for j, (s_left, s_right) in enumerate(zip(src_edges[:-1], src_edges[1:])):
            overlap = max(0.0, min(right, s_right) - max(left, s_left))
            if overlap > 0.0:
                integral += source[j] * overlap
        values[i] = integral / target_widths[i]
    if abs(float(np.sum(values * target_widths) - np.sum(total))) > 1e-6 * max(
        1.0, abs(float(np.sum(total)))
    ):
        raise DomainError("Отображение не сохранило интеграл")
    return values


class ExternalSolverAdapter:
    """Обмен с внешним кодом. Отказ внешнего решателя не подменяется локальной оценкой."""

    def __init__(self, runner: Callable[[Dict], Dict], version: str):
        self.runner = runner
        self.version = version

    def exchange(self, payload: Dict) -> Dict:
        try:
            result = self.runner(payload)
        except Exception as exc:
            return {
                "applicability": Applicability.UNSUPPORTED.value,
                "message": str(exc),
                "version": self.version,
            }
        result = dict(result)
        result["version"] = self.version
        return result


def picard_power(
    reactivity_of_power, power0: float, iterations: int = 20, tol: float = 1e-8
) -> float:
    """Неподвижная точка P = P0 * exp не используется.

    Для отрицательной обратной связи ищем P, при которой реактивность(P) = 0,
    линейным делением отрезка. Это проверка сходимости связки, не динамика.
    """
    low, high = 0.2 * power0, 2.0 * power0
    for _ in range(iterations):
        mid = 0.5 * (low + high)
        if reactivity_of_power(mid) > 0.0:
            low = mid
        else:
            high = mid
        if abs(high - low) < tol * power0:
            break
    return 0.5 * (low + high)
