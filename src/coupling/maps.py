"""Консервативное отображение мощности между сетками и внешний решатель."""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np

from core.status import Applicability, DomainError, SolverError


def conserve_integral(
    source: np.ndarray, source_widths: np.ndarray, target_widths: np.ndarray
) -> np.ndarray:
    """Кусочно-постоянное отображение с сохранением интеграла по длине."""
    from math import fsum
    from .conservative import ConservativeMap, vector
    values=vector(source)
    sw=vector(source_widths,len(values),positive=True)
    tw=vector(target_widths,positive=True)
    if max(len(sw),len(tw))>512:raise DomainError("One to 512 cells required")
    try:extent=fsum(sw);other=fsum(tw)
    except OverflowError as error:raise DomainError('Mesh length exceeds float64 range') from error
    if abs(extent-other)>8*np.finfo(float).eps*max(extent,other):
        raise DomainError("Meshes must cover the same length within roundoff")
    a=np.r_[0.,np.cumsum(sw)];b=np.r_[0.,np.cumsum(tw)]
    # Width-based legacy inputs can differ only by summation roundoff. Align
    # their terminal coordinate, never rescale an interior cell or a state.
    a[-1]=b[-1]=extent
    operator=ConservativeMap(a,b)
    result=operator.averages(values)
    try:
        with np.errstate(over='raise',invalid='raise'):
            original=values*sw;transferred=result*tw
        defect=abs(fsum(transferred)-fsum(original))
        scale=max(fsum(abs(original)),1.)
    except (FloatingPointError,OverflowError,ValueError) as error:
        raise DomainError('Integral exceeds finite float64 range') from error
    if defect>2e-12*scale:raise DomainError("Integral transfer failed")
    return result


class ExternalSolverAdapter:
    """Обмен с внешним кодом. Отказ внешнего решателя не подменяется локальной оценкой."""

    def __init__(self, runner: Callable[[Dict], Dict], version: str):
        self.runner = runner
        self.version = version

    def exchange(self, payload: Dict) -> Dict:
        try:
            result = self.runner(payload)
        except (DomainError, SolverError) as exc:
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
