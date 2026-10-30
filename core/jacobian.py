"""Масштабированные центральные разности для RHS с изменяемой алгеброй."""

from __future__ import annotations

import numpy as np

from core.status import DomainError, SolverError


class CentralJacobian:
    """Без округления состояния; на границе области — односторонняя разность.

    После вычисления RHS снова вызывается в исходном состоянии. Счётчик включает
    все попытки, в том числе отвергнутые областью и восстановление алгебры.
    Исключения solver/отмены/программы не превращаются в численную производную.
    """

    def __init__(self, rhs, relative_step: float):
        if isinstance(relative_step, bool) or not np.isfinite(relative_step) or relative_step <= 0:
            raise SolverError("Шаг якобиана должен быть положительным конечным числом")
        self.rhs = rhs
        self.relative_step = float(relative_step)
        self.rhs_evaluations = 0

    def _evaluate(self, t, y):
        self.rhs_evaluations += 1
        value = np.asarray(self.rhs(t, y), dtype=float).copy()
        if value.shape != y.shape or not np.all(np.isfinite(value)):
            raise SolverError("RHS якобиана имеет неверную форму или неконечные значения")
        return value

    def __call__(self, t, state):
        y = np.asarray(state, dtype=float).copy()
        base = self._evaluate(t, y)
        result = np.empty((y.size, y.size))
        try:
            for index in range(y.size):
                step = self.relative_step * max(abs(y[index]), 1.0)
                left, right = y.copy(), y.copy()
                left[index] -= step
                right[index] += step
                lo = hi = None
                try:
                    lo = self._evaluate(t, left)
                except DomainError:
                    pass
                try:
                    hi = self._evaluate(t, right)
                except DomainError:
                    pass
                if lo is not None and hi is not None:
                    result[:, index] = (hi - lo) / (right[index] - left[index])
                elif hi is not None:
                    result[:, index] = (hi - base) / (right[index] - y[index])
                elif lo is not None:
                    result[:, index] = (base - lo) / (y[index] - left[index])
                else:
                    raise DomainError("Обе пробы якобиана вне области для переменной %d" % index)
        finally:
            self._evaluate(t, y)
        if not np.all(np.isfinite(result)):
            raise SolverError("Якобиан содержит неконечные значения")
        return result
