"""Обёртка жёсткого интегратора.

Выбран scipy BDF (переменный порядок, адаптивный шаг): открытая реализация
формул NDF/BDF без внешней библиотеки SUNDIALS. Отказ шага не маскируется.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
from scipy.integrate import solve_ivp

from core.status import SolverError

Rhs = Callable[[float, np.ndarray], np.ndarray]


@dataclass
class IntegrationResult:
    t: np.ndarray
    y: np.ndarray
    n_steps: int
    message: str
    success: bool
    dense: object = None
    n_rhs_evaluations: int = 0
    n_jacobian_evaluations: int = 0
    n_lu_decompositions: int = 0
    n_accepted_steps: Optional[int] = None
    events: tuple = ()

    @property
    def final(self) -> np.ndarray:
        return self.y[:, -1]


def integrate(
    rhs: Rhs,
    y0: np.ndarray,
    t_span: Sequence[float],
    rtol: float = 1e-6,
    atol: float = 1e-8,
    max_step: Optional[float] = None,
    t_eval: Optional[np.ndarray] = None,
    events: Optional[list] = None,
    method: str = "BDF",
    dense_output: bool = False,
    first_step: Optional[float] = None,
    jacobian_relative_step: Optional[float] = None,
) -> IntegrationResult:
    """Интегрировать ODE. При несходимости поднимает SolverError."""
    if method not in ("BDF", "Radau", "LSODA"):
        raise SolverError("Метод %s не входит в согласованный набор жёстких решателей" % method)
    y0 = np.asarray(y0, dtype=float)
    kwargs = {
        "fun": rhs,
        "t_span": (float(t_span[0]), float(t_span[1])),
        "y0": y0,
        "method": method,
        "rtol": rtol,
        "atol": atol,
        "dense_output": dense_output,
        "t_eval": t_eval,
        "events": events,
    }
    if max_step is not None:
        kwargs["max_step"] = max_step
    if first_step is not None:
        if isinstance(first_step, bool) or not np.isfinite(first_step) or first_step <= 0.0:
            raise SolverError("Первый шаг должен быть положительным конечным числом")
        kwargs["first_step"] = min(float(first_step), abs(float(t_span[1]) - float(t_span[0])))
    jacobian = None
    if jacobian_relative_step is not None:
        from core.jacobian import CentralJacobian

        jacobian = CentralJacobian(rhs, jacobian_relative_step)
        kwargs["jac"] = jacobian
    sol = solve_ivp(**kwargs)
    if not sol.success:
        raise SolverError(sol.message)
    return IntegrationResult(
        t=sol.t,
        y=sol.y,
        n_steps=int(sol.nfev),  # legacy: исторически это nfev, не число шагов
        message=sol.message,
        success=True,
        dense=sol.sol if dense_output else None,
        n_rhs_evaluations=int(sol.nfev) + (0 if jacobian is None else jacobian.rhs_evaluations),
        n_jacobian_evaluations=int(getattr(sol, "njev", 0)),
        n_lu_decompositions=int(getattr(sol, "nlu", 0)),
        n_accepted_steps=int(sol.t.size - 1) if t_eval is None else None,
    )


def per_variable_atol(scales, relative_tolerance: float) -> np.ndarray:
    """Абсолютный допуск каждого неизвестного: atol_i = ε · масштаб_i."""
    scales = np.asarray(scales, dtype=float)
    if relative_tolerance <= 0.0 or scales.size == 0 or np.any(scales <= 0.0):
        raise ValueError("Масштабы допуска положительны, относительный допуск тоже")
    if not np.all(np.isfinite(scales)):
        raise ValueError("Масштаб допуска должен быть конечным")
    return relative_tolerance * scales


def observe_cfl(speed_m_s: float, dx_m: float, dt_s: float) -> float:
    if dx_m <= 0.0:
        raise ValueError("Шаг сетки должен быть положительным")
    return abs(speed_m_s) * dt_s / dx_m


def substep_count(cfl: float, limit: float = 1.0) -> int:
    """Число подшагов явной схемы. limit — заявленный предел Куранта, не подгонка физики."""
    if limit <= 0.0:
        raise ValueError("Предел Куранта должен быть положительным")
    if cfl < 0.0:
        raise ValueError("Число Куранта берётся по модулю скорости до вызова")
    return max(1, int(np.ceil(cfl / limit - 1e-15)))
