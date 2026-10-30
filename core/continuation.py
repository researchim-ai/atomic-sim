"""Неявные интеграторы с сохранением истории между точками вывода.

Используется публичный step/dense_output интерфейс scipy.integrate BDF/Radau.
История уничтожается при изменении уравнений/команд или непринятом шаге.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import BDF, Radau

from core.solvers import IntegrationResult
from core.status import SolverError


class DenseInterval:
    def __init__(self, times, pieces, mesh_times=None):
        self.breaks = np.asarray(times, dtype=float)
        self.ts = self.breaks if mesh_times is None else np.asarray(mesh_times, dtype=float)
        self.pieces = pieces

    def __call__(self, time):
        values = np.asarray(time, dtype=float)
        flat = values.reshape(-1)
        if np.any(flat < self.breaks[0]) or np.any(flat > self.breaks[-1]):
            raise SolverError("Запрос плотного вывода вне принятого интервала")
        indices = np.clip(
            np.searchsorted(self.breaks, flat, side="right") - 1, 0, len(self.pieces) - 1
        )
        columns = [
            np.asarray(self.pieces[int(index)](float(t))).reshape(-1)
            for index, t in zip(indices, flat)
        ]
        result = np.column_stack(columns)
        return (
            result[:, 0] if values.ndim == 0 else result.reshape((result.shape[0],) + values.shape)
        )


class ImplicitContinuation:
    def __init__(
        self,
        rhs,
        y0,
        t0: float,
        horizon: float,
        *,
        rtol: float,
        atol: float,
        max_step: float,
        first_step: float | None = None,
        jacobian_relative_step: float | None = None,
        method: str = "BDF",
        initial_target: float | None = None,
    ):
        if not np.isfinite(horizon) or horizon <= t0 or not np.isfinite(max_step) or max_step <= 0:
            raise SolverError("Недопустимый горизонт или шаг continuation")
        self.max_step = max_step
        self.horizon = float(horizon)
        initial_bound = horizon if initial_target is None else initial_target
        if not np.isfinite(initial_bound) or not t0 < initial_bound <= horizon:
            raise SolverError("Недопустимая первая граница continuation")
        if method not in ("BDF", "Radau"):
            raise SolverError("Continuation поддерживает BDF и Radau")
        engine = BDF if method == "BDF" else Radau
        self.method = method
        self.jacobian = None
        if jacobian_relative_step is not None:
            from core.jacobian import CentralJacobian

            self.jacobian = CentralJacobian(rhs, jacobian_relative_step)
        self.solver = engine(
            rhs,
            t0,
            np.asarray(y0, dtype=float),
            initial_bound,
            rtol=rtol,
            atol=atol,
            max_step=max_step,
            jac=self.jacobian,
            first_step=None
            if first_step is None
            else min(first_step, max_step, initial_bound - t0),
        )
        self._reported = (0, 0, 0)
        self._reported_nfev = 0
        self._failed = False

    @property
    def time_s(self):
        return self.solver.t

    @property
    def state(self):
        return self.solver.y

    def advance_to(self, target: float) -> IntegrationResult:
        solver = self.solver
        if self._failed or not np.isfinite(target) or not solver.t < target <= self.horizon:
            raise SolverError("Continuation не может принять этот интервал")
        times = [solver.t]
        states = [solver.y.copy()]
        pieces = []
        try:
            # Границу должен обрабатывать сам solver. Ограничение max_step
            # остатком до точки вывода оставляло иногда 1 ULP < min_step.
            solver.t_bound = target
            if solver.status == "finished":
                solver.status = "running"
            solver.max_step = self.max_step
            while solver.t < target:
                message = solver.step()
                if solver.status == "failed":
                    raise SolverError(message or "%s step failed" % self.method)
                if solver.t <= times[-1] or solver.t > target:
                    raise SolverError("Интегратор не продвинулся к заданной точке вывода")
                times.append(solver.t)
                states.append(solver.y.copy())
                pieces.append(solver.dense_output())
        except BaseException:
            self._failed = True
            raise
        jac_calls = 0 if self.jacobian is None else self.jacobian.rhs_evaluations
        counts = (solver.nfev + jac_calls, solver.njev, solver.nlu)
        delta = tuple(int(new - old) for new, old in zip(counts, self._reported))
        legacy_delta = solver.nfev - self._reported_nfev
        self._reported_nfev = solver.nfev
        self._reported = counts
        return IntegrationResult(
            t=np.asarray(times),
            y=np.column_stack(states),
            n_steps=int(legacy_delta),
            message="%s continuation accepted" % self.method,
            success=True,
            dense=DenseInterval(times, pieces),
            n_rhs_evaluations=delta[0],
            n_jacobian_evaluations=delta[1],
            n_lu_decompositions=delta[2],
            n_accepted_steps=len(pieces),
        )


# Совместимость SDK и ранее написанных проверок.
BDFContinuation = ImplicitContinuation
