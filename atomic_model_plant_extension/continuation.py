"""Public SciPy step/dense API with directional roots and retained history."""

import numpy as np
from scipy.integrate import BDF, Radau
from scipy.optimize import brentq

from core.continuation import DenseInterval
from core.jacobian import CentralJacobian
from core.solvers import IntegrationResult
from core.status import SolverError


class HybridContinuation:
    def __init__(self, fun, state, start, end, *, rtol, atol, first, max_step, jacstep, method):
        if method not in ("Radau", "BDF"):
            raise SolverError("Hybrid continuation requires Radau/BDF")
        self.jac = CentralJacobian(fun, jacstep or 1e-6)
        self.solver = (Radau if method == "Radau" else BDF)(
            fun,
            start,
            state,
            end,
            rtol=rtol,
            atol=atol,
            first_step=min(first or max_step, end - start),
            max_step=max_step,
            jac=self.jac,
        )
        self.reported = (0, 0, 0)
        self.invalid = False

    def advance_until(self, target, events):
        s = self.solver
        if self.invalid or target <= s.t:
            raise SolverError("Invalid hybrid continuation clock")
        s.t_bound = target
        if s.status == "finished":
            s.status = "running"
        times = [s.t]
        states = [s.y.copy()]
        pieces = []
        hit = []
        oldg = [float(e(s.t, s.y)) for e in events]
        for step in range(10000):
            if s.t == target:
                break
            start = float(s.t)
            message = s.step()
            if s.status == "failed" or not start < s.t <= target:
                raise SolverError(message or "Hybrid step failed")
            if s.nfev + self.jac.rhs_evaluations > 500000:
                raise SolverError("Hybrid RHS budget exhausted")
            dense = s.dense_output()
            newg = [float(e(s.t, s.y)) for e in events]
            roots = []
            for i, (e, a, b) in enumerate(zip(events, oldg, newg)):
                if not np.isfinite([a, b]).all():
                    raise SolverError("Nonfinite hybrid guard")
                cross = (
                    (a < 0 <= b or a == 0 < b) if e.direction > 0 else (a > 0 >= b or a == 0 > b)
                )
                if cross:
                    root = (
                        start
                        if a == 0
                        else brentq(
                            lambda t: float(e(t, dense(t))),
                            start,
                            s.t,
                            xtol=2e-12,
                            rtol=8 * np.finfo(float).eps,
                        )
                    )
                    roots.append((root, i))
            if roots:
                end = min(r[0] for r in roots)
                hit = [i for t, i in roots if abs(t - end) <= 1e-10]
                endpoint = dense(end)
                if end > times[-1]:
                    times.append(end)
                    states.append(endpoint.copy())
                    pieces.append(dense)
                self.invalid = True
                break
            times.append(float(s.t))
            states.append(s.y.copy())
            pieces.append(dense)
            oldg = newg
        else:
            raise SolverError("Hybrid internal step budget exhausted")
        counts = (s.nfev + self.jac.rhs_evaluations, s.njev, s.nlu)
        delta = tuple(int(a - b) for a, b in zip(counts, self.reported))
        self.reported = counts
        # For an immediate guard the caller changes the active constraint with
        # unchanged stocks before starting a new nonempty integration segment.
        output = DenseInterval(times, pieces) if pieces else None
        result = IntegrationResult(
            t=np.array(times),
            y=np.column_stack(states),
            n_steps=delta[0],
            message="Hybrid retained-history interval",
            success=True,
            dense=output,
            n_rhs_evaluations=delta[0],
            n_jacobian_evaluations=delta[1],
            n_lu_decompositions=delta[2],
            n_accepted_steps=len(pieces),
        )
        return result, hit
