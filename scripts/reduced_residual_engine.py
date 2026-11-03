"""Transactional reduced Radau backend with pure algebraic Newton projection.

Constant diagonal mass and matching algebraic variable/row partitions only.
Radau is restarted at macro boundaries; native collocation history is not
serialized. Interval admission is delegated to the registered physical system.
"""

import json
import math
from copy import deepcopy

import numpy as np
from scipy.integrate import solve_ivp

from core.network import NetworkError, number
from scripts.nonlinear_thermal_graph import digest
from scripts.registered_graph_system import BUDGET, create_system, signature
from scripts.solve_hydraulic_graph import _keys


class ReducedResidualEngine:
    def __init__(self, request, *, max_macro_step_s=2.0):
        self.system = create_system(deepcopy(request))
        self.request = deepcopy(request)
        self.budget = json.loads(BUDGET.read_text())
        self.max_step = number(max_macro_step_s, "max_macro_step_s", positive=True)
        if self.max_step > self.budget["maximum_macro_step_s"]:
            raise NetworkError("Reduced macro step exceeds budget")
        self.identity = digest(
            dict(request=request, max_macro_step_s=self.max_step, method="Radau")
        )
        self.captured = signature()
        self._increments = np.zeros(len(self.system.differential))
        self._full = self.system.origin.copy()
        self._time = 0.0
        self._validate_layout()
        self._full = self._project(self._increments, self._full)

    def _validate_layout(self):
        s = self.system
        size = len(s.origin)
        if (
            s.mass.shape != (size,)
            or not np.all(np.isfinite(s.mass))
            or np.any(s.mass < 0)
            or len(s.differential) + len(s.algebraic) != size
            or s.scales.shape != (len(s.differential),)
            or np.any(s.scales <= 0)
            or not np.all(np.isfinite(s.scales))
        ):
            raise NetworkError("Unsupported reduced residual layout")

    def _sources(self):
        if signature() != self.captured:
            raise NetworkError("Reduced engine source/budget drift")

    def _project(self, increments, seed):
        s = self.system
        full = seed.copy()
        d = s.differential
        a = s.algebraic
        full[d] = s.origin[d] + increments * s.scales
        tolerances = (
            np.array([s.contract["rows"][i]["tolerance"] for i in a])
            * self.budget["algebraic_tolerance_factor"]
        )
        columns = np.array([s.contract["variables"][i]["nominal"] for i in a])
        rows = np.array([s.contract["rows"][i]["nominal"] for i in a])
        for _ in range(self.budget["maximum_newton_iterations"]):
            residual = s.residual(full, np.zeros_like(full))[a]
            norm = float(np.max(abs(residual) / tolerances))
            if norm <= 1:
                return full
            jac = s.algebraic_jacobian(full)
            if jac.shape != (len(a), len(a)) or not np.all(np.isfinite(jac)):
                raise NetworkError("Invalid algebraic Jacobian")
            try:
                delta = columns * np.linalg.solve(jac * columns / rows[:, None], -residual / rows)
            except np.linalg.LinAlgError as error:
                raise NetworkError("Singular reduced algebraic block") from error
            accepted = False
            factor = 1.0
            for _ in range(self.budget["maximum_backtracks"]):
                trial = full.copy()
                trial[a] += factor * delta
                try:
                    r = s.residual(trial, np.zeros_like(trial))[a]
                    candidate = float(np.max(abs(r) / tolerances))
                    if candidate <= 1 or candidate < (1.0 - 1e-4 * factor) * norm:
                        full = trial
                        accepted = True
                        break
                except NetworkError:
                    pass
                factor *= 0.5
            if not accepted:
                raise NetworkError("Reduced algebraic line search exhausted")
        raise NetworkError("Reduced algebraic Newton budget exhausted")

    def _snapshot(self, increments, full, time):
        s = self.system
        return dict(
            time_s=time,
            increments=increments.tolist(),
            full_state=full.tolist(),
            state=s.observe(full, increments),
            method="Radau",
            macro_restart_policy="fresh_collocation_history_each_macro_segment",
        )

    def snapshot(self):
        return self._snapshot(self._increments, self._full, self._time)

    def advance(self, duration_s):
        self._sources()
        duration = number(duration_s, "duration_s", positive=True)
        target = self._time + duration
        if (
            duration > self.budget["maximum_duration_s"]
            or not math.isfinite(target)
            or target <= self._time
        ):
            raise NetworkError("Invalid reduced duration or unrepresentable clock")
        z = self._increments.copy()
        full = self._full.copy()
        elapsed = 0.0
        attempts = 0
        segments = []
        rejected = 0
        rhs_evaluations = 0
        s = self.system
        d = s.differential
        atol = (
            np.array(
                [
                    self.budget["temperature_atol_k"]
                    if unit == "temperature"
                    else self.budget["energy_atol_j"]
                    for unit in s.atol_units
                ]
            )
            / s.scales
        )
        while elapsed < duration:
            step, box = s.admit(full, min(self.max_step, duration - elapsed))
            while True:
                attempts += 1
                if attempts > self.budget["maximum_attempts"]:
                    raise NetworkError("Reduced integration attempt budget exhausted")
                if step < self.budget["minimum_macro_step_s"] and step < duration - elapsed:
                    raise NetworkError("Reduced minimum macro step exhausted")
                seed = full.copy()

                def rhs(time, value):
                    nonlocal rhs_evaluations
                    rhs_evaluations += 1
                    if rhs_evaluations > self.budget["maximum_rhs_evaluations_per_advance"]:
                        raise NetworkError("Reduced RHS evaluation budget exhausted")
                    candidate = self._project(value, seed)
                    s.require_box(candidate, box)
                    residual = s.residual(candidate, np.zeros_like(candidate))
                    return -residual[d] / s.mass[d] / s.scales

                try:
                    solved = solve_ivp(
                        rhs,
                        (0.0, step),
                        z,
                        method="Radau",
                        rtol=self.budget["rtol"],
                        atol=atol,
                        max_step=step,
                    )
                    if (
                        not solved.success
                        or solved.t[-1] != step
                        or not np.all(np.isfinite(solved.y))
                    ):
                        raise NetworkError("Reduced Radau failed")
                    candidate_z = solved.y[:, -1]
                    candidate_full = self._project(candidate_z, seed)
                    s.require_box(candidate_full, box)
                    if abs(s.energy_error(candidate_z)) > self.budget["maximum_energy_error_j"]:
                        raise NetworkError("Reduced energy budget exceeded")
                    break
                except NetworkError:
                    rejected += 1
                    step *= 0.5
                    if step < self.budget["minimum_macro_step_s"]:
                        raise NetworkError("Reduced segment failed; whole advance rolled back")
            delta = np.diff(solved.t)
            segments.append(
                dict(
                    duration_s=step,
                    accepted_internal_steps=len(delta),
                    minimum_internal_step_s=float(min(delta)),
                    maximum_internal_step_s=float(max(delta)),
                )
            )
            z = candidate_z
            full = candidate_full
            previous = elapsed
            elapsed = min(duration, elapsed + step)
            if elapsed <= previous:
                raise NetworkError("Reduced macro clock cannot progress")
        final = self._snapshot(z, full, target)
        self._sources()
        self._increments, self._full, self._time = z, full, target
        return dict(
            **final,
            segments=segments,
            rejected_macro_steps=rejected,
            rhs_evaluations=rhs_evaluations,
        )

    def checkpoint(self):
        payload = dict(
            schema_version=1,
            model="registered-reduced-radau-v1",
            input_sha256=self.identity,
            signature=deepcopy(self.captured),
            state=self.snapshot(),
        )
        return dict(payload=payload, sha256=digest(payload))

    def restore(self, checkpoint):
        self._sources()
        _keys(checkpoint, ("payload", "sha256"))
        p = checkpoint["payload"]
        if digest(p) != checkpoint["sha256"]:
            raise NetworkError("Reduced checkpoint digest mismatch")
        _keys(p, ("schema_version", "model", "input_sha256", "signature", "state"))
        if (
            type(p["schema_version"]) is not int
            or p["schema_version"] != 1
            or p["model"] != "registered-reduced-radau-v1"
            or p["input_sha256"] != self.identity
            or p["signature"] != self.captured
        ):
            raise NetworkError("Reduced checkpoint input/model/source mismatch")
        state = p["state"]
        _keys(
            state, ("time_s", "increments", "full_state", "state", "method", "macro_restart_policy")
        )
        time = number(state["time_s"], "time_s")
        if time < 0:
            raise NetworkError("Negative reduced checkpoint time")
        z = np.array([number(v, "increment") for v in state["increments"]])
        full = self.system.adapter._vector(state["full_state"], "state")
        if z.shape != self._increments.shape:
            raise NetworkError("Reduced checkpoint differential shape mismatch")
        projected = self._project(z, full)
        if (
            projected.tolist() != full.tolist()
            or abs(self.system.energy_error(z)) > self.budget["maximum_energy_error_j"]
        ):
            raise NetworkError("Reduced checkpoint algebraic/energy mismatch")
        fresh = self._snapshot(z, projected, time)
        if fresh != state:
            raise NetworkError("Reduced checkpoint observation mismatch")
        self._sources()
        self._increments, self._full, self._time = z, projected, time
        return fresh

    def import_legacy_checkpoint(self, checkpoint):
        self._sources()
        z, full, time = self.system.from_legacy(deepcopy(checkpoint))
        projected = self._project(z, full)
        if abs(self.system.energy_error(z)) > self.budget["maximum_energy_error_j"]:
            raise NetworkError("Handoff energy budget exceeded")
        result = self._snapshot(z, projected, time)
        self._sources()
        self._increments, self._full, self._time = z, projected, time
        return result

    def export_legacy_checkpoint(self):
        self._sources()
        checkpoint = self.system.to_legacy(self._increments, self._time)
        result = dict(
            checkpoint=checkpoint,
            engine_signature=deepcopy(self.captured),
            native_dp_proposal_reset=True,
            collocation_history_transferred=False,
        )
        self._sources()
        return result
