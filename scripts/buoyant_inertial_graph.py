"""Closed lumped Boussinesq loops with explicit thermal/mechanical exchange.

Gravity density is diagnostic, not a variable mass/EOS density. Buoyancy work
is subtracted from thermal inventories with the same endpoint split used in
momentum/energy bookkeeping. This is a declared reduced closure, not full
Boussinesq thermodynamics or compressible water properties.
"""

import json
import math
from copy import deepcopy
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from core.network import NetworkError, number
from scripts.inertial_graph import InertialGraphError, digest
from scripts.solve_hydraulic_graph import _keys, sha
from scripts.thermal_inertial_graph import LEDGERS as THERMAL_LEDGERS
from scripts.thermal_inertial_graph import ThermalInertialGraph, ThermalInertialSystem
from scripts.thermal_inertial_graph import signature as thermal_signature

BUDGET = Path("validation/ai05_buoyant_graph_budget_v1.json")
LEDGERS = THERMAL_LEDGERS + ("buoyancy_work_j",)


def signature():
    value = thermal_signature()
    value["drivers"]["scripts/buoyant_inertial_graph.py"] = sha(__file__)
    value["budgets"][str(BUDGET)] = sha(BUDGET)
    return value


class BuoyantSystem(ThermalInertialSystem):
    def __init__(self, request):
        _keys(request, ("format", "schema_version", "base", "buoyancy"))
        if (
            request["format"] != "atomic-buoyant-inertial-graph"
            or type(request["schema_version"]) is not int
            or request["schema_version"] != 1
        ):
            raise NetworkError("Expected buoyant inertial schema1")
        b = request["buoyancy"]
        _keys(b, ("expansion_per_k", "reference_temperature_k", "maximum_relative_density_change"))
        self.buoyant_budget = json.loads(BUDGET.read_text())
        self.beta = number(b["expansion_per_k"], "expansion_per_k")
        self.buoyancy_tref = number(
            b["reference_temperature_k"], "buoyancy_reference_temperature", positive=True
        )
        self.density_limit = number(
            b["maximum_relative_density_change"], "relative_density_limit", positive=True
        )
        if (
            self.beta < 0
            or self.density_limit > self.buoyant_budget["maximum_relative_density_change"]
        ):
            raise NetworkError("Invalid Boussinesq expansion or contrast limit")
        # Parent validates initial momentum using dynamic dispatch to our rates.
        # Its bootstrap has six unused ledger zeros; only this construction path
        # adapts that vector to our seventh bookkeeping coordinate.
        self._initializing = True
        super().__init__(deepcopy(request["base"]))
        self._initializing = False
        if len(self.boundary) != 1 or np.any(self.injection != 0) or self.dimension < 1:
            raise NetworkError(
                "Buoyant family requires closed cycles, one pressure gauge and no mass injection"
            )
        height = float(np.ptp(self.elevation))
        if (
            self.gravity * self.beta * height / self.cp
            > self.buoyant_budget["maximum_dissipation_number"]
        ):
            raise NetworkError(
                "Buoyant reduced thermal/mechanical closure exceeds dissipation-number limit"
            )
        self.evaluate(np.zeros(self.dimension + len(self.names) + 7), self.initial_pressures)

    def buoyancy(self, T):
        theta = T - self.buoyancy_tref
        contrast = self.beta * theta
        if (
            not np.all(np.isfinite(T))
            or np.max(abs(contrast)) > self.density_limit
            or np.max(abs(theta)) / self.buoyancy_tref
            > self.buoyant_budget["maximum_relative_temperature_change"]
        ):
            raise InertialGraphError(
                "boussinesq_domain", "Thermal/density anomaly left declared small-contrast domain"
            )
        mean = np.array([0.5 * (theta[a] + theta[b]) for a, b in self._endpoints])
        dz = np.array([self.elevation[b] - self.elevation[a] for a, b in self._endpoints])
        force = self.rho * self.gravity * self.beta * mean * dz
        return force, self.rho * (1 - self.beta * mean)

    def rates(self, flows, boundaries, temperature=None):
        q = np.asarray(flows, dtype=float)
        T = self.T0 if temperature is None else np.asarray(temperature, dtype=float)
        if q.shape != self.L.shape or not np.all(np.isfinite(q)) or np.max(abs(q)) > self.qmax:
            raise InertialGraphError("flow_domain", "Buoyant flow left declared range")
        active = self.pressures(boundaries)
        loss, _ = self.loss(q, T)
        force, _ = self.buoyancy(T)
        potential = np.zeros(len(self.names))
        potential[self.boundary] = [
            active[self.names[i]] + self.rho * self.gravity * self.elevation[i]
            for i in self.boundary
        ]
        if len(self.free):
            potential[self.free] = np.linalg.solve(
                self.gram,
                self.Af @ ((force - self.Ab.T @ potential[self.boundary] - loss) / self.L),
            )
        p = potential - self.rho * self.gravity * self.elevation
        if not np.all(np.isfinite(p)) or np.any(p < self.pmin) or np.any(p > self.pmax):
            raise InertialGraphError("pressure_domain", "Buoyant pressure left declared range")
        qd = self.Z @ (self.Z.T @ ((force - self.A.T @ potential - loss) / self.L))
        mismatch = self.L * qd + self.A.T @ potential + loss - force
        if not np.all(np.isfinite(qd)) or max(abs(mismatch)) > self.budget["momentum_atol_pa"]:
            raise InertialGraphError(
                "momentum_projection", "Buoyant momentum residual exceeded budget"
            )
        ext = -self.A @ q
        return (
            qd,
            p,
            np.array(
                [p @ ext / self.rho, self.gravity * self.elevation @ ext, q @ loss / self.rho]
            ),
        )

    def thermal_terms(self, q, T, p):
        rate, ledgers, loss, multipliers = super().thermal_terms(q, T, p)
        force, _ = self.buoyancy(T)
        work = q * force / self.rho
        # The same signed work appears in kinetic gain and thermal loss.
        for value, (a, b) in zip(work, self._endpoints):
            rate[a] -= 0.5 * value / self.capacity[a]
            rate[b] -= 0.5 * value / self.capacity[b]
        return rate, np.r_[ledgers, sum(work)], loss, multipliers

    def evaluate(self, y, boundaries):
        expected = self.dimension + len(self.names) + 7
        if self._initializing and y.shape == (expected - 1,):
            y = np.r_[y, 0.0]
        if y.shape != (expected,) or not np.all(np.isfinite(y)):
            raise NetworkError("Malformed buoyant coordinates")
        # Ledger values never enter RHS. The inherited kernel validates six of
        # them, while virtual thermal_terms supplies all seven derivatives.
        return super().evaluate(y[:-1], boundaries)

    def contract(self):
        value = super().contract()
        value.update(
            family="closed_buoyant_inertial_graph_v1",
            buoyancy=True,
            density_model="linear gravity-only anomaly; reference density for mass/inertia",
            ledger_names=list(LEDGERS),
            full_eos=False,
            buoyancy_exchange="signed work removed equally from endpoint thermal inventories",
        )
        return value


class BuoyantInertialGraph(ThermalInertialGraph):
    def __init__(self, request, *, max_step_s=1.0):
        self.system = BuoyantSystem(deepcopy(request))
        self.budget = json.loads(BUDGET.read_text())
        self.max_step = number(max_step_s, "max_step_s", positive=True)
        if self.max_step > self.budget["maximum_macro_step_s"]:
            raise NetworkError("Buoyant step exceeds budget")
        self.identity = digest(dict(request=request, max_step_s=self.max_step, method="Radau"))
        self.captured = signature()
        self._time = 0.0
        self._revision = 0
        self._history = []
        self._boundaries = deepcopy(self.system.initial_pressures)
        self._state = np.zeros(self.system.dimension + len(self.system.names) + 7)
        self.snapshot()

    def _sources(self):
        if signature() != self.captured:
            raise InertialGraphError("source_drift", "Buoyant source/budget/runtime drift")

    def _observe(self, state, time):
        s = self.system
        _, point = s.evaluate(state, self._boundaries)
        d = s.dimension
        n = len(s.names)
        force, density = s.buoyancy(point["T"])
        kinetic = float(0.5 * np.dot(s.L * point["q"], point["q"]) / s.rho)
        thermal = float(s.capacity @ state[d : d + n])
        P, G, Q, H, F, B, W = state[-7:]
        errors = np.array(
            [
                kinetic - s.k0 - P - G - W + F + B,
                thermal - Q - H - F + W,
                kinetic - s.k0 + thermal - P - G - Q - H + B,
            ]
        )
        scale = max(1.0, abs(s.k0), abs(kinetic), abs(thermal), float(max(abs(state[-7:]))))
        if (
            not math.isfinite(kinetic)
            or not np.all(np.isfinite(errors))
            or max(abs(errors)) > self.budget["energy_atol_j"] + self.budget["energy_rtol"] * scale
            or min(F, B) < -self.budget["energy_atol_j"]
        ):
            raise InertialGraphError(
                "energy",
                "Buoyant thermal/mechanical/total balance failed",
                errors_j=errors.tolist(),
            )
        mass = s.A @ point["q"]
        momentum = (
            s.L * point["qd"]
            + s.A.T @ (point["p"] + s.rho * s.gravity * s.elevation)
            + point["loss"]
            - force
        )
        if (
            max(abs(mass)) > self.budget["mass_atol_kg_s"]
            or max(abs(momentum)) > self.budget["momentum_atol_pa"]
        ):
            raise InertialGraphError("residual", "Buoyant mass or momentum residual failed")
        return dict(
            time_s=time,
            flows_kg_s=dict(zip(s.edges, map(float, point["q"]))),
            pressures_pa=dict(zip(s.names, map(float, point["p"]))),
            temperatures_k=dict(zip(s.names, map(float, point["T"]))),
            temperature_increments_k=dict(zip(s.names, map(float, state[d : d + n]))),
            resistance_multipliers=dict(zip(s.edges, map(float, point["multipliers"]))),
            gravity_density_kg_m3=dict(zip(s.edges, map(float, density))),
            kinetic_energy_j=kinetic,
            thermal_energy_change_j=thermal,
            mechanical_energy_residual_j=float(errors[0]),
            thermal_energy_residual_j=float(errors[1]),
            total_energy_residual_j=float(errors[2]),
            maximum_mass_residual_kg_s=float(max(abs(mass))),
            maximum_momentum_residual_pa=float(max(abs(momentum))),
            **dict(zip(LEDGERS, map(float, state[-7:]))),
        )

    def residual(self, state, rate):
        s = self.system
        m = len(s.edges)
        n = len(s.names)
        size = m + 2 * n + 7
        y = np.asarray(state, dtype=float)
        dy = np.asarray(rate, dtype=float)
        if (
            y.shape != (size,)
            or dy.shape != (size,)
            or not np.all(np.isfinite(y))
            or not np.all(np.isfinite(dy))
        ):
            raise NetworkError("Malformed buoyant physical residual")
        q, p, T = y[:m], y[m : m + n], y[m + n : m + 2 * n]
        tr, ledgers, loss, _ = s.thermal_terms(q, T, p)
        force, _ = s.buoyancy(T)
        node = s.A @ q
        for i in s.fixed:
            node[i] = p[i] - self._boundaries[s.names[i]]
        momentum = s.L * dy[:m] + s.A.T @ (p + s.rho * s.gravity * s.elevation) + loss - force
        return np.r_[momentum, node, s.capacity * (dy[m + n : m + 2 * n] - tr), dy[-7:] - ledgers]

    def checkpoint(self):
        self._sources()
        payload = dict(
            schema_version=1,
            model="closed-buoyant-radau-v1",
            input_sha256=self.identity,
            signature=deepcopy(self.captured),
            coordinates=self._state.tolist(),
            snapshot=self.snapshot(),
        )
        return dict(payload=payload, sha256=digest(payload))

    def _transaction(self, duration, command=None):
        self._sources()
        duration = number(duration, "duration", positive=True)
        end = self._time + duration
        if (
            duration > self.budget["maximum_duration_s"]
            or not math.isfinite(end)
            or end <= self._time
        ):
            raise NetworkError("Invalid coupled duration/clock")
        candidate = deepcopy(self)
        transition = None
        if command is not None:
            if len(self._history) >= self.budget["maximum_commands"]:
                raise NetworkError("Coupled command budget exhausted")
            pressures, cid = command
            left = self._observe(self._state, self._time)
            candidate._boundaries = candidate.system.pressures(pressures)
            right = candidate._observe(candidate._state, candidate._time)
            transition = dict(time_s=self._time, left=left, right=right)
            candidate._history.append(
                dict(
                    sequence=len(self._history) + 1,
                    command_id=cid,
                    time_s=self._time,
                    commit_revision=self.revision + 1,
                    pressures_pa=deepcopy(candidate._boundaries),
                )
            )
        evaluations = 0

        def rhs(time, y):
            nonlocal evaluations
            evaluations += 1
            if evaluations > self.budget["maximum_rhs_evaluations"]:
                raise InertialGraphError("rhs_budget", "Coupled RHS limit exceeded")
            return candidate.system.evaluate(y, candidate._boundaries)[0]

        d = candidate.system.dimension
        n = len(candidate.system.names)
        solution = solve_ivp(
            rhs,
            (self._time, end),
            candidate._state,
            method="Radau",
            rtol=1e-10,
            atol=np.r_[np.full(d + n, 1e-11), np.full(7, 1e-8)],
            max_step=self.max_step,
        )
        if not solution.success or not np.all(np.isfinite(solution.y)) or solution.t[-1] != end:
            raise InertialGraphError("integration", "Coupled integration failed")
        if len(solution.t) - 1 > self.budget["maximum_internal_steps"]:
            raise InertialGraphError("step_budget", "Coupled step limit exceeded")
        candidate._state = solution.y[:, -1].copy()
        candidate._time = end
        candidate._revision += 1
        result = candidate.snapshot()
        self._sources()
        self._install(candidate)
        return dict(
            **result,
            boundary_transition=transition,
            work=dict(
                rhs=evaluations,
                internal_steps=len(solution.t) - 1,
                maximum_internal_step_s=float(max(np.diff(solution.t))),
            ),
        )

    def restore(self, checkpoint):
        self._sources()
        _keys(checkpoint, ("payload", "sha256"))
        p = checkpoint["payload"]
        if digest(p) != checkpoint["sha256"]:
            raise NetworkError("Inertial checkpoint digest mismatch")
        _keys(
            p, ("schema_version", "model", "input_sha256", "signature", "coordinates", "snapshot")
        )
        if (
            type(p["schema_version"]) is not int
            or p["schema_version"] != 1
            or p["model"] != "closed-buoyant-radau-v1"
            or p["input_sha256"] != self.identity
            or p["signature"] != self.captured
        ):
            raise NetworkError("Inertial checkpoint input/source mismatch")
        snap = p["snapshot"]
        _keys(snap, ("revision", "active_pressures_pa", "history", "state", "contract"))
        rev = snap["revision"]
        history = snap["history"]
        if not isinstance(snap["state"], dict) or "time_s" not in snap["state"]:
            raise NetworkError("Malformed inertial checkpoint observation")
        time = number(snap["state"]["time_s"], "time_s")
        if (
            type(rev) is not int
            or rev < 0
            or time < 0
            or (rev == 0) != (time == 0)
            or not isinstance(history, list)
            or len(history) > self.budget["maximum_commands"]
        ):
            raise NetworkError("Inertial checkpoint clock/history mismatch")
        active = deepcopy(self.system.initial_pressures)
        last_time = 0.0
        last_rev = 0
        ids = set()
        for sequence, record in enumerate(history, 1):
            _keys(record, ("sequence", "command_id", "time_s", "commit_revision", "pressures_pa"))
            cid = record["command_id"]
            when = number(record["time_s"], "command_time")
            commit = record["commit_revision"]
            if (
                type(record["sequence"]) is not int
                or record["sequence"] != sequence
                or not isinstance(cid, str)
                or not 1 <= len(cid) <= 128
                or not cid.isascii()
                or any(not (c.isalnum() or c in "-_.:") for c in cid)
                or cid in ids
                or not last_time <= when < time
                or (sequence > 1 and when <= last_time)
                or ((when == 0) != (commit == 1))
                or type(commit) is not int
                or not last_rev < commit <= rev
            ):
                raise NetworkError("Inertial checkpoint command path mismatch")
            ids.add(cid)
            active = self.system.pressures(record["pressures_pa"])
            last_time, last_rev = when, commit
        values = p["coordinates"]
        if (
            not isinstance(values, list)
            or len(values) != self.system.dimension + len(self.system.names) + 7
        ):
            raise NetworkError("Inertial coordinate layout mismatch")
        values = np.array([number(v, "coordinate") for v in values])
        if time == 0 and np.any(values != 0):
            raise NetworkError("Nonzero initial checkpoint stocks")
        candidate = deepcopy(self)
        candidate._time = time
        candidate._revision = rev
        candidate._state = values
        candidate._boundaries = active
        candidate._history = deepcopy(history)
        fresh = candidate.snapshot()
        if fresh != snap:
            raise NetworkError("Inertial checkpoint observation/active frame mismatch")
        self._sources()
        self._install(candidate)
        return fresh
