"""Explicit inertial pipe registry and constraint-reduced hydraulic Radau backend.

All pipes have strictly positive inertance. Junction mass constraints are
eliminated in a source-bound null basis, not labelled a general index-1 DAE.
Dissipated mechanical energy is exported to an ideal heat bath, not silently
added to an unmodelled fluid temperature. No pump/shaft is synthesized.
"""

import hashlib
import json
import math
import platform
from copy import deepcopy
from pathlib import Path
from types import MappingProxyType

import numpy as np
import scipy
from scipy.integrate import solve_ivp
from scipy.linalg import null_space

from core.network import NetworkError, identifier, number
from scripts.solve_hydraulic_graph import _keys, sha
from sdk.experiment import _source_fingerprint

BUDGET = Path("validation/ai05_inertial_graph_budget_v1.json")
REGISTRY = MappingProxyType(
    {
        "pressure_boundary_v1": dict(ports="hydraulic", storage=False),
        "junction_v1": dict(ports="hydraulic", storage=False),
        "inertial_pipe_v1": dict(ports="hydraulic", storage="kinetic", inertance_unit="Pa s2/kg"),
    }
)


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def signature():
    return dict(
        source_sha256=_source_fingerprint(),
        drivers={
            p: sha(p)
            for p in (
                "scripts/inertial_graph.py",
                "scripts/solve_hydraulic_graph.py",
            )
        },
        budgets={str(BUDGET): sha(BUDGET)},
        python=platform.python_version(),
        numpy=np.__version__,
        scipy=scipy.__version__,
    )


class InertialGraphError(NetworkError):
    def __init__(self, criterion, message, **details):
        super().__init__(message)
        self.report = dict(accepted=False, criterion=criterion, message=message, details=details)


class InertialPipeSystem:
    """Pure physical residual with q positive from->to and injection into nodes."""

    def __init__(self, spec):
        _keys(
            spec,
            (
                "format",
                "schema_version",
                "density_kg_m3",
                "gravity_m_s2",
                "nodes",
                "edges",
                "initial_flows_kg_s",
                "min_pressure_pa",
                "max_pressure_pa",
                "max_abs_flow_kg_s",
            ),
        )
        if (
            spec["format"] != "atomic-inertial-pipe-graph"
            or type(spec["schema_version"]) is not int
            or spec["schema_version"] != 1
        ):
            raise NetworkError("Expected inertial pipe graph schema1")
        self.budget = json.loads(BUDGET.read_text())
        self.spec = deepcopy(spec)
        self.rho = number(spec["density_kg_m3"], "density", positive=True)
        self.gravity = number(spec["gravity_m_s2"], "gravity")
        self.pmin = number(spec["min_pressure_pa"], "min_pressure", positive=True)
        self.pmax = number(spec["max_pressure_pa"], "max_pressure", positive=True)
        self.qmax = number(spec["max_abs_flow_kg_s"], "max_abs_flow", positive=True)
        if self.gravity < 0 or self.pmin >= self.pmax:
            raise NetworkError("Invalid pressure/gravity bounds")
        nodes, edges = spec["nodes"], spec["edges"]
        if (
            not isinstance(nodes, list)
            or not 2 <= len(nodes) <= self.budget["maximum_nodes"]
            or not isinstance(edges, list)
            or not 1 <= len(edges) <= self.budget["maximum_edges"]
        ):
            raise NetworkError("Inertial topology resource limit")
        names = []
        elevations = []
        self.fixed = {}
        injections = []
        for node in nodes:
            _keys(node, ("id", "kind", "elevation_m"), ("pressure_pa", "injection_kg_s"))
            name = identifier(node["id"])
            if name in names:
                raise NetworkError("Duplicate node")
            index = len(names)
            names.append(name)
            elevations.append(number(node["elevation_m"], "elevation_m"))
            if node["kind"] == "pressure_boundary_v1":
                _keys(node, ("id", "kind", "elevation_m", "pressure_pa"))
                self.fixed[index] = number(node["pressure_pa"], "pressure_pa", positive=True)
                injections.append(0.0)
            elif node["kind"] == "junction_v1":
                _keys(node, ("id", "kind", "elevation_m", "injection_kg_s"))
                injections.append(number(node["injection_kg_s"], "injection"))
            else:
                raise NetworkError("Unregistered hydraulic node component")
        if not self.fixed:
            raise NetworkError("Explicit pressure boundary required")
        self.names = tuple(names)
        self.elevation = np.array(elevations)
        self.injection = np.array(injections)
        self.boundary = np.array(list(self.fixed), dtype=int)
        self.free = np.array([i for i in range(len(names)) if i not in self.fixed], dtype=int)
        self.A = np.zeros((len(names), len(edges)))
        self.edges = []
        inertance = []
        linear = []
        quadratic = []
        adjacent = {name: set() for name in names}
        for j, edge in enumerate(edges):
            _keys(
                edge,
                (
                    "id",
                    "kind",
                    "from",
                    "to",
                    "inertance_pa_s2_kg",
                    "linear_pa_s_kg",
                    "quadratic_pa_s2_kg2",
                ),
            )
            eid = identifier(edge["id"])
            a = edge["from"]
            b = edge["to"]
            if (
                edge["kind"] != "inertial_pipe_v1"
                or eid in self.edges
                or a not in names
                or b not in names
                or a == b
            ):
                raise NetworkError("Unregistered, duplicate or invalid pipe")
            self.edges.append(eid)
            self.A[names.index(a), j] = -1
            self.A[names.index(b), j] = 1
            adjacent[a].add(b)
            adjacent[b].add(a)
            inertance.append(number(edge["inertance_pa_s2_kg"], "inertance", positive=True))
            linear.append(number(edge["linear_pa_s_kg"], "linear resistance"))
            quadratic.append(number(edge["quadratic_pa_s2_kg2"], "quadratic resistance"))
        reachable = {names[0]}
        while True:
            extended = reachable | {b for a in reachable for b in adjacent[a]}
            if extended == reachable:
                break
            reachable = extended
        if reachable != set(names) or any(r < 0 for r in linear + quadratic):
            raise NetworkError("Disconnected graph or negative resistance")
        self.L = np.array(inertance)
        self.R = np.array(linear)
        self.Q = np.array(quadratic)
        self.Af = self.A[self.free]
        self.Ab = self.A[self.boundary]
        self.gram = (self.Af / self.L) @ self.Af.T
        if len(self.free) and (
            not np.all(np.isfinite(self.gram))
            or np.linalg.cond(self.gram) > self.budget["maximum_projection_condition"]
        ):
            raise NetworkError("Singular or ill-conditioned pressure projection")
        self.Z = np.ascontiguousarray(null_space(self.Af) if len(self.free) else np.eye(len(edges)))
        if np.linalg.norm(self.Af @ self.Z, ord="fro") > 1e-12:
            raise NetworkError("Constraint basis failed mass nullspace check")
        _keys(spec["initial_flows_kg_s"], tuple(self.edges))
        self.q0 = np.array(
            [number(spec["initial_flows_kg_s"][e], "initial_flow") for e in self.edges]
        )
        if (
            np.max(np.abs(self.Af @ self.q0 + self.injection[self.free]), initial=0)
            > self.budget["mass_atol_kg_s"]
        ):
            raise InertialGraphError(
                "initial_mass", "Supplied initial flows violate junction mass constraints"
            )
        self.k0 = float(0.5 * np.dot(self.L * self.q0, self.q0) / self.rho)
        self.dimension = self.Z.shape[1]
        self.initial_pressures = {names[i]: v for i, v in self.fixed.items()}
        self.basis_sha256 = digest(dict(nodes=names, edges=self.edges, basis=self.Z.tolist()))
        self.rates(self.q0, self.initial_pressures)

    def pressures(self, values):
        _keys(values, tuple(self.initial_pressures))
        normalized = {k: number(v, "boundary_pressure", positive=True) for k, v in values.items()}
        if any(not self.pmin <= v <= self.pmax for v in normalized.values()):
            raise InertialGraphError("pressure_domain", "Boundary pressure outside declared range")
        return normalized

    def rates(self, flows, boundaries):
        q = np.asarray(flows, dtype=float)
        if q.shape != self.L.shape or not np.all(np.isfinite(q)) or np.max(np.abs(q)) > self.qmax:
            raise InertialGraphError("flow_domain", "Flow left declared domain")
        boundaries = self.pressures(boundaries)
        potential = np.zeros(len(self.names))
        potential[self.boundary] = [
            boundaries[self.names[i]] + self.rho * self.gravity * self.elevation[i]
            for i in self.boundary
        ]
        loss = self.R * q + self.Q * q * np.abs(q)
        if len(self.free):
            potential[self.free] = np.linalg.solve(
                self.gram, -self.Af @ ((self.Ab.T @ potential[self.boundary] + loss) / self.L)
            )
        pressure = potential - self.rho * self.gravity * self.elevation
        if (
            not np.all(np.isfinite(pressure))
            or np.any(pressure < self.pmin)
            or np.any(pressure > self.pmax)
        ):
            raise InertialGraphError(
                "pressure_domain", "Projected node pressure left declared domain"
            )
        qdot = (-self.A.T @ potential - loss) / self.L
        qdot = self.Z @ (self.Z.T @ qdot)
        if (
            np.max(np.abs(self.L * qdot + self.A.T @ potential + loss), initial=0)
            > self.budget["momentum_atol_pa"]
        ):
            raise InertialGraphError(
                "momentum_projection", "Projected momentum residual exceeded budget"
            )
        external = -self.A @ q
        power = np.array(
            [
                float(pressure @ external / self.rho),
                float(self.gravity * self.elevation @ external),
                float(q @ loss / self.rho),
            ]
        )
        if not np.all(np.isfinite(qdot)) or not np.all(np.isfinite(power)):
            raise NetworkError("Nonfinite inertial rate")
        return qdot, pressure, power

    def residual(self, state, rate, *, boundaries=None):
        size = len(self.edges) + len(self.names) + 3
        y = np.asarray(state, dtype=float)
        dy = np.asarray(rate, dtype=float)
        if (
            y.shape != (size,)
            or dy.shape != (size,)
            or not np.all(np.isfinite(y))
            or not np.all(np.isfinite(dy))
        ):
            raise NetworkError("Malformed inertial residual vectors")
        m = len(self.edges)
        q = y[:m]
        p = y[m:-3]
        loss = self.R * q + self.Q * q * np.abs(q)
        momentum = (
            self.L * dy[:m] + self.A.T @ (p + self.rho * self.gravity * self.elevation) + loss
        )
        mass = self.A @ q + self.injection
        active = self.pressures(self.initial_pressures if boundaries is None else boundaries)
        for i in self.fixed:
            mass[i] = p[i] - active[self.names[i]]
        external = -self.A @ q
        energy = dy[-3:] - [
            p @ external / self.rho,
            self.gravity * self.elevation @ external,
            q @ loss / self.rho,
        ]
        return np.r_[momentum, mass, energy]

    def contract(self):
        return dict(
            family="inertial_pipe_graph_v1",
            node_ids=list(self.names),
            edge_ids=list(self.edges),
            differential_coordinates=self.dimension,
            pressure_boundaries=list(self.initial_pressures),
            mass_flow_unit="kg/s",
            momentum_residual_unit="Pa",
            junction_residual_unit="kg/s",
            work_unit="J",
            basis_sha256=self.basis_sha256,
            supported_components=list(REGISTRY),
            general_dae=False,
            variable_density=False,
            thermal_coupling=False,
        )


class InertialGraph:
    def __init__(self, spec, *, max_step_s=1.0):
        self.system = InertialPipeSystem(deepcopy(spec))
        self.budget = json.loads(BUDGET.read_text())
        self.max_step = number(max_step_s, "max_step_s", positive=True)
        if self.max_step > self.budget["maximum_macro_step_s"]:
            raise NetworkError("Inertial step exceeds budget")
        self.identity = digest(dict(spec=spec, max_step_s=self.max_step, method="Radau"))
        self.captured = signature()
        self._time = 0.0
        self._revision = 0
        self._history = []
        self._boundaries = deepcopy(self.system.initial_pressures)
        self._state = np.zeros(self.system.dimension + 3)
        self.snapshot()

    def _sources(self):
        if signature() != self.captured:
            raise InertialGraphError("source_drift", "Inertial sources/budget/runtime changed")

    @property
    def revision(self):
        return self._revision

    def _observe(self, state, time):
        s = self.system
        q = s.q0 + s.Z @ state[:-3]
        qd, p, power = s.rates(q, self._boundaries)
        mass = s.Af @ q + s.injection[s.free]
        momentum = (
            s.L * qd + s.A.T @ (p + s.rho * s.gravity * s.elevation) + s.R * q + s.Q * q * np.abs(q)
        )
        kinetic = float(0.5 * np.dot(s.L * q, q) / s.rho)
        error = kinetic - s.k0 - state[-3] - state[-2] + state[-1]
        if not math.isfinite(kinetic) or not math.isfinite(error):
            raise InertialGraphError("energy", "Nonfinite kinetic energy ledger")
        scale = max(1.0, abs(s.k0), abs(kinetic), float(np.max(np.abs(state[-3:]))))
        if (
            np.max(np.abs(mass), initial=0) > self.budget["mass_atol_kg_s"]
            or np.max(np.abs(momentum), initial=0) > self.budget["momentum_atol_pa"]
        ):
            raise InertialGraphError(
                "residual", "Physical mass or momentum residual exceeded budget"
            )
        if (
            abs(error) > self.budget["energy_atol_j"] + self.budget["energy_rtol"] * scale
            or state[-1] < -self.budget["energy_atol_j"]
        ):
            raise InertialGraphError(
                "energy", "Kinetic/work/heat ledger exceeded budget", error_j=float(error)
            )
        return dict(
            time_s=time,
            flows_kg_s=dict(zip(s.edges, map(float, q))),
            pressures_pa=dict(zip(s.names, map(float, p))),
            flow_rates_kg_s2=dict(zip(s.edges, map(float, qd))),
            kinetic_energy_j=kinetic,
            pressure_work_j=float(state[-3]),
            gravity_work_j=float(state[-2]),
            exported_heat_j=float(state[-1]),
            energy_residual_j=float(error),
            maximum_mass_residual_kg_s=float(np.max(np.abs(mass), initial=0)),
            maximum_momentum_residual_pa=float(np.max(np.abs(momentum), initial=0)),
            pressure_power_w=float(power[0]),
            gravity_power_w=float(power[1]),
            dissipation_w=float(power[2]),
        )

    def snapshot(self):
        return dict(
            revision=self._revision,
            active_pressures_pa=deepcopy(self._boundaries),
            history=deepcopy(self._history),
            state=self._observe(self._state, self._time),
            contract=self.system.contract(),
        )

    def residual(self, state, rate):
        return self.system.residual(state, rate, boundaries=self._boundaries)

    def _transaction(self, duration, command=None):
        self._sources()
        duration = number(duration, "duration_s", positive=True)
        end = self._time + duration
        if (
            duration > self.budget["maximum_duration_s"]
            or not math.isfinite(end)
            or end <= self._time
        ):
            raise NetworkError("Invalid inertial duration/clock")
        candidate = deepcopy(self)
        transition = None
        if command is not None:
            if len(candidate._history) >= self.budget["maximum_commands"]:
                raise NetworkError("Command history exhausted")
            pressures, cid = command
            candidate._boundaries = candidate.system.pressures(pressures)
            transition = dict(
                time_s=self._time,
                left=self._observe(self._state, self._time),
                right=candidate._observe(candidate._state, candidate._time),
            )
            candidate._history.append(
                dict(
                    sequence=len(candidate._history) + 1,
                    command_id=cid,
                    time_s=self._time,
                    commit_revision=self._revision + 1,
                    pressures_pa=deepcopy(candidate._boundaries),
                )
            )
        evaluations = 0

        def rhs(time, y):
            nonlocal evaluations
            evaluations += 1
            if evaluations > candidate.budget["maximum_rhs_evaluations"]:
                raise InertialGraphError("rhs_budget", "Inertial RHS budget exhausted")
            s = candidate.system
            q = s.q0 + s.Z @ y[:-3]
            qd, _, power = s.rates(q, candidate._boundaries)
            return np.r_[s.Z.T @ qd, power]

        solution = solve_ivp(
            rhs,
            (self._time, end),
            candidate._state,
            method="Radau",
            rtol=1e-10,
            atol=np.r_[np.full(candidate.system.dimension, 1e-11), [1e-9] * 3],
            max_step=self.max_step,
        )
        if not solution.success or not np.all(np.isfinite(solution.y)) or solution.t[-1] != end:
            raise InertialGraphError("integration", "Inertial Radau did not reach endpoint")
        if len(solution.t) - 1 > self.budget["maximum_internal_steps"]:
            raise InertialGraphError("step_budget", "Inertial internal step budget exhausted")
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
                maximum_internal_step_s=float(np.max(np.diff(solution.t))),
            ),
        )

    def _install(self, other):
        self._time, self._revision, self._history, self._boundaries, self._state = (
            other._time,
            other._revision,
            other._history,
            other._boundaries,
            other._state,
        )

    def advance(self, duration_s):
        return self._transaction(duration_s)

    def step(self, pressures_pa, duration_s, *, expected_revision, command_id):
        self._sources()
        if type(expected_revision) is not int or expected_revision != self._revision:
            raise InertialGraphError("stale_revision", "Stale inertial observation revision")
        if (
            not isinstance(command_id, str)
            or not 1 <= len(command_id) <= 128
            or not command_id.isascii()
            or any(not (c.isalnum() or c in "-_.:") for c in command_id)
        ):
            raise NetworkError("Invalid inertial command ID")
        if any(r["command_id"] == command_id for r in self._history):
            raise NetworkError("Duplicate inertial command ID")
        return self._transaction(duration_s, (deepcopy(pressures_pa), command_id))

    def checkpoint(self):
        self._sources()
        payload = dict(
            schema_version=1,
            model="inertial-pipe-radau-v1",
            input_sha256=self.identity,
            signature=deepcopy(self.captured),
            coordinates=self._state.tolist(),
            snapshot=self.snapshot(),
        )
        return dict(payload=payload, sha256=digest(payload))

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
            or p["model"] != "inertial-pipe-radau-v1"
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
        if not isinstance(values, list) or len(values) != self.system.dimension + 3:
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
