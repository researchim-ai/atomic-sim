"""Two-way thermal/resistance coupling on positive-inertance pipe graphs.

Density/cp and thermal inventories are fixed. Kinetic dissipation transferred
to fluid heat is an internal exchange, not a second external energy source.
No buoyancy, EOS, pressure waves or advective kinetic-energy transport.
"""

import json
import math
from copy import deepcopy
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from core.network import NetworkError, number
from scripts.inertial_graph import InertialGraph, InertialGraphError, InertialPipeSystem, digest
from scripts.inertial_graph import signature as hydraulic_signature
from scripts.solve_hydraulic_graph import _keys, sha

BUDGET = Path("validation/ai05_thermal_inertial_budget_v1.json")
LEDGERS = (
    "pressure_work_j",
    "gravity_work_j",
    "external_heat_j",
    "external_thermal_advection_j",
    "friction_to_fluid_j",
    "exported_friction_heat_j",
)


def signature():
    value = hydraulic_signature()
    value["drivers"]["scripts/thermal_inertial_graph.py"] = sha(__file__)
    value["budgets"][str(BUDGET)] = sha(BUDGET)
    return value


class ThermalInertialSystem(InertialPipeSystem):
    def __init__(self, request):
        _keys(request, ("format", "schema_version", "hydraulic", "thermal", "edge_laws"))
        if (
            request["format"] != "atomic-thermal-inertial-graph"
            or type(request["schema_version"]) is not int
            or request["schema_version"] != 1
        ):
            raise NetworkError("Expected thermal-inertial graph schema1")
        h = request["hydraulic"]
        t = request["thermal"]
        if (
            not isinstance(h, dict)
            or not isinstance(h.get("nodes"), list)
            or not isinstance(h.get("edges"), list)
            or any(not isinstance(n, dict) or "id" not in n for n in h["nodes"] + h["edges"])
        ):
            raise NetworkError("Malformed thermal hydraulic topology")
        names = [n["id"] for n in h["nodes"]]
        edge_ids = [e["id"] for e in h["edges"]]
        _keys(
            t,
            (
                "heat_capacity_j_kg_k",
                "reference_temperature_k",
                "min_temperature_k",
                "max_temperature_k",
                "nodes",
            ),
        )
        self.cp = number(t["heat_capacity_j_kg_k"], "cp", positive=True)
        self.tref = number(t["reference_temperature_k"], "reference_temperature", positive=True)
        self.tmin = number(t["min_temperature_k"], "minimum_temperature", positive=True)
        self.tmax = number(t["max_temperature_k"], "maximum_temperature", positive=True)
        if self.tmin >= self.tmax:
            raise NetworkError("Invalid temperature interval")
        _keys(t["nodes"], tuple(names))
        _keys(request["edge_laws"], tuple(edge_ids))
        initial = []
        volumes = []
        heat = []
        ua = []
        ambient = []
        supply = []
        for name in names:
            n = t["nodes"][name]
            _keys(
                n,
                (
                    "volume_m3",
                    "initial_temperature_k",
                    "heat_power_w",
                    "ambient_ua_w_k",
                    "ambient_temperature_k",
                    "supply_temperature_k",
                ),
            )
            initial.append(number(n["initial_temperature_k"], "initial_temperature", positive=True))
            volumes.append(number(n["volume_m3"], "volume", positive=True))
            heat.append(number(n["heat_power_w"], "heat"))
            ua.append(number(n["ambient_ua_w_k"], "ambient_ua"))
            ambient.append(number(n["ambient_temperature_k"], "ambient_temperature", positive=True))
            supply.append(number(n["supply_temperature_k"], "supply_temperature", positive=True))
        if any(v < 0 for v in ua) or any(not self.tmin <= v <= self.tmax for v in initial + supply):
            raise NetworkError("Thermal initial/source domain or UA invalid")
        self.T0 = np.array(initial)
        self.volumes = np.array(volumes)
        self.heat = np.array(heat)
        self.ua = np.array(ua)
        self.ambient = np.array(ambient)
        self.supply = np.array(supply)
        self.laws = []
        for eid in edge_ids:
            law = request["edge_laws"][eid]
            _keys(
                law,
                (
                    "reference_temperature_k",
                    "temperature_coefficient_per_k",
                    "minimum_multiplier",
                    "maximum_multiplier",
                    "fluid_heat_fraction",
                ),
            )
            law = {k: number(v, k) for k, v in law.items()}
            if (
                law["reference_temperature_k"] <= 0
                or not 0 < law["minimum_multiplier"] <= law["maximum_multiplier"]
                or not 0 <= law["fluid_heat_fraction"] <= 1
            ):
                raise NetworkError("Invalid resistance law or friction heat fraction")
            self.laws.append(law)
        self._topology_names = names
        for edge in h["edges"]:
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
            if edge["from"] not in names or edge["to"] not in names:
                raise NetworkError("Unknown coupled edge endpoint")
        self._endpoints = [(names.index(e["from"]), names.index(e["to"])) for e in h["edges"]]
        super().__init__(deepcopy(h))
        self.capacity = self.rho * self.cp * self.volumes
        if not np.all(np.isfinite(self.capacity)):
            raise NetworkError("Nonfinite thermal capacity")
        self.coupled_budget = json.loads(BUDGET.read_text())
        self.evaluate(np.zeros(self.dimension + len(self.names) + 6), self.initial_pressures)

    def loss(self, q, temperature):
        if (
            temperature.shape != self.T0.shape
            or not np.all(np.isfinite(temperature))
            or np.any(temperature < self.tmin)
            or np.any(temperature > self.tmax)
        ):
            raise InertialGraphError(
                "temperature_domain", "Thermal stage left declared temperature interval"
            )
        multipliers = []
        for law, (a, b) in zip(self.laws, self._endpoints):
            exponent = law["temperature_coefficient_per_k"] * (
                0.5 * (temperature[a] + temperature[b]) - law["reference_temperature_k"]
            )
            if (
                not math.log(law["minimum_multiplier"])
                <= exponent
                <= math.log(law["maximum_multiplier"])
            ):
                raise InertialGraphError(
                    "resistance_domain", "Temperature multiplier left declared interval"
                )
            multipliers.append(math.exp(exponent))
        loss = np.array(multipliers) * (self.R * q + self.Q * q * np.abs(q))
        return loss, np.array(multipliers)

    def rates(self, flows, boundaries, temperature=None):
        q = np.asarray(flows, dtype=float)
        temperature = self.T0 if temperature is None else np.asarray(temperature, dtype=float)
        if q.shape != self.L.shape or not np.all(np.isfinite(q)) or np.max(np.abs(q)) > self.qmax:
            raise InertialGraphError("flow_domain", "Coupled flow left declared domain")
        active = self.pressures(boundaries)
        loss, _ = self.loss(q, temperature)
        potential = np.zeros(len(self.names))
        potential[self.boundary] = [
            active[self.names[i]] + self.rho * self.gravity * self.elevation[i]
            for i in self.boundary
        ]
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
            raise InertialGraphError("pressure_domain", "Coupled pressure left declared domain")
        qd = self.Z @ (self.Z.T @ ((-self.A.T @ potential - loss) / self.L))
        mismatch = self.L * qd + self.A.T @ potential + loss
        if not np.all(np.isfinite(qd)) or np.max(abs(mismatch)) > self.budget["momentum_atol_pa"]:
            raise InertialGraphError("momentum_projection", "Coupled momentum projection failed")
        external = -self.A @ q
        return (
            qd,
            pressure,
            np.array(
                [
                    pressure @ external / self.rho,
                    self.gravity * self.elevation @ external,
                    q @ loss / self.rho,
                ]
            ),
        )

    def evaluate(self, y, boundaries):
        d = self.dimension
        n = len(self.names)
        if y.shape != (d + n + 6,) or not np.all(np.isfinite(y)):
            raise NetworkError("Malformed coupled coordinate vector")
        q = self.q0 + self.Z @ y[:d]
        temperature = self.T0 + y[d : d + n]
        qd, p, _ = self.rates(q, boundaries, temperature)
        thermal_rate, ledgers, loss, multipliers = self.thermal_terms(q, temperature, p)
        return np.r_[self.Z.T @ qd, thermal_rate, ledgers], dict(
            q=q, qd=qd, p=p, T=temperature, multipliers=multipliers, loss=loss
        )

    def thermal_terms(self, q, temperature, p):
        loss, multipliers = self.loss(q, temperature)
        node_heat = np.zeros(len(self.names))
        # Compute each internal flux once; use the same scalar with opposite signs.
        for j, (a, b) in enumerate(self._endpoints):
            flux = q[j] * self.cp * ((temperature[a] if q[j] >= 0 else temperature[b]) - self.tref)
            node_heat[a] -= flux
            node_heat[b] += flux
        external_mass = -self.A @ q
        external_mass[self.free] = self.injection[self.free]
        advective = (
            external_mass
            * self.cp
            * np.where(external_mass >= 0, self.supply - self.tref, temperature - self.tref)
        )
        sources = self.heat - self.ua * (temperature - self.ambient)
        friction = q * loss / self.rho
        alpha = np.array([law["fluid_heat_fraction"] for law in self.laws])
        fluid = alpha * friction
        for power, (a, b) in zip(fluid, self._endpoints):
            node_heat[a] += 0.5 * power
            node_heat[b] += 0.5 * power
        thermal_rate = (node_heat + advective + sources) / self.capacity
        ledgers = np.array(
            [
                -p @ (self.A @ q) / self.rho,
                -self.gravity * self.elevation @ (self.A @ q),
                sum(sources),
                sum(advective),
                sum(fluid),
                sum((1 - alpha) * friction),
            ]
        )
        if not np.all(np.isfinite(thermal_rate)) or not np.all(np.isfinite(ledgers)):
            raise NetworkError("Nonfinite coupled thermal rate")
        return thermal_rate, ledgers, loss, multipliers

    def contract(self):
        value = super().contract()
        value.update(
            family="thermal_inertial_pipe_graph_v1",
            thermal_coupling=True,
            flow_coordinates=self.dimension,
            temperature_states=len(self.names),
            ledger_names=list(LEDGERS),
            heat_deposition="equal endpoint split of recovered pipe friction",
            advected_heat="upwind cp*(T-Tref); pressure work counted only in mechanical ledger",
            buoyancy=False,
        )
        return value


class ThermalInertialGraph(InertialGraph):
    def __init__(self, request, *, max_step_s=1.0):
        self.system = ThermalInertialSystem(deepcopy(request))
        self.budget = json.loads(BUDGET.read_text())
        self.max_step = number(max_step_s, "max_step_s", positive=True)
        if self.max_step > self.budget["maximum_macro_step_s"]:
            raise NetworkError("Coupled step exceeds budget")
        self.identity = digest(dict(request=request, max_step_s=self.max_step, method="Radau"))
        self.captured = signature()
        self._time = 0.0
        self._revision = 0
        self._history = []
        self._boundaries = deepcopy(self.system.initial_pressures)
        self._state = np.zeros(self.system.dimension + len(self.system.names) + 6)
        self.snapshot()

    def _sources(self):
        if signature() != self.captured:
            raise InertialGraphError("source_drift", "Coupled sources/budget/runtime changed")

    def _observe(self, state, time):
        s = self.system
        rate, point = s.evaluate(state, self._boundaries)
        d = s.dimension
        n = len(s.names)
        kinetic = float(0.5 * np.dot(s.L * point["q"], point["q"]) / s.rho)
        thermal = float(s.capacity @ state[d : d + n])
        P, G, Q, H, F, B = state[-6:]
        errors = np.array(
            [
                kinetic - s.k0 - P - G + F + B,
                thermal - Q - H - F,
                kinetic - s.k0 + thermal - P - G - Q - H + B,
            ]
        )
        scale = max(1.0, abs(s.k0), abs(kinetic), abs(thermal), float(max(abs(state[-6:]))))
        if (
            not math.isfinite(kinetic)
            or not np.all(np.isfinite(errors))
            or max(abs(errors)) > self.budget["energy_atol_j"] + self.budget["energy_rtol"] * scale
            or min(F, B) < -self.budget["energy_atol_j"]
        ):
            raise InertialGraphError(
                "energy", "Coupled kinetic/thermal/total ledger failed", errors_j=errors.tolist()
            )
        mass = s.Af @ point["q"] + s.injection[s.free]
        momentum = (
            s.L * point["qd"]
            + s.A.T @ (point["p"] + s.rho * s.gravity * s.elevation)
            + point["loss"]
        )
        if (
            np.max(abs(mass), initial=0) > self.budget["mass_atol_kg_s"]
            or max(abs(momentum)) > self.budget["momentum_atol_pa"]
        ):
            raise InertialGraphError("residual", "Coupled physical residual failed")
        return dict(
            time_s=time,
            flows_kg_s=dict(zip(s.edges, map(float, point["q"]))),
            pressures_pa=dict(zip(s.names, map(float, point["p"]))),
            temperatures_k=dict(zip(s.names, map(float, point["T"]))),
            temperature_increments_k=dict(zip(s.names, map(float, state[d : d + n]))),
            resistance_multipliers=dict(zip(s.edges, map(float, point["multipliers"]))),
            kinetic_energy_j=kinetic,
            thermal_energy_change_j=thermal,
            mechanical_energy_residual_j=float(errors[0]),
            thermal_energy_residual_j=float(errors[1]),
            total_energy_residual_j=float(errors[2]),
            maximum_mass_residual_kg_s=float(np.max(abs(mass), initial=0)),
            maximum_momentum_residual_pa=float(max(abs(momentum))),
            **dict(zip(LEDGERS, map(float, state[-6:]))),
        )

    def residual(self, state, rate):
        s = self.system
        m = len(s.edges)
        n = len(s.names)
        size = m + 2 * n + 6
        y = np.asarray(state, dtype=float)
        dy = np.asarray(rate, dtype=float)
        if (
            y.shape != (size,)
            or dy.shape != (size,)
            or not np.all(np.isfinite(y))
            or not np.all(np.isfinite(dy))
        ):
            raise NetworkError("Malformed coupled physical residual")
        q, p, T = y[:m], y[m : m + n], y[m + n : m + 2 * n]
        thermal_rate, ledgers, loss, _ = s.thermal_terms(q, T, p)
        node = s.A @ q + s.injection
        for i in s.fixed:
            node[i] = p[i] - self._boundaries[s.names[i]]
        momentum = s.L * dy[:m] + s.A.T @ (p + s.rho * s.gravity * s.elevation) + loss
        return np.r_[
            momentum,
            node,
            s.capacity * (dy[m + n : m + 2 * n] - thermal_rate),
            dy[-6:] - ledgers,
        ]

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
            atol=np.r_[np.full(d + n, 1e-11), np.full(6, 1e-8)],
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

    def checkpoint(self):
        self._sources()
        payload = dict(
            schema_version=1,
            model="thermal-inertial-radau-v1",
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
            or p["model"] != "thermal-inertial-radau-v1"
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
            or len(values) != self.system.dimension + len(self.system.names) + 6
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
