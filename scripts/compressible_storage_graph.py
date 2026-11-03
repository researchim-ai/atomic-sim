"""Thermodynamically complete synthetic EOS and mass/energy storage graphs.

This family has true variable mass density, not gravity-only density. Links
are quasi-steady throttles with optional conduction: enthalpy is transferred
once; no extra pressure-drop heat is added. Piston volumes are prescribed.
No acoustic/inertial transport, gravity, two-phase water or IAPWS coupling.
"""

import json
import math
import platform
from copy import deepcopy

import numpy as np
import scipy
from scipy.integrate import solve_ivp

from atomic_model_runtime.resources import resource
from core.network import NetworkError, identifier, number
from scripts.inertial_graph import InertialGraphError, digest
from scripts.solve_hydraulic_graph import _keys, sha
from sdk.experiment import _source_fingerprint

BUDGET = resource("validation/ai05_compressible_storage_budget_v1.json")
LEDGERS = ("external_heat_j", "piston_work_j", "external_entropy_j_k", "generated_entropy_j_k")


def signature():
    return dict(
        source_sha256=_source_fingerprint(),
        drivers={
            p: sha(p)
            for p in (
                "scripts/compressible_storage_graph.py",
                "scripts/inertial_graph.py",
                "scripts/solve_hydraulic_graph.py",
                "atomic_model_runtime/resources.py",
                "scripts/__init__.py",
            )
        },
        budgets={str(BUDGET): sha(BUDGET)},
        python=platform.python_version(),
        numpy=np.__version__,
        scipy=scipy.__version__,
    )


class StiffenedCaloricEOS:
    def __init__(self, spec):
        _keys(
            spec,
            (
                "kind",
                "cv_j_kg_k",
                "gamma",
                "pressure_shift_pa",
                "energy_offset_j_kg",
                "entropy_reference_temperature_k",
                "entropy_reference_density_kg_m3",
            ),
        )
        if spec["kind"] != "stiffened_caloric_v1":
            raise NetworkError("Unregistered storage EOS")
        self.cv = number(spec["cv_j_kg_k"], "cv", positive=True)
        self.gamma = number(spec["gamma"], "gamma", positive=True)
        self.pinf = number(spec["pressure_shift_pa"], "pressure_shift")
        self.e0 = number(spec["energy_offset_j_kg"], "energy_offset")
        self.Tref = number(spec["entropy_reference_temperature_k"], "entropy_Tref", positive=True)
        self.rref = number(spec["entropy_reference_density_kg_m3"], "entropy_rhoref", positive=True)
        self.cp = self.gamma * self.cv
        self.R = (self.gamma - 1) * self.cv
        if self.gamma <= 1 or self.pinf < 0 or not math.isfinite(self.cp) or self.R <= 0:
            raise NetworkError("Invalid caloric stiffened-gas parameters")

    def rho_T(self, rho, T):
        rho = number(rho, "density", positive=True)
        T = number(T, "temperature", positive=True)
        pstar = self.R * rho * T
        p = pstar - self.pinf
        u = self.cv * T + self.pinf / rho + self.e0
        h = self.cp * T + self.e0
        entropy = self.cv * (math.log(T) - math.log(self.Tref)) - self.R * (
            math.log(rho) - math.log(self.rref)
        )
        values = dict(
            density_kg_m3=rho,
            temperature_k=T,
            pressure_pa=p,
            internal_energy_j_kg=u,
            enthalpy_j_kg=h,
            entropy_j_kg_k=entropy,
            sound_speed_squared_m2_s2=self.gamma * self.R * T,
            drho_dp_at_T=rho / pstar,
            drho_dT_at_p=-rho / T,
        )
        if not all(math.isfinite(v) for v in values.values()):
            raise NetworkError("Nonfinite EOS state")
        return values

    def p_T(self, p, T):
        p = number(p, "pressure")
        T = number(T, "temperature", positive=True)
        if p + self.pinf <= 0:
            raise NetworkError("EOS requires p+pinf>0")
        return self.rho_T((p + self.pinf) / (self.R * T), T)

    def rho_u(self, rho, u):
        rho = number(rho, "density", positive=True)
        u = number(u, "specific_internal_energy")
        return self.rho_T(rho, (u - self.e0 - self.pinf / rho) / self.cv)


def _convex_log(x):
    if abs(x) < 1e-4:
        return x * x * (0.5 - x / 3 + x * x / 4 - x**3 / 5 + x**4 / 6)
    return x - math.log1p(x)


class StorageSystem:
    def __init__(self, request):
        _keys(request, ("format", "schema_version", "fluid", "domain", "nodes", "edges"))
        if (
            request["format"] != "atomic-compressible-storage-graph"
            or type(request["schema_version"]) is not int
            or request["schema_version"] != 1
        ):
            raise NetworkError("Expected compressible storage schema1")
        self.spec = deepcopy(request)
        self.budget = json.loads(BUDGET.read_text())
        self.eos = StiffenedCaloricEOS(request["fluid"])
        domain = request["domain"]
        _keys(
            domain,
            (
                "min_temperature_k",
                "max_temperature_k",
                "min_pressure_pa",
                "max_pressure_pa",
                "min_density_kg_m3",
                "max_density_kg_m3",
            ),
        )
        self.domain = {k: number(v, k, positive=True) for k, v in domain.items()}
        for field in ("temperature_k", "pressure_pa", "density_kg_m3"):
            if self.domain["min_" + field] >= self.domain["max_" + field]:
                raise NetworkError("Invalid storage domain")
        nodes, edges = request["nodes"], request["edges"]
        if (
            not isinstance(nodes, list)
            or not 1 <= len(nodes) <= self.budget["maximum_nodes"]
            or not isinstance(edges, list)
            or len(edges) > self.budget["maximum_edges"]
        ):
            raise NetworkError("Storage graph resource limit")
        self.names = []
        V = []
        vmin = []
        vmax = []
        velocity = []
        heat = []
        ua = []
        ambient = []
        m = []
        A = []
        p = []
        T = []
        rho = []
        s = []
        for node in nodes:
            _keys(
                node,
                (
                    "id",
                    "kind",
                    "volume_m3",
                    "volume_rate_m3_s",
                    "min_volume_m3",
                    "max_volume_m3",
                    "initial_pressure_pa",
                    "initial_temperature_k",
                    "heat_power_w",
                    "ambient_ua_w_k",
                    "ambient_temperature_k",
                ),
            )
            name = identifier(node["id"])
            if node["kind"] != "compressible_inventory_v1" or name in self.names:
                raise NetworkError("Duplicate or unregistered storage component")
            self.names.append(name)
            volume = number(node["volume_m3"], "volume", positive=True)
            lo = number(node["min_volume_m3"], "min_volume", positive=True)
            hi = number(node["max_volume_m3"], "max_volume", positive=True)
            if not lo <= volume <= hi or lo >= hi:
                raise NetworkError("Invalid volume interval")
            state = self.eos.p_T(
                number(node["initial_pressure_pa"], "initial_pressure", positive=True),
                number(node["initial_temperature_k"], "initial_temperature", positive=True),
            )
            V.append(volume)
            vmin.append(lo)
            vmax.append(hi)
            velocity.append(number(node["volume_rate_m3_s"], "volume_rate"))
            heat.append(number(node["heat_power_w"], "heat"))
            ua.append(number(node["ambient_ua_w_k"], "UA"))
            ambient.append(
                number(node["ambient_temperature_k"], "ambient_temperature", positive=True)
            )
            if ua[-1] < 0:
                raise NetworkError("Negative heat transfer coefficient")
            m.append(state["density_kg_m3"] * volume)
            A.append(state["density_kg_m3"] * volume * self.eos.cv * state["temperature_k"])
            p.append(number(node["initial_pressure_pa"], "initial_pressure"))
            T.append(state["temperature_k"])
            rho.append(state["density_kg_m3"])
            s.append(state["entropy_j_kg_k"])
        self.n = len(nodes)
        self.V0 = np.array(V)
        self.Vmin = np.array(vmin)
        self.Vmax = np.array(vmax)
        self.vdot = np.array(velocity)
        self.initial_heat = dict(zip(self.names, heat))
        self.ua = np.array(ua)
        self.ambient = np.array(ambient)
        self.m0 = np.array(m)
        self.A0 = np.array(A)
        self.p0 = np.array(p)
        self.T0 = np.array(T)
        self.rho0 = np.array(rho)
        self.s0 = np.array(s)
        if not np.all(np.isfinite(self.m0)) or not np.all(np.isfinite(self.A0)):
            raise NetworkError("Nonfinite initial inventories")
        self.edges = []
        self.links = []
        adjacent = {name: set() for name in self.names}
        for edge in edges:
            _keys(
                edge,
                (
                    "id",
                    "kind",
                    "from",
                    "to",
                    "linear_pa_s_kg",
                    "quadratic_pa_s2_kg2",
                    "thermal_conductance_w_k",
                ),
            )
            eid = identifier(edge["id"])
            a = edge["from"]
            b = edge["to"]
            if (
                edge["kind"] != "resistive_thermal_link_v1"
                or eid in self.edges
                or a not in self.names
                or b not in self.names
                or a == b
            ):
                raise NetworkError("Invalid or unregistered storage link")
            r = number(edge["linear_pa_s_kg"], "linear", positive=True)
            k = number(edge["quadratic_pa_s2_kg2"], "quadratic")
            conduct = number(edge["thermal_conductance_w_k"], "conductance")
            if k < 0 or conduct < 0:
                raise NetworkError("Negative link coefficient")
            self.edges.append(eid)
            self.links.append((self.names.index(a), self.names.index(b), r, k, conduct))
            adjacent[a].add(b)
            adjacent[b].add(a)
        seen = {self.names[0]}
        while True:
            enlarged = seen | {b for a in seen for b in adjacent[a]}
            if enlarged == seen:
                break
            seen = enlarged
        if seen != set(self.names):
            raise NetworkError("Disconnected storage graph")
        self.evaluate(0, np.zeros(2 * self.n + 4), self.initial_heat)

    def heaters(self, values):
        _keys(values, tuple(self.names))
        return {name: number(values[name], "heater_power") for name in self.names}

    def point(self, time, y):
        if (
            y.shape != (2 * self.n + 4,)
            or not np.all(np.isfinite(y))
            or not math.isfinite(time)
            or time < 0
        ):
            raise NetworkError("Malformed storage coordinates/clock")
        dm = y[: self.n]
        dA = y[self.n : 2 * self.n]
        dv = self.vdot * time
        V = self.V0 + dv
        m = self.m0 + dm
        if np.any(V < self.Vmin) or np.any(V > self.Vmax) or np.any(m <= 0):
            raise InertialGraphError("inventory_domain", "Volume or mass left declared domain")
        rm = dm / self.m0
        rA = dA / self.A0
        rv = dv / self.V0
        if np.any(1 + rA <= 0):
            raise InertialGraphError("thermal_inventory", "Thermal energy became nonpositive")
        T = self.T0 * (1 + rA) / (1 + rm)
        rho = self.rho0 * (1 + rm) / (1 + rv)
        p = self.p0 + (self.p0 + self.eos.pinf) * (rA - rv) / (1 + rv)
        for field, values in [("temperature_k", T), ("density_kg_m3", rho), ("pressure_pa", p)]:
            if (
                not np.all(np.isfinite(values))
                or np.any(values < self.domain["min_" + field])
                or np.any(values > self.domain["max_" + field])
            ):
                raise InertialGraphError(field + "_domain", "Storage state left declared domain")
        A = self.A0 + dA
        U = A + self.eos.pinf * V + self.eos.e0 * m
        deltaS = dm * self.s0 + m * (
            self.eos.cv * np.log1p(rA) - self.eos.cp * np.log1p(rm) + self.eos.R * np.log1p(rv)
        )
        entropy = (
            self.s0
            + self.eos.cv * np.log1p(rA)
            - self.eos.cp * np.log1p(rm)
            + self.eos.R * np.log1p(rv)
        )
        if not np.all(np.isfinite(U)) or not np.all(np.isfinite(deltaS)):
            raise NetworkError("Nonfinite thermodynamic inventories")
        return dict(m=m, V=V, T=T, rho=rho, p=p, A=A, U=U, deltaS=deltaS, entropy=entropy, dv=dv)

    def evaluate(self, time, y, heaters):
        point = self.point(time, y)
        p = point["p"]
        T = point["T"]
        cp = self.eos.cp
        dm = np.zeros(self.n)
        dA = np.zeros(self.n)
        qout = []
        generation = 0.0
        for a, b, r, k, conduct in self.links:
            dp = p[a] - p[b]
            q = (
                dp / r
                if k == 0
                else dp / (0.5 * r + math.hypot(0.5 * r, math.sqrt(k) * math.sqrt(abs(dp))))
            )
            if not math.isfinite(q) or abs(r * q + k * q * abs(q) - dp) > 1e-8 + 1e-9 * max(
                1.0, abs(dp)
            ):
                raise InertialGraphError("flow_law", "Resistive link flow law not resolved")
            upstream, downstream = (a, b) if q >= 0 else (b, a)
            amount = abs(q)
            energy = q * cp * T[upstream] + conduct * (T[a] - T[b])
            dm[a] -= q
            dm[b] += q
            dA[a] -= energy
            dA[b] += energy
            qout.append(q)
            x = (T[upstream] - T[downstream]) / T[downstream]
            production = amount * (
                cp * _convex_log(float(x))
                + self.eos.R * math.log1p(abs(dp) / (p[downstream] + self.eos.pinf))
            )
            generation += production + conduct * (T[a] - T[b]) ** 2 / (T[a] * T[b])
        heat = np.array([heaters[name] for name in self.names])
        ambient = self.ua * (self.ambient - T)
        Q = heat + ambient
        dA += Q - (p + self.eos.pinf) * self.vdot
        external_entropy = sum(heat / T + ambient / self.ambient)
        generation += float(sum(self.ua * (self.ambient - T) ** 2 / (self.ambient * T)))
        ledgers = np.array([sum(Q), p @ self.vdot, external_entropy, generation])
        result = np.r_[dm, dA, ledgers]
        if not np.all(np.isfinite(result)) or generation < 0:
            raise NetworkError("Nonfinite or negative entropy production")
        point.update(
            q=np.array(qout), generation=float(generation), external_entropy=float(external_entropy)
        )
        return result, point

    def contract(self):
        return dict(
            family="compressible_storage_graph_v1",
            node_ids=list(self.names),
            edge_ids=list(self.edges),
            mass_states=self.n,
            thermal_energy_states=self.n,
            eos="stiffened_caloric_v1",
            ledger_names=list(LEDGERS),
            variable_mass_density=True,
            pressure_from_eos=True,
            prescribed_volume="V0+volume_rate*time",
            inertial_flow=False,
            gravity=False,
            enthalpy_transport="upstream cp*T; common energy offset cancels in shifted energy coordinates",
        )


class CompressibleStorageGraph:
    def __init__(self, request, *, max_step_s=0.1):
        self.system = StorageSystem(deepcopy(request))
        self.budget = json.loads(BUDGET.read_text())
        self.max_step = number(max_step_s, "max_step", positive=True)
        if self.max_step > self.budget["maximum_step_s"]:
            raise NetworkError("Storage step exceeds budget")
        self.identity = digest(dict(request=request, max_step_s=self.max_step, method="Radau"))
        self.captured = signature()
        self._state = np.zeros(2 * self.system.n + 4)
        self._time = 0.0
        self._revision = 0
        self._history = []
        self._heaters = deepcopy(self.system.initial_heat)
        self.snapshot()

    @property
    def revision(self):
        return self._revision

    def _sources(self):
        if signature() != self.captured:
            raise InertialGraphError("source_drift", "Storage sources/budget/runtime changed")

    def _observe(self, y, time):
        s = self.system
        _, p = s.evaluate(time, y, self._heaters)
        dm = y[: s.n]
        dA = y[s.n : 2 * s.n]
        Q, W, SE, SG = y[-4:]
        mass = float(sum(dm))
        energy = float(sum(dA + s.eos.pinf * p["dv"] + s.eos.e0 * dm))
        entropy = float(sum(p["deltaS"]))
        er = energy - Q + W
        sr = entropy - SE - SG
        scale = max(1.0, abs(energy), abs(Q), abs(W))
        sscale = max(1.0, abs(entropy), abs(SE), abs(SG))
        if abs(mass) > self.budget["mass_atol_kg"] + self.budget["mass_rtol"] * sum(s.m0):
            raise InertialGraphError("mass", "Closed storage mass balance failed")
        if (
            not math.isfinite(er)
            or abs(er) > self.budget["energy_atol_j"] + self.budget["energy_rtol"] * scale
        ):
            raise InertialGraphError("energy", "Storage first-law balance failed", error_j=er)
        if (
            not math.isfinite(sr)
            or abs(sr) > self.budget["entropy_atol_j_k"] + self.budget["entropy_rtol"] * sscale
            or SG < -self.budget["entropy_atol_j_k"]
        ):
            raise InertialGraphError("entropy", "Storage entropy balance failed", error_j_k=sr)
        return dict(
            time_s=time,
            masses_kg=dict(zip(s.names, map(float, p["m"]))),
            volumes_m3=dict(zip(s.names, map(float, p["V"]))),
            densities_kg_m3=dict(zip(s.names, map(float, p["rho"]))),
            temperatures_k=dict(zip(s.names, map(float, p["T"]))),
            pressures_pa=dict(zip(s.names, map(float, p["p"]))),
            internal_energies_j=dict(zip(s.names, map(float, p["U"]))),
            flows_kg_s=dict(zip(s.edges, map(float, p["q"]))),
            mass_change_kg=mass,
            internal_energy_change_j=energy,
            entropy_change_j_k=entropy,
            mass_residual_kg=mass,
            energy_residual_j=er,
            entropy_residual_j_k=sr,
            entropy_generation_w_k=p["generation"],
            **dict(zip(LEDGERS, map(float, y[-4:]))),
        )

    def snapshot(self):
        return dict(
            revision=self._revision,
            active_heat_power_w=deepcopy(self._heaters),
            history=deepcopy(self._history),
            state=self._observe(self._state, self._time),
            contract=self.system.contract(),
        )

    def residual(self, time, state, rate):
        y = np.asarray(state, dtype=float)
        dy = np.asarray(rate, dtype=float)
        if (
            y.shape != (2 * self.system.n + 4,)
            or dy.shape != y.shape
            or not np.all(np.isfinite(dy))
        ):
            raise NetworkError("Malformed storage residual")
        return dy - self.system.evaluate(time, y, self._heaters)[0]

    def _install(self, c):
        self._state, self._time, self._revision, self._history, self._heaters = (
            c._state,
            c._time,
            c._revision,
            c._history,
            c._heaters,
        )

    def _transaction(self, duration, command=None):
        self._sources()
        duration = number(duration, "duration", positive=True)
        end = self._time + duration
        if (
            duration > self.budget["maximum_duration_s"]
            or not math.isfinite(end)
            or end <= self._time
        ):
            raise NetworkError("Invalid storage duration/clock")
        candidate = deepcopy(self)
        transition = None
        if command is not None:
            if len(self._history) >= self.budget["maximum_commands"]:
                raise NetworkError("Storage command history exhausted")
            values, cid = command
            left = self._observe(self._state, self._time)
            candidate._heaters = candidate.system.heaters(values)
            right = candidate._observe(candidate._state, candidate._time)
            transition = dict(time_s=self._time, left=left, right=right)
            candidate._history.append(
                dict(
                    sequence=len(self._history) + 1,
                    time_s=self._time,
                    commit_revision=self.revision + 1,
                    command_id=cid,
                    heat_power_w=deepcopy(candidate._heaters),
                )
            )
        evaluations = 0

        def rhs(t, y):
            nonlocal evaluations
            evaluations += 1
            if evaluations > self.budget["maximum_rhs_evaluations"]:
                raise InertialGraphError("rhs_budget", "Storage RHS limit exceeded")
            return candidate.system.evaluate(t, y, candidate._heaters)[0]

        n = self.system.n
        solution = solve_ivp(
            rhs,
            (self._time, end),
            candidate._state,
            method="Radau",
            rtol=1e-10,
            atol=np.r_[np.full(n, 1e-12), np.full(n, 1e-8), [1e-8, 1e-8, 1e-10, 1e-10]],
            max_step=self.max_step,
        )
        if not solution.success or solution.t[-1] != end or not np.all(np.isfinite(solution.y)):
            raise InertialGraphError("integration", "Storage integration failed")
        if len(solution.t) - 1 > self.budget["maximum_internal_steps"]:
            raise InertialGraphError("step_budget", "Storage step limit exceeded")
        candidate._state = solution.y[:, -1].copy()
        candidate._time = end
        candidate._revision += 1
        result = candidate.snapshot()
        self._sources()
        self._install(candidate)
        return dict(
            **result,
            heater_transition=transition,
            work=dict(
                rhs=evaluations,
                internal_steps=len(solution.t) - 1,
                maximum_internal_step_s=float(max(np.diff(solution.t))),
            ),
        )

    def advance(self, duration_s):
        return self._transaction(duration_s)

    def step(self, heat_power_w, duration_s, *, expected_revision, command_id):
        self._sources()
        if type(expected_revision) is not int or expected_revision != self.revision:
            raise InertialGraphError("stale_revision", "Stale storage observation revision")
        if (
            not isinstance(command_id, str)
            or not 1 <= len(command_id) <= 128
            or not command_id.isascii()
            or any(not (c.isalnum() or c in "-_.:") for c in command_id)
        ):
            raise NetworkError("Invalid heater command ID")
        if any(r["command_id"] == command_id for r in self._history):
            raise NetworkError("Duplicate heater command ID")
        return self._transaction(duration_s, (deepcopy(heat_power_w), command_id))

    def checkpoint(self):
        self._sources()
        p = dict(
            schema_version=1,
            model="compressible-storage-radau-v1",
            input_sha256=self.identity,
            signature=deepcopy(self.captured),
            coordinates=self._state.tolist(),
            snapshot=self.snapshot(),
        )
        return dict(payload=p, sha256=digest(p))

    def restore(self, checkpoint):
        self._sources()
        _keys(checkpoint, ("payload", "sha256"))
        p = checkpoint["payload"]
        if digest(p) != checkpoint["sha256"]:
            raise NetworkError("Storage checkpoint digest mismatch")
        _keys(
            p, ("schema_version", "model", "input_sha256", "signature", "coordinates", "snapshot")
        )
        if (
            type(p["schema_version"]) is not int
            or p["schema_version"] != 1
            or p["model"] != "compressible-storage-radau-v1"
            or p["input_sha256"] != self.identity
            or p["signature"] != self.captured
        ):
            raise NetworkError("Storage checkpoint input/source mismatch")
        snap = p["snapshot"]
        _keys(snap, ("revision", "active_heat_power_w", "history", "state", "contract"))
        if not isinstance(snap["state"], dict) or "time_s" not in snap["state"]:
            raise NetworkError("Malformed storage observation")
        time = number(snap["state"]["time_s"], "time")
        rev = snap["revision"]
        history = snap["history"]
        if (
            type(rev) is not int
            or rev < 0
            or time < 0
            or (rev == 0) != (time == 0)
            or not isinstance(history, list)
            or len(history) > self.budget["maximum_commands"]
        ):
            raise NetworkError("Storage checkpoint clock/history mismatch")
        active = deepcopy(self.system.initial_heat)
        last_time = 0.0
        last_rev = 0
        ids = set()
        for seq, r in enumerate(history, 1):
            _keys(r, ("sequence", "time_s", "commit_revision", "command_id", "heat_power_w"))
            when = number(r["time_s"], "command time")
            commit = r["commit_revision"]
            cid = r["command_id"]
            if (
                type(r["sequence"]) is not int
                or r["sequence"] != seq
                or type(commit) is not int
                or not last_rev < commit <= rev
                or not last_time <= when < time
                or (seq > 1 and when <= last_time)
                or ((when == 0) != (commit == 1))
                or not isinstance(cid, str)
                or not 1 <= len(cid) <= 128
                or not cid.isascii()
                or any(not (c.isalnum() or c in "-_.:") for c in cid)
                or cid in ids
            ):
                raise NetworkError("Storage checkpoint receipt path mismatch")
            ids.add(cid)
            active = self.system.heaters(r["heat_power_w"])
            last_time, last_rev = when, commit
        values = p["coordinates"]
        if not isinstance(values, list) or len(values) != 2 * self.system.n + 4:
            raise NetworkError("Storage checkpoint layout mismatch")
        vector = np.array([number(v, "coordinate") for v in values])
        if time == 0 and np.any(vector != 0):
            raise NetworkError("Nonzero initial storage stocks")
        c = deepcopy(self)
        c._state = vector
        c._time = time
        c._revision = rev
        c._history = deepcopy(history)
        c._heaters = active
        fresh = c.snapshot()
        if fresh != snap:
            raise NetworkError("Storage checkpoint observation mismatch")
        self._sources()
        self._install(c)
        return fresh
