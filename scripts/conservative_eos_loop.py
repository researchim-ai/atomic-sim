"""Conservative periodic Euler loop: real mass, momentum, total energy and EOS.

First-order hydrostatic Rusanov spatial flux with SSPRK3 time integration.
Potential energy is carried in the shared energy flux, not added as buoyancy
heat/work bookkeeping. General branching, reservoirs and piston faces are not
part of this family. Entropy admission is checked, not an all-state proof.
"""

import json
import math
from copy import deepcopy

import numpy as np

from atomic_model_runtime.resources import resource
from core.network import NetworkError, number
from scripts.compressible_storage_graph import StiffenedCaloricEOS
from scripts.compressible_storage_graph import signature as storage_signature
from scripts.inertial_graph import InertialGraphError, digest
from scripts.solve_hydraulic_graph import _keys, sha

BUDGET = resource("validation/ai05_conservative_loop_budget_v1.json")
LEDGERS = (
    "external_heat_j",
    "exported_drag_heat_j",
    "external_entropy_j_k",
    "generated_entropy_j_k",
    "gravity_impulse_kg_m_s",
    "drag_impulse_kg_m_s",
)


def signature():
    value = storage_signature()
    value["drivers"]["scripts/conservative_eos_loop.py"] = sha(__file__)
    value["budgets"][str(BUDGET)] = sha(BUDGET)
    return value


class ConservativeLoopSystem:
    def __init__(self, request):
        _keys(
            request,
            ("format", "schema_version", "fluid", "geometry", "initial", "heat", "drag", "domain"),
        )
        if (
            request["format"] != "atomic-conservative-eos-loop"
            or type(request["schema_version"]) is not int
            or request["schema_version"] != 1
        ):
            raise NetworkError("Expected conservative EOS loop schema1")
        self.budget = json.loads(BUDGET.read_text())
        self.eos = StiffenedCaloricEOS(request["fluid"])
        self.spec = deepcopy(request)
        g = request["geometry"]
        _keys(g, ("cells", "length_m", "area_m2", "elevations_m", "gravity_m_s2"))
        self.n = g["cells"]
        if type(self.n) is not int or not 4 <= self.n <= self.budget["maximum_cells"]:
            raise NetworkError("Conservative loop cell limit")
        self.length = number(g["length_m"], "length", positive=True)
        self.area = number(g["area_m2"], "area", positive=True)
        self.gravity = number(g["gravity_m_s2"], "gravity")
        if self.gravity < 0:
            raise NetworkError("Negative gravity")
        self.dx = self.length / self.n
        self.volume = self.dx * self.area
        self.names = tuple("c" + str(i) for i in range(self.n))

        def array(values, name, positive=False):
            if not isinstance(values, list) or len(values) != self.n:
                raise NetworkError("Invalid " + name + " profile")
            return np.array([number(v, name, positive=positive) for v in values])

        self.z = array(g["elevations_m"], "elevation")
        self.phi = self.gravity * self.z
        self.face_phi = 0.5 * (self.phi + np.roll(self.phi, -1))
        initial = request["initial"]
        _keys(initial, ("pressures_pa", "temperatures_k", "velocities_m_s"))
        self.p0 = array(initial["pressures_pa"], "initial pressure", True)
        self.T0 = array(initial["temperatures_k"], "initial temperature", True)
        self.u0 = array(initial["velocities_m_s"], "initial velocity")
        self.rho0 = (self.p0 + self.eos.pinf) / (self.eos.R * self.T0)
        self.m0 = self.rho0 * self.volume
        self.P0 = self.m0 * self.u0
        self.A0 = self.m0 * self.eos.cv * self.T0
        self.K0 = 0.5 * self.m0 * self.u0**2
        self.C0 = self.A0 + self.K0 + self.m0 * self.phi
        self.s0 = self.eos.cv * np.log(self.T0 / self.eos.Tref) - self.eos.R * np.log(
            self.rho0 / self.eos.rref
        )
        if not np.all(np.isfinite(self.C0)) or not np.all(np.isfinite(self.s0)):
            raise NetworkError("Nonfinite initial conservative inventory")
        h = request["heat"]
        _keys(h, ("power_w", "ambient_ua_w_k", "ambient_temperature_k"))
        powers = array(h["power_w"], "heater")
        self.ua = array(h["ambient_ua_w_k"], "UA")
        self.ambient = array(h["ambient_temperature_k"], "ambient temperature", True)
        if np.any(self.ua < 0):
            raise NetworkError("Negative UA")
        self.initial_heat = dict(zip(self.names, map(float, powers)))
        drag = request["drag"]
        _keys(drag, ("rate_per_s", "exported_fraction"))
        self.drag = number(drag["rate_per_s"], "drag rate")
        self.export = number(drag["exported_fraction"], "export fraction")
        if self.drag < 0 or not 0 <= self.export <= 1:
            raise NetworkError("Invalid drag model")
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
                "maximum_mach",
            ),
        )
        self.domain = {k: number(v, k, positive=True) for k, v in domain.items()}
        if self.domain["maximum_mach"] > self.budget["maximum_mach"]:
            raise NetworkError("Mach domain exceeds qualification envelope")
        for k in ("temperature_k", "pressure_pa", "density_kg_m3"):
            if self.domain["min_" + k] >= self.domain["max_" + k]:
                raise NetworkError("Invalid loop domain")
        self.evaluate(np.zeros(3 * self.n + 6), self.initial_heat)

    def heaters(self, values):
        _keys(values, self.names)
        return {k: number(values[k], "heater") for k in self.names}

    def admit(self, rho, T, p, u):
        for k, a in [("density_kg_m3", rho), ("temperature_k", T), ("pressure_pa", p)]:
            if (
                not np.all(np.isfinite(a))
                or np.any(a < self.domain["min_" + k])
                or np.any(a > self.domain["max_" + k])
            ):
                raise InertialGraphError(
                    k + "_domain", "Cell or reconstructed face left declared domain"
                )
        c = np.sqrt(self.eos.gamma * self.eos.R * T)
        if not np.all(np.isfinite(u)) or np.any(abs(u) / c > self.domain["maximum_mach"]):
            raise InertialGraphError("mach_domain", "Conservative loop Mach limit exceeded")
        return c

    def point(self, y):
        n = self.n
        if y.shape != (3 * n + 6,) or not np.all(np.isfinite(y)):
            raise NetworkError("Malformed conservative loop vector")
        dm, dP, dC = y[:n], y[n : 2 * n], y[2 * n : 3 * n]
        m = self.m0 + dm
        P = self.P0 + dP
        if np.any(m <= 0):
            raise InertialGraphError("mass_domain", "Nonpositive loop mass")
        deltaK = (self.u0 * dP + 0.5 * dP * dP / self.m0 - 0.5 * self.u0**2 * dm) / (
            1 + dm / self.m0
        )
        deltaA = dC - deltaK - self.phi * dm
        if np.any(1 + deltaA / self.A0 <= 0):
            raise InertialGraphError(
                "thermal_inventory", "Nonpositive loop internal thermal energy"
            )
        rho = self.rho0 * (1 + dm / self.m0)
        T = self.T0 * (1 + deltaA / self.A0) / (1 + dm / self.m0)
        p = self.p0 + (self.p0 + self.eos.pinf) * deltaA / self.A0
        u = P / m
        c = self.admit(rho, T, p, u)
        ds = self.eos.cv * np.log1p(deltaA / self.A0) - self.eos.cp * np.log1p(dm / self.m0)
        deltaS = dm * self.s0 + m * ds
        return dict(
            m=m,
            P=P,
            rho=rho,
            T=T,
            p=p,
            u=u,
            c=c,
            deltaK=deltaK,
            deltaA=deltaA,
            deltaS=deltaS,
            s=self.s0 + ds,
        )

    def reconstruction(self, rho, T, p, u, cell_phi, face_phi):
        exponent = -(face_phi - cell_phi) / (self.eos.R * T)
        if np.max(abs(exponent)) > self.budget["maximum_hydrostatic_exponent"]:
            raise InertialGraphError(
                "hydrostatic_domain", "Hydrostatic reconstruction exponent exceeded budget"
            )
        factor = np.exp(exponent)
        r = rho * factor
        pressure = p + (p + self.eos.pinf) * np.expm1(exponent)
        c = self.admit(r, T, pressure, u)
        conserved = np.array([r, r * u, r * (self.eos.cv * T + 0.5 * u * u + face_phi)])
        flux = np.array(
            [r * u, r * u * u + pressure, r * u * (self.eos.cp * T + 0.5 * u * u + face_phi)]
        )
        return conserved, flux, abs(u) + c

    def evaluate(self, y, heaters):
        point = self.point(y)
        r, T, p, u = point["rho"], point["T"], point["p"], point["u"]
        left, fl, sl = self.reconstruction(r, T, p, u, self.phi, self.face_phi)
        right, fr, sr = self.reconstruction(
            np.roll(r, -1),
            np.roll(T, -1),
            np.roll(p, -1),
            np.roll(u, -1),
            np.roll(self.phi, -1),
            self.face_phi,
        )
        speed = np.maximum(sl, sr)
        face = self.area * (0.5 * (fl + fr) - 0.5 * speed * (right - left))
        change = np.roll(face, 1, axis=1) - face
        er = -(self.face_phi - self.phi) / (self.eos.R * T)
        el = -(np.roll(self.face_phi, 1) - self.phi) / (self.eos.R * T)
        gravity_force = self.area * (p + self.eos.pinf) * (np.expm1(er) - np.expm1(el))
        drag_force = self.drag * point["P"]
        dissipation = self.drag * point["P"] ** 2 / point["m"]
        ambient = self.ua * (self.ambient - T)
        heat = np.array([heaters[k] for k in self.names]) + ambient
        exported = self.export * dissipation
        change[1] += gravity_force - drag_force
        change[2] += heat - exported
        thermal_rate = change[2] - u * change[1] + (0.5 * u * u - self.phi) * change[0]
        entropy_rate = thermal_rate / T + (point["s"] - self.eos.cp) * change[0]
        external_entropy = sum((heat - ambient - exported) / T + ambient / self.ambient)
        generated = float(sum(entropy_rate) - external_entropy)
        scale = max(1.0, float(sum(abs(entropy_rate))), abs(float(external_entropy)))
        if (
            not math.isfinite(generated)
            or generated < -self.budget["entropy_generation_negative_atol_w_k"] - 1e-10 * scale
        ):
            raise InertialGraphError(
                "entropy_admission",
                "Semidiscrete loop entropy production is negative",
                production_w_k=generated,
            )
        ledgers = np.array(
            [
                sum(heat),
                sum(exported),
                external_entropy,
                generated,
                sum(gravity_force),
                sum(drag_force),
            ]
        )
        result = np.r_[change.ravel(), ledgers]
        if not np.all(np.isfinite(result)):
            raise NetworkError("Nonfinite loop RHS")
        point.update(
            face_flux=face, speed=float(max(speed)), generated=generated, entropy_rate=entropy_rate
        )
        return result, point

    def contract(self):
        return dict(
            family="periodic_conservative_eos_loop_v1",
            cells=self.n,
            periodic=True,
            area_m2=self.area,
            length_m=self.length,
            state_layout="mass,loop momentum,shifted total energy increments; six ledgers",
            total_energy="internal + kinetic + m*g*z",
            spatial_order=1,
            flux="hydrostatic Rusanov",
            temporal_method="SSPRK3",
            ledger_names=list(LEDGERS),
            general_graph=False,
            variable_mass_density=True,
            gravity=True,
            advective_kinetic_transport=True,
            general_entropy_stability_proof=False,
        )


class ConservativeEOSLoop:
    def __init__(self, request, *, max_step_s=0.002):
        self.system = ConservativeLoopSystem(deepcopy(request))
        self.budget = json.loads(BUDGET.read_text())
        self.max_step = number(max_step_s, "max_step", positive=True)
        if self.max_step > self.budget["maximum_step_s"]:
            raise NetworkError("Loop time step exceeds budget")
        self.identity = digest(dict(request=request, max_step_s=self.max_step, method="SSPRK3"))
        self.captured = signature()
        self._state = np.zeros(3 * self.system.n + 6)
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
            raise InertialGraphError(
                "source_drift", "Conservative loop source/budget/runtime changed"
            )

    def _observe(self, y, time):
        s = self.system
        _, p = s.evaluate(y, self._heaters)
        n = s.n
        dm, dP, dC = y[:n], y[n : 2 * n], y[2 * n : 3 * n]
        Q, E, SE, SG, IG, ID = y[-6:]
        mass = float(sum(dm))
        momentum = float(sum(dP) - IG + ID)
        energy = float(sum(dC + s.eos.e0 * dm) - Q + E)
        entropy = float(sum(p["deltaS"]) - SE - SG)
        checks = [
            (mass, self.budget["mass_atol_kg"] + self.budget["mass_rtol"] * sum(s.m0), "mass"),
            (
                momentum,
                self.budget["momentum_atol_kg_m_s"]
                + self.budget["momentum_rtol"] * max(1.0, sum(abs(s.P0)), abs(IG), abs(ID)),
                "momentum",
            ),
            (
                energy,
                self.budget["energy_atol_j"]
                + self.budget["energy_rtol"] * max(1.0, abs(Q), abs(E), sum(abs(dC))),
                "energy",
            ),
            (
                entropy,
                self.budget["entropy_atol_j_k"]
                + self.budget["entropy_rtol"] * max(1.0, abs(SE), abs(SG)),
                "entropy",
            ),
        ]
        for value, limit, name in checks:
            if not math.isfinite(value) or abs(value) > limit:
                raise InertialGraphError(
                    name,
                    "Conservative loop balance failed",
                    observed=float(value),
                    limit=float(limit),
                )
        if SG < -self.budget["entropy_atol_j_k"]:
            raise InertialGraphError("entropy", "Negative integrated loop entropy production")
        return dict(
            time_s=time,
            masses_kg=dict(zip(s.names, map(float, p["m"]))),
            densities_kg_m3=dict(zip(s.names, map(float, p["rho"]))),
            velocities_m_s=dict(zip(s.names, map(float, p["u"]))),
            temperatures_k=dict(zip(s.names, map(float, p["T"]))),
            pressures_pa=dict(zip(s.names, map(float, p["p"]))),
            mean_circulation_kg_s=float(sum(p["P"]) / s.length),
            kinetic_energy_j=float(sum(s.K0 + p["deltaK"])),
            internal_energy_change_j=float(sum(p["deltaA"] + s.eos.e0 * dm)),
            kinetic_energy_change_j=float(sum(p["deltaK"])),
            potential_energy_change_j=float(s.phi @ dm),
            total_energy_change_j=float(sum(dC + s.eos.e0 * dm)),
            entropy_change_j_k=float(sum(p["deltaS"])),
            mass_residual_kg=mass,
            momentum_residual_kg_m_s=momentum,
            energy_residual_j=energy,
            entropy_residual_j_k=entropy,
            entropy_generation_w_k=p["generated"],
            **dict(zip(LEDGERS, map(float, y[-6:]))),
        )

    def snapshot(self):
        return dict(
            revision=self.revision,
            active_heat_power_w=deepcopy(self._heaters),
            history=deepcopy(self._history),
            state=self._observe(self._state, self._time),
            contract=self.system.contract(),
        )

    def residual(self, state, rate):
        y = np.asarray(state, dtype=float)
        dy = np.asarray(rate, dtype=float)
        if dy.shape != y.shape or not np.all(np.isfinite(dy)):
            raise NetworkError("Malformed conservative residual")
        return dy - self.system.evaluate(y, self._heaters)[0]

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
            raise NetworkError("Invalid loop duration/clock")
        c = deepcopy(self)
        transition = None
        if command is not None:
            if len(self._history) >= self.budget["maximum_commands"]:
                raise NetworkError("Loop heater history exhausted")
            powers, cid = command
            left = self._observe(self._state, self._time)
            c._heaters = c.system.heaters(powers)
            transition = dict(time_s=self._time, left=left, right=c._observe(c._state, c._time))
            c._history.append(
                dict(
                    sequence=len(self._history) + 1,
                    time_s=self._time,
                    commit_revision=self.revision + 1,
                    command_id=cid,
                    heat_power_w=deepcopy(c._heaters),
                )
            )
        evaluations = 0
        steps = 0
        maximum_step = 0.0
        maximum_courant = 0.0

        def rhs(y, h, prepared=None):
            nonlocal evaluations, maximum_courant
            evaluations += 1
            if evaluations > self.budget["maximum_rhs_evaluations"]:
                raise InertialGraphError("rhs_budget", "Loop RHS budget exceeded")
            rate, point = prepared if prepared is not None else c.system.evaluate(y, c._heaters)
            courant = h * point["speed"] / c.system.dx
            maximum_courant = max(maximum_courant, courant)
            if courant > self.budget["maximum_courant"] * (1 + 1e-12):
                raise InertialGraphError("courant", "SSPRK stage violated Courant bound")
            return rate

        while c._time < end:
            if steps >= self.budget["maximum_substeps"]:
                raise InertialGraphError("step_budget", "Loop substep budget exhausted")
            prepared = c.system.evaluate(c._state, c._heaters)
            speed = prepared[1]["speed"]
            h = min(
                self.max_step,
                0.9 * self.budget["maximum_courant"] * c.system.dx / speed,
                end - c._time,
            )
            if c._time + h <= c._time:
                raise NetworkError("Loop time cannot progress")
            y = c._state
            first = y + h * rhs(y, h, prepared)
            second = 0.75 * y + 0.25 * (first + h * rhs(first, h))
            new = y / 3 + (2 / 3) * (second + h * rhs(second, h))
            c._state = new
            c._time = end if h == end - c._time else c._time + h
            steps += 1
            maximum_step = max(maximum_step, h)
        c._revision += 1
        result = c.snapshot()
        self._sources()
        self._install(c)
        return dict(
            **result,
            heater_transition=transition,
            work=dict(
                rhs=evaluations,
                substeps=steps,
                maximum_internal_step_s=maximum_step,
                maximum_courant=maximum_courant,
            ),
        )

    def advance(self, duration_s):
        return self._transaction(duration_s)

    def step(self, heat_power_w, duration_s, *, expected_revision, command_id):
        self._sources()
        if type(expected_revision) is not int or expected_revision != self.revision:
            raise InertialGraphError("stale_revision", "Stale conservative loop observation")
        if (
            not isinstance(command_id, str)
            or not 1 <= len(command_id) <= 128
            or not command_id.isascii()
            or any(not (c.isalnum() or c in "-_.:") for c in command_id)
        ):
            raise NetworkError("Invalid loop command ID")
        if any(r["command_id"] == command_id for r in self._history):
            raise NetworkError("Duplicate loop command ID")
        return self._transaction(duration_s, (deepcopy(heat_power_w), command_id))

    def checkpoint(self):
        self._sources()
        p = dict(
            schema_version=1,
            model="conservative-eos-loop-v1",
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
            raise NetworkError("Conservative loop checkpoint digest mismatch")
        _keys(
            p, ("schema_version", "model", "input_sha256", "signature", "coordinates", "snapshot")
        )
        if (
            type(p["schema_version"]) is not int
            or p["schema_version"] != 1
            or p["model"] != "conservative-eos-loop-v1"
            or p["input_sha256"] != self.identity
            or p["signature"] != self.captured
        ):
            raise NetworkError("Conservative loop checkpoint input/source mismatch")
        snap = p["snapshot"]
        _keys(snap, ("revision", "active_heat_power_w", "history", "state", "contract"))
        if not isinstance(snap["state"], dict) or "time_s" not in snap["state"]:
            raise NetworkError("Malformed conservative loop observation")
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
            raise NetworkError("Conservative loop checkpoint clock/history mismatch")
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
                raise NetworkError("Conservative loop checkpoint receipt path mismatch")
            ids.add(cid)
            active = self.system.heaters(r["heat_power_w"])
            last_time, last_rev = when, commit
        values = p["coordinates"]
        if not isinstance(values, list) or len(values) != 3 * self.system.n + 6:
            raise NetworkError("Conservative loop checkpoint layout mismatch")
        vector = np.array([number(v, "coordinate") for v in values])
        if time == 0 and np.any(vector != 0):
            raise NetworkError("Nonzero initial conservative loop stocks")
        c = deepcopy(self)
        c._state = vector
        c._time = time
        c._revision = rev
        c._history = deepcopy(history)
        c._heaters = active
        fresh = c.snapshot()
        if fresh != snap:
            raise NetworkError("Conservative loop checkpoint observation mismatch")
        self._sources()
        self._install(c)
        return fresh
