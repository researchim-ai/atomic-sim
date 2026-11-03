"""Two independent periodic Euler loops coupled by caloric finite-capacity walls.

Mass and momentum do not cross wall ports. Thermal ports use a single signed
heat flow in the fluid and wall energy equations. No branched hydraulic node.
"""

import json
import math
from copy import deepcopy
from pathlib import Path

import numpy as np

from core.network import NetworkError, number
from scripts.conservative_eos_loop import ConservativeEOSLoop
from scripts.conservative_eos_loop import signature as loop_signature
from scripts.inertial_graph import InertialGraphError, digest
from scripts.solve_hydraulic_graph import _keys, sha

BUDGET = Path("validation/ai05_eos_wall_budget_v1.json")
WALL_LEDGERS = (
    "fluid_wall_heat_j",
    "fluid_wall_entropy_j_k",
    "wall_external_heat_j",
    "wall_external_entropy_j_k",
    "interface_generated_entropy_j_k",
)


def signature():
    result = loop_signature()
    result["drivers"]["scripts/coupled_eos_wall.py"] = sha(__file__)
    result["budgets"][str(BUDGET)] = sha(BUDGET)
    return result


class CoupledEOSWall:
    def __init__(self, request, *, max_step_s=0.001):
        _keys(request, ("format", "schema_version", "loops", "walls"))
        if (
            request["format"] != "atomic-coupled-eos-wall"
            or type(request["schema_version"]) is not int
            or request["schema_version"] != 1
        ):
            raise NetworkError("Expected coupled EOS wall schema1")
        if not isinstance(request["loops"], list) or len(request["loops"]) != 2:
            raise NetworkError("Exactly two independent loops required")
        self.budget = json.loads(BUDGET.read_text())
        self.request = deepcopy(request)
        self.loops = [ConservativeEOSLoop(r) for r in request["loops"]]
        if sum(e.system.n for e in self.loops) > self.budget["maximum_total_cells"]:
            raise NetworkError("Assembly cell budget exceeded")
        self.max_step = number(max_step_s, "max step", positive=True)
        if self.max_step > self.budget["maximum_step_s"]:
            raise NetworkError("Assembly step exceeds budget")
        walls = request["walls"]
        if not isinstance(walls, list) or not 1 <= len(walls) <= self.budget["maximum_walls"]:
            raise NetworkError("Wall count outside budget")
        self.ports = []
        capacity = []
        initial = []
        conductance = []
        ambient_ua = []
        ambient = []
        for wall in walls:
            _keys(
                wall,
                (
                    "cells",
                    "capacity_j_k",
                    "initial_temperature_k",
                    "conductance_w_k",
                    "ambient_ua_w_k",
                    "ambient_temperature_k",
                ),
            )
            ports = wall["cells"]
            ua = wall["conductance_w_k"]
            if (
                not isinstance(ports, list)
                or len(ports) != 2
                or not isinstance(ua, list)
                or len(ua) != 2
            ):
                raise NetworkError("Wall requires one cell/conductance per loop")
            for i, cell in enumerate(ports):
                if type(cell) is not int or not 0 <= cell < self.loops[i].system.n:
                    raise NetworkError("Wall port cell does not exist")
            u = [number(v, "wall conductance", positive=True) for v in ua]
            C = number(wall["capacity_j_k"], "wall capacity", positive=True)
            T = number(wall["initial_temperature_k"], "wall temperature", positive=True)
            U = number(wall["ambient_ua_w_k"], "wall ambient UA")
            Ta = number(wall["ambient_temperature_k"], "wall ambient temperature", positive=True)
            if U < 0 or not math.isfinite(C * T):
                raise NetworkError("Invalid wall thermal inventory")
            for e in self.loops:
                if (
                    not e.system.domain["min_temperature_k"]
                    <= T
                    <= e.system.domain["max_temperature_k"]
                ):
                    raise NetworkError("Wall temperature outside shared declared domain")
            self.ports.append(tuple(ports))
            capacity.append(C)
            initial.append(T)
            conductance.append(u)
            ambient_ua.append(U)
            ambient.append(Ta)
        self.capacity = np.array(capacity)
        self.T0 = np.array(initial)
        self.ua = np.array(conductance)
        self.ambient_ua = np.array(ambient_ua)
        self.ambient = np.array(ambient)
        self.offsets = [0]
        for e in self.loops:
            self.offsets.append(self.offsets[-1] + 3 * e.system.n + 6)
        self.wall_start = self.offsets[-1]
        self._state = np.zeros(self.wall_start + len(walls) + 5)
        self._time = 0.0
        self._revision = 0
        self.identity = digest(dict(request=request, max_step_s=self.max_step, method="SSPRK3"))
        self.captured = signature()
        self.snapshot()

    def _sources(self):
        if signature() != self.captured:
            raise InertialGraphError("source_drift", "Coupled wall source/budget/runtime changed")

    def _parts(self, y):
        if y.shape != self._state.shape or not np.all(np.isfinite(y)):
            raise NetworkError("Malformed assembly coordinates")
        local = [y[self.offsets[i] : self.offsets[i + 1]] for i in range(2)]
        delta = y[self.wall_start : -5]
        T = self.T0 + delta / self.capacity
        for e in self.loops:
            if np.any(T < e.system.domain["min_temperature_k"]) or np.any(
                T > e.system.domain["max_temperature_k"]
            ):
                raise InertialGraphError("wall_domain", "Wall left shared temperature domain")
        return local, delta, T

    def evaluate(self, y):
        local, delta, Tw = self._parts(y)
        points = [e.system.point(z) for e, z in zip(self.loops, local)]
        heats = [np.zeros(e.system.n) for e in self.loops]
        loads = [np.zeros(e.system.n) for e in self.loops]
        wall_rate = self.ambient_ua * (self.ambient - Tw)
        external = wall_rate.copy()
        entropy_flow = 0.0
        production = float(sum(external * (1 / Tw - 1 / self.ambient)))
        total_heat = 0.0
        for w, ports in enumerate(self.ports):
            for i, cell in enumerate(ports):
                Tf = points[i]["T"][cell]
                Q = self.ua[w, i] * (Tw[w] - Tf)
                heats[i][cell] += Q
                loads[i][cell] += self.ua[w, i]
                wall_rate[w] -= Q
                total_heat += Q
                entropy_flow += Q / Tf
                production += Q * (1 / Tf - 1 / Tw[w])
        rates = []
        max_ratio = 0.0
        for i, (e, z) in enumerate(zip(self.loops, local)):
            powers = {name: e._heaters[name] + heats[i][j] for j, name in enumerate(e.system.names)}
            rate, point = e.system.evaluate(z, powers)
            rates.append(rate)
            max_ratio = max(max_ratio, point["speed"] / e.system.dx / e.budget["maximum_courant"])
            thermal = (loads[i] + e.system.ua) / (points[i]["m"] * e.system.eos.cv)
            max_ratio = max(max_ratio, float(max(thermal)) / self.budget["maximum_thermal_number"])
        max_ratio = max(
            max_ratio,
            float(max((self.ua.sum(axis=1) + self.ambient_ua) / self.capacity))
            / self.budget["maximum_thermal_number"],
        )
        ledgers = [
            total_heat,
            entropy_flow,
            float(sum(external)),
            float(sum(external / self.ambient)),
            production,
        ]
        result = np.concatenate([*rates, wall_rate, np.array(ledgers)])
        if not np.all(np.isfinite(result)) or production < -1e-12:
            raise InertialGraphError("wall_entropy", "Invalid wall heat/entropy production")
        return result, max_ratio

    def snapshot(self):
        local, delta, Tw = self._parts(self._state)
        self.evaluate(self._state)
        observations = [e._observe(z, self._time) for e, z in zip(self.loops, local)]
        Qf, Sf, Qa, Sa, Sg = self._state[-5:]
        energy = sum(o["total_energy_change_j"] for o in observations) + float(sum(delta))
        external = sum(o["external_heat_j"] for o in observations) - Qf + Qa
        exported = sum(o["exported_drag_heat_j"] for o in observations)
        entropy = sum(o["entropy_change_j_k"] for o in observations) + float(
            sum(self.capacity * np.log1p(delta / (self.capacity * self.T0)))
        )
        external_entropy = sum(o["external_entropy_j_k"] for o in observations) - Sf + Sa
        generated = sum(o["generated_entropy_j_k"] for o in observations) + Sg
        re = energy - external + exported
        rs = entropy - external_entropy - generated
        for residual, limit, label in [
            (
                re,
                self.budget["energy_atol_j"]
                + self.budget["energy_rtol"] * max(1.0, abs(energy), abs(external), abs(exported)),
                "energy",
            ),
            (
                rs,
                self.budget["entropy_atol_j_k"]
                + self.budget["entropy_rtol"]
                * max(1.0, abs(entropy), abs(external_entropy), abs(generated)),
                "entropy",
            ),
        ]:
            if not math.isfinite(residual) or abs(residual) > limit:
                raise InertialGraphError(
                    label, "Assembly balance failed", observed=float(residual), limit=float(limit)
                )
        if generated < -self.budget["entropy_atol_j_k"]:
            raise InertialGraphError("entropy", "Negative assembly entropy production")
        return dict(
            time_s=self._time,
            revision=self._revision,
            loops=observations,
            wall_temperatures_k=Tw.tolist(),
            wall_energy_changes_j=delta.tolist(),
            total_energy_change_j=float(energy),
            external_heat_j=float(external),
            exported_drag_heat_j=float(exported),
            entropy_change_j_k=float(entropy),
            external_entropy_j_k=float(external_entropy),
            generated_entropy_j_k=float(generated),
            energy_residual_j=float(re),
            entropy_residual_j_k=float(rs),
            wall_ledgers=dict(zip(WALL_LEDGERS, map(float, self._state[-5:]))),
        )

    def residual(self, state, rate):
        y = np.asarray(state, dtype=float)
        dy = np.asarray(rate, dtype=float)
        if dy.shape != y.shape or not np.all(np.isfinite(dy)):
            raise NetworkError("Malformed assembly residual")
        return dy - self.evaluate(y)[0]

    def advance(self, duration_s):
        self._sources()
        duration = number(duration_s, "duration", positive=True)
        end = self._time + duration
        if (
            duration > self.budget["maximum_duration_s"]
            or not math.isfinite(end)
            or end <= self._time
        ):
            raise NetworkError("Invalid assembly duration/clock")
        c = deepcopy(self)
        steps = rhs = 0
        maximum_step = maximum_number = 0.0

        def evaluate(y):
            nonlocal rhs
            rhs += 1
            if rhs > self.budget["maximum_rhs_evaluations"]:
                raise InertialGraphError("rhs_budget", "Assembly RHS budget exhausted")
            return c.evaluate(y)

        while c._time < end:
            if steps >= self.budget["maximum_substeps"]:
                raise InertialGraphError("step_budget", "Assembly step budget exhausted")
            rate, ratio = evaluate(c._state)
            h = min(self.max_step, 0.9 / ratio, end - c._time)
            if c._time + h <= c._time:
                raise NetworkError("Assembly time cannot progress")
            y = c._state
            first = y + h * rate
            r1, k1 = evaluate(first)
            second = 0.75 * y + 0.25 * (first + h * r1)
            r2, k2 = evaluate(second)
            stage = max(ratio, k1, k2) * h
            if stage > 1 + 1e-12:
                raise InertialGraphError("stability", "Assembly stage exceeds CFL/thermal bound")
            maximum_number = max(maximum_number, stage)
            c._state = y / 3 + (2 / 3) * (second + h * r2)
            c._time = end if h == end - c._time else c._time + h
            maximum_step = max(maximum_step, h)
            steps += 1
        c._revision += 1
        result = c.snapshot()
        self._sources()
        self._state, self._time, self._revision = c._state, c._time, c._revision
        return dict(
            **result,
            work=dict(
                rhs=rhs,
                substeps=steps,
                maximum_internal_step_s=maximum_step,
                maximum_stability_fraction=maximum_number,
            ),
        )

    def checkpoint(self):
        self._sources()
        p = dict(
            schema_version=1,
            model="coupled-eos-wall-v1",
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
            raise NetworkError("Assembly checkpoint digest mismatch")
        _keys(
            p, ("schema_version", "model", "input_sha256", "signature", "coordinates", "snapshot")
        )
        if (
            type(p["schema_version"]) is not int
            or p["schema_version"] != 1
            or p["model"] != "coupled-eos-wall-v1"
            or p["input_sha256"] != self.identity
            or p["signature"] != self.captured
        ):
            raise NetworkError("Assembly checkpoint identity mismatch")
        if not isinstance(p["coordinates"], list) or len(p["coordinates"]) != len(self._state):
            raise NetworkError("Assembly checkpoint layout mismatch")
        c = deepcopy(self)
        c._state = np.array([number(v, "coordinate") for v in p["coordinates"]])
        snap = p["snapshot"]
        if not isinstance(snap, dict) or "time_s" not in snap or "revision" not in snap:
            raise NetworkError("Malformed assembly checkpoint snapshot")
        c._time = number(snap["time_s"], "time")
        c._revision = snap["revision"]
        if (
            type(c._revision) is not int
            or c._revision < 0
            or c._time < 0
            or (c._revision == 0) != (c._time == 0)
            or (c._time == 0 and np.any(c._state != 0))
        ):
            raise NetworkError("Assembly checkpoint clock mismatch")
        fresh = c.snapshot()
        if fresh != snap:
            raise NetworkError("Assembly checkpoint observation mismatch")
        self._sources()
        self._state, self._time, self._revision = c._state, c._time, c._revision
        return fresh
