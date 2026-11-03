"""Joint finite-inventory Euler/wall assembly with explicit thermal ports.

Two hydraulically isolated circuits retain their terminal chambers. Heat
passes through finite-capacity walls. Full-mask commands change external
heaters transactionally; internal interface flows remain shared and signed.
"""

import json
import math
import threading
from copy import copy, deepcopy
from functools import wraps

import numpy as np

from atomic_model_runtime.resources import resource
from core.network import NetworkError, identifier, number
from scripts.inertial_graph import InertialGraphError, digest
from scripts.solve_hydraulic_graph import _keys, sha
from scripts.terminal_eos_chambers import TerminalEOSChambers
from scripts.terminal_eos_chambers import signature as terminal_signature

BUDGET = resource("validation/ai05_controlled_assembly_budget_v1.json")
CIRCUITS = ("primary", "secondary")
WALL_LEDGERS = (
    "fluid_wall_heat_j",
    "fluid_wall_entropy_j_k",
    "wall_external_heat_j",
    "wall_external_entropy_j_k",
    "interface_generated_entropy_j_k",
)


def _locked(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return call


def signature():
    result = terminal_signature()
    result["drivers"]["scripts/controlled_eos_assembly.py"] = sha(__file__)
    result["budgets"][str(BUDGET)] = sha(BUDGET)
    return result


class ControlledEOSAssembly:
    def __init__(self, request, *, max_step_s=0.0005):
        _keys(request, ("format", "schema_version", "circuits", "walls"))
        if (
            request["format"] != "atomic-controlled-eos-assembly"
            or type(request["schema_version"]) is not int
            or request["schema_version"] != 1
        ):
            raise NetworkError("Expected controlled EOS assembly schema1")
        _keys(request["circuits"], CIRCUITS)
        self.budget = json.loads(BUDGET.read_text())
        self.request = deepcopy(request)
        self.circuits = [TerminalEOSChambers(request["circuits"][name]) for name in CIRCUITS]
        if sum(e.channel.system.n for e in self.circuits) > self.budget["maximum_total_cells"]:
            raise NetworkError("Assembly total cell budget")
        self.max_step = number(max_step_s, "max step", positive=True)
        if self.max_step > self.budget["maximum_step_s"]:
            raise NetworkError("Assembly step exceeds budget")
        self._heaters = {}
        for name, e in zip(CIRCUITS, self.circuits):
            for cell, value in e.channel._heaters.items():
                self._heaters[f"{name}.channel.{cell}"] = value
            for i, side in enumerate(["left", "right"]):
                self._heaters[f"{name}.chamber.{side}"] = float(e.heaters[i])
        walls = request["walls"]
        if not isinstance(walls, list) or not 1 <= len(walls) <= self.budget["maximum_walls"]:
            raise NetworkError("Wall count outside assembly budget")
        self.wall_ids = []
        self.ports = []
        C = []
        T = []
        UA = []
        Ua = []
        Ta = []
        for w in walls:
            _keys(
                w,
                (
                    "id",
                    "ports",
                    "capacity_j_k",
                    "initial_temperature_k",
                    "conductance_w_k",
                    "heat_power_w",
                    "ambient_ua_w_k",
                    "ambient_temperature_k",
                ),
            )
            wid = identifier(w["id"])
            if wid in self.wall_ids:
                raise NetworkError("Duplicate wall ID")
            self.wall_ids.append(wid)
            if (
                not isinstance(w["ports"], list)
                or len(w["ports"]) != 2
                or not isinstance(w["conductance_w_k"], list)
                or len(w["conductance_w_k"]) != 2
            ):
                raise NetworkError("Two typed thermal ports and conductances required")
            ports = [self._port(p) for p in w["ports"]]
            if {p[0] for p in ports} != {0, 1}:
                raise NetworkError("Each wall must connect both circuits")
            self.ports.append(ports)
            C.append(number(w["capacity_j_k"], "wall capacity", positive=True))
            T.append(number(w["initial_temperature_k"], "wall temperature", positive=True))
            UA.append([number(v, "wall UA", positive=True) for v in w["conductance_w_k"]])
            Ua.append(number(w["ambient_ua_w_k"], "wall ambient UA"))
            Ta.append(number(w["ambient_temperature_k"], "wall ambient temperature", positive=True))
            if Ua[-1] < 0:
                raise NetworkError("Negative wall ambient UA")
            self._heaters[f"wall.{wid}"] = number(w["heat_power_w"], "wall heater")
        self.capacity = np.array(C)
        self.T0 = np.array(T)
        self.ua = np.array(UA)
        self.ambient_ua = np.array(Ua)
        self.ambient = np.array(Ta)
        if not np.all(np.isfinite(self.capacity * self.T0)):
            raise NetworkError("Nonfinite wall inventory")
        self.initial_heaters = deepcopy(self._heaters)
        self.offsets = [0]
        for e in self.circuits:
            self.offsets.append(self.offsets[-1] + len(e._state))
        self.wall_start = self.offsets[-1]
        self._state = np.zeros(self.wall_start + len(walls) + 5)
        self._time = 0.0
        self._revision = 0
        self._history = []
        self.identity = digest(dict(request=request, max_step_s=self.max_step, method="SSPRK3"))
        self.captured = signature()
        self._lock = threading.RLock()
        self.snapshot()

    def __deepcopy__(self, memo):
        result = type(self).__new__(type(self))
        memo[id(self)] = result
        for key, value in self.__dict__.items():
            if key != "_lock":
                setattr(result, key, deepcopy(value, memo))
        result._lock = threading.RLock()
        return result

    @property
    def revision(self):
        return self._revision

    def _port(self, p):
        if not isinstance(p, dict) or p.get("component") not in ["channel", "chamber"]:
            raise NetworkError("Unknown thermal component port")
        _keys(
            p,
            ("circuit", "component", "cell")
            if p["component"] == "channel"
            else ("circuit", "component", "side"),
        )
        if not isinstance(p["circuit"], str) or p["circuit"] not in CIRCUITS:
            raise NetworkError("Unknown circuit")
        i = CIRCUITS.index(p["circuit"])
        if p["component"] == "channel":
            j = p["cell"]
            if type(j) is not int or not 0 <= j < self.circuits[i].channel.system.n:
                raise NetworkError("Unknown channel heat port")
        else:
            if p["side"] not in ["left", "right"]:
                raise NetworkError("Unknown chamber heat port")
            j = ["left", "right"].index(p["side"])
        return i, p["component"], j

    def _sources(self):
        if signature() != self.captured:
            raise InertialGraphError("source_drift", "Mixed assembly source/budget/runtime changed")

    def _mask(self, values):
        _keys(values, tuple(self.initial_heaters))
        return {key: number(values[key], "heater power") for key in self.initial_heaters}

    def _parts(self, y):
        if y.shape != self._state.shape or not np.all(np.isfinite(y)):
            raise NetworkError("Malformed mixed assembly coordinates")
        local = [y[self.offsets[i] : self.offsets[i + 1]] for i in range(2)]
        delta = y[self.wall_start : -5]
        Tw = self.T0 + delta / self.capacity
        for e in self.circuits:
            d = e.channel.system.domain
            if np.any(Tw < d["min_temperature_k"]) or np.any(Tw > d["max_temperature_k"]):
                raise InertialGraphError("wall_domain", "Wall left shared temperature domain")
        points = [e.point(z) for e, z in zip(self.circuits, local)]
        channels = [
            e.channel.system.point(z[: 3 * e.channel.system.n + 6])
            for e, z in zip(self.circuits, local)
        ]
        return local, delta, Tw, points, channels

    def _view(self, i, channel_heat=None, chamber_heat=None):
        e = copy(self.circuits[i])
        e.channel = copy(e.channel)
        name = CIRCUITS[i]
        e.channel._heaters = {
            cell: self._heaters[f"{name}.channel.{cell}"]
            + (0.0 if channel_heat is None else channel_heat[j])
            for j, cell in enumerate(e.channel.system.names)
        }
        e.heaters = np.array(
            [self._heaters[f"{name}.chamber.{side}"] for side in ["left", "right"]]
        )
        if chamber_heat is not None:
            e.heaters = e.heaters + chamber_heat
        return e

    @_locked
    def evaluate(self, y):
        local, delta, Tw, points, channels = self._parts(y)
        ch_heat = [np.zeros(e.channel.system.n) for e in self.circuits]
        tank_heat = [np.zeros(2) for _ in self.circuits]
        ch_load = [e.channel.system.ua.copy() for e in self.circuits]
        tank_load = [e.ua.copy() for e in self.circuits]
        ambient = self.ambient_ua * (self.ambient - Tw)
        heaters = np.array([self._heaters[f"wall.{wid}"] for wid in self.wall_ids])
        wall_rate = heaters + ambient
        qwall = swall = 0.0
        generated = float(sum(ambient * (1 / Tw - 1 / self.ambient)))
        for w, ports in enumerate(self.ports):
            for k, (i, kind, j) in enumerate(ports):
                Tf = (channels[i] if kind == "channel" else points[i])["T"][j]
                Q = self.ua[w, k] * (Tw[w] - Tf)
                (ch_heat if kind == "channel" else tank_heat)[i][j] += Q
                (ch_load if kind == "channel" else tank_load)[i][j] += self.ua[w, k]
                wall_rate[w] -= Q
                qwall += Q
                swall += Q / Tf
                generated += Q * (1 / Tf - 1 / Tw[w])
        rates = []
        ratio = 0.0
        for i, z in enumerate(local):
            e = self._view(i, ch_heat[i], tank_heat[i])
            rate, k, _ = e.evaluate(z)
            rates.append(rate)
            cv = e.channel.system.eos.cv
            ratio = max(
                ratio,
                k,
                float(max(ch_load[i] / (channels[i]["m"] * cv)))
                / self.budget["maximum_thermal_number"],
                float(max(tank_load[i] / (points[i]["m"] * cv)))
                / self.budget["maximum_thermal_number"],
            )
        ratio = max(
            ratio,
            float(max((self.ua.sum(axis=1) + self.ambient_ua) / self.capacity))
            / self.budget["maximum_thermal_number"],
        )
        ledgers = [
            qwall,
            swall,
            float(sum(heaters + ambient)),
            float(sum(heaters / Tw + ambient / self.ambient)),
            generated,
        ]
        result = np.concatenate([*rates, wall_rate, np.array(ledgers)])
        if not np.all(np.isfinite(result)) or not math.isfinite(ratio) or generated < -1e-12:
            raise InertialGraphError("wall_entropy", "Invalid mixed assembly heat/entropy rates")
        return result, ratio

    @_locked
    def snapshot(self):
        local, delta, Tw, points, channels = self._parts(self._state)
        self.evaluate(self._state)
        states = {}
        temperatures = {}
        for i, (name, z) in enumerate(zip(CIRCUITS, local)):
            e = self._view(i)
            e._state = z
            e._time = self._time
            e._revision = self._revision
            states[name] = e.snapshot()
            for cell, T in zip(e.channel.system.names, channels[i]["T"]):
                temperatures[f"{name}.channel.{cell}"] = float(T)
            for side, T in zip(["left", "right"], points[i]["T"]):
                temperatures[f"{name}.chamber.{side}"] = float(T)
        for wid, T in zip(self.wall_ids, Tw):
            temperatures[f"wall.{wid}"] = float(T)
        Qf, Sf, Qa, Sa, Sg = self._state[-5:]
        energy = sum(o["total_energy_change_j"] for o in states.values()) + float(sum(delta))
        external = sum(o["external_heat_j"] for o in states.values()) - Qf + Qa
        exported = sum(o["channel"]["exported_drag_heat_j"] for o in states.values())
        entropy = sum(o["entropy_change_j_k"] for o in states.values()) + float(
            sum(self.capacity * np.log1p(delta / (self.capacity * self.T0)))
        )
        external_entropy = sum(o["external_entropy_j_k"] for o in states.values()) - Sf + Sa
        generation = sum(o["generated_entropy_j_k"] for o in states.values()) + Sg
        re = energy - external + exported
        rs = entropy - external_entropy - generation
        for value, limit, label in [
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
                * max(1.0, abs(entropy), abs(external_entropy), abs(generation)),
                "entropy",
            ),
        ]:
            if not math.isfinite(value) or abs(value) > limit:
                raise InertialGraphError(
                    label,
                    "Mixed assembly balance failed",
                    observed=float(value),
                    limit=float(limit),
                )
        if generation < -self.budget["entropy_atol_j_k"]:
            raise InertialGraphError("entropy", "Negative mixed assembly generation")
        return dict(
            time_s=self._time,
            revision=self.revision,
            circuits=states,
            wall_temperatures_k=dict(zip(self.wall_ids, map(float, Tw))),
            port_temperatures_k=temperatures,
            active_heat_power_w=deepcopy(self._heaters),
            history=deepcopy(self._history),
            total_energy_change_j=float(energy),
            external_heat_j=float(external),
            exported_drag_heat_j=float(exported),
            entropy_change_j_k=float(entropy),
            external_entropy_j_k=float(external_entropy),
            generated_entropy_j_k=float(generation),
            energy_residual_j=float(re),
            entropy_residual_j_k=float(rs),
            wall_ledgers=dict(zip(WALL_LEDGERS, map(float, self._state[-5:]))),
        )

    @_locked
    def residual(self, state, rate):
        y = np.asarray(state, dtype=float)
        dy = np.asarray(rate, dtype=float)
        if dy.shape != y.shape or not np.all(np.isfinite(dy)):
            raise NetworkError("Malformed mixed assembly residual")
        return dy - self.evaluate(y)[0]

    def _install(self, c):
        self._state, self._time, self._revision, self._history, self._heaters = (
            c._state,
            c._time,
            c._revision,
            c._history,
            c._heaters,
        )

    @_locked
    def _transaction(self, duration_s, command=None):
        self._sources()
        duration = number(duration_s, "duration", positive=True)
        end = self._time + duration
        if (
            duration > self.budget["maximum_duration_s"]
            or not math.isfinite(end)
            or end <= self._time
        ):
            raise NetworkError("Invalid mixed assembly duration/clock")
        c = deepcopy(self)
        transition = None
        if command is not None:
            if len(c._history) >= self.budget["maximum_commands"]:
                raise NetworkError("Heater history exhausted")
            powers, cid = command
            left = c.snapshot()
            c._heaters = c._mask(powers)
            c._history.append(
                dict(
                    sequence=len(c._history) + 1,
                    time_s=c._time,
                    commit_revision=c.revision + 1,
                    command_id=cid,
                    heat_power_w=deepcopy(c._heaters),
                )
            )
            transition = dict(time_s=c._time, left=left, right=c.snapshot())
        steps = rhs = 0
        maxstep = stability = 0.0

        def evaluate(y):
            nonlocal rhs
            rhs += 1
            if rhs > self.budget["maximum_rhs_evaluations"]:
                raise InertialGraphError("rhs_budget", "Mixed assembly RHS budget exhausted")
            return c.evaluate(y)

        while c._time < end:
            if steps >= self.budget["maximum_substeps"]:
                raise InertialGraphError("step_budget", "Mixed assembly step budget exhausted")
            rate, k = evaluate(c._state)
            h = min(self.max_step, 0.9 / k, end - c._time)
            if c._time + h <= c._time:
                raise NetworkError("Mixed assembly time cannot progress")
            y = c._state
            first = y + h * rate
            r1, k1 = evaluate(first)
            second = 0.75 * y + 0.25 * (first + h * r1)
            r2, k2 = evaluate(second)
            stage = h * max(k, k1, k2)
            if stage > 1 + 1e-12:
                raise InertialGraphError("stability", "Mixed assembly stage bound exceeded")
            c._state = y / 3 + (2 / 3) * (second + h * r2)
            c._time = end if h == end - c._time else c._time + h
            maxstep = max(maxstep, h)
            stability = max(stability, stage)
            steps += 1
        c._revision += 1
        result = c.snapshot()
        self._sources()
        self._install(c)
        return dict(
            **result,
            heater_transition=transition,
            work=dict(
                rhs=rhs,
                substeps=steps,
                maximum_internal_step_s=maxstep,
                maximum_stability_fraction=stability,
            ),
        )

    def advance(self, duration_s):
        return self._transaction(duration_s)

    @_locked
    def step(self, heat_power_w, duration_s, *, expected_revision, command_id):
        self._sources()
        if type(expected_revision) is not int or expected_revision != self.revision:
            raise InertialGraphError("stale_revision", "Stale assembly observation")
        if (
            not isinstance(command_id, str)
            or not 1 <= len(command_id) <= 128
            or not command_id.isascii()
            or any(not (c.isalnum() or c in "-_.:") for c in command_id)
        ):
            raise NetworkError("Invalid command ID")
        if any(r["command_id"] == command_id for r in self._history):
            raise NetworkError("Duplicate command ID")
        return self._transaction(duration_s, (deepcopy(heat_power_w), command_id))

    @_locked
    def checkpoint(self):
        self._sources()
        p = dict(
            schema_version=1,
            model="controlled-eos-assembly-v1",
            input_sha256=self.identity,
            signature=deepcopy(self.captured),
            coordinates=self._state.tolist(),
            snapshot=self.snapshot(),
        )
        return dict(payload=p, sha256=digest(p))

    @_locked
    def restore(self, checkpoint):
        self._sources()
        _keys(checkpoint, ("payload", "sha256"))
        p = checkpoint["payload"]
        if digest(p) != checkpoint["sha256"]:
            raise NetworkError("Mixed assembly checkpoint digest mismatch")
        _keys(
            p, ("schema_version", "model", "input_sha256", "signature", "coordinates", "snapshot")
        )
        if (
            type(p["schema_version"]) is not int
            or p["schema_version"] != 1
            or p["model"] != "controlled-eos-assembly-v1"
            or p["input_sha256"] != self.identity
            or p["signature"] != self.captured
        ):
            raise NetworkError("Mixed assembly checkpoint identity mismatch")
        snap = p["snapshot"]
        if (
            not isinstance(snap, dict)
            or "time_s" not in snap
            or "revision" not in snap
            or "history" not in snap
        ):
            raise NetworkError("Malformed mixed assembly observation")
        time = number(snap["time_s"], "time")
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
            raise NetworkError("Mixed assembly clock/history mismatch")
        active = deepcopy(self.initial_heaters)
        ids = set()
        last_time = 0.0
        last_rev = 0
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
                raise NetworkError("Mixed assembly receipt path mismatch")
            ids.add(cid)
            active = self._mask(r["heat_power_w"])
            last_time, last_rev = when, commit
        values = p["coordinates"]
        if not isinstance(values, list) or len(values) != len(self._state):
            raise NetworkError("Mixed assembly checkpoint layout mismatch")
        vector = np.array([number(v, "coordinate") for v in values])
        if time == 0 and np.any(vector != 0):
            raise NetworkError("Nonzero initial mixed stocks")
        c = deepcopy(self)
        c._state = vector
        c._time = time
        c._revision = rev
        c._history = deepcopy(history)
        c._heaters = active
        fresh = c.snapshot()
        if fresh != snap:
            raise NetworkError("Mixed assembly observation mismatch")
        self._sources()
        self._install(c)
        return fresh
