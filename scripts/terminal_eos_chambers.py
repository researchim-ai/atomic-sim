"""Finite same-area terminal Euler volumes and stationary reflecting endwalls.

Terminal chambers retain mass, momentum and total energy, not just p/T.
The channel and chambers use one signed numerical flux on each interface.
This is a 1-D nonuniform-volume closure, not an abrupt-area plenum or junction.
"""

import json
import math
from copy import copy, deepcopy

import numpy as np

from atomic_model_runtime.resources import resource
from core.network import NetworkError, number
from scripts.inertial_graph import InertialGraphError, digest
from scripts.open_eos_channel import OpenEOSChannel
from scripts.open_eos_channel import signature as channel_signature
from scripts.solve_hydraulic_graph import _keys, sha

BUDGET = resource("validation/ai05_terminal_chambers_budget_v1.json")
CHAMBER_LEDGERS = (
    "chamber_external_heat_j",
    "chamber_external_entropy_j_k",
    "chamber_generated_entropy_j_k",
    "endwall_impulse_kg_m_s",
)


def signature():
    result = channel_signature()
    result["drivers"]["scripts/terminal_eos_chambers.py"] = sha(__file__)
    result["budgets"][str(BUDGET)] = sha(BUDGET)
    return result


class TerminalEOSChambers:
    def __init__(self, request, *, max_step_s=0.0005):
        _keys(request, ("format", "schema_version", "channel", "chambers"))
        if (
            request["format"] != "atomic-terminal-eos-chambers"
            or type(request["schema_version"]) is not int
            or request["schema_version"] != 1
        ):
            raise NetworkError("Expected terminal chambers schema1")
        self.budget = json.loads(BUDGET.read_text())
        self.request = deepcopy(request)
        channel = deepcopy(request["channel"])
        _keys(
            channel,
            ("format", "schema_version", "fluid", "geometry", "initial", "heat", "drag", "domain"),
        )
        if (
            channel["format"] != "atomic-conservative-eos-loop"
            or type(channel["schema_version"]) is not int
            or channel["schema_version"] != 1
        ):
            raise NetworkError("Expected versioned bare channel data")
        chambers = request["chambers"]
        if not isinstance(chambers, list) or len(chambers) != 2:
            raise NetworkError("Two terminal chambers required")
        self.specs = []
        boundaries = {}
        for side, spec in zip(["left", "right"], chambers):
            _keys(
                spec,
                (
                    "volume_m3",
                    "initial_pressure_pa",
                    "initial_temperature_k",
                    "initial_velocity_m_s",
                    "elevation_m",
                    "heat_power_w",
                    "ambient_ua_w_k",
                    "ambient_temperature_k",
                ),
            )
            values = {
                k: number(
                    v,
                    k,
                    positive=k
                    in [
                        "volume_m3",
                        "initial_pressure_pa",
                        "initial_temperature_k",
                        "ambient_temperature_k",
                    ],
                )
                for k, v in spec.items()
            }
            if values["ambient_ua_w_k"] < 0:
                raise NetworkError("Negative chamber UA")
            self.specs.append(values)
            boundaries[side] = dict(
                pressure_pa=values["initial_pressure_pa"],
                temperature_k=values["initial_temperature_k"],
                velocity_m_s=values["initial_velocity_m_s"],
                elevation_m=values["elevation_m"],
            )
        channel["format"] = "atomic-open-eos-channel"
        channel["boundaries"] = boundaries
        self.channel = OpenEOSChannel(channel)
        s = self.channel.system
        if s.n > self.budget["maximum_cells"]:
            raise NetworkError("Assembly cell budget exceeded")
        self.max_step = number(max_step_s, "max step", positive=True)
        if self.max_step > self.budget["maximum_step_s"]:
            raise NetworkError("Assembly step exceeds budget")
        self.V = np.array([c["volume_m3"] for c in self.specs])
        self.T0 = np.array([c["initial_temperature_k"] for c in self.specs])
        self.p0 = np.array([c["initial_pressure_pa"] for c in self.specs])
        self.u0 = np.array([c["initial_velocity_m_s"] for c in self.specs])
        self.phi = s.gravity * np.array([c["elevation_m"] for c in self.specs])
        self.rho0 = (self.p0 + s.eos.pinf) / (s.eos.R * self.T0)
        self.m0 = self.rho0 * self.V
        self.P0 = self.m0 * self.u0
        self.A0 = self.m0 * s.eos.cv * self.T0
        self.s0 = s.eos.cv * np.log(self.T0 / s.eos.Tref) - s.eos.R * np.log(self.rho0 / s.eos.rref)
        self.ua = np.array([c["ambient_ua_w_k"] for c in self.specs])
        self.ambient = np.array([c["ambient_temperature_k"] for c in self.specs])
        self.heaters = np.array([c["heat_power_w"] for c in self.specs])
        self.start = 3 * s.n + 14
        self._state = np.zeros(self.start + 10)
        self._time = 0.0
        self._revision = 0
        self.identity = digest(dict(request=request, max_step_s=self.max_step, method="SSPRK3"))
        self.captured = signature()
        self.snapshot()

    def _sources(self):
        if signature() != self.captured:
            raise InertialGraphError("source_drift", "Terminal assembly sources/budget changed")

    def point(self, y):
        if y.shape != self._state.shape or not np.all(np.isfinite(y)):
            raise NetworkError("Malformed terminal assembly state")
        s = self.channel.system
        dm, dP, dC = y[self.start : self.start + 6].reshape(3, 2)
        m = self.m0 + dm
        if np.any(m <= 0):
            raise InertialGraphError("mass_domain", "Nonpositive chamber mass")
        deltaK = (self.u0 * dP + 0.5 * dP * dP / self.m0 - 0.5 * self.u0**2 * dm) / (
            1 + dm / self.m0
        )
        deltaA = dC - deltaK - self.phi * dm
        if np.any(1 + deltaA / self.A0 <= 0):
            raise InertialGraphError("thermal_inventory", "Nonpositive chamber thermal energy")
        T = self.T0 * (1 + deltaA / self.A0) / (1 + dm / self.m0)
        p = self.p0 + (self.p0 + s.eos.pinf) * deltaA / self.A0
        rho = self.rho0 * (1 + dm / self.m0)
        u = (self.P0 + dP) / m
        c = s.admit(rho, T, p, u)
        ds = s.eos.cv * np.log1p(deltaA / self.A0) - s.eos.cp * np.log1p(dm / self.m0)
        deltaS = dm * self.s0 + m * ds
        dynamic = copy(self.channel.system)
        dynamic.baths = [(rho[i], T[i], p[i], u[i], self.phi[i]) for i in range(2)]
        return dict(
            m=m,
            P=self.P0 + dP,
            rho=rho,
            u=u,
            T=T,
            p=p,
            c=c,
            s=self.s0 + ds,
            deltaS=deltaS,
            deltaE=dC + s.eos.e0 * dm,
            system=dynamic,
        )

    def evaluate(self, y):
        point = self.point(y)
        s = point["system"]
        channel_rate, flow = s.evaluate(y[: self.start], self.channel._heaters)
        transfer = np.array(
            [
                channel_rate[self.start - 8 : self.start - 4],
                channel_rate[self.start - 4 : self.start],
            ]
        )
        rho, u, p = point["rho"], point["u"], point["p"]
        a = abs(u) + point["c"]
        wall = s.area * (rho * u * u + p + np.array([-1.0, 1.0]) * a * rho * u)
        dm = -transfer[:, 0]
        dP = np.array([wall[0], -wall[1]]) - transfer[:, 1]
        ambient = self.ua * (self.ambient - point["T"])
        heat = self.heaters + ambient
        dC = -transfer[:, 2] + s.eos.e0 * transfer[:, 0] + heat
        thermal = dC - point["u"] * dP + (0.5 * point["u"] ** 2 - self.phi) * dm
        entropy_rate = thermal / point["T"] + (point["s"] - s.eos.cp) * dm
        SE = float(sum(self.heaters / point["T"] + ambient / self.ambient))
        SG = float(sum(entropy_rate) - SE + sum(transfer[:, 3]))
        scale = max(1.0, float(sum(abs(entropy_rate))), abs(SE), float(sum(abs(transfer[:, 3]))))
        if (
            not math.isfinite(SG)
            or SG < -self.budget["entropy_generation_negative_atol_w_k"] - 1e-10 * scale
        ):
            raise InertialGraphError(
                "entropy_admission",
                "Terminal chamber entropy production negative",
                production_w_k=SG,
            )
        result = np.r_[channel_rate, dm, dP, dC, [sum(heat), SE, SG, wall[0] - wall[1]]]
        if not np.all(np.isfinite(result)):
            raise NetworkError("Nonfinite terminal assembly RHS")
        ratio = max(
            flow["speed"] / s.dx / self.budget["maximum_courant"],
            float(max((abs(u) + point["c"]) / (self.V / s.area))) / self.budget["maximum_courant"],
        )
        ratio = max(
            ratio,
            float(max(self.ua / (point["m"] * s.eos.cv))) / self.budget["maximum_thermal_number"],
            float(max(s.ua / (flow["m"] * s.eos.cv))) / self.budget["maximum_thermal_number"],
        )
        return result, ratio, point

    def snapshot(self):
        _, _, p = self.evaluate(self._state)
        view = copy(self.channel)
        view.system = p["system"]
        channel = view._observe(self._state[: self.start], self._time)
        dm, dP, dC = self._state[self.start : self.start + 6].reshape(3, 2)
        Q, SE, SG, IW = self._state[-4:]
        mass = float(sum(self._state[: view.system.n]) + sum(dm))
        momentum = float(
            sum(self._state[view.system.n : 2 * view.system.n])
            + sum(dP)
            - channel["gravity_impulse_kg_m_s"]
            + channel["drag_impulse_kg_m_s"]
            - IW
        )
        energy = channel["total_energy_change_j"] + float(sum(p["deltaE"]))
        heat = channel["external_heat_j"] + Q
        entropy = channel["entropy_change_j_k"] + float(sum(p["deltaS"]))
        external_entropy = channel["external_entropy_j_k"] + SE
        generated = channel["generated_entropy_j_k"] + SG
        re = energy - heat + channel["exported_drag_heat_j"]
        rs = entropy - external_entropy - generated
        values = [mass, momentum, re, rs]
        limits = [
            self.budget["mass_atol_kg"]
            + self.budget["mass_rtol"] * float(sum(self.m0) + sum(view.system.m0)),
            self.budget["momentum_atol_kg_m_s"]
            + self.budget["momentum_rtol"]
            * max(
                1.0,
                abs(IW),
                abs(channel["gravity_impulse_kg_m_s"]),
                abs(channel["drag_impulse_kg_m_s"]),
                float(sum(abs(self.P0)) + sum(abs(view.system.P0))),
            ),
            self.budget["energy_atol_j"]
            + self.budget["energy_rtol"]
            * max(1.0, abs(energy), abs(heat), abs(channel["exported_drag_heat_j"])),
            self.budget["entropy_atol_j_k"]
            + self.budget["entropy_rtol"]
            * max(1.0, abs(entropy), abs(external_entropy), abs(generated)),
        ]
        names = [
            "mass_residual_kg",
            "momentum_residual_kg_m_s",
            "energy_residual_j",
            "entropy_residual_j_k",
        ]
        for name, value, limit in zip(names, values, limits):
            if not math.isfinite(value) or abs(value) > limit:
                raise InertialGraphError(
                    name,
                    "Terminal assembly balance failed",
                    observed=float(value),
                    limit=float(limit),
                )
        if generated < -self.budget["entropy_atol_j_k"]:
            raise InertialGraphError("entropy", "Negative assembly entropy generation")
        return dict(
            time_s=self._time,
            revision=self._revision,
            channel=channel,
            chambers={
                side: {key: float(p[key][i]) for key in ["m", "P", "rho", "u", "T", "p"]}
                for i, side in enumerate(["left", "right"])
            },
            total_energy_change_j=float(energy),
            external_heat_j=float(heat),
            entropy_change_j_k=float(entropy),
            external_entropy_j_k=float(external_entropy),
            generated_entropy_j_k=float(generated),
            chamber_ledgers=dict(zip(CHAMBER_LEDGERS, map(float, self._state[-4:]))),
            **dict(zip(names, map(float, values))),
        )

    def residual(self, state, rate):
        y = np.asarray(state, dtype=float)
        dy = np.asarray(rate, dtype=float)
        if dy.shape != y.shape or not np.all(np.isfinite(dy)):
            raise NetworkError("Malformed terminal residual")
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
            rate, ratio, _ = evaluate(c._state)
            h = min(self.max_step, 0.9 / ratio, end - c._time)
            if c._time + h <= c._time:
                raise NetworkError("Assembly time cannot progress")
            y = c._state
            first = y + h * rate
            r1, k1, _ = evaluate(first)
            second = 0.75 * y + 0.25 * (first + h * r1)
            r2, k2, _ = evaluate(second)
            stage = max(ratio, k1, k2) * h
            if stage > 1 + 1e-12:
                raise InertialGraphError("courant", "Assembly stage exceeds Courant bound")
            c._state = y / 3 + (2 / 3) * (second + h * r2)
            c._time = end if h == end - c._time else c._time + h
            maximum_step = max(maximum_step, h)
            maximum_number = max(maximum_number, stage)
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
            model="terminal-eos-chambers-v1",
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
            raise NetworkError("Terminal checkpoint digest mismatch")
        _keys(
            p, ("schema_version", "model", "input_sha256", "signature", "coordinates", "snapshot")
        )
        if (
            type(p["schema_version"]) is not int
            or p["schema_version"] != 1
            or p["model"] != "terminal-eos-chambers-v1"
            or p["input_sha256"] != self.identity
            or p["signature"] != self.captured
        ):
            raise NetworkError("Terminal checkpoint identity mismatch")
        if not isinstance(p["coordinates"], list) or len(p["coordinates"]) != len(self._state):
            raise NetworkError("Terminal checkpoint layout mismatch")
        snap = p["snapshot"]
        if not isinstance(snap, dict) or "time_s" not in snap or "revision" not in snap:
            raise NetworkError("Malformed terminal checkpoint observation")
        c = deepcopy(self)
        c._state = np.array([number(v, "coordinate") for v in p["coordinates"]])
        c._time = number(snap["time_s"], "time")
        c._revision = snap["revision"]
        if (
            type(c._revision) is not int
            or c._revision < 0
            or c._time < 0
            or (c._revision == 0) != (c._time == 0)
            or (c._time == 0 and np.any(c._state != 0))
        ):
            raise NetworkError("Terminal checkpoint clock mismatch")
        fresh = c.snapshot()
        if fresh != snap:
            raise NetworkError("Terminal checkpoint observation mismatch")
        self._sources()
        self._state, self._time, self._revision = c._state, c._time, c._revision
        return fresh
