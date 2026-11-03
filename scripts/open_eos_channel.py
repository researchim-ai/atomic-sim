"""Open hydrostatic Rusanov channel with prescribed exterior face states.

Exterior states define a numerical bath Riemann interface, not characteristic
nonreflecting boundaries or finite dynamic tank inventories. Four boundary
transport ledgers close mass, momentum, physical total energy and entropy.
"""

import json
import math
from copy import deepcopy

import numpy as np

from atomic_model_runtime.resources import resource
from core.network import NetworkError, number
from scripts.conservative_eos_loop import ConservativeEOSLoop, ConservativeLoopSystem
from scripts.conservative_eos_loop import signature as loop_signature
from scripts.inertial_graph import InertialGraphError, digest
from scripts.solve_hydraulic_graph import _keys, sha

BUDGET = resource("validation/ai05_open_channel_budget_v1.json")
BOUNDARY_LEDGERS = (
    "boundary_mass_kg",
    "boundary_momentum_kg_m_s",
    "boundary_energy_j",
    "boundary_entropy_j_k",
)


def signature():
    result = loop_signature()
    result["drivers"]["scripts/open_eos_channel.py"] = sha(__file__)
    result["budgets"][str(BUDGET)] = sha(BUDGET)
    return result


class OpenChannelSystem(ConservativeLoopSystem):
    def __init__(self, request):
        _keys(
            request,
            (
                "format",
                "schema_version",
                "fluid",
                "geometry",
                "initial",
                "heat",
                "drag",
                "domain",
                "boundaries",
            ),
        )
        if (
            request["format"] != "atomic-open-eos-channel"
            or type(request["schema_version"]) is not int
            or request["schema_version"] != 1
        ):
            raise NetworkError("Expected open EOS channel schema1")
        periodic = deepcopy(request)
        periodic.pop("boundaries")
        periodic["format"] = "atomic-conservative-eos-loop"
        # Reuse EOS/inventory validation, without evaluating an artificial periodic seam.
        self._initializing = True
        super().__init__(periodic)
        self._initializing = False
        self.spec = deepcopy(request)
        self.budget = json.loads(BUDGET.read_text())
        if self.n > self.budget["maximum_cells"]:
            raise NetworkError("Open channel cell budget")
        baths = request["boundaries"]
        _keys(baths, ("left", "right"))
        self.baths = []
        for side in ["left", "right"]:
            bath = baths[side]
            _keys(bath, ("pressure_pa", "temperature_k", "velocity_m_s", "elevation_m"))
            p = number(bath["pressure_pa"], "bath pressure", positive=True)
            T = number(bath["temperature_k"], "bath temperature", positive=True)
            u = number(bath["velocity_m_s"], "bath velocity")
            phi = self.gravity * number(bath["elevation_m"], "bath elevation")
            rho = (p + self.eos.pinf) / (self.eos.R * T)
            self.admit(np.array([rho]), np.array([T]), np.array([p]), np.array([u]))
            if not math.isfinite(phi):
                raise NetworkError("Nonfinite bath potential")
            self.baths.append((rho, T, p, u, phi))
        self.faces = np.r_[self.baths[0][4], 0.5 * (self.phi[:-1] + self.phi[1:]), self.baths[1][4]]
        self.evaluate(np.zeros(3 * self.n + 14), self.initial_heat)

    def evaluate(self, y, heaters):
        if self._initializing:
            return np.zeros_like(y), self.point(y)
        n = self.n
        if y.shape != (3 * n + 14,) or not np.all(np.isfinite(y)):
            raise NetworkError("Malformed open channel vector")
        point = self.point(y[: 3 * n + 6])
        rho, T, p, u = point["rho"], point["T"], point["p"], point["u"]
        inner_left, fl, sl = self.reconstruction(rho, T, p, u, self.phi, self.faces[1:])
        inner_right, fr, sr = self.reconstruction(rho, T, p, u, self.phi, self.faces[:-1])
        exterior = []
        for r, t, pr, v, phi in self.baths:
            exterior.append(
                self.reconstruction(
                    np.array([r]),
                    np.array([t]),
                    np.array([pr]),
                    np.array([v]),
                    np.array([phi]),
                    np.array([phi]),
                )
            )
        left = np.column_stack([exterior[0][0][:, 0], inner_left])
        right = np.column_stack([inner_right, exterior[1][0][:, 0]])
        fleft = np.column_stack([exterior[0][1][:, 0], fl])
        fright = np.column_stack([fr, exterior[1][1][:, 0]])
        speeds = np.maximum(np.r_[exterior[0][2], sl], np.r_[sr, exterior[1][2]])
        face = self.area * (0.5 * (fleft + fright) - 0.5 * speeds * (right - left))

        def entropy_density(states, temperatures):
            r = states[0]
            entropy = self.eos.cv * np.log(temperatures / self.eos.Tref) - self.eos.R * np.log(
                r / self.eos.rref
            )
            return r * entropy, states[1] / r

        sleft, vleft = entropy_density(left, np.r_[self.baths[0][1], T])
        sright, vright = entropy_density(right, np.r_[T, self.baths[1][1]])
        entropy_face = self.area * (
            0.5 * (sleft * vleft + sright * vright) - 0.5 * speeds * (sright - sleft)
        )
        change = face[:, :-1] - face[:, 1:]
        er = -(self.faces[1:] - self.phi) / (self.eos.R * T)
        el = -(self.faces[:-1] - self.phi) / (self.eos.R * T)
        gravity = self.area * (p + self.eos.pinf) * (np.expm1(er) - np.expm1(el))
        drag = self.drag * point["P"]
        dissipation = self.drag * point["P"] ** 2 / point["m"]
        ambient = self.ua * (self.ambient - T)
        heat = np.array([heaters[k] for k in self.names]) + ambient
        exported = self.export * dissipation
        change[1] += gravity - drag
        change[2] += heat - exported
        thermal = change[2] - u * change[1] + (0.5 * u * u - self.phi) * change[0]
        entropy_rate = thermal / T + (point["s"] - self.eos.cp) * change[0]
        external_entropy = float(sum((heat - ambient - exported) / T + ambient / self.ambient))
        left_transport = [
            face[0, 0],
            face[1, 0],
            face[2, 0] + self.eos.e0 * face[0, 0],
            entropy_face[0],
        ]
        right_transport = [
            -face[0, -1],
            -face[1, -1],
            -face[2, -1] - self.eos.e0 * face[0, -1],
            -entropy_face[-1],
        ]
        bentropy = float(left_transport[3] + right_transport[3])
        generated = float(sum(entropy_rate) - external_entropy - bentropy)
        scale = max(1.0, float(sum(abs(entropy_rate))), abs(external_entropy), abs(bentropy))
        if (
            not math.isfinite(generated)
            or generated < -self.budget["entropy_generation_negative_atol_w_k"] - 1e-10 * scale
        ):
            raise InertialGraphError(
                "entropy_admission",
                "Open-channel semidiscrete entropy production negative",
                production_w_k=generated,
            )
        ledger = [
            sum(heat),
            sum(exported),
            external_entropy,
            generated,
            sum(gravity),
            sum(drag),
            *left_transport,
            *right_transport,
        ]
        result = np.r_[change.ravel(), ledger]
        if not np.all(np.isfinite(result)):
            raise NetworkError("Nonfinite open channel RHS")
        point.update(
            speed=float(max(speeds)),
            generated=generated,
            face_flux=face,
            entropy_face_flux=entropy_face,
        )
        return result, point

    def contract(self):
        return dict(
            family="open_eos_channel_v1",
            periodic=False,
            cells=self.n,
            spatial_order=1,
            temporal_method="SSPRK3",
            boundary_kind="prescribed exterior face states, Rusanov flux",
            dynamic_reservoirs=False,
            characteristic_nonreflecting=False,
            general_graph=False,
            boundary_transport_ledgers=list(BOUNDARY_LEDGERS),
            separate_boundary_transport=True,
            general_entropy_stability_proof=False,
        )


class OpenEOSChannel(ConservativeEOSLoop):
    def __init__(self, request, *, max_step_s=0.001):
        self.system = OpenChannelSystem(deepcopy(request))
        self.budget = json.loads(BUDGET.read_text())
        self.max_step = number(max_step_s, "max step", positive=True)
        if self.max_step > self.budget["maximum_step_s"]:
            raise NetworkError("Open channel step exceeds budget")
        self.identity = digest(dict(request=request, max_step_s=self.max_step, method="SSPRK3"))
        self.captured = signature()
        self._state = np.zeros(3 * self.system.n + 14)
        self._time = 0.0
        self._revision = 0
        self._history = []
        self._heaters = deepcopy(self.system.initial_heat)
        self.snapshot()

    def _sources(self):
        if signature() != self.captured:
            raise InertialGraphError("source_drift", "Open channel source/budget/runtime changed")

    def _observe(self, y, time):
        s = self.system
        n = s.n
        _, p = s.evaluate(y, self._heaters)
        dm, dP, dC = y[:n], y[n : 2 * n], y[2 * n : 3 * n]
        Q, E, SE, SG, IG, ID = y[3 * n : 3 * n + 6]
        BM, BP, BE, BS = y[-8:-4] + y[-4:]
        energy = float(sum(dC + s.eos.e0 * dm))
        entropy = float(sum(p["deltaS"]))
        residuals = dict(
            mass_residual_kg=float(sum(dm) - BM),
            momentum_residual_kg_m_s=float(sum(dP) - IG + ID - BP),
            energy_residual_j=float(energy - Q + E - BE),
            entropy_residual_j_k=float(entropy - SE - SG - BS),
        )
        limits = [
            self.budget["mass_atol_kg"] + self.budget["mass_rtol"] * max(float(sum(s.m0)), abs(BM)),
            self.budget["momentum_atol_kg_m_s"]
            + self.budget["momentum_rtol"]
            * max(1.0, float(sum(abs(s.P0))), abs(IG), abs(ID), abs(BP)),
            self.budget["energy_atol_j"]
            + self.budget["energy_rtol"] * max(1.0, abs(Q), abs(E), abs(BE), float(sum(abs(dC)))),
            self.budget["entropy_atol_j_k"]
            + self.budget["entropy_rtol"] * max(1.0, abs(SE), abs(SG), abs(BS)),
        ]
        for (name, value), limit in zip(residuals.items(), limits):
            if not math.isfinite(value) or abs(value) > limit:
                raise InertialGraphError(
                    name, "Open channel balance failed", observed=value, limit=float(limit)
                )
        if SG < -self.budget["entropy_atol_j_k"]:
            raise InertialGraphError(
                "entropy", "Negative open channel integrated entropy generation"
            )
        return dict(
            time_s=time,
            masses_kg=dict(zip(s.names, map(float, p["m"]))),
            densities_kg_m3=dict(zip(s.names, map(float, p["rho"]))),
            velocities_m_s=dict(zip(s.names, map(float, p["u"]))),
            temperatures_k=dict(zip(s.names, map(float, p["T"]))),
            pressures_pa=dict(zip(s.names, map(float, p["p"]))),
            total_energy_change_j=energy,
            entropy_change_j_k=entropy,
            boundary_flows_into_channel_kg_s=dict(
                left=float(p["face_flux"][0, 0]), right=float(-p["face_flux"][0, -1])
            ),
            mean_velocity_m_s=float(sum(p["P"]) / sum(p["m"])),
            external_heat_j=float(Q),
            exported_drag_heat_j=float(E),
            external_entropy_j_k=float(SE),
            generated_entropy_j_k=float(SG),
            gravity_impulse_kg_m_s=float(IG),
            drag_impulse_kg_m_s=float(ID),
            **dict(zip(BOUNDARY_LEDGERS, map(float, y[-8:-4] + y[-4:]))),
            boundary_transport={
                side: dict(zip(BOUNDARY_LEDGERS, map(float, values)))
                for side, values in [("left", y[-8:-4]), ("right", y[-4:])]
            },
            **residuals,
        )

    def checkpoint(self):
        p = dict(
            schema_version=1,
            model="open-eos-channel-v1",
            input_sha256=self.identity,
            signature=deepcopy(self.captured),
            coordinates=self._state.tolist(),
            snapshot=self.snapshot(),
        )
        self._sources()
        return dict(payload=p, sha256=digest(p))

    def restore(self, checkpoint):
        self._sources()
        _keys(checkpoint, ("payload", "sha256"))
        p = checkpoint["payload"]
        if digest(p) != checkpoint["sha256"]:
            raise NetworkError("Open channel checkpoint digest mismatch")
        _keys(
            p, ("schema_version", "model", "input_sha256", "signature", "coordinates", "snapshot")
        )
        if (
            type(p["schema_version"]) is not int
            or p["schema_version"] != 1
            or p["model"] != "open-eos-channel-v1"
            or p["input_sha256"] != self.identity
            or p["signature"] != self.captured
        ):
            raise NetworkError("Open channel checkpoint identity mismatch")
        snap = p["snapshot"]
        _keys(snap, ("revision", "active_heat_power_w", "history", "state", "contract"))
        if not isinstance(snap["state"], dict) or "time_s" not in snap["state"]:
            raise NetworkError("Malformed channel observation")
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
            raise NetworkError("Open channel checkpoint clock/history mismatch")
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
                raise NetworkError("Open channel checkpoint receipt path mismatch")
            ids.add(cid)
            active = self.system.heaters(r["heat_power_w"])
            last_time, last_rev = when, commit
        values = p["coordinates"]
        if not isinstance(values, list) or len(values) != 3 * self.system.n + 14:
            raise NetworkError("Open channel checkpoint layout mismatch")
        vector = np.array([number(v, "coordinate") for v in values])
        if time == 0 and np.any(vector != 0):
            raise NetworkError("Nonzero initial channel stocks")
        c = deepcopy(self)
        c._state = vector
        c._time = time
        c._revision = rev
        c._history = deepcopy(history)
        c._heaters = active
        fresh = c.snapshot()
        if fresh != snap:
            raise NetworkError("Open channel checkpoint observation mismatch")
        self._sources()
        self._install(c)
        return fresh
