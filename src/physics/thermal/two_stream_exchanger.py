"""Constant-property, mixed-cell two-stream exchanger with wall storage.

Flows are prescribed boundary data, not a hydraulic solution. Cells share a
physical left-to-right coordinate; signed flows select the donor and outlet.
Every accepted interval has constant controls and is advanced by a matrix
exponential, including independent per-cell energy integrals.
"""

import hashlib
import json
from copy import deepcopy

import numpy as np
from scipy.linalg import expm

from core.network import NetworkError, number
from core.status import ModelPassport, ModelStatus

PASSPORT = ModelPassport(
    model_id="two-stream-wall-exchanger",
    version="1.0.0",
    status=ModelStatus.IMPLEMENTED,
    phenomenon="Two advecting liquid streams exchanging heat through a finite-capacity wall",
    approximations=(
        "Constant positive cp and fixed fluid masses; perfectly mixed finite-volume cells",
        "Prescribed signed mass flows; no pressure, momentum or pumping work model",
        "Positive constant film conductances; no axial conduction, phase change or radiation",
        "Piecewise constant boundary controls; exact linear-system propagation in time",
    ),
    applicability="Single-phase constant-property thermal transients; positive temperatures",
    data_sources=(
        "m_cell*cp*dT/dt=abs(mdot)*cp*(T_donor-T)+UA*(Tw-T)",
        "Cw*dTw/dt=UAa*(Ta-Tw)+UAb*(Tb-Tw)+P-UAamb*(Tw-Tamb)",
    ),
)
CONTROL_FIELDS = {
    "a_mass_flow_kg_s",
    "b_mass_flow_kg_s",
    "a_inlet_temperature_k",
    "b_inlet_temperature_k",
}
CONFIG_FIELDS = {
    "cells",
    "a_cp_j_kg_k",
    "b_cp_j_kg_k",
    "a_cell_mass_kg",
    "b_cell_mass_kg",
    "wall_capacity_j_k",
    "a_wall_ua_w_k",
    "b_wall_ua_w_k",
    "ambient_ua_w_k",
    "ambient_temperature_k",
    "wall_power_w",
    "energy_tolerance_j",
}


def controls(raw):
    if not isinstance(raw, dict) or set(raw) != CONTROL_FIELDS:
        raise NetworkError("Complete exchanger boundary control vector required")
    return {key: number(value, key, positive="temperature" in key) for key, value in raw.items()}


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


class TwoStreamExchanger:
    def __init__(self, configuration, boundary_controls, initialization):
        if not isinstance(configuration, dict) or set(configuration) != CONFIG_FIELDS:
            raise NetworkError("Exact exchanger configuration required")
        n = configuration["cells"]
        if type(n) is not int or not 1 <= n <= 32:
            raise NetworkError("Exchanger requires 1..32 cells")
        config = {"cells": n}
        for key in ("a_cp_j_kg_k", "b_cp_j_kg_k", "ambient_temperature_k", "energy_tolerance_j"):
            config[key] = number(configuration[key], key, positive=True)
        for key in CONFIG_FIELDS - set(config):
            raw = configuration[key]
            if not isinstance(raw, list) or len(raw) != n:
                raise NetworkError(f"{key} requires one value per cell")
            positive = key in {"a_cell_mass_kg", "b_cell_mass_kg", "wall_capacity_j_k"}
            values = [number(v, key, positive=positive) for v in raw]
            if not positive and min(values) < 0:
                raise NetworkError(f"{key} must be nonnegative")
            config[key] = values
        self._configuration = config
        self.n = n
        self._capacity = np.concatenate(
            (
                np.array(config["a_cell_mass_kg"]) * config["a_cp_j_kg_k"],
                np.array(config["b_cell_mass_kg"]) * config["b_cp_j_kg_k"],
                config["wall_capacity_j_k"],
            )
        )
        if not np.isfinite(self._capacity).all():
            raise NetworkError("Exchanger heat capacities overflow")
        self._controls = controls(boundary_controls)
        self._initial_controls = deepcopy(self._controls)
        self._matrix, self._drive = self._assemble(self._controls)
        self._initial = self._initialize(initialization)
        self._temperature = self._initial.copy()
        self._integrals = np.zeros(3 * n)
        self.time_s = 0.0
        self._events = []
        self._initialization = deepcopy(initialization)
        self.observation()

    @property
    def configuration(self):
        return deepcopy(self._configuration)

    @property
    def boundary_controls(self):
        return deepcopy(self._controls)

    @property
    def events(self):
        return deepcopy(self._events)

    def _assemble(self, boundary):
        n, cfg = self.n, self._configuration
        # Power matrix (W/K) and drive (W), before division by cell capacity.
        power = np.zeros((3 * n, 3 * n))
        drive = np.zeros(3 * n)
        for side, offset in (("a", 0), ("b", n)):
            flow = boundary[f"{side}_mass_flow_kg_s"]
            rate = abs(flow) * cfg[f"{side}_cp_j_kg_k"]
            for i in range(n):
                row = offset + i
                power[row, row] -= rate
                donor = i - 1 if flow >= 0 else i + 1
                if 0 <= donor < n:
                    power[row, offset + donor] += rate
                else:
                    drive[row] += rate * boundary[f"{side}_inlet_temperature_k"]
                ua = cfg[f"{side}_wall_ua_w_k"][i]
                power[row, row] -= ua
                power[row, 2 * n + i] += ua
                power[2 * n + i, row] += ua
                power[2 * n + i, 2 * n + i] -= ua
        for i in range(n):
            ambient = cfg["ambient_ua_w_k"][i]
            power[2 * n + i, 2 * n + i] -= ambient
            drive[2 * n + i] += ambient * cfg["ambient_temperature_k"] + cfg["wall_power_w"][i]
        matrix, rhs = power / self._capacity[:, None], drive / self._capacity
        if not np.isfinite(matrix).all() or not np.isfinite(rhs).all():
            raise NetworkError("Exchanger coefficients overflow")
        return matrix, rhs

    def _initialize(self, initialization):
        if not isinstance(initialization, dict):
            raise NetworkError("Explicit exchanger initialization required")
        if initialization == {"mode": "steady"}:
            # Singular disconnected stores have no unique prescribed steady state.
            if np.linalg.matrix_rank(self._matrix) != 3 * self.n:
                raise NetworkError("Steady initialization is unanchored or numerically singular")
            try:
                state = np.linalg.solve(self._matrix, -self._drive)
            except np.linalg.LinAlgError as exc:
                raise NetworkError("Steady initialization is singular") from exc
        elif set(initialization) == {"mode", "temperature_k"} and initialization["mode"] == "given":
            raw = initialization["temperature_k"]
            if not isinstance(raw, dict) or set(raw) != {"a", "b", "wall"}:
                raise NetworkError("Initial a/b/wall temperature arrays required")
            state = []
            for name in ("a", "b", "wall"):
                if not isinstance(raw[name], list) or len(raw[name]) != self.n:
                    raise NetworkError("Initial temperature count differs from cells")
                state.extend(number(v, "initial temperature", positive=True) for v in raw[name])
            state = np.array(state)
        else:
            raise NetworkError("Unknown exchanger initialization schema")
        if not np.isfinite(state).all() or np.min(state) <= 0:
            raise NetworkError("Invalid initialized absolute temperatures")
        residual = self._capacity * (self._matrix @ state + self._drive)
        if initialization["mode"] == "steady" and max(abs(residual)) > 1e-6:
            raise NetworkError("Steady initialization residual exceeds 1e-6 W per cell")
        return state

    def initialization_report(self):
        return dict(
            mode=self._initialization["mode"],
            energy_rate_residual_w=(
                self._capacity
                * (
                    self._assemble(self._initial_controls)[0] @ self._initial
                    + self._assemble(self._initial_controls)[1]
                )
            )
            .reshape(3, self.n)
            .tolist(),
            unique_steady_required=self._initialization["mode"] == "steady",
        )

    def _observe(self, temperature, integrals, stamp):
        if (
            temperature.shape != (3 * self.n,)
            or integrals.shape != (3 * self.n,)
            or not np.isfinite(temperature).all()
            or not np.isfinite(integrals).all()
            or np.any(temperature <= 0)
        ):
            raise NetworkError("Invalid exchanger state")
        energy = self._capacity * (temperature - self._initial)
        residual = energy - integrals
        global_residual = float(sum(residual))
        tolerance = self._configuration["energy_tolerance_j"]
        if max(abs(residual)) > tolerance or abs(global_residual) > tolerance:
            raise NetworkError("Exchanger local/global energy ledger exceeded")
        n, cfg = self.n, self._configuration
        fields = dict(zip(("a", "b", "wall"), temperature.reshape(3, n).tolist()))
        powers = {}
        external = 0.0
        for side in ("a", "b"):
            flow = self._controls[f"{side}_mass_flow_kg_s"]
            outlet = fields[side][-1 if flow >= 0 else 0]
            rate = abs(flow) * cfg[f"{side}_cp_j_kg_k"]
            inlet_power = rate * self._controls[f"{side}_inlet_temperature_k"]
            outlet_power = rate * outlet
            powers[side] = dict(
                inlet_enthalpy_w=inlet_power,
                outlet_enthalpy_w=outlet_power,
                net_enthalpy_w=inlet_power - outlet_power,
                outlet_temperature_k=outlet,
                outlet_end="right" if flow > 0 else "left" if flow < 0 else None,
            )
            external += inlet_power - outlet_power
        ambient = float(
            np.array(cfg["ambient_ua_w_k"]) @ (temperature[2 * n :] - cfg["ambient_temperature_k"])
        )
        external += sum(cfg["wall_power_w"]) - ambient
        rates = self._capacity * (self._matrix @ temperature + self._drive)
        power_residual = float(sum(rates) - external)
        if not np.isfinite(power_residual) or abs(power_residual) > 1e-6:
            raise NetworkError("Exchanger instantaneous port power mismatch")
        return dict(
            time_s=stamp,
            temperature_k=fields,
            controls=self.boundary_controls,
            stored_energy_change_j=float(sum(energy)),
            integrated_net_cell_energy_j=integrals.reshape(3, n).tolist(),
            cell_energy_residual_j=residual.reshape(3, n).tolist(),
            energy_residual_j=global_residual,
            port_power_residual_w=power_residual,
            ports=powers,
            ambient_heat_out_w=ambient,
            a_to_wall_w=(
                np.array(cfg["a_wall_ua_w_k"]) * (temperature[:n] - temperature[2 * n :])
            ).tolist(),
            wall_to_b_w=(
                np.array(cfg["b_wall_ua_w_k"]) * (temperature[2 * n :] - temperature[n : 2 * n])
            ).tolist(),
        )

    def observation(self):
        return self._observe(self._temperature, self._integrals, self.time_s)

    def advance(self, dt_s):
        dt = number(dt_s, "dt_s", positive=True)
        end = self.time_s + dt
        if not np.isfinite(end) or end <= self.time_s:
            raise NetworkError("Exchanger clock cannot advance")
        size = 3 * self.n
        # Deviations from the accepted temperature avoid cancellation in the
        # per-cell integrals: integrate C*d(deltaT)/dt alongside deltaT.
        augmented = np.zeros((2 * size + 1, 2 * size + 1))
        drive = self._matrix @ self._temperature + self._drive
        augmented[:size, :size] = self._matrix
        augmented[:size, -1] = drive
        augmented[size : 2 * size, :size] = self._capacity[:, None] * self._matrix
        augmented[size : 2 * size, -1] = self._capacity * drive
        seed = np.zeros(2 * size + 1)
        seed[-1] = 1
        result = expm(augmented * dt) @ seed
        candidate = self._temperature + result[:size]
        integrals = self._integrals + result[size : 2 * size]
        observed = self._observe(candidate, integrals, end)
        self._temperature, self._integrals, self.time_s = candidate, integrals, end
        return observed

    def set_controls(self, boundary_controls):
        if len(self._events) >= 10000:
            raise NetworkError("Exchanger control history resource limit exceeded")
        boundary = controls(boundary_controls)
        matrix, drive = self._assemble(boundary)
        previous = self._controls, self._matrix, self._drive
        try:
            self._controls, self._matrix, self._drive = boundary, matrix, drive
            self.observation()
        except Exception:
            self._controls, self._matrix, self._drive = previous
            raise
        self._events.append(
            dict(t=self.time_s, before=deepcopy(previous[0]), after=deepcopy(boundary))
        )

    def snapshot(self):
        return dict(
            schema_version=1,
            model_id=PASSPORT.model_id,
            model_version=PASSPORT.version,
            configuration=self.configuration,
            configuration_sha256=digest(self.configuration),
            initialization=deepcopy(self._initialization),
            initial_controls=deepcopy(self._initial_controls),
            controls=self.boundary_controls,
            time_s=self.time_s,
            temperature_k=self._temperature.tolist(),
            cell_energy_integrals_j=self._integrals.tolist(),
            events=self.events,
        )

    @classmethod
    def from_snapshot(cls, payload):
        fields = {
            "schema_version",
            "model_id",
            "model_version",
            "configuration",
            "configuration_sha256",
            "initialization",
            "initial_controls",
            "controls",
            "time_s",
            "temperature_k",
            "cell_energy_integrals_j",
            "events",
        }
        if (
            not isinstance(payload, dict)
            or set(payload) != fields
            or type(payload["schema_version"]) is not int
            or payload["schema_version"] != 1
            or payload["model_id"] != PASSPORT.model_id
            or payload["model_version"] != PASSPORT.version
            or payload["configuration_sha256"] != digest(payload["configuration"])
        ):
            raise NetworkError("Exchanger snapshot schema/configuration mismatch")
        model = cls(
            payload["configuration"], payload["initial_controls"], payload["initialization"]
        )
        stamp = number(payload["time_s"], "snapshot time")
        if stamp < 0 or not isinstance(payload["events"], list) or len(payload["events"]) > 10000:
            raise NetworkError("Invalid exchanger snapshot time/history")
        previous_time, boundary = -1.0, model.boundary_controls
        for event in payload["events"]:
            if not isinstance(event, dict) or set(event) != {"t", "before", "after"}:
                raise NetworkError("Invalid exchanger control history")
            t = number(event["t"], "event time")
            if not 0 <= t <= stamp or t < previous_time or controls(event["before"]) != boundary:
                raise NetworkError("Broken exchanger control history")
            previous_time, boundary = t, controls(event["after"])
        if boundary != controls(payload["controls"]):
            raise NetworkError("Exchanger active controls differ from history")
        model._controls = boundary
        model._matrix, model._drive = model._assemble(boundary)
        arrays = []
        for key in ("temperature_k", "cell_energy_integrals_j"):
            raw = payload[key]
            if not isinstance(raw, list) or len(raw) != 3 * model.n:
                raise NetworkError("Invalid exchanger snapshot array")
            arrays.append(np.array([number(v, key) for v in raw]))
        model._temperature, model._integrals = arrays
        model.time_s, model._events = stamp, deepcopy(payload["events"])
        model.observation()
        if stamp == 0 and (
            not np.array_equal(model._temperature, model._initial) or np.any(model._integrals != 0)
        ):
            raise NetworkError("Noninitial exchanger state at zero time")
        return model
