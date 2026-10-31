"""Signed displacement pump/shaft coupled to inertial gas-cushion storage.

Reciprocal law q=D*w-C*H, tau_h=D*H. Leakage C*H**2 is routed through
an explicit thermal port to a fixed vessel, so its destination is continuous
at flow reversal. Bearing heat leaves the assembly. No pump map qualification.
"""

import hashlib
import json
from copy import copy, deepcopy

import numpy as np
from scipy.optimize import brentq
from scipy.sparse import csc_matrix

from core.continuation import ImplicitContinuation
from core.network import NetworkError, PortSpec, number
from core.status import ModelPassport, ModelStatus
from physics.hydraulics.inertial_storage_pair import InertialStoragePair

PASSPORT = ModelPassport(
    model_id="pump-shaft-gas-storage-pair",
    version="1.0.0",
    status=ModelStatus.IMPLEMENTED,
    phenomenon="Reciprocal pump/shaft power coupled to gas storage and hydraulic kinetic energy",
    approximations=(
        "constant displacement and positive linear leakage; signed flow/speed/head",
        "explicit leakage heat port to a fixed vessel; bearing loss to an external sink",
        "ideal hydraulic inertance, zero link thermal inventory; constant-density liquid",
        "sealed ideal gas cushions; no steam pressurizer, cavitation or acoustic waves",
        "no centrifugal/manufacturer map, ideal shutoff or check-valve contact",
    ),
    applicability="Positive gas/liquid inventories, finite nonnegative hydraulic resistance and shaft drag",
    data_sources=("I*mdot=dp+H-R*m*abs(m); J*wdot=tau_motor-D*H-B*w",),
)
CONFIG_FIELDS = {"hydraulic_configuration", "pump", "initial_omega_rad_s", "motor_torque_nm"}
PUMP_FIELDS = {
    "displacement_m3_rad",
    "leakage_m3_s_pa",
    "inertia_kg_m2",
    "drag_nm_s_rad",
    "leak_heat_vessel",
}
CONTROL_FIELDS = {"resistance_pa_s2_kg2", "motor_torque_nm"}
SHAFT_PORT = PortSpec("shaft", "mechanical")
LEAK_HEAT_PORT = PortSpec("leak_heat", "thermal")


class PumpStoragePair:
    def __init__(self, hydraulic_configuration, pump, initial_omega_rad_s, motor_torque_nm):
        if not isinstance(hydraulic_configuration, dict):
            raise NetworkError("Explicit hydraulic configuration required")
        self._hydraulic = InertialStoragePair(**deepcopy(hydraulic_configuration))
        if not isinstance(pump, dict) or set(pump) != PUMP_FIELDS:
            raise NetworkError("Exact storage pump parameters required")
        params = {
            k: number(v, k, positive=k != "drag_nm_s_rad")
            for k, v in pump.items()
            if k != "leak_heat_vessel"
        }
        if params["drag_nm_s_rad"] < 0:
            raise NetworkError("Negative shaft drag is unsupported")
        if (
            not isinstance(pump["leak_heat_vessel"], str)
            or pump["leak_heat_vessel"] not in self._hydraulic.names
        ):
            raise NetworkError("Explicit existing leakage heat receiver required")
        params["leak_heat_vessel"] = pump["leak_heat_vessel"]
        omega = number(initial_omega_rad_s, "initial_omega_rad_s")
        torque = number(motor_torque_nm, "motor_torque_nm")
        self._config = dict(
            hydraulic_configuration=self._hydraulic.configuration,
            pump=params,
            initial_omega_rad_s=omega,
            motor_torque_nm=torque,
        )
        self.names = self._hydraulic.names
        self.rho = self._hydraulic.rho
        self._heat_index = self.names.index(params["leak_heat_vessel"])
        self._resistance = self._hydraulic._resistance
        self._torque = torque
        self._state = np.r_[self._hydraulic._state, omega, np.zeros(4)]
        self.time_s = 0.0
        self.events = []
        self.last_events = ()
        self._continuation = None
        self._policy = None
        self._horizon = None
        self.observation()

    @property
    def configuration(self):
        return deepcopy(self._config)

    @property
    def controls(self):
        return dict(resistance_pa_s2_kg2=self._resistance, motor_torque_nm=self._torque)

    def configuration_sha256(self):
        return hashlib.sha256(
            json.dumps(self.configuration, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()

    def _view(self):
        view = copy(self._hydraulic)
        view._resistance = self._resistance
        return view

    def _machine(self, y):
        y = np.asarray(y, dtype=float)
        if y.shape != (14,) or not np.isfinite(y).all():
            raise NetworkError("Invalid pump-storage state")
        p = self._config["pump"]
        q = float(y[8]) / self.rho
        omega = float(y[9])
        head = (p["displacement_m3_rad"] * omega - q) / p["leakage_m3_s_pa"]
        torque = p["displacement_m3_rad"] * head
        leak = p["leakage_m3_s_pa"] * head**2
        bearing = p["drag_nm_s_rad"] * omega**2
        if not np.isfinite([head, torque, leak, bearing]).all():
            raise NetworkError("Nonfinite storage pump power")
        return q, omega, head, torque, leak, bearing

    def derivatives(self, t, y):
        q, omega, head, torque, leak, bearing = self._machine(y)
        p = self._config["pump"]
        rates = self._view().derivatives(t, y[:9])
        rates[8] += head / self._hydraulic._inertia
        rates[1 + self._heat_index] += leak
        # Internal ledger of hydraulic subsystem includes shaft work, not only heat.
        rates[5] += torque * omega
        result = np.r_[
            rates,
            (self._torque - torque - p["drag_nm_s_rad"] * omega) / p["inertia_kg_m2"],
            self._torque * omega,
            bearing,
            leak,
            head * q,
        ]
        if not np.isfinite(result).all():
            raise NetworkError("Nonfinite pump-storage rates")
        return result

    def analytic_jacobian(self, t, y):
        q, omega, head, torque, leak, bearing = self._machine(y)
        p = self._config["pump"]
        jac = np.zeros((14, 14))
        jac[:9, :9] = self._view().analytic_jacobian(t, y[:9]).toarray()
        hm = -1 / (self.rho * p["leakage_m3_s_pa"])
        hw = p["displacement_m3_rad"] / p["leakage_m3_s_pa"]
        for col, dh in ((8, hm), (9, hw)):
            dl = 2 * p["leakage_m3_s_pa"] * head * dh
            jac[8, col] += dh / self._hydraulic._inertia
            jac[1 + self._heat_index, col] += dl
            jac[12, col] = dl
            jac[9, col] = -p["displacement_m3_rad"] * dh / p["inertia_kg_m2"]
        jac[5, 8] += p["displacement_m3_rad"] * hm * omega
        jac[5, 9] += p["displacement_m3_rad"] * (hw * omega + head)
        jac[9, 9] -= p["drag_nm_s_rad"] / p["inertia_kg_m2"]
        jac[10, 9] = self._torque
        jac[11, 9] = 2 * p["drag_nm_s_rad"] * omega
        jac[13, 8] = head / self.rho + q * hm
        jac[13, 9] = q * hw
        if not np.isfinite(jac).all():
            raise NetworkError("Nonfinite pump-storage Jacobian")
        return csc_matrix(jac)

    def observation(self, y=None, time_s=None):
        y = self._state if y is None else np.asarray(y, dtype=float)
        q, omega, head, torque, leak, bearing = self._machine(y)
        p = self._config["pump"]
        view = self._view()
        o = view.observation(y[:9], self.time_s if time_s is None else time_s)
        rate = self.derivatives(o["time_s"], y)
        r = view._rates(y[:9])
        storage = view._storage
        shaft_delta = (
            0.5
            * p["inertia_kg_m2"]
            * (omega - self._config["initial_omega_rad_s"])
            * (omega + self._config["initial_omega_rad_s"])
        )
        shaft_defect = shaft_delta - y[10] + y[11] + y[12] + y[13]
        external_heat = float(y[5] - y[12] - y[13])
        total_defect = float(
            sum(y[1:5]) + o["kinetic_energy_change_j"] + shaft_delta - y[10] + y[11] - external_heat
        )
        transfer_defect = torque * omega - head * q - leak
        kinetic_rate = view._inertia * y[8] * rate[8] / self.rho
        link_defect = float(
            r["face_flux"][0] - r["face_flux"][1] + torque * omega - leak - kinetic_rate
        )
        local = {}
        for i, name in enumerate(self.names):
            sign = -1 if i == 0 else 1
            local[name] = float(
                rate[1 + i]
                + rate[3 + i]
                - sign * r["face_flux"][i]
                - storage._heater[i]
                - r["ambient_heat"][i]
                - (leak if i == self._heat_index else 0)
            )
        if (
            not np.isfinite(
                [
                    shaft_delta,
                    shaft_defect,
                    total_defect,
                    transfer_defect,
                    link_defect,
                    *local.values(),
                    self._torque * omega,
                ]
            ).all()
            or min(y[11], y[12]) < -storage._config["energy_tolerance_j"]
            or max(abs(total_defect), abs(shaft_defect)) > storage._config["energy_tolerance_j"]
            or max(abs(transfer_defect), abs(link_defect), max(abs(v) for v in local.values()))
            > 1e-6
        ):
            raise NetworkError("Pump-storage total/local/shaft energy closure exceeded")
        receiver = p["leak_heat_vessel"]
        temperature = o["liquid_temperature_k"][receiver]
        o["ports"].update(
            {
                "pump.leak_heat": dict(temperature_k=temperature, heat_flow_w=-leak),
                receiver + ".pump_heat": dict(temperature_k=temperature, heat_flow_w=leak),
                "shaft.to_hydraulics": dict(omega_rad_s=omega, torque_nm=-torque),
                "hydraulics.shaft": dict(omega_rad_s=omega, torque_nm=torque),
            }
        )
        o.update(
            omega_rad_s=omega,
            head_pa=head,
            hydraulic_torque_nm=torque,
            motor_torque_nm=self._torque,
            motor_power_w=self._torque * omega,
            shaft_to_pump_power_w=torque * omega,
            hydraulic_power_w=head * q,
            leakage_power_w=leak,
            bearing_power_w=bearing,
            shaft_energy_change_j=shaft_delta,
            motor_work_j=float(y[10]),
            bearing_heat_j=float(y[11]),
            leakage_heat_j=float(y[12]),
            head_work_j=float(y[13]),
            hydraulic_input_ledger_j=float(y[5]),
            external_heat_j=external_heat,
            energy_residual_j=total_defect,
            hydraulic_energy_residual_j=o["energy_residual_j"],
            shaft_energy_residual_j=float(shaft_defect),
            power_transfer_residual_w=float(transfer_defect),
            link_energy_residual_w=link_defect,
            local_energy_residual_w=local,
            flow_acceleration_kg_s2=float(rate[8]),
        )
        return o

    def enable_continuation(self, horizon_s):
        horizon = number(horizon_s, "horizon_s", positive=True)
        if horizon <= self.time_s:
            raise NetworkError("Future pump-storage horizon required")
        self._horizon = horizon

    def advance(self, dt_s, *, method="Radau", rtol=1e-9, max_step_s=0.05, atol_scale=1.0):
        dt = number(dt_s, "dt_s", positive=True)
        rtol = number(rtol, "rtol", positive=True)
        step = number(max_step_s, "max_step_s", positive=True)
        scale = number(atol_scale, "atol_scale", positive=True)
        if method not in ("Radau", "BDF") or not 1e-12 <= rtol <= 1e-2 or not 1e-6 <= scale <= 1:
            raise NetworkError("Unsupported pump-storage solver")
        target = self.time_s + dt
        if not np.isfinite(target) or target <= self.time_s:
            raise NetworkError("Target time cannot advance")
        saved = self._state.copy(), self.time_s, deepcopy(self.events), self.last_events
        policy = method, rtol, step, scale
        try:
            if (
                self._continuation is None
                or policy != self._policy
                or target > self._continuation.horizon
            ):
                self._continuation = ImplicitContinuation(
                    self.derivatives,
                    self._state,
                    self.time_s,
                    max(target, self._horizon or target),
                    rtol=rtol,
                    atol=np.r_[1e-12, np.full(7, 1e-8), 1e-12, 1e-11, np.full(4, 1e-8)] * scale,
                    max_step=step,
                    method=method,
                    initial_target=target,
                    jacobian=self.analytic_jacobian,
                )
                self._policy = policy
            result = self._continuation.advance_to(target)
            observed = self.observation(result.final, target)
            events = []
            for index, kind in ((8, "flow_zero"), (9, "shaft_zero")):
                for i in range(len(result.t) - 1):
                    a, b = result.y[index, i : i + 2]
                    direction = 1 if a < 0 <= b else -1 if a > 0 >= b else 0
                    if direction:
                        root = (
                            float(result.t[i + 1])
                            if b == 0
                            else brentq(
                                lambda t: float(result.dense(t)[index]),
                                float(result.t[i]),
                                float(result.t[i + 1]),
                                xtol=1e-12,
                            )
                        )
                        events.append(dict(kind=kind, t=root, direction=direction))
            events.sort(key=lambda e: (e["t"], e["kind"]))
            self._state, self.time_s = result.final.copy(), target
            self.events.extend(events)
            self.last_events = tuple(events)
        except BaseException:
            self._state, self.time_s, self.events, self.last_events = saved
            self._continuation = None
            raise
        return observed

    def set_controls(self, controls):
        if not isinstance(controls, dict) or set(controls) != CONTROL_FIELDS:
            raise NetworkError("Complete pump-storage control vector required")
        resistance = number(controls["resistance_pa_s2_kg2"], "resistance")
        torque = number(controls["motor_torque_nm"], "motor torque")
        if resistance < 0:
            raise NetworkError("Nonnegative finite resistance required")
        previous = self.controls
        saved = self._resistance, self._torque
        try:
            self._resistance, self._torque = resistance, torque
            self.derivatives(self.time_s, self._state)
            observed = self.observation()
        except BaseException:
            self._resistance, self._torque = saved
            raise
        event = dict(kind="controls_changed", t=self.time_s, previous=previous, value=self.controls)
        self.events.append(event)
        self.last_events = (event,)
        self._continuation = None
        self._policy = None
        return observed

    def snapshot(self):
        return dict(
            schema_version=1,
            model_id=PASSPORT.model_id,
            model_version=PASSPORT.version,
            configuration=self.configuration,
            configuration_sha256=self.configuration_sha256(),
            state=self._state.tolist(),
            time_s=self.time_s,
            active_controls=self.controls,
            events=deepcopy(self.events),
        )

    @classmethod
    def from_snapshot(cls, payload):
        fields = {
            "schema_version",
            "model_id",
            "model_version",
            "configuration",
            "configuration_sha256",
            "state",
            "time_s",
            "active_controls",
            "events",
        }
        if (
            not isinstance(payload, dict)
            or set(payload) != fields
            or type(payload["schema_version"]) is not int
            or payload["schema_version"] != 1
            or payload["model_id"] != PASSPORT.model_id
            or payload["model_version"] != PASSPORT.version
        ):
            raise NetworkError("Incompatible pump-storage checkpoint")
        if (
            not isinstance(payload["configuration"], dict)
            or set(payload["configuration"]) != CONFIG_FIELDS
        ):
            raise NetworkError("Invalid pump-storage configuration")
        model = cls(**payload["configuration"])
        if model.configuration_sha256() != payload["configuration_sha256"]:
            raise NetworkError("Pump-storage configuration hash mismatch")
        t = number(payload["time_s"], "checkpoint time")
        if t < 0 or not isinstance(payload["events"], list):
            raise NetworkError("Invalid pump-storage time/events")

        def validated(c):
            if not isinstance(c, dict) or set(c) != CONTROL_FIELDS:
                raise NetworkError("Invalid checkpoint controls")
            result = {k: number(v, k) for k, v in c.items()}
            if result["resistance_pa_s2_kg2"] < 0:
                raise NetworkError("Negative checkpoint resistance")
            return result

        active = validated(payload["active_controls"])
        expected = model.controls
        previous_t = 0.0
        for event in payload["events"]:
            if not isinstance(event, dict):
                raise NetworkError("Invalid pump-storage event")
            stamp = number(event.get("t"), "event time")
            if not previous_t <= stamp <= t:
                raise NetworkError("Unordered pump-storage event")
            previous_t = stamp
            if event.get("kind") == "controls_changed" and set(event) == {
                "kind",
                "t",
                "previous",
                "value",
            }:
                if validated(event["previous"]) != expected:
                    raise NetworkError("Control history mismatch")
                expected = validated(event["value"])
            elif event.get("kind") in ("flow_zero", "shaft_zero") and set(event) == {
                "kind",
                "t",
                "direction",
            }:
                if type(event["direction"]) is not int or event["direction"] not in (-1, 1):
                    raise NetworkError("Invalid crossing direction")
            else:
                raise NetworkError("Unknown pump-storage event")
        if active != expected:
            raise NetworkError("Active controls do not match history")
        if not isinstance(payload["state"], list) or any(
            type(v) not in (int, float) for v in payload["state"]
        ):
            raise NetworkError("Numeric pump-storage state required")
        state = np.array(payload["state"], dtype=float)
        model._resistance = active["resistance_pa_s2_kg2"]
        model._torque = active["motor_torque_nm"]
        model.observation(state, t)
        if t == 0 and not np.array_equal(state, model._state):
            raise NetworkError("Modified initial pump-storage state")
        model._state, model.time_s, model.events = state, t, deepcopy(payload["events"])
        return model
