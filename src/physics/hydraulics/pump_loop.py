"""Passive displacement pump with leakage, shaft inertia and a closed liquid loop.

q=D*omega-C*H, torque_hyd=D*H; no divisions by speed or efficiency.
Leakage C*H**2 heats the selected casing wall; shaft drag heats an external
sink. This is a lumped displacement machine, not a centrifugal pump curve.
"""

import hashlib
import json
from copy import deepcopy

import numpy as np
from scipy.optimize import brentq
from scipy.sparse import csc_matrix

from core.continuation import ImplicitContinuation
from core.network import NetworkError, PortSpec, number
from core.status import ModelPassport, ModelStatus
from physics.hydraulics.circulation_loop import CirculationLoop

PASSPORT = ModelPassport(
    model_id="displacement-pump-circulation-loop",
    version="1.0.0",
    status=ModelStatus.IMPLEMENTED,
    phenomenon="Reciprocal hydraulic/shaft power transfer with passive leakage and drag",
    approximations=(
        "fixed displacement, positive linear leakage, linear bearing drag",
        "signed flow, speed and head; no efficiency or speed division",
        "leakage heat enters the selected casing wall; drag heat leaves to external sink",
        "constant-property uniform-area single cycle; no cavitation, acoustics or phase change",
        "no centrifugal characteristics, manufacturer map or plant calibration",
    ),
    applicability="Reduced displacement machine and one closed incompressible liquid cycle",
    data_sources=("q=D*omega-C*H; tau=D*H; J*omega_dot=tau_motor-tau-B*omega",),
)
SHAFT_PORT = PortSpec("shaft", "mechanical")
PARAMETERS = {"segment", "displacement_m3_rad", "leakage_m3_s_pa", "inertia_kg_m2", "drag_nm_s_rad"}


class PumpCirculationLoop:
    def __init__(self, loop_initial, pump, initial_omega_rad_s, motor_torque_nm):
        # Detached immutable configuration: integration never mutates the embedded loop.
        self._loop = CirculationLoop.from_snapshot(deepcopy(loop_initial))
        if (
            self._loop.time_s != 0
            or self._loop.events
            or not np.array_equal(
                self._loop._state, np.r_[self._loop.initial_flow, np.zeros(2 * self._loop.n + 2)]
            )
        ):
            raise NetworkError("Pump requires an initial loop checkpoint, not an evolved state")
        if np.any(self._loop._initial_heads) or np.any(self._loop._heads):
            raise NetworkError("External ideal heads are not allowed in a shaft-driven loop")
        if not isinstance(pump, dict) or set(pump) != PARAMETERS:
            raise NetworkError("Exact displacement pump parameters required")
        if not isinstance(pump["segment"], str) or pump["segment"] not in self._loop.names:
            raise NetworkError("Unknown pump segment")
        self._pump = dict(segment=pump["segment"])
        for key in PARAMETERS - {"segment"}:
            self._pump[key] = number(pump[key], key, positive=key != "drag_nm_s_rad")
        if self._pump["drag_nm_s_rad"] < 0:
            raise NetworkError("Negative bearing drag is unsupported")
        self._index = self._loop.names.index(pump["segment"])
        self._size = len(self._loop._state)
        self._omega0 = number(initial_omega_rad_s, "initial_omega_rad_s")
        self._torque0 = number(motor_torque_nm, "motor_torque_nm")
        self._torque = self._torque0
        self._state = np.r_[self._loop._state, self._omega0, 0.0, 0.0, 0.0]
        self.time_s = 0.0
        self.events = []
        self.last_events = ()
        self._continuation = None
        self._policy = None
        self._horizon = None
        self.observation()

    @property
    def pump(self):
        return dict(self._pump)

    def _machine(self, y):
        y = np.asarray(y, dtype=float)
        if y.shape != (self._size + 4,) or not np.isfinite(y).all():
            raise NetworkError("Invalid pump/shaft state")
        p = self._pump
        q = float(y[0]) / self._loop.rho
        omega = float(y[self._size])
        head = (p["displacement_m3_rad"] * omega - q) / p["leakage_m3_s_pa"]
        torque = p["displacement_m3_rad"] * head
        leak = p["leakage_m3_s_pa"] * head**2
        bearing = p["drag_nm_s_rad"] * omega**2
        heads = np.zeros(self._loop.n)
        heads[self._index] = head
        if not np.isfinite([head, torque, leak, bearing]).all():
            raise NetworkError("Nonfinite pump power or head")
        return q, omega, head, torque, leak, bearing, heads

    def derivatives(self, _t, y):
        q, omega, head, torque, leak, bearing, heads = self._machine(y)
        loop = self._loop
        rates = loop._rates(y[: self._size], heads)["rhs"]
        # Leakage is deposited in the pump casing, not counted as liquid friction.
        rates[1 + loop.n + self._index] += leak / loop.wall_capacity[self._index]
        rates[-1] += leak
        result = np.r_[
            rates,
            (self._torque - torque - self._pump["drag_nm_s_rad"] * omega)
            / self._pump["inertia_kg_m2"],
            self._torque * omega,
            bearing,
            leak,
        ]
        if not np.isfinite(result).all():
            raise NetworkError("Nonfinite pump/shaft derivatives")
        return result

    def analytic_jacobian(self, _t, y):
        q, omega, head, torque, leak, bearing, heads = self._machine(y)
        loop, p, k = self._loop, self._pump, self._size
        loop._rates(y[:k], heads)  # Validate the actual pressure domain before differentiation.
        j = np.zeros((k + 4, k + 4))
        j[:k, :k] = loop.analytic_jacobian(_t, y[:k]).toarray()
        hm = -1 / (loop.rho * p["leakage_m3_s_pa"])
        hw = p["displacement_m3_rad"] / p["leakage_m3_s_pa"]
        j[0, 0] += hm / loop.inertia_total
        j[0, k] = hw / loop.inertia_total
        j[k - 2, 0] = head / loop.rho + q * hm
        j[k - 2, k] = q * hw
        for col, dh in ((0, hm), (k, hw)):
            dl = 2 * p["leakage_m3_s_pa"] * head * dh
            j[1 + loop.n + self._index, col] += dl / loop.wall_capacity[self._index]
            j[k - 1, col] += dl
            j[k + 3, col] = dl
            j[k, col] = -p["displacement_m3_rad"] * dh / p["inertia_kg_m2"]
        j[k, k] -= p["drag_nm_s_rad"] / p["inertia_kg_m2"]
        j[k + 1, k] = self._torque
        j[k + 2, k] = 2 * p["drag_nm_s_rad"] * omega
        if not np.isfinite(j).all():
            raise NetworkError("Nonfinite pump/shaft Jacobian")
        return csc_matrix(j)

    def observation(self, y=None, time_s=None):
        y = self._state if y is None else np.asarray(y, dtype=float)
        q, omega, head, torque, leak, bearing, heads = self._machine(y)
        loop, p, k = self._loop, self._pump, self._size
        if min(y[k + 2], y[k + 3]) < -loop.energy_tolerance_j:
            raise NetworkError("Negative cumulative passive heat")
        observed = loop.observation(y[:k], self.time_s if time_s is None else time_s, heads)
        shaft_change = 0.5 * p["inertia_kg_m2"] * (omega - self._omega0) * (omega + self._omega0)
        external_heat = observed["external_heat_j"] - y[k + 3]
        residual = (
            observed["stored_energy_change_j"] + shaft_change - y[k + 1] + y[k + 2] - external_heat
        )
        shaft_residual = (
            shaft_change - y[k + 1] + y[k + 2] + observed["external_head_work_j"] + y[k + 3]
        )
        transfer_residual = torque * omega - head * q - leak
        if (
            not np.isfinite([residual, shaft_residual, transfer_residual]).all()
            or max(abs(residual), abs(shaft_residual)) > loop.energy_tolerance_j
            or abs(transfer_residual) > 1e-6
        ):
            raise NetworkError("Pump/shaft total-energy or reciprocal power closure exceeded")
        if not np.isfinite([self._torque * omega, head * q, torque * omega]).all():
            raise NetworkError("Nonfinite shaft power")
        return dict(
            time_s=observed["time_s"],
            loop=observed,
            omega_rad_s=omega,
            head_pa=head,
            hydraulic_torque_nm=torque,
            motor_torque_nm=self._torque,
            motor_power_w=self._torque * omega,
            hydraulic_power_w=head * q,
            shaft_to_pump_power_w=torque * omega,
            leakage_power_w=leak,
            bearing_power_w=bearing,
            shaft_energy_change_j=shaft_change,
            motor_work_j=float(y[k + 1]),
            bearing_heat_j=float(y[k + 2]),
            leakage_heat_j=float(y[k + 3]),
            external_ambient_heater_heat_j=float(external_heat),
            total_energy_residual_j=float(residual),
            shaft_energy_residual_j=float(shaft_residual),
            power_transfer_residual_w=float(transfer_residual),
        )

    def enable_continuation(self, horizon_s):
        horizon = number(horizon_s, "horizon_s", positive=True)
        if horizon <= self.time_s:
            raise NetworkError("Future pump horizon required")
        self._horizon = horizon

    def advance(self, dt_s, *, method="Radau", rtol=1e-9, max_step_s=0.05, atol_scale=1.0):
        dt = number(dt_s, "dt_s", positive=True)
        rtol = number(rtol, "rtol", positive=True)
        step = number(max_step_s, "max_step_s", positive=True)
        scale = number(atol_scale, "atol_scale", positive=True)
        if method not in ("Radau", "BDF") or not 1e-12 <= rtol <= 1e-2 or not 1e-6 <= scale <= 1:
            raise NetworkError("Unsupported pump solver policy")
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
                atol = (
                    np.r_[
                        1e-11, np.full(2 * self._loop.n, 1e-12), 1e-8, 1e-8, 1e-11, 1e-8, 1e-8, 1e-8
                    ]
                    * scale
                )
                self._continuation = ImplicitContinuation(
                    self.derivatives,
                    self._state,
                    self.time_s,
                    max(target, self._horizon or target),
                    rtol=rtol,
                    atol=atol,
                    max_step=step,
                    method=method,
                    initial_target=target,
                    jacobian=self.analytic_jacobian,
                )
                self._policy = policy
            result = self._continuation.advance_to(target)
            obs = self.observation(result.final, target)
            events = []
            for index, kind in ((0, "flow_zero"), (self._size, "shaft_zero")):
                for i in range(len(result.t) - 1):
                    a, b = result.y[index, i : i + 2]
                    direction = 1 if a < 0 <= b else -1 if a > 0 >= b else 0
                    if direction:
                        t = (
                            float(result.t[i + 1])
                            if b == 0
                            else brentq(
                                lambda t: float(result.dense(t)[index]),
                                float(result.t[i]),
                                float(result.t[i + 1]),
                                xtol=1e-12,
                            )
                        )
                        events.append(dict(kind=kind, t=t, direction=direction))
            events.sort(key=lambda e: (e["t"], e["kind"]))
            self._state, self.time_s = result.final.copy(), target
            self.events.extend(events)
            self.last_events = tuple(events)
        except BaseException:
            self._state, self.time_s, self.events, self.last_events = saved
            self._continuation = None
            raise
        return obs

    def set_motor_torque(self, torque_nm):
        torque = number(torque_nm, "motor_torque_nm")
        # Validate the current physical state before committing a discontinuity.
        self.observation()
        event = dict(
            kind="motor_torque_changed", t=self.time_s, previous_nm=self._torque, torque_nm=torque
        )
        old = self._torque
        try:
            self._torque = torque
            self.derivatives(self.time_s, self._state)
            observed = self.observation()
        except BaseException:
            self._torque = old
            raise
        self._continuation = None
        self._policy = None
        self.events.append(event)
        self.last_events = (event,)
        return observed

    def configuration_sha256(self):
        return hashlib.sha256(
            json.dumps(
                dict(
                    loop_initial=self._loop.snapshot(),
                    pump=self.pump,
                    initial_omega_rad_s=self._omega0,
                    motor_torque_nm=self._torque0,
                ),
                sort_keys=True,
                allow_nan=False,
            ).encode()
        ).hexdigest()

    def snapshot(self):
        return dict(
            schema_version=1,
            model_id=PASSPORT.model_id,
            model_version=PASSPORT.version,
            loop_initial=self._loop.snapshot(),
            pump=self.pump,
            initial_omega_rad_s=self._omega0,
            initial_motor_torque_nm=self._torque0,
            active_motor_torque_nm=self._torque,
            state=self._state.tolist(),
            time_s=self.time_s,
            events=deepcopy(self.events),
            configuration_sha256=self.configuration_sha256(),
        )

    @classmethod
    def from_snapshot(cls, payload):
        fields = {
            "schema_version",
            "model_id",
            "model_version",
            "loop_initial",
            "pump",
            "initial_omega_rad_s",
            "initial_motor_torque_nm",
            "active_motor_torque_nm",
            "state",
            "time_s",
            "events",
            "configuration_sha256",
        }
        if (
            not isinstance(payload, dict)
            or set(payload) != fields
            or type(payload["schema_version"]) is not int
            or payload["schema_version"] != 1
            or payload["model_id"] != PASSPORT.model_id
            or payload["model_version"] != PASSPORT.version
        ):
            raise NetworkError("Incompatible pump checkpoint")
        model = cls(
            payload["loop_initial"],
            payload["pump"],
            payload["initial_omega_rad_s"],
            payload["initial_motor_torque_nm"],
        )
        if model.configuration_sha256() != payload["configuration_sha256"]:
            raise NetworkError("Pump configuration hash mismatch")
        t = number(payload["time_s"], "time_s")
        torque = number(payload["active_motor_torque_nm"], "active_motor_torque_nm")
        if t < 0 or not isinstance(payload["events"], list):
            raise NetworkError("Invalid pump checkpoint time/events")
        previous_t, expected = 0.0, model._torque0
        for e in payload["events"]:
            if not isinstance(e, dict):
                raise NetworkError("Invalid pump event")
            stamp = number(e.get("t"), "event time")
            if not previous_t <= stamp <= t:
                raise NetworkError("Unordered pump event")
            previous_t = stamp
            if e.get("kind") == "motor_torque_changed" and set(e) == {
                "kind",
                "t",
                "previous_nm",
                "torque_nm",
            }:
                if number(e["previous_nm"], "previous torque") != expected:
                    raise NetworkError("Pump torque event chain mismatch")
                expected = number(e["torque_nm"], "torque")
            elif e.get("kind") in ("flow_zero", "shaft_zero") and set(e) == {
                "kind",
                "t",
                "direction",
            }:
                if type(e["direction"]) is not int or e["direction"] not in (-1, 1):
                    raise NetworkError("Invalid crossing direction")
            else:
                raise NetworkError("Unknown pump checkpoint event")
        if expected != torque:
            raise NetworkError("Active torque does not match event chain")
        if not isinstance(payload["state"], list) or any(
            type(v) not in (float, int) for v in payload["state"]
        ):
            raise NetworkError("Numeric pump checkpoint state required")
        state = np.array(payload["state"], dtype=float)
        model._torque = torque
        model.observation(state, t)
        if t == 0 and not np.array_equal(state, model._state):
            raise NetworkError("Modified initial pump state")
        model._state, model.time_s, model.events = state, t, deepcopy(payload["events"])
        return model


def steady_pump_loop(loop_model, pump, motor_torque_nm):
    """Solve signed mechanical equilibrium and actual thermal residual, without warmup."""
    from physics.hydraulics.circulation_loop import steady_circulation

    if not isinstance(loop_model, dict) or set(loop_model) != {
        "topology",
        "fluid",
        "reference",
        "ambient_temperature_k",
        "energy_tolerance_j",
    }:
        raise NetworkError("Exact loop model required for pump equilibrium")
    names = [node["id"] for node in loop_model["topology"]["nodes"]]
    temperatures = dict.fromkeys(names, loop_model["ambient_temperature_k"])
    probe = CirculationLoop(
        **loop_model,
        initial_mass_flow_kg_s=0.0,
        initial_temperature_k=temperatures,
        initial_wall_temperature_k=temperatures,
    )
    validated = PumpCirculationLoop(probe.snapshot(), pump, 0.0, motor_torque_nm)
    p, torque = validated.pump, validated._torque
    displacement, leakage, drag = p["displacement_m3_rad"], p["leakage_m3_s_pa"], p["drag_nm_s_rad"]
    alpha = probe.resistance_total * (displacement + drag * leakage / displacement)
    beta = drag / (displacement * probe.rho)
    magnitude = (
        0.0
        if torque == 0
        else 2 * abs(torque) / (beta + np.sqrt(beta**2 + 4 * alpha * abs(torque)))
    )
    flow = float(np.copysign(magnitude, torque))
    head = probe.resistance_total * flow * abs(flow)
    omega = (flow / probe.rho + leakage * head) / displacement
    heated = deepcopy(loop_model)
    for node in heated["topology"]["nodes"]:
        if node["id"] == p["segment"]:
            node["parameters"]["drive_pressure_pa"] = head
            node["parameters"]["wall_power_w"] += leakage * head**2
    thermal = steady_circulation(**heated)
    initial = thermal.observation()
    loop = CirculationLoop(
        **loop_model,
        initial_mass_flow_kg_s=flow,
        initial_temperature_k=initial["fluid_temperature_k"],
        initial_wall_temperature_k=initial["wall_temperature_k"],
    )
    result = PumpCirculationLoop(loop.snapshot(), p, float(omega), torque)
    rates = result.derivatives(0.0, result._state)
    if (
        abs(probe.inertia_total * rates[0]) > 1e-7
        or abs(p["inertia_kg_m2"] * rates[result._size]) > 1e-9
        or max(abs(np.r_[probe.capacity, probe.wall_capacity] * rates[1 : 2 * probe.n + 1])) > 1e-6
    ):
        raise NetworkError("Pump steady dimensional residual acceptance failed")
    return result
