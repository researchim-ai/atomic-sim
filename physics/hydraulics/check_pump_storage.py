"""Event-driven unilateral flow constraint on the coupled pump/storage model.

Open: I*mdot=dp+H-p_crack-R*m^2; p_crack*m/rho heats right liquid.
Closed: m=0, inertial constraint reaction balances drive; gas/shaft still evolve.
This is an ideal zero-speed seat, not flap mechanics, impact or water hammer.
"""

import hashlib
import json
from copy import copy, deepcopy

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import csc_matrix

from core.network import NetworkError, number
from core.status import ModelPassport, ModelStatus
from physics.hydraulics.pump_storage_pair import CONTROL_FIELDS, PumpStoragePair

RELEASE_PA = 1e-6
PRESSURE_ROOT_PA = 1e-5
FLOW_ROOT_KG_S = 1e-10
PROJECTION_J = 1e-12
CONFIG_FIELDS = {"plant_configuration", "cracking_pressure_pa"}
PASSPORT = ModelPassport(
    model_id="check-valve-pump-storage",
    version="1.0.0",
    status=ModelStatus.IMPLEMENTED,
    phenomenon="Hybrid unilateral inertial flow with zero-speed seating and pressure-driven release",
    approximations=(
        "ideal massless check valve; open m>=0, closed m=0, no flap/impact energy",
        "cracking-pressure work heats the right vessel while open",
        "resolved release at positive drive 1e-6 Pa; event pressure residual <=1e-5 Pa",
        "zero-flow root residual <=1e-10 kg/s; kinetic projection <=1e-12 J",
        "coupled reduced displacement pump, ideal hydraulic inertance and gas cushions",
        "no ideal commanded shutoff, water hammer, cavitation or steam pressurizer",
    ),
    applicability="Nonnegative accepted flow, finite nonnegative cracking pressure and positive inventories",
    data_sources=("m>=0; closed reaction balances dp+H-p_crack; seat reached at m=0",),
)


class CheckPumpStorage:
    def __init__(self, plant_configuration, cracking_pressure_pa):
        if not isinstance(plant_configuration, dict):
            raise NetworkError("Explicit pump-storage configuration required")
        self._plant = PumpStoragePair(**deepcopy(plant_configuration))
        self._crack = number(cracking_pressure_pa, "cracking_pressure_pa")
        if self._crack < 0:
            raise NetworkError("Nonnegative cracking pressure required")
        self._config = dict(
            plant_configuration=self._plant.configuration, cracking_pressure_pa=self._crack
        )
        self.names = self._plant.names
        self.rho = self._plant.rho
        self._state = self._plant._state.copy()
        self._controls = self._plant.controls
        if self._state[8] < 0:
            raise NetworkError("Check-valve initial flow must be nonnegative")
        self.time_s = 0.0
        self.events = []
        self.last_events = ()
        self._mode = (
            "open" if self._state[8] > 0 or self.drive(self._state) > RELEASE_PA else "closed"
        )
        self._initial_mode = self._mode
        self.observation()

    @property
    def configuration(self):
        return deepcopy(self._config)

    @property
    def controls(self):
        return dict(self._controls)

    @property
    def mode(self):
        return self._mode

    def configuration_sha256(self):
        return hashlib.sha256(
            json.dumps(self.configuration, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()

    def _view(self):
        view = copy(self._plant)
        view._resistance = self._controls["resistance_pa_s2_kg2"]
        view._torque = self._controls["motor_torque_nm"]
        return view

    def drive(self, y):
        view = self._view()
        q, omega, head, torque, leak, bearing = view._machine(y)
        r = view._view()._rates(np.asarray(y)[:9])
        value = float(r["pressure"][0] - r["pressure"][1] + head - self._crack)
        if not np.isfinite(value):
            raise NetworkError("Nonfinite check drive")
        return value

    def derivatives(self, t, y, mode=None):
        phase = self._mode if mode is None else mode
        if phase not in ("open", "closed"):
            raise NetworkError("Unknown check phase")
        state = np.asarray(y, dtype=float).copy()
        if state.shape != (14,) or not np.isfinite(state).all():
            raise NetworkError("Invalid check-valve state")
        if phase == "closed":
            state[8] = 0.0  # Smooth extension of the constrained m=0 subspace.
        rate = self._view().derivatives(t, state)
        if phase == "open":
            rate[8] -= self._crack / self._plant._hydraulic._inertia
            rate[2] += self._crack * state[8] / self.rho
        else:
            rate[8] = 0.0
        if not np.isfinite(rate).all():
            raise NetworkError("Nonfinite check phase rates")
        return rate

    def analytic_jacobian(self, t, y, mode=None):
        phase = self._mode if mode is None else mode
        if phase not in ("open", "closed"):
            raise NetworkError("Unknown check phase")
        state = np.asarray(y, dtype=float).copy()
        if state.shape != (14,) or not np.isfinite(state).all():
            raise NetworkError("Invalid check-valve state")
        if phase == "closed":
            state[8] = 0.0
        jac = self._view().analytic_jacobian(t, state).toarray()
        if phase == "open":
            jac[2, 8] += self._crack / self.rho
        else:
            jac[:, 8] = 0.0
            jac[8, :] = 0.0
        if not np.isfinite(jac).all():
            raise NetworkError("Nonfinite check phase Jacobian")
        return csc_matrix(jac)

    def observation(self, y=None, time_s=None, mode=None):
        state = self._state if y is None else np.asarray(y, dtype=float)
        phase = self._mode if mode is None else mode
        if phase not in ("open", "closed"):
            raise NetworkError("Unknown check phase")
        if (
            state.shape != (14,)
            or not np.isfinite(state).all()
            or state[8] < 0
            or (phase == "closed" and state[8] != 0)
        ):
            raise NetworkError(
                "Accepted check state must satisfy nonnegative flow and exact closed constraint"
            )
        view = self._view()
        o = view.observation(state, self.time_s if time_s is None else time_s)
        drive = self.drive(state)
        if phase == "closed" and drive > RELEASE_PA + 1e-8:
            raise NetworkError("Closed valve violates resolved release condition")
        rate = self.derivatives(o["time_s"], state, phase)
        loss = self._crack * state[8] / self.rho if phase == "open" else 0.0
        right = self.names[1]
        o["ports"][right + ".transfer"]["energy_flow_w"] += loss
        o["ports"]["connection.right"]["energy_flow_w"] -= loss
        kinetic_rate = self._plant._hydraulic._inertia * state[8] * rate[8] / self.rho
        link = (
            sum(o["ports"]["connection." + end]["energy_flow_w"] for end in ("left", "right"))
            + o["shaft_to_pump_power_w"]
            - o["leakage_power_w"]
            - kinetic_rate
        )
        core = view._view()
        r = core._rates(state[:9])
        local = {}
        receiver = view._heat_index
        for i, name in enumerate(self.names):
            local[name] = float(
                rate[1 + i]
                + rate[3 + i]
                - o["ports"][name + ".transfer"]["energy_flow_w"]
                - core._storage._heater[i]
                - r["ambient_heat"][i]
                - (o["leakage_power_w"] if i == receiver else 0)
            )
        if (
            not np.isfinite([drive, loss, link, *local.values()]).all()
            or max(abs(link), max(abs(x) for x in local.values())) > 1e-6
        ):
            raise NetworkError("Check-valve local energy closure exceeded")
        o.update(
            check_mode=phase,
            check_drive_pa=drive,
            check_reaction_pa=max(-drive, 0.0) if phase == "closed" else 0.0,
            check_constraint_residual_pa=max(drive, 0.0) if phase == "closed" else 0.0,
            check_dissipation_w=float(loss),
            flow_acceleration_kg_s2=float(rate[8]),
            link_energy_residual_w=float(link),
            local_energy_residual_w=local,
        )
        return o

    def advance(self, dt_s, *, method="Radau", rtol=1e-9, max_step_s=0.02, atol_scale=1.0):
        dt = number(dt_s, "dt_s", positive=True)
        rtol = number(rtol, "rtol", positive=True)
        step = number(max_step_s, "max_step_s", positive=True)
        scale = number(atol_scale, "atol_scale", positive=True)
        if method not in ("Radau", "BDF") or not 1e-12 <= rtol <= 1e-2 or not 1e-6 <= scale <= 1:
            raise NetworkError("Unsupported check-valve solver")
        target = self.time_s + dt
        if not np.isfinite(target) or target <= self.time_s:
            raise NetworkError("Target time cannot advance")
        saved = self._state.copy(), self.time_s, self._mode, deepcopy(self.events), self.last_events
        state = self._state.copy()
        now = self.time_s
        phase = self._mode
        new_events = []
        try:
            for _ in range(128):
                if now >= target:
                    break
                start = now

                def switch_event(t, y):
                    value = self.drive(y) - RELEASE_PA if phase == "closed" else y[8]
                    return (-1e-30 if phase == "closed" else 1e-30) if value == 0 else value

                switch_event.terminal = True
                switch_event.direction = 1 if phase == "closed" else -1

                def shaft_event(t, y):
                    return float(y[9]) if y[9] != 0 else 1e-30

                shaft_event.terminal = False
                shaft_event.direction = 0
                result = solve_ivp(
                    lambda t, y: self.derivatives(t, y, phase),
                    (now, target),
                    state,
                    method=method,
                    rtol=rtol,
                    atol=np.r_[1e-12, np.full(7, 1e-8), 1e-12, 1e-11, np.full(4, 1e-8)] * scale,
                    max_step=step,
                    jac=lambda t, y: self.analytic_jacobian(t, y, phase),
                    dense_output=False,
                    events=(switch_event, shaft_event),
                )
                if not result.success:
                    raise NetworkError("Check-valve segment solver failed: " + result.message)
                for t, y in zip(result.t_events[1], result.y_events[1]):
                    if t > start + 1e-12:
                        speed = self.derivatives(float(t), y, phase)[9]
                        if speed != 0:
                            new_events.append(
                                dict(
                                    kind="shaft_zero", t=float(t), direction=1 if speed > 0 else -1
                                )
                            )
                state = result.y[:, -1].copy()
                now = float(result.t[-1])
                if len(result.t_events[0]) == 0:
                    self.observation(state, now, phase)
                    break
                if now < start:
                    raise NetworkError("Check event moved backwards in time")
                drive = self.drive(state)
                if phase == "open":
                    residual = float(state[8])
                    projection = self._plant._hydraulic._inertia * residual**2 / (2 * self.rho)
                    if (
                        abs(residual) > FLOW_ROOT_KG_S
                        or projection > PROJECTION_J
                        or drive > RELEASE_PA + 1e-8
                    ):
                        raise NetworkError(
                            "Check seating root exceeded flow/energy/force tolerance"
                        )
                    state[8] = 0.0
                    phase = "closed"
                    new_events.append(
                        dict(
                            kind="check_closed",
                            t=now,
                            flow_residual_kg_s=residual,
                            kinetic_projection_j=projection,
                            drive_pa=drive,
                        )
                    )
                else:
                    if abs(drive - RELEASE_PA) > PRESSURE_ROOT_PA or drive <= 0:
                        raise NetworkError("Check release root residual exceeded")
                    phase = "open"
                    new_events.append(dict(kind="check_opened", t=now, drive_pa=drive))
                self.observation(state, now, phase)
            else:
                raise NetworkError("Check-valve transition resource limit exceeded")
            if now != target:
                raise NetworkError("Incomplete check-valve interval")
            observed = self.observation(state, target, phase)
            new_events.sort(key=lambda e: (e["t"], e["kind"]))
            self._state, self.time_s, self._mode = state, target, phase
            self.events.extend(new_events)
            self.last_events = tuple(new_events)
        except BaseException:
            self._state, self.time_s, self._mode, self.events, self.last_events = saved
            raise
        return observed

    def set_controls(self, controls):
        if not isinstance(controls, dict) or set(controls) != CONTROL_FIELDS:
            raise NetworkError("Complete check-assembly controls required")
        proposed = {k: number(v, k) for k, v in controls.items()}
        if proposed["resistance_pa_s2_kg2"] < 0:
            raise NetworkError("Nonnegative finite resistance required")
        previous = self.controls
        try:
            self._controls = proposed
            self.derivatives(self.time_s, self._state)
            o = self.observation()
        except BaseException:
            self._controls = previous
            raise
        event = dict(kind="controls_changed", t=self.time_s, previous=previous, value=self.controls)
        self.events.append(event)
        self.last_events = (event,)
        return o

    def snapshot(self):
        return dict(
            schema_version=1,
            model_id=PASSPORT.model_id,
            model_version=PASSPORT.version,
            configuration=self.configuration,
            configuration_sha256=self.configuration_sha256(),
            state=self._state.tolist(),
            time_s=self.time_s,
            initial_mode=self._initial_mode,
            mode=self._mode,
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
            "initial_mode",
            "mode",
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
            raise NetworkError("Incompatible check-valve checkpoint")
        if (
            not isinstance(payload["configuration"], dict)
            or set(payload["configuration"]) != CONFIG_FIELDS
        ):
            raise NetworkError("Invalid check configuration")
        model = cls(**payload["configuration"])
        if (
            model.configuration_sha256() != payload["configuration_sha256"]
            or payload["initial_mode"] != model._initial_mode
        ):
            raise NetworkError("Check configuration or initial mode mismatch")
        t = number(payload["time_s"], "checkpoint time")
        if t < 0 or not isinstance(payload["events"], list):
            raise NetworkError("Invalid check checkpoint time/events")

        def controls(c):
            if not isinstance(c, dict) or set(c) != CONTROL_FIELDS:
                raise NetworkError("Invalid check controls")
            d = {k: number(v, k) for k, v in c.items()}
            if d["resistance_pa_s2_kg2"] < 0:
                raise NetworkError("Negative checkpoint resistance")
            return d

        active = controls(payload["active_controls"])
        expected = model.controls
        mode = model._initial_mode
        previous_t = 0.0
        for e in payload["events"]:
            if not isinstance(e, dict):
                raise NetworkError("Invalid check event")
            stamp = number(e.get("t"), "event time")
            if not previous_t <= stamp <= t:
                raise NetworkError("Unordered check event")
            previous_t = stamp
            kind = e.get("kind")
            if kind == "controls_changed" and set(e) == {"kind", "t", "previous", "value"}:
                if controls(e["previous"]) != expected:
                    raise NetworkError("Check control history mismatch")
                expected = controls(e["value"])
            elif kind == "check_closed" and set(e) == {
                "kind",
                "t",
                "flow_residual_kg_s",
                "kinetic_projection_j",
                "drive_pa",
            }:
                flow = number(e["flow_residual_kg_s"], "root flow")
                energy = number(e["kinetic_projection_j"], "root energy")
                drive = number(e["drive_pa"], "drive")
                exact = model._plant._hydraulic._inertia * flow**2 / (2 * model.rho)
                if (
                    mode != "open"
                    or abs(flow) > FLOW_ROOT_KG_S
                    or not 0 <= energy <= PROJECTION_J
                    or abs(energy - exact) > 1e-18
                    or drive > RELEASE_PA + 1e-8
                ):
                    raise NetworkError("Invalid seating event")
                mode = "closed"
            elif kind == "check_opened" and set(e) == {"kind", "t", "drive_pa"}:
                drive = number(e["drive_pa"], "drive")
                if mode != "closed" or drive <= 0 or abs(drive - RELEASE_PA) > PRESSURE_ROOT_PA:
                    raise NetworkError("Invalid release event")
                mode = "open"
            elif kind == "shaft_zero" and set(e) == {"kind", "t", "direction"}:
                if type(e["direction"]) is not int or e["direction"] not in (-1, 1):
                    raise NetworkError("Invalid shaft crossing")
            else:
                raise NetworkError("Unknown check checkpoint event")
        if expected != active or mode != payload["mode"]:
            raise NetworkError("Check active controls/mode history mismatch")
        if not isinstance(payload["state"], list) or any(
            type(v) not in (int, float) for v in payload["state"]
        ):
            raise NetworkError("Numeric check state required")
        state = np.array(payload["state"], dtype=float)
        model._controls = active
        model.observation(state, t, mode)
        if t == 0 and not np.array_equal(state, model._state):
            raise NetworkError("Modified initial check state")
        model._state, model.time_s, model._mode, model.events = (
            state,
            t,
            mode,
            deepcopy(payload["events"]),
        )
        return model
