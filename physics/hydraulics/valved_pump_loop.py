"""Finite-leakage modulating restriction and exact first-order signal actuator.

K(a)=Kopen+(Kclosed-Kopen)*(1-a)^2. The restriction heats the fluid once.
No perfect shutoff/check-valve contact, swept-volume work or actuator energy.
"""

import hashlib
import json
from copy import copy, deepcopy

import numpy as np

from core.network import NetworkError, number
from core.status import ModelPassport, ModelStatus
from physics.hydraulics.pump_loop import PASSPORT as PUMP_PASSPORT
from physics.hydraulics.pump_loop import PumpCirculationLoop

PASSPORT = ModelPassport(
    model_id="valved-displacement-pump-circulation-loop",
    version="1.0.0",
    status=ModelStatus.IMPLEMENTED,
    phenomenon="Passive finite-leakage restriction with modulating signal actuator",
    approximations=(
        "finite Kclosed >= Kopen >= 0; both flow signs supported",
        "exact first-order opening response between piecewise constant commands",
        "zero swept volume and no actuator energy; valve dissipation heats liquid",
        "constant-property single-cycle displacement-pump model",
        "no perfect shutoff, check-valve contact, cavitation, acoustics or plant calibration",
    ),
    applicability="Finite resistance valve in the declared reduced liquid circuit",
    data_sources=("K(a)=Kopen+(Kclosed-Kopen)*(1-a)^2; da/dt=(command-a)/tau",),
)
VALVE_FIELDS = {
    "segment",
    "open_loss_coefficient",
    "closed_loss_coefficient",
    "initial_opening",
    "actuator_time_s",
}


class ValvedPumpLoop(PumpCirculationLoop):
    def __init__(self, loop_initial, pump, initial_omega_rad_s, motor_torque_nm, valve):
        if not isinstance(valve, dict) or set(valve) != VALVE_FIELDS:
            raise NetworkError("Exact finite-leakage valve parameters required")
        self._valve = {k: number(v, k) for k, v in valve.items() if k != "segment"}
        self._valve["segment"] = valve["segment"]
        p = self._valve
        if (
            not isinstance(p["segment"], str)
            or p["open_loss_coefficient"] < 0
            or p["closed_loss_coefficient"] < p["open_loss_coefficient"]
            or p["closed_loss_coefficient"] <= 0
            or p["actuator_time_s"] <= 0
            or not 0 <= p["initial_opening"] <= 1
        ):
            raise NetworkError("Invalid finite-leakage valve coefficient, opening or time constant")
        self._epoch = 0.0
        self._epoch_opening = p["initial_opening"]
        self._target = p["initial_opening"]
        super().__init__(loop_initial, pump, initial_omega_rad_s, motor_torque_nm)

    @property
    def valve(self):
        return dict(self._valve)

    def opening(self, time_s=None):
        t = self.time_s if time_s is None else number(time_s, "valve time")
        if t < self._epoch:
            raise NetworkError("Valve evaluation before current command epoch")
        return float(
            self._target
            + (self._epoch_opening - self._target)
            * np.exp(-(t - self._epoch) / self._valve["actuator_time_s"])
        )

    def _effective(self, time_s):
        if self._valve["segment"] not in self._loop.names:
            raise NetworkError("Unknown valve segment")
        a = self.opening(time_s)
        loss = (
            self._valve["open_loss_coefficient"]
            + (self._valve["closed_loss_coefficient"] - self._valve["open_loss_coefficient"])
            * (1 - a) ** 2
        )
        resistance = loss / (2 * self._loop.rho * self._loop.area**2)
        if not np.isfinite(resistance):
            raise NetworkError("Nonfinite valve resistance")
        loop = copy(self._loop)
        loop.resistance = self._loop.resistance.copy()
        index = loop.names.index(self._valve["segment"])
        loop.resistance[index] += resistance
        loop.resistance.setflags(write=False)
        loop.resistance_total = float(sum(loop.resistance))
        # A plain evaluator avoids recursive virtual dispatch and mutates no live state.
        model = object.__new__(PumpCirculationLoop)
        model.__dict__ = {
            k: v
            for k, v in self.__dict__.items()
            if k not in ("derivatives", "analytic_jacobian", "observation")
        }
        model._loop = loop
        return model, a, resistance

    def derivatives(self, t, y):
        return self._effective(t)[0].derivatives(t, y)

    def analytic_jacobian(self, t, y):
        # Opening is an exact function of time, not an additional solver unknown.
        return self._effective(t)[0].analytic_jacobian(t, y)

    def observation(self, y=None, time_s=None):
        t = self.time_s if time_s is None else time_s
        model, a, resistance = self._effective(t)
        o = model.observation(y, t)
        flow = o["loop"]["mass_flow_kg_s"]
        o.update(
            valve_opening=a,
            valve_target=self._target,
            valve_pressure_loss_pa=resistance * flow * abs(flow),
            valve_dissipation_w=resistance * abs(flow) ** 3 / self._loop.rho,
        )
        if not np.isfinite([o["valve_pressure_loss_pa"], o["valve_dissipation_w"]]).all():
            raise NetworkError("Nonfinite valve pressure or dissipation")
        return o

    def set_valve_opening(self, target):
        target = number(target, "valve target")
        if not 0 <= target <= 1:
            raise NetworkError("Valve target must be between zero and one")
        opening = self.opening()
        previous = self._target
        saved = self._epoch, self._epoch_opening, self._target
        try:
            self._epoch, self._epoch_opening, self._target = self.time_s, opening, target
            self.derivatives(self.time_s, self._state)
            o = self.observation()
        except BaseException:
            self._epoch, self._epoch_opening, self._target = saved
            raise
        event = dict(
            kind="valve_target_changed",
            t=self.time_s,
            previous_target=previous,
            opening=opening,
            target=target,
        )
        self._continuation = None
        self._policy = None
        self.events.append(event)
        self.last_events = (event,)
        return o

    def configuration_sha256(self):
        plain = object.__new__(PumpCirculationLoop)
        plain.__dict__ = self.__dict__.copy()
        return hashlib.sha256(
            json.dumps(
                dict(pump_configuration_sha256=plain.configuration_sha256(), valve=self.valve),
                sort_keys=True,
                allow_nan=False,
            ).encode()
        ).hexdigest()

    def snapshot(self):
        payload = PumpCirculationLoop.snapshot(self)
        payload.update(model_id=PASSPORT.model_id, model_version=PASSPORT.version, valve=self.valve)
        return payload

    @classmethod
    def from_snapshot(cls, payload):
        if (
            not isinstance(payload, dict)
            or payload.get("model_id") != PASSPORT.model_id
            or payload.get("model_version") != PASSPORT.version
        ):
            raise NetworkError("Incompatible valved-loop checkpoint")
        model = cls(
            payload.get("loop_initial"),
            payload.get("pump"),
            payload.get("initial_omega_rad_s"),
            payload.get("initial_motor_torque_nm"),
            payload.get("valve"),
        )
        if payload.get("configuration_sha256") != model.configuration_sha256():
            raise NetworkError("Valved-loop configuration hash mismatch")
        raw = deepcopy(payload)
        raw.pop("valve")
        raw.update(
            model_id=PUMP_PASSPORT.model_id,
            model_version=PUMP_PASSPORT.version,
            configuration_sha256=PumpCirculationLoop.configuration_sha256(model),
        )
        if not isinstance(raw.get("events"), list):
            raise NetworkError("Valved-loop events required")
        t = number(raw.get("time_s"), "checkpoint time")
        previous_t = 0.0
        filtered = []
        for event in raw["events"]:
            if not isinstance(event, dict):
                raise NetworkError("Invalid valved-loop event")
            stamp = number(event.get("t"), "event time")
            if not previous_t <= stamp <= t:
                raise NetworkError("Unordered valved-loop event")
            previous_t = stamp
            if event.get("kind") == "valve_target_changed":
                if set(event) != {"kind", "t", "previous_target", "opening", "target"}:
                    raise NetworkError("Invalid valve command record")
                previous = number(event["previous_target"], "previous valve target")
                opening = number(event["opening"], "recorded valve opening")
                target = number(event["target"], "valve target")
                if (
                    previous != model._target
                    or not 0 <= target <= 1
                    or abs(opening - model.opening(stamp)) > 1e-12
                ):
                    raise NetworkError("Valve command history mismatch")
                model._epoch, model._epoch_opening, model._target = stamp, opening, target
            else:
                filtered.append(event)
        raw["events"] = filtered
        plain = PumpCirculationLoop.from_snapshot(raw)
        model._state, model.time_s = plain._state.copy(), plain.time_s
        model._torque = plain._torque
        model.events = deepcopy(payload["events"])
        model.observation()
        return model
