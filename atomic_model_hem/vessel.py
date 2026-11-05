"""Closed rigid equilibrium-water vessel: exact prescribed heat inventory evolution.

Mass and volume fixed, no flux, momentum, moving boundary, or empirical heat
transfer closure. Phase events are intersections of fixed v with saturation
branches. This is a thermodynamic HEM building block, not a flow engine.
"""

import json
import math
import threading
from copy import deepcopy

from scipy.optimize import brentq

from atomic_model_hem import eos
from atomic_model_workspace.common import digest, keys

MAX_CALLS, MAX_SAMPLES, MAX_EVENTS = 128, 512, 256


class RigidEquilibriumVessel:
    def __init__(self, request):
        keys(
            request,
            (
                "schema_version",
                "volume_m3",
                "initial",
                "initial_power_w",
                "maximum_energy_segment_j_kg",
            ),
        )
        if type(request["schema_version"]) is not int or request["schema_version"] != 1:
            raise ValueError("Vessel schema1 required")
        self.request = deepcopy(request)
        self.volume = eos.number(request["volume_m3"], "volume")
        self.segment = eos.number(request["maximum_energy_segment_j_kg"], "energy segment")
        if not 1e-6 <= self.volume <= 100 or not 100 <= self.segment <= 20000:
            raise ValueError("Vessel volume/segment outside supported budget")
        initial = request["initial"]
        if initial.get("kind") == "pt":
            keys(initial, ("kind", "pressure_pa", "temperature_k"))
            state = eos.pt(initial["pressure_pa"], initial["temperature_k"])
        elif initial.get("kind") == "px":
            keys(initial, ("kind", "pressure_pa", "quality"))
            state = eos.px(initial["pressure_pa"], initial["quality"])
        else:
            raise ValueError("Initial state must be explicit pt or px")
        self.initial = state
        self.mass = self.volume * state.density_kg_m3
        self._power = eos.number(request["initial_power_w"], "power")
        if abs(self._power) > 1e8:
            raise ValueError("Vessel heat-power budget")
        self._time = 0.0
        self._heat = 0.0
        self._events = []
        self._calls = []
        self._controller = None
        self._state = state
        self._boundaries = self._phase_boundaries()
        self.identity = digest(request)
        self.source = eos.source_identity()
        self._lock = threading.RLock()

    def _sources(self):
        if eos.source_identity() != self.source:
            raise ValueError("HEM source changed")

    def _phase_boundaries(self):
        target = 1 / self.initial.density_kg_m3
        result = []
        for quality, name in [(0.0, "liquid"), (1.0, "vapor")]:

            def residual(p):
                return 1 / eos.px(float(p), quality).density_kg_m3 - target

            lo, hi = residual(eos.P_MIN), residual(eos.P_MAX)
            if lo * hi < 0:
                p = brentq(residual, eos.P_MIN, eos.P_MAX, xtol=1e-5, rtol=1e-13)
            elif lo == 0:
                p = eos.P_MIN
            elif hi == 0:
                p = eos.P_MAX
            else:
                continue
            state = eos.px(float(p), quality)
            result.append(
                dict(
                    branch=name,
                    heat_j=self.mass
                    * (state.internal_energy_j_kg - self.initial.internal_energy_j_kg),
                    state=state,
                )
            )
        return sorted(result, key=lambda r: r["heat_j"])

    def _observation(self, time, heat, power, state, events, controller, revision):
        return dict(
            time_s=time,
            mass_kg=self.mass,
            volume_m3=self.volume,
            internal_energy_j=self.mass * self.initial.internal_energy_j_kg + heat,
            internal_energy_change_j=heat,
            external_heat_j=heat,
            energy_balance_residual_j=0.0,
            mass_balance_residual_kg=0.0,
            eos_energy_residual_j=self.mass * state.internal_energy_j_kg
            - (self.mass * self.initial.internal_energy_j_kg + heat),
            eos_mass_residual_kg=state.density_kg_m3 * self.volume - self.mass,
            power_w=power,
            thermodynamic_state=state.as_dict(),
            entropy_change_j_k=self.mass * (state.entropy_j_kg_k - self.initial.entropy_j_kg_k),
            entropy_accounting="State diagnostic only; no independent entropy-transfer/production ledger",
            revision=revision,
            events=deepcopy(events),
            controller_state=deepcopy(controller),
            validated=False,
            plant_referenced=False,
        )

    def snapshot(self):
        with self._lock:
            self._sources()
            return self._observation(
                self._time,
                self._heat,
                self._power,
                self._state,
                self._events,
                self._controller,
                len(self._calls),
            )

    def _checkpoint(self, time, heat, power, state, events, controller, calls):
        payload = dict(
            schema_version=1,
            model="if97-rigid-equilibrium-vessel-v1",
            input_sha256=self.identity,
            source=deepcopy(self.source),
            calls=deepcopy(calls),
            observation=self._observation(time, heat, power, state, events, controller, len(calls)),
        )
        raw = json.dumps(payload, allow_nan=False)
        if len(raw.encode()) > 4 * 1024**2:
            raise ValueError("Vessel checkpoint size budget")
        return dict(payload=payload, sha256=digest(payload))

    def _execute(self, duration, power, controller, command):
        self._sources()
        dt = eos.number(duration, "duration")
        power = eos.number(power, "power")
        if (
            not 0 < dt <= 10000
            or abs(power) > 1e8
            or len(self._calls) >= MAX_CALLS
            or self._time + dt <= self._time
        ):
            raise ValueError("Vessel call/time/power budget")
        target = self._heat + power * dt
        if power != 0 and (power * dt == 0 or target == self._heat):
            raise ValueError("Unrepresentable heat increment")
        if not math.isfinite(target) or not math.isfinite(self._time + dt):
            raise ValueError("Nonfinite inventory/clock")
        count = max(1, math.ceil(abs(target - self._heat) / self.mass / self.segment))
        if count > MAX_SAMPLES:
            raise ValueError("Vessel state-sample budget exceeded")
        fractions = {i / count for i in range(1, count + 1)}
        crossing = []
        for boundary in self._boundaries:
            q = boundary["heat_j"]
            if (self._heat < q <= target) or (target <= q < self._heat):
                fraction = (q - self._heat) / (target - self._heat)
                fractions.add(fraction)
                crossing.append((fraction, boundary))
        state = self._state
        events = deepcopy(self._events)
        samples = []
        for fraction in sorted(fractions):
            boundary = next((b for f, b in crossing if f == fraction), None)
            heat = boundary["heat_j"] if boundary else self._heat + (target - self._heat) * fraction
            u = self.initial.internal_energy_j_kg + heat / self.mass
            state = (
                boundary["state"]
                if boundary
                else eos.rho_u(
                    self.initial.density_kg_m3,
                    u,
                    pressure_seed_pa=state.pressure_pa,
                    temperature_seed_k=state.temperature_k,
                )
            )
            if (
                abs(state.density_kg_m3 - self.initial.density_kg_m3)
                > eos.RHO_RTOL * self.initial.density_kg_m3
                or abs(state.internal_energy_j_kg - u) > eos.U_ATOL
            ):
                raise ValueError("Vessel EOS inventory residual")
            time = self._time + dt * fraction
            if boundary:
                if time <= self._time:
                    raise ValueError("Unrepresentable phase-event clock")
                events.append(
                    dict(
                        sequence=len(events) + 1,
                        time_s=time,
                        branch=boundary["branch"],
                        direction="heating" if power > 0 else "cooling",
                        external_heat_j=heat,
                        state=state.as_dict(),
                    )
                )
            samples.append(dict(time_s=time, state=state.as_dict()))
        if len(events) > MAX_EVENTS:
            raise ValueError("Vessel phase-event budget")
        calls = self._calls + [dict(duration_s=dt, command=deepcopy(command))]
        cp = self._checkpoint(self._time + dt, target, power, state, events, controller, calls)
        self._sources()
        self._time += dt
        self._heat = target
        self._power = power
        self._state = state
        self._events, self._calls, self._controller = events, calls, deepcopy(controller)
        return dict(**cp["payload"]["observation"], samples=samples)

    def advance(self, duration_s):
        with self._lock:
            return self._execute(duration_s, self._power, self._controller, None)

    def step(self, power_w, duration_s, *, expected_revision, controller_state):
        with self._lock:
            if type(expected_revision) is not int or expected_revision != len(self._calls):
                raise ValueError("Stale vessel revision")
            # Explicit finite JSON state, identical before and after serialization.
            raw = json.dumps(controller_state, allow_nan=False, sort_keys=True)
            if len(raw.encode()) > 32768 or json.loads(raw) != controller_state:
                raise ValueError("Invalid controller state")
            normalized = json.loads(raw)
            return self._execute(
                duration_s,
                power_w,
                normalized,
                dict(
                    power_w=eos.number(power_w, "power"),
                    expected_revision=expected_revision,
                    controller_state=normalized,
                ),
            )

    def checkpoint(self):
        with self._lock:
            self._sources()
            return self._checkpoint(
                self._time,
                self._heat,
                self._power,
                self._state,
                self._events,
                self._controller,
                self._calls,
            )

    def restore(self, checkpoint):
        with self._lock:
            if len(json.dumps(checkpoint, allow_nan=False).encode()) > 4 * 1024**2:
                raise ValueError("Vessel checkpoint size budget")
            self._sources()
            keys(checkpoint, ("payload", "sha256"))
            p = checkpoint["payload"]
            if (
                digest(p) != checkpoint["sha256"]
                or p.get("source") != self.source
                or p.get("input_sha256") != self.identity
            ):
                raise ValueError("Vessel checkpoint identity mismatch")
            calls = p.get("calls")
            if not isinstance(calls, list) or len(calls) > MAX_CALLS:
                raise ValueError("Vessel replay call budget")
            other = RigidEquilibriumVessel(self.request)
            for call in calls:
                keys(call, ("duration_s", "command"))
                command = call["command"]
                if command is None:
                    other.advance(call["duration_s"])
                else:
                    keys(command, ("power_w", "expected_revision", "controller_state"))
                    other.step(
                        command["power_w"],
                        call["duration_s"],
                        expected_revision=command["expected_revision"],
                        controller_state=command["controller_state"],
                    )
            if other.checkpoint() != checkpoint:
                raise ValueError("Vessel checkpoint full replay differs")
            self._sources()
            self._time, self._heat, self._power, self._state = (
                other._time,
                other._heat,
                other._power,
                other._state,
            )
            self._events, self._calls, self._controller = (
                other._events,
                other._calls,
                other._controller,
            )
            return self.snapshot()
