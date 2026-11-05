"""Shared-flux periodic IF97 equilibrium Euler cells with atomic SSPRK3 steps.

Flat constant-area periodic domain, one velocity and equilibrium temperature.
No slip, gravity, wall capacitance, inlet/outlet, friction, choking or CHF model.
"""

import hashlib
import math
import threading
from copy import deepcopy
from pathlib import Path

import numpy as np

from atomic_model_hem import eos
from atomic_model_workspace.common import digest, keys
from physics.materials import iapws

MAX_STEPS = 4096
MAX_CALLS = 128


def signature():
    return dict(
        hem=eos.source_identity(),
        flow={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.glob("*.py"))
        },
    )


def characteristic_speeds(state):
    """Local characteristic bound; coexistence endpoint uses both one-sided branches.

    The phase-chart x column is exact, so no quality stencil leaves [0,1].
    No mass-averaged single-phase sound speed is used for a mixture.
    """
    p = state.pressure_pa
    if not eos.P_MIN + 100 < p < eos.P_MAX - 100:
        raise ValueError("Flow pressure outside derivative-margin domain")
    if state.region != 4:
        c = float(iapws.properties_pt(p, state.temperature_k).sound_speed_m_s)
        return c, c
    x = state.quality
    dp = p * 1e-5
    liquid, vapor = iapws.saturation_properties(p)

    def chart_speed_squared(step):
        a, b = eos.px(p + step, x), eos.px(p - step, x)
        j = np.array(
            [
                [
                    (1 / a.density_kg_m3 - 1 / b.density_kg_m3) / (2 * step),
                    vapor.v_m3_kg - liquid.v_m3_kg,
                ],
                [
                    (a.internal_energy_j_kg - b.internal_energy_j_kg) / (2 * step),
                    vapor.u_j_kg - liquid.u_j_kg,
                ],
            ]
        )
        inv = np.linalg.solve(j, np.array([[-1 / state.density_kg_m3**2, 0.0], [0.0, 1.0]]))
        return float(inv[0, 0] + inv[0, 1] * p / state.density_kg_m3**2)

    coarse, c2 = chart_speed_squared(dp), chart_speed_squared(dp / 2)
    if (
        not math.isfinite(c2)
        or not math.isfinite(coarse)
        or c2 <= 0
        or abs(coarse - c2) / c2 > 2e-3
    ):
        raise ValueError("Nonhyperbolic or unresolved equilibrium chart")
    speeds = [math.sqrt(c2)]
    if x <= 1e-5:
        speeds.append(float(liquid.sound_speed_m_s))
    if x >= 1 - 1e-5:
        speeds.append(float(vapor.sound_speed_m_s))
    return min(speeds), max(speeds)


def sound_bound(state):
    return characteristic_speeds(state)[1]


class PeriodicHEMChannel:
    def __init__(self, request):
        keys(
            request,
            (
                "schema_version",
                "geometry",
                "initial",
                "heat_power_w",
                "wave_speed_bound_m_s",
                "max_step_s",
                "cfl",
            ),
        )
        if type(request["schema_version"]) is not int or request["schema_version"] != 1:
            raise ValueError("HEM channel schema1 required")
        self.request = deepcopy(request)
        g = request["geometry"]
        keys(g, ("cells", "length_m", "area_m2"))
        self.n = g["cells"]
        if type(self.n) is not int or not 4 <= self.n <= 16:
            raise ValueError("HEM channel cell budget")
        length = eos.number(g["length_m"], "length")
        area = eos.number(g["area_m2"], "area")
        if not 0.1 <= length <= 100 or not 1e-5 <= area <= 1:
            raise ValueError("HEM geometry domain")
        self.dx = length / self.n
        self.area = area
        self.volume = self.dx * area
        self.wave = eos.number(request["wave_speed_bound_m_s"], "wave bound")
        self.step_cap = eos.number(request["max_step_s"], "max step")
        self.cfl = eos.number(request["cfl"], "CFL")
        if (
            not 100 <= self.wave <= 4000
            or not 1e-8 <= self.step_cap <= 0.001
            or not 0 < self.cfl <= 0.3
        ):
            raise ValueError("HEM step/wave budget")
        initial = request["initial"]
        if not isinstance(initial, list) or len(initial) != self.n:
            raise ValueError("HEM initial profile length")
        states = []
        velocities = []
        for point in initial:
            if not isinstance(point, dict):
                raise ValueError("HEM initial chart must be an object")
            if point.get("kind") == "pt":
                keys(point, ("kind", "pressure_pa", "temperature_k", "velocity_m_s"))
                s = eos.pt(point["pressure_pa"], point["temperature_k"])
            elif point.get("kind") == "px":
                keys(point, ("kind", "pressure_pa", "quality", "velocity_m_s"))
                s = eos.px(point["pressure_pa"], point["quality"])
            elif point.get("kind") == "rho_u":
                keys(
                    point,
                    (
                        "kind",
                        "density_kg_m3",
                        "internal_energy_j_kg",
                        "pressure_seed_pa",
                        "temperature_seed_k",
                        "velocity_m_s",
                    ),
                )
                s = eos.rho_u(
                    point["density_kg_m3"],
                    point["internal_energy_j_kg"],
                    pressure_seed_pa=point["pressure_seed_pa"],
                    temperature_seed_k=point["temperature_seed_k"],
                )
            else:
                raise ValueError("Unknown HEM initial phase chart")
            states.append(s)
            velocities.append(eos.number(point["velocity_m_s"], "velocity"))
        self.states0 = tuple(states)
        self.eref = states[0].internal_energy_j_kg
        self.m0 = np.array([s.density_kg_m3 * self.volume for s in states])
        v = np.array(velocities)
        self.P0 = self.m0 * v
        self.C0 = (
            np.array([m * (s.internal_energy_j_kg - self.eref) for m, s in zip(self.m0, states)])
            + 0.5 * self.m0 * v * v
        )
        self.s0 = np.array([s.entropy_j_kg_k for s in states])
        self._heat = self._heaters(request["heat_power_w"])
        self._y = np.zeros(3 * self.n + 2)
        self._time = 0.0
        self._calls = []
        self._controller = None
        self.identity = digest(request)
        self.source = signature()
        self._lock = threading.RLock()
        self.evaluate(self._y, self._heat)

    def _sources(self):
        if signature() != self.source:
            raise ValueError("HEM flow source changed")

    def _heaters(self, values):
        if not isinstance(values, list) or len(values) != self.n:
            raise ValueError("Full HEM heater profile required")
        result = np.array([eos.number(v, "heat power") for v in values])
        if max(abs(result)) > 1e6:
            raise ValueError("HEM heater budget")
        return result

    def primitive(self, y):
        if (
            not isinstance(y, np.ndarray)
            or y.shape != (3 * self.n + 2,)
            or not np.isfinite(y).all()
        ):
            raise ValueError("Invalid HEM coordinate shape/value")
        m = self.m0 + y[: self.n]
        momentum = self.P0 + y[self.n : 2 * self.n]
        c = self.C0 + y[2 * self.n : 3 * self.n]
        if np.any(m <= 0):
            raise ValueError("Nonpositive HEM cell mass")
        velocity = momentum / m
        u = self.eref + (c - 0.5 * m * velocity * velocity) / m
        states = []
        speeds = []
        cache = {}
        for i in range(self.n):
            rho = float(m[i] / self.volume)
            energy = float(u[i])
            seed = self.states0[i]
            key = (rho, energy, seed.pressure_pa, seed.temperature_k)
            if key not in cache:
                state = eos.rho_u(
                    rho,
                    energy,
                    pressure_seed_pa=seed.pressure_pa,
                    temperature_seed_k=seed.temperature_k,
                )
                cache[key] = (state, *characteristic_speeds(state))
            state, mach_speed, speed = cache[key]
            if abs(velocity[i]) > 0.2 * mach_speed or abs(velocity[i]) + speed > self.wave:
                raise ValueError("HEM Mach/wave bound exceeded")
            states.append(state)
            speeds.append(speed)
        return m, momentum, c, velocity, states, np.array(speeds)

    def evaluate(self, y, heaters):
        if (
            not isinstance(heaters, np.ndarray)
            or heaters.shape != (self.n,)
            or not np.isfinite(heaters).all()
        ):
            raise ValueError("Invalid HEM source profile")
        m, momentum, c, v, states, speeds = self.primitive(y)
        pressure = np.array([s.pressure_pa for s in states])
        temperature = np.array([s.temperature_k for s in states])
        w = np.array([m, momentum, c]) / self.volume
        flux = np.array(
            [
                m * v / self.volume,
                m * v * v / self.volume + pressure,
                v * (c / self.volume + pressure),
            ]
        )
        shared = 0.5 * (flux + np.roll(flux, -1, axis=1)) - 0.5 * self.wave * (
            np.roll(w, -1, axis=1) - w
        )
        rate = -self.area * (shared - np.roll(shared, 1, axis=1))
        rate[2] += heaters
        derivative = np.r_[rate.reshape(-1), sum(heaters), sum(heaters / temperature)]
        if not np.isfinite(derivative).all():
            raise ValueError("Nonfinite HEM flux")
        return derivative, shared

    def residual(self, state, rate):
        with self._lock:
            self._sources()
            if (
                not isinstance(rate, np.ndarray)
                or rate.shape != self._y.shape
                or not np.isfinite(rate).all()
            ):
                raise ValueError("Invalid HEM rates")
            return rate - self.evaluate(state, self._heat)[0]

    def _observe(self, y, time, heat, controller, revision):
        m, momentum, c, v, states, speeds = self.primitive(y)
        dm = float(sum(y[: self.n]))
        dp = float(sum(y[self.n : 2 * self.n]))
        denergy = float(sum(y[2 * self.n : 3 * self.n]) + self.eref * dm)
        ds = float(
            sum(
                (m - self.m0) * self.s0
                + m * (np.array([s.entropy_j_kg_k for s in states]) - self.s0)
            )
        )
        mass_tol = 1e-10 + 1e-10 * float(sum(self.m0))
        energy_tol = 1e-5 + 1e-9 * max(1.0, abs(float(y[-2])))
        if abs(dm) > mass_tol or abs(dp) > 1e-8 or abs(denergy - y[-2]) > energy_tol:
            raise ValueError("HEM conservative ledger exceeded")
        entropy_generated = ds - float(y[-1])
        return dict(
            time_s=float(time),
            revision=revision,
            heaters_w=list(map(float, heat)),
            controller_state=deepcopy(controller),
            cells=[
                dict(
                    mass_kg=float(m[i]),
                    momentum_kg_m_s=float(momentum[i]),
                    total_energy_j=float(c[i] + self.eref * m[i]),
                    velocity_m_s=float(v[i]),
                    **states[i].as_dict(),
                )
                for i in range(self.n)
            ],
            ledgers=dict(
                mass_residual_kg=dm,
                momentum_residual_kg_m_s=dp,
                total_energy_change_j=denergy,
                external_heat_j=float(y[-2]),
                energy_residual_j=denergy - float(y[-2]),
                entropy_change_j_k=ds,
                external_heat_entropy_j_k=float(y[-1]),
                numerical_entropy_production_j_k=entropy_generated,
            ),
            entropy_acceptance="Per-adiabatic-call total entropy inequality only; heated-history entropy accounting diagnostic, not a qualified second-law closure",
            maximum_characteristic_speed_m_s=float(max(abs(v) + speeds)),
            validated=False,
            plant_referenced=False,
        )

    def snapshot(self):
        with self._lock:
            self._sources()
            return self._observe(
                self._y, self._time, self._heat, self._controller, len(self._calls)
            )

    def _checkpoint(self, y, time, heat, controller, calls):
        payload = dict(
            schema_version=1,
            model="periodic-if97-hem-euler-v1",
            input_sha256=self.identity,
            source=deepcopy(self.source),
            calls=deepcopy(calls),
            coordinates=list(map(float, y)),
            observation=self._observe(y, time, heat, controller, len(calls)),
        )
        import json

        if len(json.dumps(payload, allow_nan=False).encode()) > 4 * 1024**2:
            raise ValueError("HEM checkpoint budget")
        return dict(payload=payload, sha256=digest(payload))

    def _execute(self, duration, heaters, controller, command):
        self._sources()
        duration = eos.number(duration, "duration")
        target = self._time + duration
        if not 0 < duration <= 0.1 or target <= self._time or len(self._calls) >= MAX_CALLS:
            raise ValueError("HEM duration/call budget")
        y = self._y.copy()
        time = self._time
        steps = 0
        maximum = 0.0
        entropy_start = self._observe(y, time, heaters, controller, len(self._calls))["ledgers"][
            "entropy_change_j_k"
        ]
        entropy_previous = entropy_start
        while time < target:
            dt = min(self.step_cap, self.cfl * self.dx / self.wave, target - time)
            if time + dt <= time or steps >= MAX_STEPS:
                raise ValueError("HEM step/clock budget")
            k1 = self.evaluate(y, heaters)[0]
            a = y + dt * k1
            k2 = self.evaluate(a, heaters)[0]
            b = 0.75 * y + 0.25 * (a + dt * k2)
            k3 = self.evaluate(b, heaters)[0]
            next_y = y / 3 + 2 * (b + dt * k3) / 3
            next_time = min(target, time + dt)
            observed = self._observe(next_y, next_time, heaters, controller, len(self._calls) + 1)
            entropy_current = observed["ledgers"]["entropy_change_j_k"]
            if np.all(heaters == 0) and entropy_current < max(entropy_start, entropy_previous) - (
                1e-5 + 2e-7 * float(sum(self.m0))
            ):
                raise ValueError("HEM adiabatic call entropy decreased")
            entropy_previous = entropy_current
            y, time = next_y, next_time
            steps += 1
            maximum = max(maximum, dt)
        calls = self._calls + [dict(duration_s=duration, command=deepcopy(command))]
        cp = self._checkpoint(y, time, heaters, controller, calls)
        self._sources()
        self._y, self._time, self._heat, self._controller, self._calls = (
            y,
            time,
            heaters.copy(),
            deepcopy(controller),
            calls,
        )
        return dict(**cp["payload"]["observation"], steps=steps, maximum_step_s=maximum)

    def advance(self, duration_s):
        with self._lock:
            return self._execute(duration_s, self._heat, self._controller, None)

    def step(self, heat_power_w, duration_s, *, expected_revision, controller_state):
        with self._lock:
            if type(expected_revision) is not int or expected_revision != len(self._calls):
                raise ValueError("Stale HEM flow revision")
            import json

            raw = json.dumps(controller_state, allow_nan=False, sort_keys=True)
            if len(raw.encode()) > 32768 or json.loads(raw) != controller_state:
                raise ValueError("Invalid HEM controller state")
            heat = self._heaters(heat_power_w)
            state = json.loads(raw)
            return self._execute(
                duration_s,
                heat,
                state,
                dict(
                    heat_power_w=list(map(float, heat)),
                    expected_revision=expected_revision,
                    controller_state=state,
                ),
            )

    def checkpoint(self):
        with self._lock:
            self._sources()
            return self._checkpoint(self._y, self._time, self._heat, self._controller, self._calls)

    def restore(self, checkpoint):
        with self._lock:
            import json

            self._sources()
            keys(checkpoint, ("payload", "sha256"))
            p = checkpoint["payload"]
            if (
                len(json.dumps(checkpoint, allow_nan=False).encode()) > 4 * 1024**2
                or digest(p) != checkpoint["sha256"]
                or p.get("source") != self.source
                or p.get("input_sha256") != self.identity
            ):
                raise ValueError("HEM checkpoint identity/size")
            calls = p.get("calls")
            if not isinstance(calls, list) or len(calls) > MAX_CALLS:
                raise ValueError("HEM replay budget")
            other = PeriodicHEMChannel(self.request)
            for call in calls:
                keys(call, ("duration_s", "command"))
                c = call["command"]
                if c is None:
                    other.advance(call["duration_s"])
                else:
                    keys(c, ("heat_power_w", "expected_revision", "controller_state"))
                    other.step(
                        c["heat_power_w"],
                        call["duration_s"],
                        expected_revision=c["expected_revision"],
                        controller_state=c["controller_state"],
                    )
            if other.checkpoint() != checkpoint:
                raise ValueError("HEM full checkpoint replay differs")
            self._sources()
            self._y, self._time, self._heat, self._controller, self._calls = (
                other._y,
                other._time,
                other._heat,
                other._controller,
                other._calls,
            )
            return self.snapshot()
