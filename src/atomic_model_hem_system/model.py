"""Joint conservative IF97 fluid inventories and finite thermal walls.

Two terminal control volumes are same-area 1D Euler cells, not stationary
well-mixed plenums. Their outer ends reflect velocity and do no energy work.
All interface transports and wall exchanges are integrated with the stocks.
"""

import hashlib
import json
import threading
from copy import deepcopy
from pathlib import Path

import numpy as np

from atomic_model_hem import eos
from atomic_model_hem_flow.model import PeriodicHEMChannel, characteristic_speeds
from atomic_model_hem_flow.model import signature as flow_signature
from atomic_model_workspace.common import digest, keys
from physics.materials import iapws

MAX_STEPS, MAX_CALLS = 4096, 128


def signature():
    return dict(
        flow=flow_signature(),
        system={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.glob("*.py"))
        },
    )


def isochoric_capacity(state):
    """Positive branch-specific cv for explicit conductive stiffness admission."""
    if state.region != 4:
        value = float(iapws.properties_pt(state.pressure_pa, state.temperature_k).cv_j_kg_k)
    else:
        p, x = state.pressure_pa, state.quality
        step = p * 1e-5
        a, b = eos.px(p + step, x), eos.px(p - step, x)
        f, g = iapws.saturation_properties(p)
        vp = (1 / a.density_kg_m3 - 1 / b.density_kg_m3) / (2 * step)
        up = (a.internal_energy_j_kg - b.internal_energy_j_kg) / (2 * step)
        tp = (a.temperature_k - b.temperature_k) / (2 * step)
        value = float((up - (g.u_j_kg - f.u_j_kg) * vp / (g.v_m3_kg - f.v_m3_kg)) / tp)
        if x <= 1e-5:
            value = min(value, float(f.cv_j_kg_k))
        if x >= 1 - 1e-5:
            value = min(value, float(g.cv_j_kg_k))
    if not np.isfinite(value) or value <= 0:
        raise ValueError("Invalid isochoric thermal capacity")
    return value


class FiniteHEMSystem:
    def __init__(self, request):
        keys(request, ("schema_version", "channel", "terminals", "walls"))
        if type(request["schema_version"]) is not int or request["schema_version"] != 1:
            raise ValueError("Finite HEM system schema1 required")
        self.request = deepcopy(request)
        channel = deepcopy(request["channel"])
        keys(request["terminals"], ("left", "right"))
        internal = channel["geometry"]["cells"]
        if type(internal) is not int or not 4 <= internal <= 12:
            raise ValueError("Finite HEM internal cell budget")
        left, right = request["terminals"]["left"], request["terminals"]["right"]
        for terminal in [left, right]:
            keys(terminal, ("volume_m3", "initial", "heat_power_w"))
            if not 1e-5 <= eos.number(terminal["volume_m3"], "terminal volume") <= 1:
                raise ValueError("Terminal volume domain")
        self.n = internal + 2
        self.names = ("left",) + tuple("c" + str(i) for i in range(internal)) + ("right",)
        area = eos.number(channel["geometry"]["area_m2"], "area")
        length = eos.number(channel["geometry"]["length_m"], "length")
        if not 1e-5 <= area <= 1 or not 0.1 <= length <= 100:
            raise ValueError("System channel geometry domain")
        combined = deepcopy(channel)
        combined["geometry"] = dict(
            cells=self.n,
            length_m=length + (left["volume_m3"] + right["volume_m3"]) / area,
            area_m2=area,
        )
        combined["initial"] = (
            [deepcopy(left["initial"])]
            + deepcopy(channel["initial"])
            + [deepcopy(right["initial"])]
        )
        combined["heat_power_w"] = (
            [left["heat_power_w"]] + list(channel["heat_power_w"]) + [right["heat_power_w"]]
        )
        parsed = PeriodicHEMChannel(combined)
        self.area = parsed.area
        self.wave = parsed.wave
        self.cfl = parsed.cfl
        self.step_cap = parsed.step_cap
        self.volumes = np.array(
            [left["volume_m3"]] + [length / internal * area] * internal + [right["volume_m3"]],
            dtype=float,
        )
        self.minimum_length = float(min(self.volumes) / area)
        self.states0 = parsed.states0
        self.eref = parsed.eref
        v = parsed.P0 / parsed.m0
        self.m0 = np.array([s.density_kg_m3 * vol for s, vol in zip(self.states0, self.volumes)])
        self.P0 = self.m0 * v
        self.C0 = (
            np.array(
                [m * (s.internal_energy_j_kg - self.eref) for m, s in zip(self.m0, self.states0)]
            )
            + 0.5 * self.m0 * v * v
        )
        self.s0 = np.array([s.entropy_j_kg_k for s in self.states0])
        walls = request["walls"]
        if not isinstance(walls, list) or not 1 <= len(walls) <= 16:
            raise ValueError("Finite wall count budget")
        ids = []
        ports = []
        capacity = []
        temperature = []
        conductance = []
        ua = []
        ambient = []
        wall_power = []
        for wall in walls:
            keys(
                wall,
                (
                    "id",
                    "cell",
                    "capacity_j_k",
                    "initial_temperature_k",
                    "conductance_w_k",
                    "heat_power_w",
                    "ambient_ua_w_k",
                    "ambient_temperature_k",
                ),
            )
            name = wall["id"]
            if (
                not isinstance(name, str)
                or not name.isascii()
                or not name.replace("_", "").isalnum()
                or len(name) > 32
                or name in ids
                or wall["cell"] not in self.names
            ):
                raise ValueError("Invalid wall id/port")
            c = eos.number(wall["capacity_j_k"], "wall capacity")
            t = eos.number(wall["initial_temperature_k"], "wall temperature")
            k = eos.number(wall["conductance_w_k"], "wall conductance")
            a = eos.number(wall["ambient_ua_w_k"], "ambient conductance")
            ta = eos.number(wall["ambient_temperature_k"], "ambient temperature")
            if (
                not 100 <= c <= 1e7
                or not 0 <= k <= 1e5
                or not 0 <= a <= 1e5
                or not 300 <= t <= 900
                or not 300 <= ta <= 900
            ):
                raise ValueError("Wall parameter domain")
            ids.append(name)
            ports.append(self.names.index(wall["cell"]))
            capacity.append(c)
            temperature.append(t)
            conductance.append(k)
            ua.append(a)
            ambient.append(ta)
            wall_power.append(wall["heat_power_w"])
        self.wall_ids = tuple(ids)
        self.w = len(ids)
        self.ports = np.array(ports)
        self.capacity = np.array(capacity)
        self.tw0 = np.array(temperature)
        self.conductance = np.array(conductance)
        self.ua = np.array(ua)
        self.ambient = np.array(ambient)
        self.wall_slice = slice(3 * self.n, 3 * self.n + self.w)
        cursor = self.wall_slice.stop
        self.face_slice = slice(cursor, cursor + 3 * (self.n + 1))
        cursor = self.face_slice.stop
        self.cell_heat_slice = slice(cursor, cursor + self.n)
        cursor = self.cell_heat_slice.stop
        self.wall_exchange_slice = slice(cursor, cursor + self.w)
        cursor = self.wall_exchange_slice.stop
        self.wall_external_slice = slice(cursor, cursor + self.w)
        cursor = self.wall_external_slice.stop
        self.entropy_index, self.production_index = cursor, cursor + 1
        self.size = cursor + 2
        self._y = np.zeros(self.size)
        self._time = 0.0
        self._calls = []
        self._controller = None
        self._controls = self._mask(
            dict(
                fluid=dict(zip(self.names, combined["heat_power_w"])),
                walls=dict(zip(self.wall_ids, wall_power)),
            )
        )
        self.identity = digest(request)
        self.source = signature()
        self._lock = threading.RLock()
        self.evaluate(self._y, self._controls)

    def _mask(self, values):
        keys(values, ("fluid", "walls"))
        keys(values["fluid"], self.names)
        keys(values["walls"], self.wall_ids)
        fluid = np.array([eos.number(values["fluid"][name], "fluid power") for name in self.names])
        wall = np.array([eos.number(values["walls"][name], "wall power") for name in self.wall_ids])
        if max(abs(fluid)) > 1e6 or max(abs(wall)) > 1e6:
            raise ValueError("System heater budget")
        return dict(fluid=fluid, walls=wall)

    def control(self):
        with self._lock:
            self._sources()
            return dict(
                fluid=dict(zip(self.names, map(float, self._controls["fluid"]))),
                walls=dict(zip(self.wall_ids, map(float, self._controls["walls"]))),
            )

    def contract(self):
        return dict(
            model="finite-if97-channel-wall-system-v1",
            cells=list(self.names),
            walls=list(self.wall_ids),
            coordinates=[
                dict(name="fluid_mass_increment", count=self.n, unit="kg"),
                dict(name="fluid_momentum_increment", count=self.n, unit="kg m/s"),
                dict(name="fluid_shifted_energy_increment", count=self.n, unit="J"),
                dict(name="wall_temperature_increment", count=self.w, unit="K"),
            ],
            ledger_units=dict(
                face_mass="kg",
                face_momentum="kg m/s",
                face_energy="J",
                cell_heat="J",
                wall_exchange="J",
                wall_external="J",
                external_entropy="J/K",
                thermal_entropy_production="J/K",
            ),
            control_units=dict(fluid="W", walls="W"),
            outer_boundary="Fixed reflecting walls; no external mass/energy flux",
        )

    def _sources(self):
        if signature() != self.source:
            raise ValueError("Finite HEM system source changed")

    def primitive(self, y):
        if not isinstance(y, np.ndarray) or y.shape != (self.size,) or not np.isfinite(y).all():
            raise ValueError("Invalid system coordinate vector")
        m = self.m0 + y[: self.n]
        P = self.P0 + y[self.n : 2 * self.n]
        C = self.C0 + y[2 * self.n : 3 * self.n]
        if np.any(m <= 0):
            raise ValueError("Nonpositive system fluid mass")
        v = P / m
        u = self.eref + (C - 0.5 * m * v * v) / m
        states = []
        speeds = []
        cache = {}
        for i, seed in enumerate(self.states0):
            key = (float(m[i] / self.volumes[i]), float(u[i]), seed.pressure_pa, seed.temperature_k)
            if key not in cache:
                state = eos.rho_u(
                    key[0],
                    key[1],
                    pressure_seed_pa=seed.pressure_pa,
                    temperature_seed_k=seed.temperature_k,
                )
                cache[key] = (state, *characteristic_speeds(state))
            state, lower, upper = cache[key]
            if abs(v[i]) > 0.2 * lower or abs(v[i]) + upper > self.wave:
                raise ValueError("System Mach/wave bound exceeded")
            states.append(state)
            speeds.append(upper)
        tw = self.tw0 + y[self.wall_slice]
        if np.any(tw < 300) or np.any(tw > 900):
            raise ValueError("Wall temperature left domain")
        return m, P, C, v, states, tw, np.array(speeds)

    def evaluate(self, y, controls):
        keys(controls, ("fluid", "walls"))
        if any(
            not isinstance(controls[k], np.ndarray)
            or controls[k].shape != (size,)
            or not np.isfinite(controls[k]).all()
            for k, size in [("fluid", self.n), ("walls", self.w)]
        ):
            raise ValueError("Invalid system source profile")
        return self._evaluate_point(y, controls, self.primitive(y))

    def _evaluate_point(self, y, controls, point):
        m, P, C, v, states, tw, speeds = point
        pressure = np.array([s.pressure_pa for s in states])
        temperature = np.array([s.temperature_k for s in states])
        W = np.array([m, P, C]) / self.volumes
        F = np.array(
            [P / self.volumes, P * v / self.volumes + pressure, v * (C / self.volumes + pressure)]
        )
        ghost_left = W[:, 0] * np.array([1.0, -1.0, 1.0])
        ghost_right = W[:, -1] * np.array([1.0, -1.0, 1.0])
        fl = F[:, 0] * np.array([-1.0, 1.0, -1.0])
        fr = F[:, -1] * np.array([-1.0, 1.0, -1.0])
        wl = np.column_stack((ghost_left, W))
        wr = np.column_stack((W, ghost_right))
        flux = 0.5 * (np.column_stack((fl, F)) + np.column_stack((F, fr))) - 0.5 * self.wave * (
            wr - wl
        )
        if min(flux[1, 0], flux[1, -1]) <= 0:
            raise ValueError("Nonpositive numerical endwall force; reflected regime unsupported")
        exchange = self.conductance * (tw - temperature[self.ports])
        external = controls["walls"] + self.ua * (self.ambient - tw)
        fluid_heat = controls["fluid"].copy()
        np.add.at(fluid_heat, self.ports, exchange)
        rhs = np.zeros_like(y)
        rhs[: 3 * self.n] = (-self.area * np.diff(flux, axis=1)).reshape(-1)
        rhs[2 * self.n : 3 * self.n] += fluid_heat
        rhs[self.wall_slice] = (external - exchange) / self.capacity
        physical_flux = flux.copy()
        physical_flux[2] += self.eref * physical_flux[0]
        rhs[self.face_slice] = (self.area * physical_flux).reshape(-1)
        rhs[self.cell_heat_slice] = controls["fluid"]
        rhs[self.wall_exchange_slice] = exchange
        rhs[self.wall_external_slice] = external
        rhs[self.entropy_index] = sum(controls["fluid"] / temperature) + sum(
            controls["walls"] / tw + self.ua * (self.ambient - tw) / self.ambient
        )
        rhs[self.production_index] = sum(
            self.conductance * (tw - temperature[self.ports]) ** 2 / (tw * temperature[self.ports])
            + self.ua * (self.ambient - tw) ** 2 / (self.ambient * tw)
        )
        if not np.isfinite(rhs).all() or rhs[self.production_index] < 0:
            raise ValueError("Invalid finite HEM system rates")
        return rhs, dict(shared_flux=flux, wall_to_fluid_w=exchange, wall_external_w=external)

    def residual(self, state, rate):
        with self._lock:
            self._sources()
            if (
                not isinstance(rate, np.ndarray)
                or rate.shape != (self.size,)
                or not np.isfinite(rate).all()
            ):
                raise ValueError("Invalid system rate vector")
            return rate - self.evaluate(state, self._controls)[0]

    def _observe(self, y, time, controls, controller, revision, _point=None):
        m, P, C, v, states, tw, speeds = self.primitive(y) if _point is None else _point
        face = y[self.face_slice].reshape(3, self.n + 1)
        if max(abs(face[0, [0, -1]])) > 1e-10 or max(abs(face[2, [0, -1]])) > 1e-7:
            raise ValueError("Closed endwall transported mass or energy")
        dm = y[: self.n]
        dP = y[self.n : 2 * self.n]
        dE = y[2 * self.n : 3 * self.n] + self.eref * dm
        exchanges = np.zeros(self.n)
        np.add.at(exchanges, self.ports, y[self.wall_exchange_slice])
        mass_error = dm + np.diff(face[0])
        momentum_error = dP + np.diff(face[1])
        energy_error = dE + np.diff(face[2]) - y[self.cell_heat_slice] - exchanges
        wall_energy = self.capacity * y[self.wall_slice]
        wall_error = wall_energy - y[self.wall_external_slice] + y[self.wall_exchange_slice]
        global_energy = float(
            sum(dE)
            + sum(wall_energy)
            - sum(y[self.cell_heat_slice])
            - sum(y[self.wall_external_slice])
            - face[2, 0]
            + face[2, -1]
        )
        scale = max(
            1.0,
            float(max(abs(face[2]))),
            float(sum(abs(y[self.cell_heat_slice])) + sum(abs(y[self.wall_external_slice]))),
        )
        if (
            max(abs(mass_error)) > 1e-9
            or max(abs(momentum_error)) > 1e-8
            or max(abs(energy_error)) > 1e-5 + 1e-8 * scale
            or max(abs(wall_error)) > 1e-5 + 1e-8 * scale
            or abs(global_energy) > 1e-5 + 1e-8 * scale
        ):
            raise ValueError("System local/global conservative ledger exceeded")
        fluid_ds = float(
            sum(
                (m - self.m0) * self.s0
                + m * (np.array([s.entropy_j_kg_k for s in states]) - self.s0)
            )
        )
        wall_ds = float(sum(self.capacity * np.log1p(y[self.wall_slice] / self.tw0)))
        ds = fluid_ds + wall_ds
        return dict(
            time_s=float(time),
            revision=revision,
            controller_state=deepcopy(controller),
            controls=dict(
                fluid=dict(zip(self.names, map(float, controls["fluid"]))),
                walls=dict(zip(self.wall_ids, map(float, controls["walls"]))),
            ),
            cells=[
                dict(
                    id=self.names[i],
                    volume_m3=float(self.volumes[i]),
                    mass_kg=float(m[i]),
                    momentum_kg_m_s=float(P[i]),
                    total_energy_j=float(C[i] + self.eref * m[i]),
                    velocity_m_s=float(v[i]),
                    **states[i].as_dict(),
                )
                for i in range(self.n)
            ],
            walls=[
                dict(
                    id=name,
                    cell=self.names[self.ports[i]],
                    temperature_k=float(tw[i]),
                    energy_change_j=float(wall_energy[i]),
                    heat_to_fluid_j=float(y[self.wall_exchange_slice][i]),
                    external_heat_j=float(y[self.wall_external_slice][i]),
                    energy_residual_j=float(wall_error[i]),
                )
                for i, name in enumerate(self.wall_ids)
            ],
            local_ledgers=dict(
                mass_residual_kg=list(map(float, mass_error)),
                momentum_residual_kg_m_s=list(map(float, momentum_error)),
                energy_residual_j=list(map(float, energy_error)),
                integrated_face_mass_kg=list(map(float, face[0])),
                integrated_face_momentum_kg_m_s=list(map(float, face[1])),
                integrated_face_energy_j=list(map(float, face[2])),
            ),
            global_ledger=dict(
                mass_residual_kg=float(sum(dm) - face[0, 0] + face[0, -1]),
                momentum_residual_kg_m_s=float(sum(dP) - face[1, 0] + face[1, -1]),
                endwall_impulse_kg_m_s=float(face[1, 0] - face[1, -1]),
                external_fluid_heat_j=float(sum(y[self.cell_heat_slice])),
                external_wall_heat_j=float(sum(y[self.wall_external_slice])),
                energy_residual_j=global_energy,
                total_entropy_change_j_k=ds,
                external_entropy_j_k=float(y[self.entropy_index]),
                thermal_entropy_production_j_k=float(y[self.production_index]),
                numerical_entropy_residual_j_k=ds
                - float(y[self.entropy_index])
                - float(y[self.production_index]),
            ),
            maximum_characteristic_speed_m_s=float(max(abs(v) + speeds)),
            entropy_acceptance="Unforced closed-system total entropy nondecrease; forced entropy balance remains diagnostic",
            validated=False,
            plant_referenced=False,
        )

    def snapshot(self):
        with self._lock:
            self._sources()
            return self._observe(
                self._y, self._time, self._controls, self._controller, len(self._calls)
            )

    def _checkpoint(self, y, time, controls, controller, calls):
        payload = dict(
            schema_version=1,
            model="finite-if97-channel-wall-system-v1",
            input_sha256=self.identity,
            source=deepcopy(self.source),
            calls=deepcopy(calls),
            coordinates=list(map(float, y)),
            observation=self._observe(y, time, controls, controller, len(calls)),
        )
        if len(json.dumps(payload, allow_nan=False).encode()) > 4 * 1024**2:
            raise ValueError("System checkpoint budget")
        return dict(payload=payload, sha256=digest(payload))

    def _execute(self, duration, controls, controller, command):
        self._sources()
        duration = eos.number(duration, "duration")
        target = self._time + duration
        if not 0 < duration <= 0.1 or target <= self._time or len(self._calls) >= MAX_CALLS:
            raise ValueError("System duration/call budget")
        y = self._y.copy()
        time = self._time
        steps = 0
        maximum = 0.0
        minimum = float("inf")
        entropy_start = self._observe(y, time, controls, controller, len(self._calls))[
            "global_ledger"
        ]["total_entropy_change_j_k"]
        previous_entropy = entropy_start
        unforced = bool(
            np.all(controls["fluid"] == 0)
            and np.all(controls["walls"] == 0)
            and np.all(self.ua == 0)
        )
        point = self.primitive(y)
        while time < target:
            k1 = self._evaluate_point(y, controls, point)[0]
            m, P, C, v, states, tw, speeds = point
            internal = np.array([s.internal_energy_j_kg - self.eref for s in states])
            du = (
                k1[2 * self.n : 3 * self.n]
                - v * k1[self.n : 2 * self.n]
                + (0.5 * v * v - internal) * k1[: self.n]
            ) / m
            wall_rate = max(abs(k1[self.wall_slice]))
            conductance_by_cell = np.zeros(self.n)
            np.add.at(conductance_by_cell, self.ports, self.conductance)
            fluid_thermal_rate = max(
                (
                    k / (mass * isochoric_capacity(state))
                    for k, mass, state in zip(conductance_by_cell, m, states)
                    if k > 0
                ),
                default=0.0,
            )
            wall_thermal_rate = float(max((self.conductance + self.ua) / self.capacity))
            thermal_relax = 0.1 / max(fluid_thermal_rate, wall_thermal_rate, 1e-30)
            dt = min(
                self.step_cap,
                self.cfl * self.minimum_length / self.wave,
                target - time,
                thermal_relax,
                500 / max(float(max(abs(du))), 1e-30),
                0.25 / max(float(wall_rate), 1e-30),
            )
            if time + dt <= time or steps >= MAX_STEPS:
                raise ValueError("System step/clock budget")
            a = y + dt * k1
            k2 = self.evaluate(a, controls)[0]
            b = 0.75 * y + 0.25 * (a + dt * k2)
            k3 = self.evaluate(b, controls)[0]
            candidate = y / 3 + 2 * (b + dt * k3) / 3
            next_time = min(target, time + dt)
            point = self.primitive(candidate)
            observed = self._observe(
                candidate, next_time, controls, controller, len(self._calls) + 1, _point=point
            )
            entropy = observed["global_ledger"]["total_entropy_change_j_k"]
            if unforced and entropy < max(entropy_start, previous_entropy) - (
                1e-5 + 2e-7 * float(sum(self.m0))
            ):
                raise ValueError("System unforced entropy decreased")
            y, time = candidate, next_time
            previous_entropy = entropy
            steps += 1
            maximum = max(maximum, dt)
            minimum = min(minimum, dt)
        calls = self._calls + [dict(duration_s=duration, command=deepcopy(command))]
        cp = self._checkpoint(y, time, controls, controller, calls)
        self._sources()
        self._y, self._time, self._controls, self._controller, self._calls = (
            y,
            time,
            deepcopy(controls),
            deepcopy(controller),
            calls,
        )
        return dict(
            **cp["payload"]["observation"],
            steps=steps,
            maximum_step_s=maximum,
            minimum_step_s=minimum,
        )

    def advance(self, duration_s):
        with self._lock:
            return self._execute(duration_s, self._controls, self._controller, None)

    def step(self, controls, duration_s, *, expected_revision, controller_state):
        with self._lock:
            if type(expected_revision) is not int or expected_revision != len(self._calls):
                raise ValueError("Stale system revision")
            raw = json.dumps(controller_state, allow_nan=False, sort_keys=True)
            if len(raw.encode()) > 32768 or json.loads(raw) != controller_state:
                raise ValueError("Invalid system controller state")
            mask = self._mask(controls)
            state = json.loads(raw)
            return self._execute(
                duration_s,
                mask,
                state,
                dict(
                    controls=deepcopy(controls),
                    expected_revision=expected_revision,
                    controller_state=state,
                ),
            )

    def checkpoint(self):
        with self._lock:
            self._sources()
            return self._checkpoint(
                self._y, self._time, self._controls, self._controller, self._calls
            )

    def restore(self, checkpoint):
        with self._lock:
            self._sources()
            keys(checkpoint, ("payload", "sha256"))
            p = checkpoint["payload"]
            if (
                len(json.dumps(checkpoint, allow_nan=False).encode()) > 4 * 1024**2
                or digest(p) != checkpoint["sha256"]
                or p.get("source") != self.source
                or p.get("input_sha256") != self.identity
            ):
                raise ValueError("System checkpoint identity/size")
            calls = p.get("calls")
            if not isinstance(calls, list) or len(calls) > MAX_CALLS:
                raise ValueError("System replay budget")
            other = FiniteHEMSystem(self.request)
            for call in calls:
                keys(call, ("duration_s", "command"))
                c = call["command"]
                if c is None:
                    other.advance(call["duration_s"])
                else:
                    keys(c, ("controls", "expected_revision", "controller_state"))
                    other.step(
                        c["controls"],
                        call["duration_s"],
                        expected_revision=c["expected_revision"],
                        controller_state=c["controller_state"],
                    )
            if other.checkpoint() != checkpoint:
                raise ValueError("System full checkpoint replay differs")
            self._sources()
            self._y, self._time, self._controls, self._controller, self._calls = (
                other._y,
                other._time,
                other._controls,
                other._controller,
                other._calls,
            )
            return self.snapshot()
