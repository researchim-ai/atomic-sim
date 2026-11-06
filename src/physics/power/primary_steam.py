"""One ODE for a shaft-driven liquid loop and an equilibrium steam power cycle.

The primary liquid has prescribed constant rho/cp, with a separate IF97
subcooling guard. The model is a reduced component assembly, not a reactor
core, centrifugal pump map, pressurizer or compressible primary circuit.
"""

import math
from copy import deepcopy

import numpy as np

from core.bounded_step import advance_with_limits
from core.network import NetworkError, number
from core.status import DomainError, ModelPassport, ModelStatus
from physics.hydraulics.pump_loop import PumpCirculationLoop
from physics.materials.iapws import saturation_properties
from physics.power.steam_cycle import SteamPowerCycle
from physics.power.steam_cycle import controls as steam_controls
from physics.thermal.two_stream_exchanger import digest

PASSPORT = ModelPassport(
    model_id="primary-loop-steam-power-cycle",
    version="1.0.0",
    status=ModelStatus.IMPLEMENTED,
    phenomenon="Monolithic pump/liquid/wall/steam/turbine assembly with reciprocal heat transfer",
    approximations=(
        "Constant-density/cp primary liquid; fixed pressure gauge, incompressible momentum",
        "Displacement pump with shaft inertia, leakage-to-casing and external bearing heat",
        "Finite primary walls and steam-generator wall; reciprocal linear thermal conductance",
        "IF97 saturation guard on primary fluid temperatures; no variable-property primary EOS",
        "Equilibrium two-vessel steam cycle with empirical admission and prescribed feed",
        "Local first-order rate continuation below 1e-12 initial physical scales; exact domain stocks",
        "No reactor neutronics, distributed boiling, pressurizer, acoustics or manufacturer maps",
    ),
    applicability="Reduced subcooled-water/steam component transients within explicit limits",
    data_sources=(
        "Q=UA*(Tprimary-Tsgwall); identical opposite thermal-port powers and integrals",
        "Primary loop total energy + steam cycle total energy; internal heat cancels once",
    ),
)
COUPLING_FIELDS = {
    "steam_generator_segment",
    "heater_segment",
    "ua_w_k",
    "minimum_subcooling_k",
    "min_primary_temperature_k",
    "min_primary_pressure_pa",
    "max_primary_pressure_pa",
    "energy_tolerance_j",
}


def controls(raw):
    if not isinstance(raw, dict) or set(raw) != {
        "primary_motor_torque_nm",
        "primary_wall_heater_power_w",
        "steam_cycle",
    }:
        raise NetworkError("Complete primary/steam control vector required")
    heater = number(raw["primary_wall_heater_power_w"], "primary wall heater")
    if heater < 0:
        raise NetworkError("Negative primary heater power")
    return dict(
        primary_motor_torque_nm=number(raw["primary_motor_torque_nm"], "primary motor torque"),
        primary_wall_heater_power_w=heater,
        steam_cycle=steam_controls(raw["steam_cycle"]),
    )


class PrimarySteamCycle:
    def __init__(self, primary_initial, steam_initial, coupling, boundary_controls):
        self._primary = PumpCirculationLoop.from_snapshot(primary_initial)
        self._steam = SteamPowerCycle.from_snapshot(steam_initial)
        p, s = self._primary, self._steam
        if p.time_s != 0 or p.events or s.time_s != 0 or s.events or np.any(s._state != 0):
            raise NetworkError("Untouched primary and steam initial snapshots required")
        if not np.array_equal(p._state, np.r_[p._loop._state, p._omega0, 0.0, 0.0, 0.0]):
            raise NetworkError("Evolved primary initial state")
        if not isinstance(coupling, dict) or set(coupling) != COUPLING_FIELDS:
            raise NetworkError("Exact primary/steam thermal coupling required")
        cfg = {}
        for key in ("steam_generator_segment", "heater_segment"):
            if not isinstance(coupling[key], str) or coupling[key] not in p._loop.names:
                raise NetworkError("Unknown primary heat-exchange/heater segment")
            cfg[key] = coupling[key]
        for key in COUPLING_FIELDS - set(cfg):
            cfg[key] = number(coupling[key], key, positive=True)
        if not (
            1e5 <= cfg["min_primary_pressure_pa"] < cfg["max_primary_pressure_pa"] <= 16e6
            and cfg["min_primary_temperature_k"] >= 273.15
        ):
            raise NetworkError("Primary pressure/temperature limits outside declared water scope")
        self._initials = deepcopy(
            dict(primary_initial=primary_initial, steam_initial=steam_initial)
        )
        self._coupling = cfg
        self._offset = len(p._state)
        self._exchange_index = p._loop.names.index(cfg["steam_generator_segment"])
        self._heater_index = p._loop.names.index(cfg["heater_segment"])
        self._controls = controls(boundary_controls)
        self._initial_controls = deepcopy(self._controls)
        if (
            self._controls["primary_motor_torque_nm"] != p._torque
            or self._controls["steam_cycle"] != s.boundary_controls
        ):
            raise NetworkError("Initial component commands differ from coupled controls")
        self._state = np.r_[p._state, np.zeros(14)]
        self.time_s, self._events = 0.0, []
        self._heat_reference = (
            p._loop._temperature_reference[self._exchange_index]
            - s.configuration["hot_wall"]["initial_temperature_k"]
        )
        # Fixed initial physical origin; counters never enter the rates.
        self._rate_anchor = self._state.copy()
        self._rate_scales = np.full(len(self._state), np.inf)
        indices, scales = self._physical_indices_scales()
        self._rate_scales[indices] = np.abs(scales) * 1e-12
        self._rate_cache = None
        self.observation()

    @property
    def configuration(self):
        return dict(**deepcopy(self._initials), coupling=deepcopy(self._coupling))

    @property
    def boundary_controls(self):
        return deepcopy(self._controls)

    @property
    def events(self):
        return deepcopy(self._events)

    @property
    def absolute_tolerances(self):
        p = self._primary
        return np.r_[
            1e-9,
            np.full(2 * p._loop.n, 1e-11),
            1e-4,
            1e-4,
            1e-9,
            1e-4,
            1e-4,
            1e-4,
            1e-10,
            1e-4,
            1e-10,
            1e-4,
            1e-11,
            1e-11,
            1e-4,
            np.full(7, 1e-4),
        ]

    def _heat(self, state):
        y = np.asarray(state, dtype=float)
        if y.shape != self._state.shape or not np.isfinite(y).all():
            raise NetworkError("Invalid coupled primary/steam state")
        return self._coupling["ua_w_k"] * (
            self._heat_reference + (y[1 + self._exchange_index] - y[self._offset + 4])
        )

    def _exact_derivatives(self, stamp, state):
        q = self._heat(state)
        p, loop, offset = self._primary, self._primary._loop, self._offset
        primary = p.derivatives(stamp, state[:offset])
        primary[1 + self._exchange_index] -= q / loop.capacity[self._exchange_index]
        primary[1 + loop.n + self._heater_index] += (
            self._controls["primary_wall_heater_power_w"] / loop.wall_capacity[self._heater_index]
        )
        primary[p._size - 1] += self._controls["primary_wall_heater_power_w"] - q
        steam = self._steam.derivatives(stamp, state[offset : offset + 13], hot_wall_port_w=q)
        return np.r_[primary, steam, q]

    def derivatives(self, stamp, state):
        """Smooth sub-resolution rate response around the fixed initial origin.

        Absolute-stock reconstruction can round tiny increments to different
        floating-point bins. Use a local first-order continuation within 1e-12
        of initial physical scales, with a C2 blend to the nonlinear equations.
        Domain checks and observations still use the original physical stocks.
        The cache depends only on immutable configuration and current controls.
        """
        exact = self._exact_derivatives(stamp, state)
        delta = np.asarray(state) - self._rate_anchor
        radius = float(np.max(np.abs(delta / self._rate_scales)))
        if radius >= 1:
            return exact
        cache = self._rate_cache
        if cache is None or cache[0] != self._controls:
            cache = (
                deepcopy(self._controls),
                self._exact_derivatives(stamp, self._rate_anchor),
                self._jacobian(stamp, self._rate_anchor, self._exact_derivatives),
            )
            self._rate_cache = cache
        linear = cache[1] + cache[2] @ delta
        q = max(0.0, 2 * radius - 1)
        weight = q**3 * (10 - 15 * q + 6 * q**2)
        return (1 - weight) * linear + weight * exact

    def _margins(self, state):
        self._heat(state)
        p, loop = self._primary, self._primary._loop
        heads = p._machine(state[: self._offset])[-1]
        r = loop._rates(state[: p._size], heads)
        cfg = self._coupling
        result = list(self._steam._margins(state[self._offset : self._offset + 13]))
        for i, name in enumerate(loop.names):
            pressure, temperature = float(r["pressure"][i]), float(r["fluid_temperature"][i])
            saturated, _ = saturation_properties(pressure)
            result.extend(
                [
                    (
                        f"primary.{name}.pressure_low",
                        pressure - cfg["min_primary_pressure_pa"],
                        "Pa",
                        1e-3,
                    ),
                    (
                        f"primary.{name}.pressure_high",
                        cfg["max_primary_pressure_pa"] - pressure,
                        "Pa",
                        1e-3,
                    ),
                    (
                        f"primary.{name}.temperature_low",
                        temperature - cfg["min_primary_temperature_k"],
                        "K",
                        1e-8,
                    ),
                    (
                        f"primary.{name}.subcooling",
                        saturated.T_k - temperature - cfg["minimum_subcooling_k"],
                        "K",
                        1e-8,
                    ),
                ]
            )
        return result

    def observation(self, state=None, time_s=None):
        y = self._state if state is None else np.asarray(state, dtype=float)
        stamp = self.time_s if time_s is None else number(time_s, "coupled observation time")
        q = float(self._heat(y))
        margins = self._margins(y)
        for name, value, _, tol in margins:
            if value < -tol:
                raise DomainError("Primary/steam passport exceeded: " + name)
        p, k = self._primary, self._offset
        primary = p.observation(y[:k], stamp)
        steam = self._steam.observation(
            y[k : k + 13], stamp, hot_wall_port_w=q, hot_wall_port_energy_j=float(y[-1])
        )
        # The primitive primary ledger contains ALL thermal ports. Restore the
        # external-only meaning explicitly when presenting the coupled view.
        primary["net_thermal_heat_including_ports_j"] = primary.pop(
            "external_ambient_heater_heat_j"
        )
        primary["external_ambient_heater_heat_j"] = math.fsum(
            (primary["net_thermal_heat_including_ports_j"], float(y[-1]))
        )
        stored = math.fsum(
            (
                primary["loop"]["stored_energy_change_j"],
                primary["shaft_energy_change_j"],
                steam["stored_energy_change_j"],
            )
        )
        integral = steam["energy_integrals_j"]
        external = math.fsum(
            (
                primary["motor_work_j"],
                -primary["bearing_heat_j"],
                primary["external_ambient_heater_heat_j"],
                integral["heater"],
                integral["feed_pump"],
                -integral["electric"],
                -integral["generator_heat"],
                -integral["bearing_heat"],
                -integral["sink_heat"],
            )
        )
        residual = stored - external
        if abs(residual) > self._coupling["energy_tolerance_j"]:
            raise NetworkError("Coupled primary/steam energy ledger exceeded")
        rates = self.derivatives(stamp, y)
        loop = p._loop
        fluid_power = float(loop.capacity @ rates[1 : 1 + loop.n])
        wall_power = float(loop.wall_capacity @ rates[1 + loop.n : 1 + 2 * loop.n])
        kinetic_power = loop.inertia_total * y[0] * rates[0] / loop.rho
        shaft_power = p._pump["inertia_kg_m2"] * y[p._size] * rates[p._size]
        cfg = self._steam.configuration
        steam_stored_power = math.fsum(
            (
                rates[k + 1],
                rates[k + 3],
                cfg["hot_wall"]["heat_capacity_j_k"] * rates[k + 4],
                cfg["cold_wall"]["heat_capacity_j_k"] * rates[k + 5],
                rates[k + 6],
            )
        )
        primary_wall = loop._wall_reference + y[1 + loop.n : 1 + 2 * loop.n]
        external_power = math.fsum(
            (
                primary["motor_power_w"],
                -primary["bearing_power_w"],
                float(
                    sum(loop.heater + loop.ambient_ua * (loop.ambient_temperature_k - primary_wall))
                ),
                self._controls["primary_wall_heater_power_w"],
                rates[k + 7],
                rates[k + 8],
                *(-rates[k + 9 : k + 13]),
            )
        )
        power_residual = math.fsum(
            (
                fluid_power,
                wall_power,
                kinetic_power,
                shaft_power,
                steam_stored_power,
                -external_power,
            )
        )
        if abs(power_residual) > 1e-5:
            raise NetworkError("Coupled primary/steam instantaneous power mismatch")
        return dict(
            time_s=stamp,
            controls=self.boundary_controls,
            primary=primary,
            steam_cycle=steam,
            heat_transfer_w=q,
            heat_transfer_energy_j=float(y[-1]),
            stored_energy_change_j=stored,
            external_energy_j=external,
            energy_residual_j=residual,
            mass_residual_kg=steam["mass_residual_kg"],
            power_residual_w=float(power_residual),
            external_power_w=external_power,
            thermal_ports=dict(
                primary_into_w=-q,
                steam_into_w=q,
                primary_into_j=-float(y[-1]),
                steam_into_j=float(y[-1]),
            ),
            domain_margins={
                name: dict(value=float(v), unit=unit, root_tolerance=tol)
                for name, v, unit, tol in margins
            },
        )

    def _physical_indices_scales(self):
        p, k = self._primary, self._offset
        indices = [*range(1 + 2 * p._loop.n), p._size, *range(k, k + 7)]
        scales = [
            max(abs(p._loop.initial_flow), 1),
            *p._loop._temperature_reference,
            *p._loop._wall_reference,
            max(abs(p._omega0), 1),
            self._steam._origins[0][0],
            self._steam._origins[0][1],
            self._steam._origins[1][0],
            self._steam._origins[1][1],
            300,
            300,
            self._steam._rotor0,
        ]
        return indices, np.asarray(scales, dtype=float)

    def jacobian(self, stamp, state):
        return self._jacobian(stamp, state, self.derivatives)

    def _jacobian(self, stamp, state, rates):
        indices, scales = self._physical_indices_scales()
        base = rates(stamp, state)
        jac = np.zeros((len(state), len(state)))
        for index, scale in zip(indices, scales, strict=True):
            delta = max(abs(state[index]), scale) * 1e-6
            left, right = state.copy(), state.copy()
            left[index] -= delta
            right[index] += delta
            try:
                lo = rates(stamp, left)
            except (DomainError, NetworkError):
                lo = None
            try:
                hi = rates(stamp, right)
            except (DomainError, NetworkError):
                hi = None
            if lo is not None and hi is not None:
                jac[:, index] = (hi - lo) / (2 * delta)
            elif hi is not None:
                jac[:, index] = (hi - base) / delta
            elif lo is not None:
                jac[:, index] = (base - lo) / delta
            else:
                raise DomainError("Both coupled Jacobian probes outside domain")
        return jac

    def advance_until_limit(self, dt_s, **solver):
        return advance_with_limits(self, dt_s, **solver)

    def advance(self, dt_s, **solver):
        saved = self._state.copy(), self.time_s, deepcopy(self._events)
        try:
            result = self.advance_until_limit(dt_s, **solver)
            if result["boundary"] is not None:
                raise DomainError("Coupled interval crosses " + result["boundary"]["criterion"])
            return result["observation"]
        except Exception:
            self._state, self.time_s, self._events = saved
            raise

    def initialization_report(self):
        if self._controls != self._initial_controls:
            return type(self)(
                **self.configuration, boundary_controls=self._initial_controls
            ).initialization_report()
        rates = self.derivatives(0, self._state)
        p, k = self._primary, self._offset
        return dict(
            mode="given_component_states",
            primary_flow_rate_kg_s2=float(rates[0]),
            primary_shaft_acceleration_rad_s2=float(rates[p._size]),
            primary_fluid_temperature_rates_k_s=rates[1 : 1 + p._loop.n].tolist(),
            steam_inventory_mass_rates_kg_s=[float(rates[k]), float(rates[k + 2])],
            steam_inventory_energy_rates_w=[float(rates[k + 1]), float(rates[k + 3])],
            steam_wall_temperature_rates_k_s=[float(rates[k + 4]), float(rates[k + 5])],
            rotor_energy_rate_w=float(rates[k + 6]),
            initial_heat_transfer_w=float(rates[-1]),
        )

    def set_controls(self, raw):
        proposed = controls(raw)
        if len(self._events) >= 10000:
            raise NetworkError("Coupled control history resource limit exceeded")
        previous = self.boundary_controls
        try:
            self._controls = proposed
            self._primary._torque = proposed["primary_motor_torque_nm"]
            self._steam._controls = proposed["steam_cycle"]
            self.observation()
        except Exception:
            self._controls = previous
            self._primary._torque = previous["primary_motor_torque_nm"]
            self._steam._controls = previous["steam_cycle"]
            raise
        self._events.append(
            dict(kind="controls", t=self.time_s, before=previous, after=deepcopy(proposed))
        )

    def snapshot(self):
        return dict(
            schema_version=1,
            model_id=PASSPORT.model_id,
            model_version=PASSPORT.version,
            configuration=self.configuration,
            configuration_sha256=digest(self.configuration),
            initial_controls=deepcopy(self._initial_controls),
            controls=self.boundary_controls,
            state=self._state.tolist(),
            time_s=self.time_s,
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
            "initial_controls",
            "controls",
            "state",
            "time_s",
            "events",
        }
        if (
            not isinstance(payload, dict)
            or set(payload) != fields
            or type(payload["schema_version"]) is not int
            or payload["schema_version"] != 1
            or payload["model_id"] != PASSPORT.model_id
            or payload["model_version"] != PASSPORT.version
            or not isinstance(payload["configuration"], dict)
            or set(payload["configuration"]) != {"primary_initial", "steam_initial", "coupling"}
            or digest(payload["configuration"]) != payload["configuration_sha256"]
        ):
            raise NetworkError("Incompatible coupled primary/steam checkpoint")
        model = cls(**payload["configuration"], boundary_controls=payload["initial_controls"])
        stamp = number(payload["time_s"], "snapshot time")
        if stamp < 0 or not isinstance(payload["events"], list) or len(payload["events"]) > 10000:
            raise NetworkError("Invalid coupled time/history")
        previous_time = -1.0
        active = deepcopy(model._initial_controls)
        allowed = {name: (unit, tol) for name, _, unit, tol in model._margins(model._state)}
        for event in payload["events"]:
            if not isinstance(event, dict):
                raise NetworkError("Invalid coupled event")
            t = number(event.get("t"), "event time")
            if not 0 <= t <= stamp or t < previous_time:
                raise NetworkError("Unordered coupled event")
            previous_time = t
            if event.get("kind") == "controls" and set(event) == {"kind", "t", "before", "after"}:
                if controls(event["before"]) != active:
                    raise NetworkError("Broken coupled control chain")
                active = controls(event["after"])
            elif event.get("kind") == "domain_limit" and set(event) == {
                "kind",
                "t",
                "criterion",
                "residual",
                "unit",
                "root_tolerance",
            }:
                if event["criterion"] not in allowed:
                    raise NetworkError("Unknown coupled domain criterion")
                unit, tol = allowed[event["criterion"]]
                if (
                    event["unit"] != unit
                    or event["root_tolerance"] != tol
                    or abs(number(event["residual"], "root residual")) > tol
                ):
                    raise NetworkError("Invalid coupled domain root")
            else:
                raise NetworkError("Unknown coupled event schema")
        if active != controls(payload["controls"]):
            raise NetworkError("Coupled controls differ from history")
        raw = payload["state"]
        if not isinstance(raw, list) or len(raw) != len(model._state):
            raise NetworkError("Invalid coupled state shape")
        y = np.array([number(v, "state") for v in raw])
        if stamp == 0 and not np.array_equal(y, model._state):
            raise NetworkError("Noninitial coupled state at zero time")
        model._controls = active
        model._primary._torque = active["primary_motor_torque_nm"]
        model._steam._controls = active["steam_cycle"]
        model.observation(y, stamp)
        model._state, model.time_s, model._events = y, stamp, deepcopy(payload["events"])
        return model
