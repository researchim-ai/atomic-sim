"""Saturated-steam expansion and generator/shaft energy in an operating-speed band.

This is an algebraic, constant-efficiency steam path with prescribed inlet
mass flow and boundary pressures. Rotor kinetic energy is the dynamic state:
Kdot=P_steam-(G_connected+B)*2K/J. Cold start, stall, reverse rotation,
choked-flow maps and electrical network dynamics are outside the passport.
"""

import math
from copy import deepcopy

import numpy as np

from core.network import NetworkError, number
from core.status import ModelLimitError, ModelPassport, ModelStatus
from physics.materials.iapws import properties_ph, saturation_properties
from physics.thermal.two_stream_exchanger import digest

PASSPORT = ModelPassport(
    model_id="saturated-steam-turbine-generator",
    version="1.0.0",
    status=ModelStatus.IMPLEMENTED,
    phenomenon="IF97 saturated-steam expansion, rotor kinetic energy and dissipative generator load",
    approximations=(
        "Saturated dry inlet; isentropic exhaust reference lies inside the saturation dome",
        "Constant specified isentropic efficiency; no blade/nozzle/choked-flow or manufacturer map",
        "Prescribed nonnegative mass flow and fixed boundary pressures; no steam-path storage",
        "Constant inertia, viscous bearing drag and generator torque G*omega",
        "Positive declared operating-speed band; no cold start, stall or reverse rotation",
        "Generator efficiency constant; no voltage, reactive power or network synchronization",
    ),
    applicability="Reduced steam power train inside explicit thermodynamic and rotor-speed limits",
    data_sources=(
        "IAPWS-IF97: https://iapws.org/technical-guidance/release/IF97-Rev",
        "h_out=h_in-eta_is*(h_in-h_is); Kdot=P_steam-2*(G+B)*K/J",
    ),
)
CONFIG_FIELDS = {
    "inlet_pressure_pa",
    "outlet_pressure_pa",
    "isentropic_efficiency",
    "inertia_kg_m2",
    "bearing_drag_nm_s_rad",
    "generator_efficiency",
    "min_omega_rad_s",
    "max_omega_rad_s",
    "energy_tolerance_j",
}
CONTROL_FIELDS = {"mass_flow_kg_s", "generator_conductance_nm_s_rad", "generator_connected"}


def controls(raw):
    if (
        not isinstance(raw, dict)
        or set(raw) != CONTROL_FIELDS
        or type(raw["generator_connected"]) is not bool
    ):
        raise NetworkError("Complete turbine/generator controls with boolean connection required")
    result = {key: number(raw[key], key) for key in CONTROL_FIELDS - {"generator_connected"}}
    if min(result.values()) < 0:
        raise NetworkError("Steam flow and generator conductance must be nonnegative")
    result["generator_connected"] = raw["generator_connected"]
    return result


def expansion(configuration):
    """Thermodynamic port contract; independent of rotor speed and steam flow."""
    pin, pout = configuration["inlet_pressure_pa"], configuration["outlet_pressure_pa"]
    if not 0 < pout < pin:
        raise NetworkError("Turbine requires positive outlet pressure below inlet pressure")
    _, inlet = saturation_properties(pin)
    liquid, vapor = saturation_properties(pout)
    quality = (inlet.s_j_kg_k - liquid.s_j_kg_k) / (vapor.s_j_kg_k - liquid.s_j_kg_k)
    if not 0 <= quality <= 1:
        raise NetworkError("Isentropic turbine reference outside supported saturation dome")
    his = liquid.h_j_kg + quality * (vapor.h_j_kg - liquid.h_j_kg)
    drop = configuration["isentropic_efficiency"] * (inlet.h_j_kg - his)
    outlet = properties_ph(pout, inlet.h_j_kg - drop)
    if drop <= 0 or outlet.s_j_kg_k < inlet.s_j_kg_k - 1e-7:
        raise NetworkError("Invalid expansion work or negative entropy production")
    return dict(
        inlet_enthalpy_j_kg=inlet.h_j_kg,
        outlet_enthalpy_j_kg=outlet.h_j_kg,
        isentropic_outlet_enthalpy_j_kg=his,
        isentropic_quality=quality,
        inlet_entropy_j_kg_k=inlet.s_j_kg_k,
        outlet_entropy_j_kg_k=outlet.s_j_kg_k,
        inlet_temperature_k=inlet.T_k,
        outlet_temperature_k=outlet.T_k,
        outlet_quality=outlet.quality,
        work_j_kg=drop,
    )


class SteamTurbineTrain:
    def __init__(self, configuration, boundary_controls, initial_omega_rad_s):
        if not isinstance(configuration, dict) or set(configuration) != CONFIG_FIELDS:
            raise NetworkError("Exact steam turbine train configuration required")
        cfg = {
            key: number(value, key, positive=key != "bearing_drag_nm_s_rad")
            for key, value in configuration.items()
        }
        if (
            cfg["bearing_drag_nm_s_rad"] < 0
            or not 0 < cfg["isentropic_efficiency"] <= 1
            or not 0 < cfg["generator_efficiency"] <= 1
            or cfg["min_omega_rad_s"] >= cfg["max_omega_rad_s"]
        ):
            raise NetworkError("Invalid turbine efficiency, drag or operating-speed band")
        self._configuration = cfg
        self._expansion = expansion(cfg)
        self._controls = controls(boundary_controls)
        self._initial_controls = deepcopy(self._controls)
        self._omega0 = number(initial_omega_rad_s, "initial_omega_rad_s", positive=True)
        if not cfg["min_omega_rad_s"] <= self._omega0 <= cfg["max_omega_rad_s"]:
            raise NetworkError("Initial rotor speed outside declared operating band")
        self._initial_energy = 0.5 * cfg["inertia_kg_m2"] * self._omega0**2
        self._limits = np.array(
            [
                0.5 * cfg["inertia_kg_m2"] * cfg[key] ** 2
                for key in ("min_omega_rad_s", "max_omega_rad_s")
            ]
        )
        if not np.isfinite(self._initial_energy) or not np.isfinite(self._limits).all():
            raise NetworkError("Rotor inventory overflow")
        self._state = np.array([self._initial_energy, 0.0, 0.0, 0.0, 0.0])
        self.time_s, self._events = 0.0, []
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

    @property
    def thermodynamics(self):
        return deepcopy(self._expansion)

    def _powers(self, energy):
        cfg, boundary = self._configuration, self._controls
        g = boundary["generator_conductance_nm_s_rad"] if boundary["generator_connected"] else 0.0
        squared_speed = 2 * float(energy) / cfg["inertia_kg_m2"]
        steam = boundary["mass_flow_kg_s"] * self._expansion["work_j_kg"]
        generator = g * squared_speed
        bearing = cfg["bearing_drag_nm_s_rad"] * squared_speed
        if not np.isfinite([steam, generator, bearing, g]).all():
            raise NetworkError("Nonfinite turbine/generator power")
        return steam, generator, bearing, g

    def _candidate(self, dt):
        cfg = self._configuration
        steam, _, _, g = self._powers(self._state[0])
        decay = 2 * (g + cfg["bearing_drag_nm_s_rad"]) / cfg["inertia_kg_m2"]
        x = decay * dt
        if decay == 0:
            psi = dt
            chi = 0.5 * dt**2
        else:
            psi = -math.expm1(-x) / decay
            # Stable double time integral for small decay*dt.
            chi = (
                dt**2 * (0.5 - x / 6 + x * x / 24 - x**3 / 120 + x**4 / 720)
                if abs(x) < 1e-3
                else (dt - psi) / decay
            )
        energy = self._state[0] * math.exp(-x) + steam * psi
        integral_squared_speed = 2 * (self._state[0] * psi + steam * chi) / cfg["inertia_kg_m2"]
        result = self._state.copy()
        result[0] = energy
        result[1] += steam * dt
        result[2] += cfg["generator_efficiency"] * g * integral_squared_speed
        result[3] += (1 - cfg["generator_efficiency"]) * g * integral_squared_speed
        result[4] += cfg["bearing_drag_nm_s_rad"] * integral_squared_speed
        if not np.isfinite(result).all():
            raise NetworkError("Nonfinite turbine propagation")
        return result

    def _crossing(self, dt):
        candidate = self._candidate(dt)
        index = (
            0 if candidate[0] < self._limits[0] else 1 if candidate[0] > self._limits[1] else None
        )
        if index is None:
            return None
        cfg = self._configuration
        steam, _, _, g = self._powers(self._state[0])
        decay = 2 * (g + cfg["bearing_drag_nm_s_rad"]) / cfg["inertia_kg_m2"]
        rate = steam - decay * self._state[0]
        difference = self._limits[index] - self._state[0]
        stamp = difference / rate if decay == 0 else -math.log1p(-decay * difference / rate) / decay
        if not np.isfinite(stamp) or not 0 <= stamp <= dt:
            raise NetworkError("Invalid analytic rotor boundary time")
        return index, float(stamp)

    def _observe(self, state, stamp):
        if (
            state.shape != (5,)
            or not np.isfinite(state).all()
            or not self._limits[0] <= state[0] <= self._limits[1]
        ):
            raise NetworkError("Invalid turbine state or out-of-band rotor energy")
        cfg = self._configuration
        tolerance = cfg["energy_tolerance_j"]
        if min(state[1:]) < -tolerance:
            raise NetworkError("Negative accumulated turbine work or passive loss")
        residual = math.fsum(
            (state[0], -self._initial_energy, -state[1], state[2], state[3], state[4])
        )
        if abs(residual) > tolerance:
            raise NetworkError("Turbine/generator energy ledger exceeded")
        omega = math.sqrt(2 * state[0] / cfg["inertia_kg_m2"])
        steam, generator, bearing, g = self._powers(state[0])
        mass = self._controls["mass_flow_kg_s"]
        inlet = mass * self._expansion["inlet_enthalpy_j_kg"]
        outlet = mass * self._expansion["outlet_enthalpy_j_kg"]
        fluid_residual = math.fsum((inlet, -outlet, -steam))
        if not np.isfinite(fluid_residual) or abs(fluid_residual) > 1e-5:
            raise NetworkError("Turbine enthalpy/shaft port mismatch")
        return dict(
            time_s=stamp,
            omega_rad_s=omega,
            rotor_energy_j=float(state[0]),
            controls=self.boundary_controls,
            steam_power_w=steam,
            turbine_torque_nm=steam / omega,
            generator_torque_nm=g * omega,
            electrical_power_w=cfg["generator_efficiency"] * generator,
            generator_heat_w=(1 - cfg["generator_efficiency"]) * generator,
            bearing_heat_w=bearing,
            steam_work_j=float(state[1]),
            electric_energy_j=float(state[2]),
            generator_heat_j=float(state[3]),
            bearing_heat_j=float(state[4]),
            energy_residual_j=residual,
            fluid_power_residual_w=fluid_residual,
            inlet_enthalpy_power_w=inlet,
            outlet_enthalpy_power_w=outlet,
            entropy_production_w_k=mass
            * (self._expansion["outlet_entropy_j_kg_k"] - self._expansion["inlet_entropy_j_kg_k"]),
        )

    def observation(self):
        return self._observe(self._state, self.time_s)

    def advance(self, dt_s):
        dt = number(dt_s, "dt_s", positive=True)
        end = self.time_s + dt
        if not np.isfinite(end) or end <= self.time_s:
            raise NetworkError("Turbine clock cannot advance")
        crossing = self._crossing(dt)
        if crossing is not None:
            index, offset = crossing
            speed = self._configuration["min_omega_rad_s" if index == 0 else "max_omega_rad_s"]
            error = ModelLimitError(
                "Rotor left declared operating-speed band",
                criterion="omega_rad_s",
                node="turbine_shaft",
                observed=math.sqrt(
                    2 * self._candidate(dt)[0] / self._configuration["inertia_kg_m2"]
                ),
                limit=speed,
            )
            error.evaluation_time_s = self.time_s + offset
            error.evaluation_phase = "analytic_boundary"
            raise error
        candidate = self._candidate(dt)
        row = self._observe(candidate, end)
        self._state, self.time_s = candidate, end
        return row

    def advance_until_limit(self, dt_s):
        dt = number(dt_s, "dt_s", positive=True)
        if not np.isfinite(self.time_s + dt) or self.time_s + dt <= self.time_s:
            raise NetworkError("Turbine clock cannot advance")
        crossing = self._crossing(dt)
        if crossing is None:
            return dict(observation=self.advance(dt), boundary=None)
        if len(self._events) >= 10000:
            raise NetworkError("Turbine event history resource limit exceeded")
        index, offset = crossing
        candidate = self._candidate(offset)
        projection = self._limits[index] - candidate[0]
        if abs(projection) > self._configuration["energy_tolerance_j"] * 0.1:
            raise NetworkError("Analytic boundary projection exceeds roundoff budget")
        candidate[0] = self._limits[index]
        stamp = self.time_s + offset
        row = self._observe(candidate, stamp)
        speed = self._configuration["min_omega_rad_s" if index == 0 else "max_omega_rad_s"]
        event = dict(
            kind="speed_limit",
            t=stamp,
            limit="lower" if index == 0 else "upper",
            omega_rad_s=speed,
            projection_j=float(projection),
        )
        self._state, self.time_s = candidate, stamp
        self._events.append(event)
        return dict(observation=row, boundary=deepcopy(event))

    def set_controls(self, boundary_controls):
        proposed = controls(boundary_controls)
        if len(self._events) >= 10000:
            raise NetworkError("Turbine event history resource limit exceeded")
        previous = self.boundary_controls
        try:
            self._controls = proposed
            self.observation()
        except Exception:
            self._controls = previous
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
            initial_omega_rad_s=self._omega0,
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
            "initial_omega_rad_s",
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
            or digest(payload["configuration"]) != payload["configuration_sha256"]
        ):
            raise NetworkError("Incompatible turbine snapshot")
        model = cls(
            payload["configuration"], payload["initial_controls"], payload["initial_omega_rad_s"]
        )
        stamp = number(payload["time_s"], "snapshot time")
        events = payload["events"]
        if stamp < 0 or not isinstance(events, list) or len(events) > 10000:
            raise NetworkError("Invalid turbine snapshot time/history")
        previous_time, boundary = -1.0, deepcopy(model._initial_controls)
        for event in events:
            if not isinstance(event, dict):
                raise NetworkError("Invalid turbine event")
            t = number(event.get("t"), "event time")
            if not previous_time <= t <= stamp or t < 0:
                raise NetworkError("Unordered turbine event")
            previous_time = t
            if event.get("kind") == "controls" and set(event) == {"kind", "t", "before", "after"}:
                if controls(event["before"]) != boundary:
                    raise NetworkError("Broken turbine control chain")
                boundary = controls(event["after"])
            elif event.get("kind") == "speed_limit" and set(event) == {
                "kind",
                "t",
                "limit",
                "omega_rad_s",
                "projection_j",
            }:
                if event["limit"] not in ("lower", "upper"):
                    raise NetworkError("Unknown turbine speed limit")
                expected = model._configuration[
                    "min_omega_rad_s" if event["limit"] == "lower" else "max_omega_rad_s"
                ]
                if (
                    number(event["omega_rad_s"], "boundary speed") != expected
                    or abs(number(event["projection_j"], "projection"))
                    > model._configuration["energy_tolerance_j"] * 0.1
                ):
                    raise NetworkError("Invalid turbine boundary record")
            else:
                raise NetworkError("Unknown turbine event schema")
        if boundary != controls(payload["controls"]):
            raise NetworkError("Turbine controls differ from history")
        raw = payload["state"]
        if not isinstance(raw, list) or len(raw) != 5:
            raise NetworkError("Invalid turbine snapshot state shape")
        state = np.array([number(v, "snapshot state") for v in raw])
        if stamp == 0 and not np.array_equal(state, model._state):
            raise NetworkError("Noninitial turbine state at zero time")
        model._controls = boundary
        model._observe(state, stamp)
        model._state, model.time_s, model._events = state, stamp, deepcopy(events)
        return model
