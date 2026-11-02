"""Conservative saturated two-vessel steam cycle with walls, feed pump and rotor.

Pure algebra recovers each pressure from M,U,V. Steam admission is an explicit
empirical linear pressure-drop law, not a nozzle/manufacturer flow map. Both
reservoirs must remain inside their declared two-phase operating margins.
"""

import math
from copy import deepcopy

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from core.network import NetworkError, number
from core.status import DomainError, ModelPassport, ModelStatus, SolverError
from physics.materials.iapws import saturation_properties
from physics.power.steam_turbine_train import expansion
from physics.thermal.two_stream_exchanger import digest

PASSPORT = ModelPassport(
    model_id="equilibrium-steam-power-cycle",
    version="1.0.0",
    status=ModelStatus.IMPLEMENTED,
    phenomenon="Dynamic steam-generator/condenser inventories, thermal walls, turbine and generator",
    approximations=(
        "Two fixed-volume equilibrium IF97 two-phase stores with ideal vapor/liquid separation",
        "Empirical linear steam-admission conductance; no nozzle choking, momentum or pipe storage",
        "Prescribed feed mass flow; incompressible feed-pump work with specified efficiency heats feed",
        "Constant turbine/generator efficiencies; rotor kinetic energy in a positive-speed band",
        "Finite-capacity hot and cold walls; linear heat-transfer coefficients",
        "No primary reactor, distributed boiling channel, entrainment, noncondensable gas or electrical network",
    ),
    applicability="Reduced closed steam-cycle transients inside declared p/x/rotor operating limits",
    data_sources=(
        "IAPWS-IF97 saturation properties; V=M*((1-x)*vf+x*vg), U=M*((1-x)*uf+x*ug)",
        "dMh=mf-ms; dUh=mf*hfeed-ms*hg+Qh; dMc=-dMh; dUc=ms*hout-mf*hf-Qc",
    ),
)
VESSEL_FIELDS = {
    "volume_m3",
    "initial_pressure_pa",
    "initial_quality",
    "min_pressure_pa",
    "max_pressure_pa",
    "min_quality",
    "max_quality",
}
CONTROL_FIELDS = {
    "heater_power_w",
    "valve_opening",
    "feed_mass_flow_kg_s",
    "generator_conductance_nm_s_rad",
    "generator_connected",
    "sink_temperature_k",
}
TURBINE_FIELDS = {
    "isentropic_efficiency",
    "inertia_kg_m2",
    "bearing_drag_nm_s_rad",
    "generator_efficiency",
    "min_omega_rad_s",
    "max_omega_rad_s",
    "initial_omega_rad_s",
}
CONFIG_FIELDS = {
    "steam_generator",
    "condenser",
    "hot_wall",
    "cold_wall",
    "turbine",
    "steam_conductance_kg_s_pa",
    "feed_efficiency",
    "min_pressure_difference_pa",
    "mass_tolerance_kg",
    "energy_tolerance_j",
}


def controls(raw):
    if (
        not isinstance(raw, dict)
        or set(raw) != CONTROL_FIELDS
        or type(raw["generator_connected"]) is not bool
    ):
        raise NetworkError(
            "Complete steam-cycle controls with boolean generator connection required"
        )
    result = {
        key: number(raw[key], key, positive=key == "sink_temperature_k")
        for key in CONTROL_FIELDS - {"generator_connected"}
    }
    if min(result.values()) < 0 or result["valve_opening"] > 1:
        raise NetworkError("Negative steam-cycle command or valve opening outside [0,1]")
    result["generator_connected"] = raw["generator_connected"]
    return result


def inventory(configuration, mass, energy):
    """Pure scalar IF97 inversion; phase bounds are checked at acceptance/events."""
    if not np.isfinite([mass, energy]).all() or mass <= 0:
        raise DomainError("Steam-cycle inventory mass/energy is invalid")
    volume = configuration["volume_m3"]
    specific_volume = volume / mass

    def at(pressure):
        liquid, vapor = saturation_properties(float(pressure))
        quality = (specific_volume - liquid.v_m3_kg) / (vapor.v_m3_kg - liquid.v_m3_kg)
        calculated = mass * (liquid.u_j_kg + quality * (vapor.u_j_kg - liquid.u_j_kg))
        return calculated - energy, quality, liquid, vapor

    lower = max(620.0, configuration["min_pressure_pa"] * 0.5)
    upper = min(16e6, configuration["max_pressure_pa"] * 1.5)
    try:
        pressure = brentq(lambda p: at(p)[0], lower, upper, xtol=1e-9, rtol=1e-14)
    except ValueError as exc:
        raise DomainError("Equilibrium inventory pressure root not bracketed") from exc
    residual, quality, liquid, vapor = at(pressure)
    return dict(
        pressure_pa=float(pressure),
        quality=float(quality),
        temperature_k=float(liquid.T_k),
        mass_kg=float(mass),
        internal_energy_j=float(energy),
        energy_closure_j=float(residual),
        volume_closure_m3=float(
            mass * ((1 - quality) * liquid.v_m3_kg + quality * vapor.v_m3_kg) - volume
        ),
        liquid=liquid,
        vapor=vapor,
    )


class SteamPowerCycle:
    def __init__(self, configuration, boundary_controls):
        if not isinstance(configuration, dict) or set(configuration) != CONFIG_FIELDS:
            raise NetworkError("Exact steam-cycle configuration required")
        cfg = {}
        self._origins = []
        for name in ("steam_generator", "condenser"):
            raw = configuration[name]
            if not isinstance(raw, dict) or set(raw) != VESSEL_FIELDS:
                raise NetworkError("Exact vessel geometry, initialization and margins required")
            vessel = {key: number(value, key, positive=True) for key, value in raw.items()}
            if not (
                1500 <= vessel["min_pressure_pa"] < vessel["max_pressure_pa"] <= 15e6
                and vessel["min_pressure_pa"]
                <= vessel["initial_pressure_pa"]
                <= vessel["max_pressure_pa"]
                and 0 < vessel["min_quality"] < vessel["max_quality"] < 1
                and vessel["min_quality"] <= vessel["initial_quality"] <= vessel["max_quality"]
            ):
                raise NetworkError("Invalid initial vessel state or declared two-phase margins")
            liquid, vapor = saturation_properties(vessel["initial_pressure_pa"])
            x = vessel["initial_quality"]
            mass = vessel["volume_m3"] / ((1 - x) * liquid.v_m3_kg + x * vapor.v_m3_kg)
            energy = mass * ((1 - x) * liquid.u_j_kg + x * vapor.u_j_kg)
            self._origins.append((float(mass), float(energy)))
            cfg[name] = vessel
        for name, fields in (
            ("hot_wall", {"heat_capacity_j_k", "fluid_ua_w_k", "initial_temperature_k"}),
            (
                "cold_wall",
                {"heat_capacity_j_k", "fluid_ua_w_k", "ambient_ua_w_k", "initial_temperature_k"},
            ),
            ("turbine", TURBINE_FIELDS),
        ):
            raw = configuration[name]
            if not isinstance(raw, dict) or set(raw) != fields:
                raise NetworkError("Exact wall/turbine configuration required")
            values = {
                key: number(
                    value,
                    key,
                    positive=key not in {"fluid_ua_w_k", "ambient_ua_w_k", "bearing_drag_nm_s_rad"},
                )
                for key, value in raw.items()
            }
            if min(values.values()) < 0:
                raise NetworkError("Negative wall/turbine parameter")
            cfg[name] = values
        for key in CONFIG_FIELDS - set(cfg):
            cfg[key] = number(configuration[key], key, positive=True)
        t = cfg["turbine"]
        if (
            not 0 < t["isentropic_efficiency"] <= 1
            or not 0 < t["generator_efficiency"] <= 1
            or not 0 < cfg["feed_efficiency"] <= 1
            or not t["min_omega_rad_s"] < t["max_omega_rad_s"]
            or not t["min_omega_rad_s"] <= t["initial_omega_rad_s"] <= t["max_omega_rad_s"]
        ):
            raise NetworkError("Invalid steam-cycle efficiency or rotor speed band")
        self._configuration = cfg
        self._temperature_reference = [
            inventory(cfg[name], mass, energy)["temperature_k"]
            for name, (mass, energy) in zip(("steam_generator", "condenser"), self._origins)
        ]
        self._controls = controls(boundary_controls)
        self._initial_controls = deepcopy(self._controls)
        self._rotor0 = 0.5 * t["inertia_kg_m2"] * t["initial_omega_rad_s"] ** 2
        # dMh,dUh,dMc,dUc,dT_hot_wall,dT_cold_wall,dK; six external energy integrals.
        self._state = np.zeros(13)
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

    def _rates(self, state, *, hot_wall_port_w=0.0):
        hot_wall_port_w = number(hot_wall_port_w, "hot_wall_port_w")
        y = np.asarray(state, dtype=float)
        if y.shape != (13,) or not np.isfinite(y).all():
            raise NetworkError("Invalid steam-cycle state vector")
        cfg, c = self._configuration, self._controls
        hot = inventory(
            cfg["steam_generator"], self._origins[0][0] + y[0], self._origins[0][1] + y[1]
        )
        cold = inventory(cfg["condenser"], self._origins[1][0] + y[2], self._origins[1][1] + y[3])
        gap = hot["pressure_pa"] - cold["pressure_pa"]
        if gap <= 0:
            raise DomainError("Steam-cycle pressure ordering lost")
        turbine = expansion(
            dict(
                inlet_pressure_pa=hot["pressure_pa"],
                outlet_pressure_pa=cold["pressure_pa"],
                isentropic_efficiency=cfg["turbine"]["isentropic_efficiency"],
            )
        )
        steam = cfg["steam_conductance_kg_s_pa"] * c["valve_opening"] * gap
        feed = c["feed_mass_flow_kg_s"]
        pump_work_specific = gap / (cold["liquid"].rho_kg_m3 * cfg["feed_efficiency"])
        feed_enthalpy = cold["liquid"].h_j_kg + pump_work_specific
        pump_power = feed * pump_work_specific
        power = steam * turbine["work_j_kg"]
        rotor = self._rotor0 + y[6]
        if rotor <= 0:
            raise DomainError("Nonpositive rotor energy")
        squared_speed = 2 * rotor / cfg["turbine"]["inertia_kg_m2"]
        g = c["generator_conductance_nm_s_rad"] if c["generator_connected"] else 0.0
        generator = g * squared_speed
        electric = cfg["turbine"]["generator_efficiency"] * generator
        generator_heat = generator - electric
        bearing = cfg["turbine"]["bearing_drag_nm_s_rad"] * squared_speed
        wh = cfg["hot_wall"]["initial_temperature_k"] + y[4]
        wc = cfg["cold_wall"]["initial_temperature_k"] + y[5]
        if min(wh, wc) <= 0:
            raise DomainError("Nonpositive steam-cycle wall temperature")
        # Preserve small wall deviations instead of adding them to a large
        # absolute temperature and immediately subtracting another large value.
        hot_base = cfg["hot_wall"]["fluid_ua_w_k"] * (
            cfg["hot_wall"]["initial_temperature_k"] - self._temperature_reference[0]
        )
        hot_delta = cfg["hot_wall"]["fluid_ua_w_k"] * (
            y[4] - (hot["temperature_k"] - self._temperature_reference[0])
        )
        cold_base = cfg["cold_wall"]["fluid_ua_w_k"] * (
            self._temperature_reference[1] - cfg["cold_wall"]["initial_temperature_k"]
        )
        cold_delta = cfg["cold_wall"]["fluid_ua_w_k"] * (
            (cold["temperature_k"] - self._temperature_reference[1]) - y[5]
        )
        sink_base = cfg["cold_wall"]["ambient_ua_w_k"] * (
            cfg["cold_wall"]["initial_temperature_k"] - c["sink_temperature_k"]
        )
        sink_delta = cfg["cold_wall"]["ambient_ua_w_k"] * y[5]
        hot_heat = hot_base + hot_delta
        cold_heat = cold_base + cold_delta
        sink_heat = sink_base + sink_delta
        feed_power = feed * feed_enthalpy
        saturated_power = steam * hot["vapor"].h_j_kg
        exhaust_power = steam * turbine["outlet_enthalpy_j_kg"]
        condensate_power = feed * cold["liquid"].h_j_kg
        rhs = np.array(
            [
                feed - steam,
                (feed_power - saturated_power + hot_base) + hot_delta,
                steam - feed,
                (exhaust_power - condensate_power - cold_base) - cold_delta,
                ((c["heater_power_w"] - hot_base) + (hot_wall_port_w - hot_delta))
                / cfg["hot_wall"]["heat_capacity_j_k"],
                ((cold_base - sink_base) + (cold_delta - sink_delta))
                / cfg["cold_wall"]["heat_capacity_j_k"],
                power - generator - bearing,
                c["heater_power_w"],
                pump_power,
                electric,
                generator_heat,
                bearing,
                sink_heat,
            ]
        )
        if not np.isfinite(rhs).all():
            raise NetworkError("Nonfinite steam-cycle powers/rates")
        return dict(
            rhs=rhs,
            hot=hot,
            cold=cold,
            turbine=turbine,
            steam=steam,
            feed=feed,
            hot_wall=wh,
            cold_wall=wc,
            rotor=rotor,
            omega=math.sqrt(squared_speed),
            steam_power=power,
            electric=electric,
            generator_heat=generator_heat,
            bearing=bearing,
            pump_power=pump_power,
            feed_power=feed_power,
            saturated_power=saturated_power,
            exhaust_power=exhaust_power,
            condensate_power=condensate_power,
            hot_heat=hot_heat,
            cold_heat=cold_heat,
            sink_heat=sink_heat,
        )

    def derivatives(self, _time, state, *, hot_wall_port_w=0.0):
        """Pure rates; a coupled owner integrates the separate heat-port ledger."""
        return self._rates(state, hot_wall_port_w=hot_wall_port_w)["rhs"]

    def initialization_report(self):
        if self._controls != self._initial_controls:
            return type(self)(self.configuration, self._initial_controls).initialization_report()
        rates = self.derivatives(0.0, np.zeros(13))
        return dict(
            mass_rate_kg_s=dict(steam_generator=float(rates[0]), condenser=float(rates[2])),
            internal_energy_rate_w=dict(steam_generator=float(rates[1]), condenser=float(rates[3])),
            wall_energy_rate_w=dict(
                hot=float(rates[4] * self._configuration["hot_wall"]["heat_capacity_j_k"]),
                cold=float(rates[5] * self._configuration["cold_wall"]["heat_capacity_j_k"]),
            ),
            rotor_energy_rate_w=float(rates[6]),
            mode="explicit_given_state",
        )

    def _margins(self, state):
        r = self._rates(state)
        values = []
        for name, observed in (("steam_generator", r["hot"]), ("condenser", r["cold"])):
            cfg = self._configuration[name]
            values.extend(
                [
                    (
                        name + ".pressure_low",
                        observed["pressure_pa"] - cfg["min_pressure_pa"],
                        "Pa",
                        1e-3,
                    ),
                    (
                        name + ".pressure_high",
                        cfg["max_pressure_pa"] - observed["pressure_pa"],
                        "Pa",
                        1e-3,
                    ),
                    (name + ".quality_low", observed["quality"] - cfg["min_quality"], "1", 1e-10),
                    (name + ".quality_high", cfg["max_quality"] - observed["quality"], "1", 1e-10),
                ]
            )
        t = self._configuration["turbine"]
        values.extend(
            [
                ("shaft.speed_low", r["omega"] - t["min_omega_rad_s"], "rad/s", 1e-8),
                ("shaft.speed_high", t["max_omega_rad_s"] - r["omega"], "rad/s", 1e-8),
                (
                    "steam.pressure_difference",
                    r["hot"]["pressure_pa"]
                    - r["cold"]["pressure_pa"]
                    - self._configuration["min_pressure_difference_pa"],
                    "Pa",
                    1e-3,
                ),
            ]
        )
        return values

    def observation(
        self, state=None, time_s=None, *, hot_wall_port_w=0.0, hot_wall_port_energy_j=0.0
    ):
        """Validate an explicit heat-port context, positive into the hot wall.

        The coupling owner supplies both instantaneous power and its integrated
        energy. Neither is counted as the cycle's external heater. Standalone
        snapshots deliberately retain their original zero-port contract.
        """
        hot_wall_port_energy_j = number(hot_wall_port_energy_j, "hot_wall_port_energy_j")
        y = self._state if state is None else np.asarray(state, dtype=float)
        stamp = self.time_s if time_s is None else number(time_s, "observation time")
        r = self._rates(y, hot_wall_port_w=hot_wall_port_w)
        # Margins use the same pure algebra; there is no mutated EOS seed/cache state.
        margins = self._margins(y)
        for name, value, unit, tol in margins:
            if value < -tol:
                raise DomainError("Steam-cycle passport exceeded: " + name)
        cfg = self._configuration
        energy = math.fsum(
            (
                y[1],
                y[3],
                cfg["hot_wall"]["heat_capacity_j_k"] * y[4],
                cfg["cold_wall"]["heat_capacity_j_k"] * y[5],
                y[6],
            )
        )
        residual = math.fsum(
            (energy, -y[7], -y[8], y[9], y[10], y[11], y[12], -hot_wall_port_energy_j)
        )
        mass = float(y[0] + y[2])
        if abs(residual) > cfg["energy_tolerance_j"] or abs(mass) > cfg["mass_tolerance_kg"]:
            raise NetworkError("Steam-cycle mass/energy ledger exceeded")
        rhs = r["rhs"]
        stored_power = math.fsum(
            (
                rhs[1],
                rhs[3],
                cfg["hot_wall"]["heat_capacity_j_k"] * rhs[4],
                cfg["cold_wall"]["heat_capacity_j_k"] * rhs[5],
                rhs[6],
            )
        )
        external_power = math.fsum(
            (rhs[7], rhs[8], -rhs[9], -rhs[10], -rhs[11], -rhs[12], hot_wall_port_w)
        )
        power_residual = stored_power - external_power
        if abs(power_residual) > 1e-5:
            raise NetworkError("Steam-cycle instantaneous power mismatch")
        vessels = {}
        for name, record in (("steam_generator", r["hot"]), ("condenser", r["cold"])):
            if abs(record["energy_closure_j"]) > 0.01 or abs(record["volume_closure_m3"]) > 1e-9:
                raise NetworkError("Steam-cycle equilibrium closure exceeded")
            vessels[name] = {
                key: value for key, value in record.items() if key not in {"liquid", "vapor"}
            }
        return dict(
            time_s=stamp,
            vessels=vessels,
            controls=self.boundary_controls,
            omega_rad_s=r["omega"],
            rotor_energy_j=r["rotor"],
            hot_wall_temperature_k=r["hot_wall"],
            cold_wall_temperature_k=r["cold_wall"],
            steam_mass_flow_kg_s=r["steam"],
            feed_mass_flow_kg_s=r["feed"],
            steam_power_w=r["steam_power"],
            electric_power_w=r["electric"],
            generator_heat_w=r["generator_heat"],
            bearing_heat_w=r["bearing"],
            feed_pump_power_w=r["pump_power"],
            hot_wall_heat_w=r["hot_heat"],
            cold_wall_heat_w=r["cold_heat"],
            sink_heat_w=r["sink_heat"],
            internal_enthalpy_ports_w=dict(
                feed_to_hot=r["feed_power"],
                hot_to_turbine=r["saturated_power"],
                turbine_to_condenser=r["exhaust_power"],
                condenser_to_feed=r["condensate_power"],
            ),
            thermodynamics=r["turbine"],
            stored_energy_change_j=float(energy),
            mass_residual_kg=mass,
            energy_residual_j=residual,
            power_residual_w=float(power_residual),
            energy_integrals_j=dict(
                zip(
                    (
                        "heater",
                        "feed_pump",
                        "electric",
                        "generator_heat",
                        "bearing_heat",
                        "sink_heat",
                    ),
                    map(float, y[7:]),
                )
            ),
            **(
                dict(
                    hot_wall_port=dict(
                        power_into_cycle_w=float(hot_wall_port_w),
                        energy_into_cycle_j=float(hot_wall_port_energy_j),
                    )
                )
                if hot_wall_port_w != 0 or hot_wall_port_energy_j != 0
                else {}
            ),
            domain_margins={
                name: dict(value=float(value), unit=unit, root_tolerance=tol)
                for name, value, unit, tol in margins
            },
        )

    def jacobian(self, time_s, state):
        """Scaled finite differences of the seven physical states; ledgers do not feed back."""
        y = np.asarray(state, dtype=float)
        base = self.derivatives(time_s, y)
        result = np.zeros((13, 13))
        scales = [
            self._origins[0][0],
            self._origins[0][1],
            self._origins[1][0],
            self._origins[1][1],
            300,
            300,
            self._rotor0,
        ]
        for index, scale in enumerate(scales):
            step = max(abs(y[index]), scale) * 1e-6
            left, right = y.copy(), y.copy()
            left[index] -= step
            right[index] += step
            lo = hi = None
            try:
                lo = self.derivatives(time_s, left)
            except DomainError:
                pass
            try:
                hi = self.derivatives(time_s, right)
            except DomainError:
                pass
            if lo is not None and hi is not None:
                result[:, index] = (hi - lo) / (2 * step)
            elif hi is not None:
                result[:, index] = (hi - base) / step
            elif lo is not None:
                result[:, index] = (base - lo) / step
            else:
                raise DomainError("Both steam-cycle Jacobian probes outside EOS domain")
        if not np.isfinite(result).all():
            raise NetworkError("Nonfinite steam-cycle Jacobian")
        return result

    def advance_until_limit(
        self, dt_s, *, method="Radau", rtol=1e-8, max_step_s=0.1, atol_scale=0.1
    ):
        dt = number(dt_s, "dt_s", positive=True)
        rtol = number(rtol, "rtol", positive=True)
        step = number(max_step_s, "max_step_s", positive=True)
        scale = number(atol_scale, "atol_scale", positive=True)
        if method not in ("Radau", "BDF") or not 1e-12 <= rtol <= 1e-2 or not 1e-6 <= scale <= 1:
            raise NetworkError("Unsupported steam-cycle solver policy")
        end = self.time_s + dt
        if not np.isfinite(end) or end <= self.time_s:
            raise NetworkError("Steam-cycle clock cannot advance")
        self.observation()
        # A localized terminal state can be a few root-resolution units past
        # the mathematical surface. Re-enter only if the new vector field
        # demonstrably points into the domain; otherwise retain the same limit.
        for previous in reversed(self._events):
            if previous["t"] < self.time_s:
                break
            if previous["kind"] != "domain_limit":
                continue
            margins = {name: (value, tol) for name, value, _, tol in self._margins(self._state)}
            value, tolerance = margins[previous["criterion"]]
            if value > tolerance:
                continue
            probe_dt = min(dt, 1e-3)
            probe = self._state + probe_dt * self.derivatives(self.time_s, self._state)
            try:
                next_value = {name: value for name, value, _, _ in self._margins(probe)}[
                    previous["criterion"]
                ]
            except DomainError:
                next_value = value
            if next_value <= value + tolerance:
                return dict(
                    observation=self.observation(),
                    boundary=deepcopy(previous),
                    solver=dict(
                        schema_version=1,
                        backend="boundary_hold",
                        method=method,
                        start_time_s=self.time_s,
                        end_time_s=self.time_s,
                        max_step_s=step,
                        rtol=rtol,
                        atol_scale=scale,
                        scipy_nfev=0,
                        scipy_njev=0,
                        scipy_nlu=0,
                        mesh_times_s=[self.time_s],
                    ),
                )
            break
        events = []
        for index in range(11):

            def event(t, y, index=index):
                margin = self._margins(y)[index]
                # Half of the declared root resolution avoids false terminal
                # events at a stationary boundary represented with EOS roundoff.
                return margin[1] + 0.5 * margin[3]

            event.terminal = True
            event.direction = -1
            events.append(event)
        result = solve_ivp(
            self.derivatives,
            (self.time_s, end),
            self._state,
            method=method,
            rtol=rtol,
            atol=np.array([1e-10, 1e-4, 1e-10, 1e-4, 1e-11, 1e-11, 1e-4, *([1e-4] * 6)]) * scale,
            max_step=step,
            jac=self.jacobian,
            events=events,
            dense_output=False,
        )
        if not result.success:
            raise SolverError(result.message)
        candidate = result.y[:, -1].copy()
        stamp = float(result.t[-1])
        observed = self.observation(candidate, stamp)
        boundary = None
        for index, roots in enumerate(result.t_events):
            if len(roots):
                name, value, unit, tolerance = self._margins(candidate)[index]
                if abs(value) > tolerance:
                    raise NetworkError("Steam-cycle event root residual exceeded")
                boundary = dict(
                    kind="domain_limit",
                    t=stamp,
                    criterion=name,
                    residual=float(value),
                    unit=unit,
                    root_tolerance=tolerance,
                )
                break
        if boundary is not None and len(self._events) >= 10000:
            raise NetworkError("Steam-cycle event history resource limit exceeded")
        diagnostic = dict(
            schema_version=1,
            backend="scipy.solve_ivp",
            method=method,
            start_time_s=self.time_s,
            end_time_s=stamp,
            max_step_s=step,
            rtol=rtol,
            atol_scale=scale,
            scipy_nfev=int(result.nfev),
            scipy_njev=int(result.njev),
            scipy_nlu=int(result.nlu),
            mesh_times_s=result.t.tolist(),
        )
        self._state, self.time_s = candidate, stamp
        if boundary is not None:
            self._events.append(boundary)
        return dict(observation=observed, boundary=deepcopy(boundary), solver=diagnostic)

    def advance(self, dt_s, **solver):
        backup = self.snapshot()
        try:
            result = self.advance_until_limit(dt_s, **solver)
            if result["boundary"] is not None:
                raise DomainError("Steam-cycle interval crosses " + result["boundary"]["criterion"])
            return result["observation"]
        except Exception:
            self._state = np.array(backup["state"])
            self.time_s = backup["time_s"]
            self._events = backup["events"]
            raise

    def set_controls(self, boundary_controls):
        proposed = controls(boundary_controls)
        if len(self._events) >= 10000:
            raise NetworkError("Steam-cycle event history resource limit exceeded")
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
            or digest(payload["configuration"]) != payload["configuration_sha256"]
        ):
            raise NetworkError("Incompatible steam-cycle checkpoint")
        model = cls(payload["configuration"], payload["initial_controls"])
        stamp = number(payload["time_s"], "snapshot time")
        if stamp < 0 or not isinstance(payload["events"], list) or len(payload["events"]) > 10000:
            raise NetworkError("Invalid steam-cycle snapshot time/history")
        previous_time, boundary = -1.0, deepcopy(model._initial_controls)
        for event in payload["events"]:
            if not isinstance(event, dict):
                raise NetworkError("Invalid steam-cycle event")
            t = number(event.get("t"), "event time")
            if not 0 <= t <= stamp or t < previous_time:
                raise NetworkError("Unordered steam-cycle event")
            previous_time = t
            if event.get("kind") == "controls" and set(event) == {"kind", "t", "before", "after"}:
                if controls(event["before"]) != boundary:
                    raise NetworkError("Broken steam-cycle control chain")
                boundary = controls(event["after"])
            elif event.get("kind") == "domain_limit" and set(event) == {
                "kind",
                "t",
                "criterion",
                "residual",
                "unit",
                "root_tolerance",
            }:
                allowed = {name: (unit, tol) for name, _, unit, tol in model._margins(model._state)}
                if event["criterion"] not in allowed:
                    raise NetworkError("Unknown steam-cycle domain criterion")
                unit, tolerance = allowed[event["criterion"]]
                if (
                    event["unit"] != unit
                    or event["root_tolerance"] != tolerance
                    or abs(number(event["residual"], "root residual")) > tolerance
                ):
                    raise NetworkError("Invalid steam-cycle boundary residual")
            else:
                raise NetworkError("Unknown steam-cycle event schema")
        if boundary != controls(payload["controls"]):
            raise NetworkError("Steam-cycle active controls differ from history")
        raw = payload["state"]
        if not isinstance(raw, list) or len(raw) != 13:
            raise NetworkError("Invalid steam-cycle checkpoint state shape")
        state = np.array([number(value, "state") for value in raw])
        if stamp == 0 and np.any(state != 0):
            raise NetworkError("Noninitial steam-cycle state at zero time")
        model._controls = boundary
        model.observation(state, stamp)
        model._state, model.time_s, model._events = state, stamp, deepcopy(payload["events"])
        return model


def balanced_steam_cycle(configuration, *, valve_opening=1.0):
    """Design a steady set point at declared p/x; return all chosen boundaries explicitly.

    This chooses heater/feed/load/sink and wall initial temperatures. It does
    not claim to solve equilibrium for arbitrary fixed boundary commands.
    """
    opening = number(valve_opening, "valve_opening")
    probe = SteamPowerCycle(
        configuration,
        dict(
            heater_power_w=0,
            valve_opening=opening,
            feed_mass_flow_kg_s=0,
            generator_conductance_nm_s_rad=0,
            generator_connected=False,
            sink_temperature_k=295,
        ),
    )
    cfg = probe.configuration
    r = probe._rates(probe._state)
    hot, cold, turbine = r["hot"], r["cold"], cfg["turbine"]
    flow = r["steam"]
    pump_work = (hot["pressure_pa"] - cold["pressure_pa"]) / (
        cold["liquid"].rho_kg_m3 * cfg["feed_efficiency"]
    )
    hot_power = flow * (hot["vapor"].h_j_kg - cold["liquid"].h_j_kg - pump_work)
    cold_power = flow * (r["turbine"]["outlet_enthalpy_j_kg"] - cold["liquid"].h_j_kg)
    if (
        min(
            cfg["hot_wall"]["fluid_ua_w_k"],
            cfg["cold_wall"]["fluid_ua_w_k"],
            cfg["cold_wall"]["ambient_ua_w_k"],
        )
        <= 0
    ):
        raise NetworkError("Balanced steam-cycle design requires positive wall conductances")
    wh = hot["temperature_k"] + hot_power / cfg["hot_wall"]["fluid_ua_w_k"]
    wc = cold["temperature_k"] - cold_power / cfg["cold_wall"]["fluid_ua_w_k"]
    sink = wc - cold_power / cfg["cold_wall"]["ambient_ua_w_k"]
    generator = (
        r["steam_power"] / turbine["initial_omega_rad_s"] ** 2 - turbine["bearing_drag_nm_s_rad"]
    )
    boundary = dict(
        heater_power_w=float(hot_power),
        valve_opening=opening,
        feed_mass_flow_kg_s=float(flow),
        generator_conductance_nm_s_rad=float(generator),
        generator_connected=True,
        sink_temperature_k=float(sink),
    )
    cfg["hot_wall"]["initial_temperature_k"] = float(wh)
    cfg["cold_wall"]["initial_temperature_k"] = float(wc)
    result = SteamPowerCycle(cfg, boundary)
    rates = result.derivatives(0.0, result._state)[:7].copy()
    rates[4] *= cfg["hot_wall"]["heat_capacity_j_k"]
    rates[5] *= cfg["cold_wall"]["heat_capacity_j_k"]
    if max(abs(rates[[0, 2]])) > 1e-9 or max(abs(rates[[1, 3, 4, 5, 6]])) > 1e-4:
        raise NetworkError("Balanced steam-cycle design residual exceeds budget")
    return result
