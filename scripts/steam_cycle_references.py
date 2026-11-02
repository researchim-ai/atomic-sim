"""Independent absolute-inventory DOP853 cycle; two-variable log(p),x EOS root."""

from functools import lru_cache

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root

from physics.materials.iapws import saturation_properties


def reference(spec, *, initial_state=None):
    cfg = spec["model"]
    initial = np.zeros(13)
    for index, name in enumerate(("steam_generator", "condenser")):
        vessel = cfg[name]
        liquid, vapor = saturation_properties(vessel["initial_pressure_pa"])
        x = vessel["initial_quality"]
        mass = vessel["volume_m3"] / ((1 - x) * liquid.v_m3_kg + x * vapor.v_m3_kg)
        initial[2 * index] = mass
        initial[2 * index + 1] = mass * ((1 - x) * liquid.u_j_kg + x * vapor.u_j_kg)
    initial[4:7] = [
        cfg["hot_wall"]["initial_temperature_k"],
        cfg["cold_wall"]["initial_temperature_k"],
        0.5 * cfg["turbine"]["inertia_kg_m2"] * cfg["turbine"]["initial_omega_rad_s"] ** 2,
    ]
    if initial_state is not None:
        initial = np.array(initial_state, dtype=float).copy()

    @lru_cache(maxsize=128)
    def vessel(index, mass, energy):
        settings = cfg[("steam_generator", "condenser")[index]]
        vtarget = settings["volume_m3"] / mass
        utarget = energy / mass

        def residual(z):
            p = settings["initial_pressure_pa"] * np.exp(z[0])
            lf, vg = saturation_properties(float(p))
            return [
                ((1 - z[1]) * lf.v_m3_kg + z[1] * vg.v_m3_kg) / vtarget - 1,
                ((1 - z[1]) * lf.u_j_kg + z[1] * vg.u_j_kg) / utarget - 1,
            ]

        solved = root(residual, [0, settings["initial_quality"]], tol=1e-11)
        if max(abs(v) for v in residual(solved.x)) > 1e-11:
            raise ValueError("Independent log-pressure/quality inventory solve failed")
        p = float(settings["initial_pressure_pa"] * np.exp(solved.x[0]))
        lf, vg = saturation_properties(p)
        return p, float(solved.x[1]), lf, vg

    def margins(y):
        values = []
        for index, name in enumerate(("steam_generator", "condenser")):
            p, x, _, _ = vessel(index, float(y[2 * index]), float(y[2 * index + 1]))
            c = cfg[name]
            values.extend(
                [
                    (name + ".pressure_low", p - c["min_pressure_pa"]),
                    (name + ".pressure_high", c["max_pressure_pa"] - p),
                    (name + ".quality_low", x - c["min_quality"]),
                    (name + ".quality_high", c["max_quality"] - x),
                ]
            )
        omega = np.sqrt(2 * y[6] / cfg["turbine"]["inertia_kg_m2"])
        ph = vessel(0, float(y[0]), float(y[1]))[0]
        pc = vessel(1, float(y[2]), float(y[3]))[0]
        values.extend(
            [
                ("shaft.speed_low", omega - cfg["turbine"]["min_omega_rad_s"]),
                ("shaft.speed_high", cfg["turbine"]["max_omega_rad_s"] - omega),
                ("steam.pressure_difference", ph - pc - cfg["min_pressure_difference_pa"]),
            ]
        )
        return values

    def rhs(t, y, c):
        ph, xh, lh, vh = vessel(0, float(y[0]), float(y[1]))
        pc, xc, lc, vc = vessel(1, float(y[2]), float(y[3]))
        ms = cfg["steam_conductance_kg_s_pa"] * c["valve_opening"] * (ph - pc)
        mf = c["feed_mass_flow_kg_s"]
        is_quality = (vh.s_j_kg_k - lc.s_j_kg_k) / (vc.s_j_kg_k - lc.s_j_kg_k)
        his = lc.h_j_kg + is_quality * (vc.h_j_kg - lc.h_j_kg)
        hout = vh.h_j_kg - cfg["turbine"]["isentropic_efficiency"] * (vh.h_j_kg - his)
        pump_work = (ph - pc) / (lc.rho_kg_m3 * cfg["feed_efficiency"])
        feed_h = lc.h_j_kg + pump_work
        steam_power = ms * (vh.h_j_kg - hout)
        omega_squared = 2 * y[6] / cfg["turbine"]["inertia_kg_m2"]
        g = c["generator_conductance_nm_s_rad"] if c["generator_connected"] else 0.0
        generator = g * omega_squared
        electric = cfg["turbine"]["generator_efficiency"] * generator
        bearing = cfg["turbine"]["bearing_drag_nm_s_rad"] * omega_squared
        qh = cfg["hot_wall"]["fluid_ua_w_k"] * (y[4] - lh.T_k)
        qc = cfg["cold_wall"]["fluid_ua_w_k"] * (lc.T_k - y[5])
        sink = cfg["cold_wall"]["ambient_ua_w_k"] * (y[5] - c["sink_temperature_k"])
        return [
            mf - ms,
            mf * feed_h - ms * vh.h_j_kg + qh,
            ms - mf,
            ms * hout - mf * lc.h_j_kg - qc,
            (c["heater_power_w"] - qh) / cfg["hot_wall"]["heat_capacity_j_k"],
            (qc - sink) / cfg["cold_wall"]["heat_capacity_j_k"],
            steam_power - generator - bearing,
            c["heater_power_w"],
            mf * pump_work,
            electric,
            generator - electric,
            bearing,
            sink,
        ]

    events = []
    for index in range(11):

        def event(t, y, index=index):
            return margins(y)[index][1]

        event.terminal = True
        event.direction = -1
        events.append(event)
    start, state, active = 0.0, initial.copy(), spec["controls"]
    segments = []
    boundary = None
    for command in [*spec["timeline"], dict(t=spec["duration_s"], controls=None)]:
        end = command["t"]
        if end > start:
            c = dict(active)
            solved = solve_ivp(
                lambda t, y: rhs(t, y, c),
                (start, end),
                state,
                method="DOP853",
                rtol=2e-12,
                atol=[1e-11, 1e-5, 1e-11, 1e-5, 1e-10, 1e-10, 1e-6, *([1e-6] * 6)],
                max_step=0.05,
                dense_output=True,
                events=events,
            )
            if not solved.success:
                raise ValueError(solved.message)
            segments.append((start, float(solved.t[-1]), solved.sol))
            state = solved.y[:, -1]
            for index, times in enumerate(solved.t_events):
                if len(times):
                    boundary = dict(t=float(times[0]), criterion=margins(state)[index][0])
            if boundary:
                break
        start, active = end, command["controls"]

    def at(stamp):
        if stamp == 0:
            state = initial.copy()
        else:
            for start, end, solution in segments:
                if start <= stamp <= end:
                    state = solution(stamp)
                    break
            else:
                if (
                    boundary is None
                    or not segments
                    or not segments[-1][1] < stamp <= segments[-1][1] + 1e-5
                ):
                    raise ValueError("Independent steam-cycle time outside domain")
                # Extend only the final physical root's smooth polynomial.
                # A control discontinuity must select its actual right segment.
                state = segments[-1][2](stamp)
        stocks = []
        for i in (0, 1):
            p, x, lf, vg = vessel(i, float(state[2 * i]), float(state[2 * i + 1]))
            stocks.append(
                dict(
                    pressure_pa=p,
                    quality=x,
                    temperature_k=float(lf.T_k),
                    mass_kg=float(state[2 * i]),
                    internal_energy_j=float(state[2 * i + 1]),
                )
            )
        return dict(state=state.copy(), vessels=dict(zip(("steam_generator", "condenser"), stocks)))

    return at, boundary
