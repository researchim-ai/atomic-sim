"""Independent absolute-state primary/steam equations; shared IF97 backend only.

No production model, derivative, snapshot or bounded-step helper is called.
Primary flow/speed/absolute temperatures and absolute steam M/U are integrated.
"""

import json
import math
from copy import deepcopy
from functools import lru_cache

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root

from physics.materials.iapws import saturation_properties

REFERENCE_POLICY = dict(rtol=5e-14, mechanical_atol=2e-13, max_step_s=0.00625)


def steam_system(spec, *, initial_state=None):
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
        if (
            not np.isfinite(solved.x).all()
            or not np.isfinite(residual(solved.x)).all()
            or max(abs(v) for v in residual(solved.x)) > 1e-11
        ):
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

    return initial, vessel, margins, rhs


def primary_coefficients(spec):
    raw = spec["primary"]
    model = raw["loop"]["model"]
    init = raw["loop"]["initialization"]
    if init["mode"] != "given":
        raise ValueError("Independent primary reference requires explicit initial states")
    nodes = {node["id"]: node["parameters"] for node in model["topology"]["nodes"]}
    successors = {
        c["from"].split(".")[0]: c["to"].split(".")[0] for c in model["topology"]["connections"]
    }
    names = [model["reference"]["segment"]]
    while len(names) < len(nodes):
        names.append(successors[names[-1]])
    rho, cp, area = (
        model["fluid"][k] for k in ("density_kg_m3", "heat_capacity_j_kg_k", "area_m2")
    )

    def values(key):
        return np.array([nodes[name][key] for name in names])

    n = len(names)
    initial = np.r_[
        init["mass_flow_kg_s"],
        raw["initial_omega_rad_s"],
        [init["fluid_temperature_k"][name] for name in names],
        [init["wall_temperature_k"][name] for name in names],
        np.zeros(5),
    ]
    return dict(
        names=names,
        n=n,
        rho=rho,
        area=area,
        steam_inertia=spec["steam_model"]["turbine"]["inertia_kg_m2"],
        cp=cp,
        model=model,
        pump=raw["pump"],
        initial=initial,
        inertia=values("length_m") / area,
        resistance=values("loss_coefficient") / (2 * rho * area**2),
        capacity=rho * area * values("length_m") * cp,
        wall_capacity=values("wall_capacity_j_k"),
        ua=values("fluid_wall_ua_w_k"),
        ambient=values("ambient_ua_w_k"),
        heater=values("wall_power_w"),
        pump_index=names.index(raw["pump"]["segment"]),
        exchange_index=names.index(spec["coupling"]["steam_generator_segment"]),
        heater_index=names.index(spec["coupling"]["heater_segment"]),
    )


def system(spec):
    data = primary_coefficients(spec)
    steam_initial, vessel, steam_margins, steam_rhs = steam_system(dict(model=spec["steam_model"]))
    n, offset = data["n"], len(data["initial"])
    initial = np.r_[data["initial"], steam_initial, 0.0]
    cfg = spec["coupling"]

    def primary(state, boundary):
        flow, omega = state[:2]
        tf, tw = state[2 : 2 + n], state[2 + n : 2 + 2 * n]
        p = data["pump"]
        head = (p["displacement_m3_rad"] * omega - flow / data["rho"]) / p["leakage_m3_s_pa"]
        leak = p["leakage_m3_s_pa"] * head**2
        bearing = p["drag_nm_s_rad"] * omega**2
        dm = (head - sum(data["resistance"]) * flow * abs(flow)) / sum(data["inertia"])
        heads = np.zeros(n)
        heads[data["pump_index"]] = head
        drop = data["inertia"] * dm + data["resistance"] * flow * abs(flow) - heads
        pressure = np.r_[
            data["model"]["reference"]["pressure_pa"],
            data["model"]["reference"]["pressure_pa"] - np.cumsum(drop)[:-1],
        ]
        fluid, wall = np.zeros(n), np.zeros(n)
        for i in range(n):
            donor = (i - 1 if flow >= 0 else i + 1) % n
            transfer = data["ua"][i] * (tw[i] - tf[i])
            fluid[i] = (
                data["cp"] * abs(flow) * (tf[donor] - tf[i])
                + data["resistance"][i] * abs(flow) ** 3 / data["rho"]
                + transfer
            )
            wall[i] = (
                data["heater"][i]
                - transfer
                + data["ambient"][i] * (data["model"]["ambient_temperature_k"] - tw[i])
            )
        wall[data["pump_index"]] += leak
        wall[data["heater_index"]] += boundary["primary_wall_heater_power_w"]
        q = cfg["ua_w_k"] * (tf[data["exchange_index"]] - state[offset + 4])
        fluid[data["exchange_index"]] -= q
        torque = boundary["primary_motor_torque_nm"]
        exterior = sum(
            data["heater"] + data["ambient"] * (data["model"]["ambient_temperature_k"] - tw)
        )
        exterior += boundary["primary_wall_heater_power_w"]
        rates = np.r_[
            dm,
            (torque - p["displacement_m3_rad"] * head - p["drag_nm_s_rad"] * omega)
            / p["inertia_kg_m2"],
            fluid / data["capacity"],
            wall / data["wall_capacity"],
            torque * omega,
            bearing,
            leak,
            head * flow / data["rho"],
            exterior,
        ]
        return rates, pressure, q

    def rhs(t, state, boundary):
        rates, _, q = primary(state, boundary)
        steam = np.asarray(steam_rhs(t, state[offset : offset + 13], boundary["steam_cycle"]))
        steam[4] += q / spec["steam_model"]["hot_wall"]["heat_capacity_j_k"]
        return np.r_[rates, steam, q]

    def margins(state, boundary):
        _, pressure, _ = primary(state, boundary)
        values = list(steam_margins(state[offset : offset + 13]))
        for i, name in enumerate(data["names"]):
            sat, _ = saturation_properties(float(pressure[i]))
            temperature = state[2 + i]
            values.extend(
                [
                    (f"primary.{name}.pressure_low", pressure[i] - cfg["min_primary_pressure_pa"]),
                    (f"primary.{name}.pressure_high", cfg["max_primary_pressure_pa"] - pressure[i]),
                    (
                        f"primary.{name}.temperature_low",
                        temperature - cfg["min_primary_temperature_k"],
                    ),
                    (
                        f"primary.{name}.subcooling",
                        sat.T_k - temperature - cfg["minimum_subcooling_k"],
                    ),
                ]
            )
        return values

    return initial, rhs, margins, vessel, data, offset


def _reference(spec, *, initial_state=None, precision):
    initial, rhs, margins, vessel, data, offset = system(spec)
    if initial_state is not None:
        initial = np.asarray(initial_state, dtype=float).copy()
    segments = []
    state, start, active = initial.copy(), 0.0, spec["controls"]
    boundary = None
    for command in [*spec["timeline"], dict(t=spec["duration_s"], controls=None)]:
        end = command["t"]
        if end > start:
            c = dict(active)
            events = []
            for index in range(len(margins(state, c))):

                def event(t, y, index=index):
                    return margins(y, c)[index][1]

                event.terminal = True
                event.direction = -1
                events.append(event)
            result = solve_ivp(
                lambda t, y: rhs(t, y, c),
                (start, end),
                state,
                method="DOP853",
                rtol=precision["rtol"],
                atol=np.r_[
                    precision["mechanical_atol"],
                    precision["mechanical_atol"],
                    np.full(2 * data["n"], 1e-10),
                    np.full(5, 1e-6),
                    1e-11,
                    1e-5,
                    1e-11,
                    1e-5,
                    1e-10,
                    1e-10,
                    1e-6,
                    np.full(7, 1e-6),
                ],
                max_step=precision["max_step_s"],
                dense_output=True,
                events=events,
            )
            if not result.success:
                raise ValueError(result.message)
            segments.append((start, float(result.t[-1]), result.sol))
            state = result.y[:, -1]
            for index, roots in enumerate(result.t_events):
                if len(roots):
                    boundary = dict(t=float(roots[0]), criterion=margins(state, c)[index][0])
            if boundary:
                break
        start, active = end, command["controls"]

    def at(stamp):
        if stamp == 0:
            return initial.copy()
        for a, b, solution in segments:
            if a <= stamp <= b:
                return solution(stamp)
        if boundary is not None and segments and segments[-1][1] < stamp <= segments[-1][1] + 1e-5:
            return segments[-1][2](stamp)
        raise ValueError("Independent primary/steam time outside accepted domain")

    return at, boundary, (rhs, margins, vessel, data, offset, initial)


@lru_cache(maxsize=8)
def _cached_reference(payload):
    parameters = json.loads(payload)
    return _reference(
        parameters["spec"],
        initial_state=parameters["initial_state"],
        precision=parameters["precision"],
    )


def reference(spec, *, initial_state=None, precision=None):
    """Reuse only identical independent physical IVPs, never assessment verdicts.

    Solver/output policy and human name do not enter the independent equations.
    Every physical parameter, timeline, horizon and optional absolute initial
    state enters the key. JSON detaches inputs; returned trajectory samples
    are copies. The bounded cache also limits long-trajectory memory.
    """
    policy = deepcopy(REFERENCE_POLICY if precision is None else precision)
    if (
        not isinstance(policy, dict)
        or set(policy) != set(REFERENCE_POLICY)
        or any(
            isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) or v <= 0
            for v in policy.values()
        )
        or policy["rtol"] < 2.5e-14
    ):
        raise ValueError("Finite positive independent reference policy with rtol>=2.5e-14 required")
    physical = {
        key: value for key, value in spec.items() if key not in ("solver", "step_s", "name")
    }
    payload = json.dumps(
        dict(
            spec=physical,
            precision=policy,
            initial_state=None if initial_state is None else np.asarray(initial_state).tolist(),
        ),
        sort_keys=True,
        allow_nan=False,
    )
    at, boundary, details = _cached_reference(payload)
    rhs, margins, vessel, data, offset, initial = details
    return at, deepcopy(boundary), (rhs, margins, vessel, deepcopy(data), offset, initial.copy())


def recorded_state(row, data):
    """Map reported physical quantities to the independent absolute-state order."""
    p = row["primary"]
    loop = p["loop"]
    s = row["steam_cycle"]
    integral = s["energy_integrals_j"]
    inventories = [
        value
        for name in ("steam_generator", "condenser")
        for value in (s["vessels"][name]["mass_kg"], s["vessels"][name]["internal_energy_j"])
    ]
    energies = [
        integral[name]
        for name in (
            "heater",
            "feed_pump",
            "electric",
            "generator_heat",
            "bearing_heat",
            "sink_heat",
        )
    ]
    values = [
        loop["mass_flow_kg_s"],
        p["omega_rad_s"],
        *[loop["fluid_temperature_k"][name] for name in data["names"]],
        *[loop["wall_temperature_k"][name] for name in data["names"]],
        p["motor_work_j"],
        p["bearing_heat_j"],
        p["leakage_heat_j"],
        loop["external_head_work_j"],
        p["external_ambient_heater_heat_j"],
        *inventories,
        s["hot_wall_temperature_k"],
        s["cold_wall_temperature_k"],
        s["rotor_energy_j"],
        *energies,
        row["heat_transfer_energy_j"],
    ]
    if any(
        isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v)
        for v in values
    ):
        raise ValueError("Finite numeric recorded physical states required")
    return np.asarray(values, dtype=float)


def energy_balances(state, initial, data, offset, spec):
    n = data["n"]
    cfg = spec["steam_model"]
    y = state
    origin = initial
    primary_stored = math.fsum(
        (
            float(data["capacity"] @ (y[2 : 2 + n] - origin[2 : 2 + n])),
            float(data["wall_capacity"] @ (y[2 + n : 2 + 2 * n] - origin[2 + n : 2 + 2 * n])),
            sum(data["inertia"]) * (y[0] - origin[0]) * (y[0] + origin[0]) / (2 * data["rho"]),
            0.5 * data["pump"]["inertia_kg_m2"] * (y[1] - origin[1]) * (y[1] + origin[1]),
        )
    )
    s, si = y[offset : offset + 13], origin[offset : offset + 13]
    steam_stored = math.fsum(
        (
            s[1] - si[1],
            s[3] - si[3],
            cfg["hot_wall"]["heat_capacity_j_k"] * (s[4] - si[4]),
            cfg["cold_wall"]["heat_capacity_j_k"] * (s[5] - si[5]),
            s[6] - si[6],
        )
    )
    motor, bearing, _, _, heat = y[2 + 2 * n : offset] - origin[2 + 2 * n : offset]
    e = s[7:] - si[7:]
    port = y[-1] - origin[-1]
    primary_residual = math.fsum((primary_stored, -motor, bearing, -heat, port))
    steam_residual = math.fsum((steam_stored, -e[0], -e[1], e[2], e[3], e[4], e[5], -port))
    return dict(
        primary_stored_j=primary_stored,
        steam_stored_j=steam_stored,
        primary_residual_j=primary_residual,
        steam_residual_j=steam_residual,
        total_residual_j=primary_residual + steam_residual,
        mass_residual_kg=float((s[0] - si[0]) + (s[2] - si[2])),
    )


def power_balance(state, rates, data, offset, spec):
    n = data["n"]
    y = state
    r = rates
    cfg = spec["steam_model"]
    stored = math.fsum(
        (
            float(data["capacity"] @ r[2 : 2 + n]),
            float(data["wall_capacity"] @ r[2 + n : 2 + 2 * n]),
            sum(data["inertia"]) * y[0] * r[0] / data["rho"],
            data["pump"]["inertia_kg_m2"] * y[1] * r[1],
            r[offset + 1],
            r[offset + 3],
            cfg["hot_wall"]["heat_capacity_j_k"] * r[offset + 4],
            cfg["cold_wall"]["heat_capacity_j_k"] * r[offset + 5],
            r[offset + 6],
        )
    )
    ledgers = r[2 + 2 * n : offset]
    s = r[offset : offset + 13]
    external = math.fsum(
        (ledgers[0], -ledgers[1], ledgers[4], s[7], s[8], -s[9], -s[10], -s[11], -s[12])
    )
    return stored - external


def observables(state, boundary, data, offset, vessel, spec):
    """Independent pressure, total-enthalpy faces and machine/thermal powers."""
    y = state
    n = data["n"]
    p = data["pump"]
    cfg = spec["steam_model"]
    c = boundary["steam_cycle"]
    flow, omega = y[:2]
    tf = y[2 : 2 + n]
    head = (p["displacement_m3_rad"] * omega - flow / data["rho"]) / p["leakage_m3_s_pa"]
    dm = (head - sum(data["resistance"]) * flow * abs(flow)) / sum(data["inertia"])
    heads = np.zeros(n)
    heads[data["pump_index"]] = head
    drop = data["inertia"] * dm + data["resistance"] * flow * abs(flow) - heads
    pressure = np.r_[
        data["model"]["reference"]["pressure_pa"],
        data["model"]["reference"]["pressure_pa"] - np.cumsum(drop)[:-1],
    ]
    face_donor = [(i - 1) % n if flow >= 0 else i for i in range(n)]
    velocity = flow / (data["rho"] * data["area"])
    flux = flow * (data["cp"] * tf[face_donor] + pressure / data["rho"] + 0.5 * velocity**2)
    ports = {}
    for i, name in enumerate(data["names"]):
        j = (i + 1) % n
        ports[name + ".left"] = dict(
            pressure_pa=float(pressure[i]), mass_flow_kg_s=float(flow), energy_flow_w=float(flux[i])
        )
        ports[name + ".right"] = dict(
            pressure_pa=float(pressure[j]),
            mass_flow_kg_s=float(-flow),
            energy_flow_w=float(-flux[j]),
        )
    s = y[offset : offset + 13]
    ph, xh, lh, vh = vessel(0, float(s[0]), float(s[1]))
    pc, xc, lc, vc = vessel(1, float(s[2]), float(s[3]))
    ms = cfg["steam_conductance_kg_s_pa"] * c["valve_opening"] * (ph - pc)
    mf = c["feed_mass_flow_kg_s"]
    xis = (vh.s_j_kg_k - lc.s_j_kg_k) / (vc.s_j_kg_k - lc.s_j_kg_k)
    his = lc.h_j_kg + xis * (vc.h_j_kg - lc.h_j_kg)
    hout = vh.h_j_kg - cfg["turbine"]["isentropic_efficiency"] * (vh.h_j_kg - his)
    xout = (hout - lc.h_j_kg) / (vc.h_j_kg - lc.h_j_kg)
    sout = lc.s_j_kg_k + xout * (vc.s_j_kg_k - lc.s_j_kg_k)
    pumpwork = (ph - pc) / (lc.rho_kg_m3 * cfg["feed_efficiency"])
    omega2 = 2 * s[6] / cfg["turbine"]["inertia_kg_m2"]
    generator = (c["generator_conductance_nm_s_rad"] if c["generator_connected"] else 0) * omega2
    electric = generator * cfg["turbine"]["generator_efficiency"]
    q = spec["coupling"]["ua_w_k"] * (tf[data["exchange_index"]] - s[4])
    return dict(
        heat_transfer_w=float(q),
        primary=dict(
            head_pa=float(head),
            hydraulic_torque_nm=float(p["displacement_m3_rad"] * head),
            motor_power_w=float(boundary["primary_motor_torque_nm"] * omega),
            hydraulic_power_w=float(head * flow / data["rho"]),
            shaft_to_pump_power_w=float(p["displacement_m3_rad"] * head * omega),
            leakage_power_w=float(p["leakage_m3_s_pa"] * head**2),
            bearing_power_w=float(p["drag_nm_s_rad"] * omega**2),
            loop=dict(ports=ports, stored_mass_kg=float(sum(data["capacity"]) / data["cp"])),
        ),
        steam_cycle=dict(
            vessels=dict(
                steam_generator=dict(pressure_pa=ph, quality=xh, temperature_k=lh.T_k),
                condenser=dict(pressure_pa=pc, quality=xc, temperature_k=lc.T_k),
            ),
            thermodynamics=dict(
                inlet_enthalpy_j_kg=vh.h_j_kg,
                outlet_enthalpy_j_kg=hout,
                isentropic_outlet_enthalpy_j_kg=his,
                isentropic_quality=xis,
                inlet_entropy_j_kg_k=vh.s_j_kg_k,
                outlet_entropy_j_kg_k=sout,
                inlet_temperature_k=lh.T_k,
                outlet_temperature_k=lc.T_k,
                outlet_quality=xout,
                work_j_kg=vh.h_j_kg - hout,
            ),
            steam_mass_flow_kg_s=ms,
            feed_mass_flow_kg_s=mf,
            steam_power_w=ms * (vh.h_j_kg - hout),
            electric_power_w=electric,
            generator_heat_w=generator - electric,
            bearing_heat_w=cfg["turbine"]["bearing_drag_nm_s_rad"] * omega2,
            feed_pump_power_w=mf * pumpwork,
            hot_wall_heat_w=cfg["hot_wall"]["fluid_ua_w_k"] * (s[4] - lh.T_k),
            cold_wall_heat_w=cfg["cold_wall"]["fluid_ua_w_k"] * (lc.T_k - s[5]),
            sink_heat_w=cfg["cold_wall"]["ambient_ua_w_k"] * (s[5] - c["sink_temperature_k"]),
            internal_enthalpy_ports_w=dict(
                feed_to_hot=mf * (lc.h_j_kg + pumpwork),
                hot_to_turbine=ms * vh.h_j_kg,
                turbine_to_condenser=ms * hout,
                condenser_to_feed=mf * lc.h_j_kg,
            ),
        ),
    )
