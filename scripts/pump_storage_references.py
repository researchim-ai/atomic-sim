"""Independent absolute-temperature equations for pump, shaft and gas storage."""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from scripts.storage_references import coefficients


def reference(spec):
    hydraulic = spec["model"]["hydraulic_configuration"]
    pump = spec["model"]["pump"]
    model, rho, cv, m0, volume, vg0, gamma, p0, tg0, tl0, cg, array = coefficients(
        dict(model=hydraulic["storage_configuration"])
    )
    signs = np.array([-1.0, 1.0])
    ua = array("gas_liquid_ua_w_k")
    ambient = array("gas_ambient_ua_w_k")
    heater = array("liquid_heater_w")
    inertia = hydraulic["line"]["inertance_m_inv"]
    receiver = [v["id"] for v in model["vessels"]].index(pump["leak_heat_vessel"])

    def rhs(t, z, controls):
        x = z[0]
        tl = z[1:3]
        tg = z[3:5]
        m = z[8]
        omega = z[9]
        mass = m0 + signs * x
        vg = volume - mass / rho
        pressure = p0 * (tg / tg0) * (vg0 / vg)
        head = (pump["displacement_m3_rad"] * omega - m / rho) / pump["leakage_m3_s_pa"]
        leak = pump["leakage_m3_s_pa"] * head**2
        bearing = pump["drag_nm_s_rad"] * omega**2
        reaction = pump["displacement_m3_rad"] * head
        donor = 0 if m >= 0 else 1
        heat = ua * (tg - tl)
        ext = ambient * (model["ambient_temperature_k"] - tg)
        work = signs * pressure * m / rho
        loss = np.zeros(2)
        loss[1 - donor] = controls["resistance_pa_s2_kg2"] * abs(m) ** 3 / rho
        loss[receiver] += leak
        liquid_rate = (signs * m * cv * (tl[donor] - tl) + loss + heat + heater) / (mass * cv)
        return np.r_[
            m,
            liquid_rate,
            (work - heat + ext) / cg,
            work,
            sum(heater + ext),
            (pressure[0] - pressure[1] + head - controls["resistance_pa_s2_kg2"] * m * abs(m))
            / inertia,
            (controls["motor_torque_nm"] - reaction - pump["drag_nm_s_rad"] * omega)
            / pump["inertia_kg_m2"],
            controls["motor_torque_nm"] * omega,
            bearing,
            leak,
            head * m / rho,
        ]

    initial = np.r_[
        0.0,
        tl0,
        tg0,
        np.zeros(3),
        hydraulic["initial_mass_flow_kg_s"],
        spec["model"]["initial_omega_rad_s"],
        np.zeros(4),
    ]
    state = initial.copy()
    pieces = []
    events = []
    start = 0.0
    controls = dict(
        resistance_pa_s2_kg2=hydraulic["line"]["resistance_pa_s2_kg2"],
        motor_torque_nm=spec["model"]["motor_torque_nm"],
    )
    for command in spec["timeline"] + [dict(t=spec["duration_s"], controls=controls)]:
        end = command["t"]
        if end > start:
            result = solve_ivp(
                lambda t, z: rhs(t, z, controls),
                (start, end),
                state,
                method="DOP853",
                rtol=2e-13,
                atol=1e-12,
                max_step=0.01,
                dense_output=True,
            )
            if not result.success:
                raise ValueError("Independent pump-storage reference failed")
            for index, kind in ((8, "flow_zero"), (9, "shaft_zero")):
                for i in range(len(result.t) - 1):
                    a, b = result.y[index, i : i + 2]
                    direction = 1 if a < 0 <= b else -1 if a > 0 >= b else 0
                    if direction:
                        root = (
                            float(result.t[i + 1])
                            if b == 0
                            else brentq(
                                lambda t: result.sol(t)[index],
                                result.t[i],
                                result.t[i + 1],
                                xtol=1e-12,
                            )
                        )
                        events.append(dict(kind=kind, t=root, direction=direction))
            pieces.append((start, end, result.sol))
            state = result.y[:, -1]
        controls = command["controls"]
        start = end
    events.sort(key=lambda e: (e["t"], e["kind"]))

    def at(t):
        z = initial.copy() if t == 0 else next(sol(t) for a, b, sol in pieces if a <= t <= b)
        vg = vg0 - signs * z[0] / rho
        return dict(
            transfer_mass_kg=float(z[0]),
            mass_flow_kg_s=float(z[8]),
            omega_rad_s=float(z[9]),
            pressure=p0 * (z[3:5] / tg0) * (vg0 / vg),
            liquid_temperature=z[1:3],
            gas_temperature=z[3:5],
            gas_boundary_work=z[5:7],
            external_heat_j=float(z[7]),
            motor_work_j=float(z[10]),
            bearing_heat_j=float(z[11]),
            leakage_heat_j=float(z[12]),
            head_work_j=float(z[13]),
        )

    return at, events
