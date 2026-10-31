"""Independent T-based hybrid reference; closed mode eliminates the flow unknown."""

import numpy as np
from scipy.integrate import solve_ivp

from scripts.storage_references import coefficients


def reference(spec):
    plant = spec["model"]["plant_configuration"]
    hydraulic = plant["hydraulic_configuration"]
    pump = plant["pump"]
    crack = spec["model"]["cracking_pressure_pa"]
    model, rho, cv, m0, volume, vg0, gamma, p0, tg0, tl0, cg, array = coefficients(
        dict(model=hydraulic["storage_configuration"])
    )
    signs = np.array([-1.0, 1.0])
    ua = array("gas_liquid_ua_w_k")
    ambient = array("gas_ambient_ua_w_k")
    heater = array("liquid_heater_w")
    inertia = hydraulic["line"]["inertance_m_inv"]
    receiver = [v["id"] for v in model["vessels"]].index(pump["leak_heat_vessel"])

    def pressure(z):
        return p0 * (z[3:5] / tg0) * (vg0 / (vg0 - signs * z[0] / rho))

    def drive(z):
        p = pressure(z)
        head = (pump["displacement_m3_rad"] * z[9] - z[8] / rho) / pump["leakage_m3_s_pa"]
        return float(p[0] - p[1] + head - crack)

    def unpack(z, closed):
        return np.insert(z, 8, 0.0) if closed else np.asarray(z)

    def full_rhs(t, z, controls, closed):
        x = z[0]
        tl = z[1:3]
        tg = z[3:5]
        m = 0.0 if closed else z[8]
        omega = z[9]
        mass = m0 + signs * x
        p = pressure(z)
        head = (pump["displacement_m3_rad"] * omega - m / rho) / pump["leakage_m3_s_pa"]
        leak = pump["leakage_m3_s_pa"] * head**2
        bearing = pump["drag_nm_s_rad"] * omega**2
        heat = ua * (tg - tl)
        ext = ambient * (model["ambient_temperature_k"] - tg)
        work = signs * p * m / rho
        donor = 0 if m >= 0 else 1
        loss = np.zeros(2)
        loss[1 - donor] = controls["resistance_pa_s2_kg2"] * abs(m) ** 3 / rho
        loss[receiver] += leak
        if not closed:
            loss[1] += crack * m / rho
        tl_rate = (signs * m * cv * (tl[donor] - tl) + loss + heat + heater) / (mass * cv)
        return np.r_[
            m,
            tl_rate,
            (work - heat + ext) / cg,
            work,
            sum(heater + ext),
            0.0
            if closed
            else (p[0] - p[1] + head - crack - controls["resistance_pa_s2_kg2"] * m * abs(m))
            / inertia,
            (
                controls["motor_torque_nm"]
                - pump["displacement_m3_rad"] * head
                - pump["drag_nm_s_rad"] * omega
            )
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
        plant["initial_omega_rad_s"],
        np.zeros(4),
    ]
    state = initial.copy()
    closed = not (state[8] > 0 or drive(state) > 1e-6)
    initial_closed = closed
    pieces = []
    events = []
    boundaries = {}
    now = 0.0
    controls = dict(
        resistance_pa_s2_kg2=hydraulic["line"]["resistance_pa_s2_kg2"],
        motor_torque_nm=plant["motor_torque_nm"],
    )
    for command in spec["timeline"] + [dict(t=spec["duration_s"], controls=controls)]:
        end = command["t"]
        for _ in range(256):
            if now >= end:
                break
            # Resolve roots within the absolute event-time resolution before requesting
            # a public OdeSolution (which forbids duplicate initial timestamps).
            force = drive(state)
            rate0 = full_rhs(now, state, controls, closed)
            if closed:
                pg = pressure(state)
                force_rate = float(
                    pg[0] / state[3] * rate0[3]
                    - pg[1] / state[4] * rate0[4]
                    + pump["displacement_m3_rad"] / pump["leakage_m3_s_pa"] * rate0[9]
                )
                if force >= 1e-6 or (
                    force > 0 and force_rate > 0 and (1e-6 - force) / force_rate <= 1e-12
                ):
                    closed = False
                    events.append(dict(kind="check_opened", t=now))
                    boundaries[now] = (state.copy(), closed)
                    continue
            elif (
                force < 0
                and 0 <= state[8] <= 1e-10
                and rate0[8] < 0
                and state[8] / (-rate0[8]) <= 1e-12
            ):
                state[8] = 0.0
                closed = True
                events.append(dict(kind="check_closed", t=now))
                boundaries[now] = (state.copy(), closed)
                continue
            start = now
            was_closed = closed
            z0 = np.delete(state, 8) if closed else state.copy()

            def rhs(t, z):
                rate = full_rhs(t, unpack(z, was_closed), controls, was_closed)
                return np.delete(rate, 8) if was_closed else rate

            def switch(t, z):
                full = unpack(z, was_closed)
                value = drive(full) - 1e-6 if was_closed else full[8]
                return (-1e-30 if was_closed else 1e-30) if value == 0 else value

            switch.terminal = True
            switch.direction = 1 if closed else -1

            def shaft(t, z):
                omega = unpack(z, was_closed)[9]
                return float(omega) if omega != 0 else 1e-30

            shaft.terminal = False
            shaft.direction = 0
            result = solve_ivp(
                rhs,
                (now, end),
                z0,
                method="DOP853",
                rtol=2e-13,
                atol=1e-12,
                max_step=0.01,
                dense_output=True,
                events=(switch, shaft),
            )
            if not result.success:
                raise ValueError("Independent hybrid reference failed")
            for t, z in zip(result.t_events[1], result.y_events[1]):
                if t > start + 1e-12:
                    acceleration = full_rhs(t, unpack(z, was_closed), controls, was_closed)[9]
                    if acceleration != 0:
                        events.append(
                            dict(
                                kind="shaft_zero",
                                t=float(t),
                                direction=1 if acceleration > 0 else -1,
                            )
                        )
            now = float(result.t[-1])
            state = unpack(result.y[:, -1], was_closed).copy()
            pieces.append((start, now, result.sol, was_closed))
            if len(result.t_events[0]) == 0:
                break
            if now <= start:
                raise ValueError("Independent check event made no progress")
            if closed:
                closed = False
                events.append(dict(kind="check_opened", t=now))
            else:
                closed = True
                state[8] = 0.0
                events.append(dict(kind="check_closed", t=now))
            boundaries[now] = (state.copy(), closed)
        else:
            raise ValueError("Independent transition limit exceeded")
        controls = command["controls"]
    events.sort(key=lambda e: (e["t"], e["kind"]))

    def at(t):
        if t == 0:
            z = initial.copy()
            is_closed = initial_closed
        elif t in boundaries:
            z, is_closed = boundaries[t]
        else:
            a, b, sol, is_closed = next(
                p for p in pieces if p[0] <= t < p[1] or t == spec["duration_s"] and p[1] == t
            )
            z = unpack(sol(t), is_closed)
        return dict(
            transfer_mass_kg=float(z[0]),
            mass_flow_kg_s=float(z[8]),
            omega_rad_s=float(z[9]),
            pressure=pressure(z),
            liquid_temperature=z[1:3],
            gas_temperature=z[3:5],
            gas_boundary_work=z[5:7],
            external_heat_j=float(z[7]),
            motor_work_j=float(z[10]),
            bearing_heat_j=float(z[11]),
            leakage_heat_j=float(z[12]),
            head_work_j=float(z[13]),
            check_mode="closed" if is_closed else "open",
        )

    return at, events
