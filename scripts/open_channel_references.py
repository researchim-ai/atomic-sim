"""Independent absolute-energy reference for prescribed exterior-state channels."""

from copy import deepcopy

import numpy as np
from scipy.integrate import solve_ivp

from scripts.conservative_loop_references import IndependentLoop
from scripts.conservative_loop_references import request as loop_request


def request(*, velocity=0.0, bias=0.0, height=0.0, shift=1e5, heat=0.0):
    r = loop_request(velocity=velocity, shift=shift)
    n = r["geometry"]["cells"]
    z = (np.arange(n) + 0.5) * height / n
    g = 9.81 if height else 0.0
    r["geometry"].update(elevations_m=z.tolist(), gravity_m_s2=g)
    r["initial"]["pressures_pa"] = (
        200000.0 + (200000.0 + shift) * np.expm1(-g * z / (400 * 300))
    ).tolist()
    r["heat"]["power_w"][3] = heat
    r["format"] = "atomic-open-eos-channel"
    r["boundaries"] = {}
    for side, elevation, delta in [("left", 0.0, bias / 2), ("right", height, -bias / 2)]:
        pressure = 200000.0 + (200000.0 + shift) * np.expm1(-g * elevation / (400 * 300)) + delta
        r["boundaries"][side] = dict(
            pressure_pa=float(pressure),
            temperature_k=300.0,
            velocity_m_s=velocity if not bias else 0.0,
            elevation_m=elevation,
        )
    return r


def cases():
    yield "rest", request(), []
    yield "forward_uniform", request(velocity=5), []
    yield "reverse_uniform", request(velocity=-5), []
    yield "pressure_forward", request(bias=1000), []
    yield "pressure_reverse", request(bias=-1000), []
    yield "hydrostatic", request(height=10), []
    yield "stiff_heated", request(shift=1e8, heat=500), []
    yield "opposed_reversal", request(velocity=2, bias=-10000), []
    r = request(bias=1000, heat=500)
    powers = {f"c{i}": 500.0 if i == 6 else 0.0 for i in range(8)}
    yield "heater_command", r, [(0.1, powers)]
    yield "gravity_heated", request(height=10, bias=1000, heat=500), []
    r = request(velocity=3, bias=500, heat=500)
    r["drag"] = dict(rate_per_s=0.4, exported_fraction=0.25)
    r["heat"]["ambient_ua_w_k"][5] = 10.0
    yield "drag_ambient", r, []
    r = request()
    r["boundaries"]["left"]["temperature_k"] = 303.0
    yield "thermal_bath", r, []


class IndependentOpen:
    def __init__(self, spec):
        self.spec = deepcopy(spec)
        self.base = IndependentLoop(spec)
        e = self.base
        self.n = e.n
        self.faces = np.r_[
            e.spec["geometry"]["gravity_m_s2"] * spec["boundaries"]["left"]["elevation_m"],
            0.5 * (e.phi[:-1] + e.phi[1:]),
            e.spec["geometry"]["gravity_m_s2"] * spec["boundaries"]["right"]["elevation_m"],
        ]
        self.initial = np.r_[e.initial, np.zeros(8)].astype(np.longdouble)
        self.heat = e.heat.copy()

    def primitive(self, y):
        return self.base.primitive(y[: 3 * self.n + 6])

    def rhs(self, t, y, heaters):
        e = self.base
        n = self.n
        m, P, E, rho, u, T, p, s = self.primitive(y)
        face = np.zeros((n + 1, 3), dtype=np.longdouble)
        entropy_face = np.zeros(n + 1, dtype=np.longdouble)

        def cell(i, phi):
            factor = np.exp(-(phi - e.phi[i]) / (e.R * T[i]))
            return rho[i] * factor, T[i], (p[i] + e.shift) * factor - e.shift, u[i]

        def bath(side):
            b = self.spec["boundaries"][side]
            T = np.longdouble(b["temperature_k"])
            p = np.longdouble(b["pressure_pa"])
            return (p + e.shift) / (e.R * T), T, p, np.longdouble(b["velocity_m_s"])

        def values(primitive, phi):
            r, temp, pressure, v = primitive
            energy = r * (e.cv * temp + e.offset + 0.5 * v * v + phi) + e.shift
            conserved = np.array([r, r * v, energy])
            flux = np.array(
                [r * v, r * v * v + pressure, r * v * (e.cp * temp + e.offset + 0.5 * v * v + phi)]
            )
            entropy = e.cv * np.log(temp / e.Tref) - e.R * np.log(r / e.rref)
            return (
                conserved,
                flux,
                abs(v) + np.sqrt(e.gamma * e.R * temp),
                r * entropy,
                r * v * entropy,
            )

        for j, phi in enumerate(self.faces):
            left = bath("left") if j == 0 else cell(j - 1, phi)
            right = bath("right") if j == n else cell(j, phi)
            L, FL, aL, SL, FSL = values(left, phi)
            R, FR, aR, SR, FSR = values(right, phi)
            a = max(aL, aR)
            face[j] = e.area * ((FL + FR) / 2 - a * (R - L) / 2)
            entropy_face[j] = e.area * ((FSL + FSR) / 2 - a * (SR - SL) / 2)
        change = face[:-1] - face[1:]
        grav = np.zeros(n, dtype=np.longdouble)
        for i in range(n):
            grav[i] = e.area * (
                (p[i] + e.shift) * np.exp(-(self.faces[i + 1] - e.phi[i]) / (e.R * T[i]))
                - (p[i] + e.shift) * np.exp(-(self.faces[i] - e.phi[i]) / (e.R * T[i]))
            )
        drag = self.spec["drag"]["rate_per_s"] * P
        export = (
            self.spec["drag"]["exported_fraction"] * self.spec["drag"]["rate_per_s"] * P * P / m
        )
        ambient = np.array(self.spec["heat"]["ambient_ua_w_k"]) * (
            np.array(self.spec["heat"]["ambient_temperature_k"]) - T
        )
        Q = heaters + ambient
        change[:, 1] += grav - drag
        change[:, 2] += Q - export
        Udot = change[:, 2] - u * change[:, 1] + (0.5 * u * u - e.phi) * change[:, 0]
        entropy_rate = (Udot - (e.cp * T + e.offset - T * s) * change[:, 0]) / T
        SE = sum(
            (heaters - export) / T + ambient / np.array(self.spec["heat"]["ambient_temperature_k"])
        )
        BS = entropy_face[0] - entropy_face[-1]
        generated = sum(entropy_rate) - SE - BS
        return np.asarray(
            np.r_[
                change[:, 0],
                change[:, 1],
                change[:, 2],
                [
                    sum(Q),
                    sum(export),
                    SE,
                    generated,
                    sum(grav),
                    sum(drag),
                    *face[0],
                    entropy_face[0],
                    *(-face[-1]),
                    -entropy_face[-1],
                ],
            ],
            dtype=float,
        )

    def integrate(self, times, schedule, method):
        y = self.initial.copy()
        heat = self.heat.copy()
        current = 0.0
        rows = []
        evaluations = 0
        for end in times:
            for when, powers in schedule:
                if when == current:
                    heat = np.array([powers[f"c{i}"] for i in range(self.n)])
            base = y.copy()

            def fun(t, z):
                nonlocal evaluations
                evaluations += 1
                if evaluations > 300000:
                    raise ValueError("Open reference RHS budget")
                return self.rhs(t, base + z.astype(np.longdouble), heat)

            sol = solve_ivp(
                fun,
                (current, end),
                np.zeros(len(y)),
                method=method,
                rtol=2e-10,
                atol=1e-9,
                max_step=0.001,
            )
            if not sol.success:
                raise ValueError("Open reference solve failed")
            y = base + sol.y[:, -1].astype(np.longdouble)
            current = end
            m, P, E, r, u, T, p, s = self.primitive(y)
            rows.append(
                dict(
                    masses_kg=m.astype(float).tolist(),
                    velocities_m_s=u.astype(float).tolist(),
                    temperatures_k=T.astype(float).tolist(),
                    pressures_pa=p.astype(float).tolist(),
                    ledgers=y[-14:].astype(float).tolist(),
                )
            )
        return rows
