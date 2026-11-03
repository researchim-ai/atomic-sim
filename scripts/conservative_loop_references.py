"""Manufactured conservative Euler loops and independent absolute-state MOL."""

from copy import deepcopy

import numpy as np
from scipy.integrate import solve_ivp

from scripts.storage_graph_references import fluid


def request(
    n=8,
    *,
    height=0.0,
    gravity=0.0,
    velocity=0.0,
    wave=0.0,
    heat=0.0,
    drag=0.0,
    export=0.0,
    shift=1e5,
):
    length = 100.0
    x = (np.arange(n) + 0.5) * length / n
    z = 0.5 * height * (1 - np.cos(2 * np.pi * x / length))
    p = 200000.0 + (200000.0 + shift) * np.expm1(-gravity * z / (400.0 * 300.0))
    T = np.full(n, 300.0)
    if wave:
        p[:] = 200000.0
        T = 300 / (1 + wave * np.sinc(1 / n) * np.sin(2 * np.pi * x / length))
    powers = np.zeros(n)
    ua = np.zeros(n)
    if heat:
        powers[1] = heat
        ua[5 * n // 8] = 20.0
    return dict(
        format="atomic-conservative-eos-loop",
        schema_version=1,
        fluid=fluid(shift),
        geometry=dict(
            cells=n, length_m=length, area_m2=0.01, elevations_m=z.tolist(), gravity_m_s2=gravity
        ),
        initial=dict(
            pressures_pa=p.astype(float).tolist(),
            temperatures_k=T.astype(float).tolist(),
            velocities_m_s=[velocity] * n,
        ),
        heat=dict(
            power_w=powers.tolist(), ambient_ua_w_k=ua.tolist(), ambient_temperature_k=[300.0] * n
        ),
        drag=dict(rate_per_s=drag, exported_fraction=export),
        domain=dict(
            min_temperature_k=100.0,
            max_temperature_k=1000.0,
            min_pressure_pa=1000.0,
            max_pressure_pa=1e9,
            min_density_kg_m3=0.001,
            max_density_kg_m3=5000.0,
            maximum_mach=0.2,
        ),
    )


def cases():
    yield "uniform_rest", request(), []
    yield "hydrostatic_rest", request(height=10, gravity=9.81), []
    yield "stiff_hydrostatic", request(height=10, gravity=9.81, shift=1e8), []
    yield "uniform_drag", request(velocity=10, drag=0.4, export=0.25), []
    yield "entropy_wave", request(velocity=10, wave=0.01), []
    yield "heated_gravity", request(height=10, gravity=9.81, heat=500, drag=0.1), []
    yield "heated_gravity_off", request(height=10, gravity=0, heat=500, drag=0.1), []
    yield "stiff_heated", request(height=10, gravity=9.81, heat=500, drag=0.1, shift=1e8), []
    r = request(height=10, gravity=9.81, heat=500, drag=0.1)
    change = {f"c{i}": (500.0 if i == 6 else 0.0) for i in range(8)}
    yield "heater_command", r, [(0.5, change)]


class IndependentLoop:
    def __init__(self, spec):
        self.spec = deepcopy(spec)
        g = spec["geometry"]
        f = spec["fluid"]
        initial = spec["initial"]
        self.n = g["cells"]
        self.dx = g["length_m"] / self.n
        self.area = g["area_m2"]
        self.V = self.area * self.dx
        self.cv = f["cv_j_kg_k"]
        self.gamma = f["gamma"]
        self.R = (self.gamma - 1) * self.cv
        self.cp = self.gamma * self.cv
        self.shift = f["pressure_shift_pa"]
        self.offset = f["energy_offset_j_kg"]
        self.Tref = f["entropy_reference_temperature_k"]
        self.rref = f["entropy_reference_density_kg_m3"]
        self.phi = g["gravity_m_s2"] * np.array(g["elevations_m"])
        self.face = 0.5 * (self.phi + np.roll(self.phi, -1))
        T = np.array(initial["temperatures_k"])
        p = np.array(initial["pressures_pa"])
        u = np.array(initial["velocities_m_s"])
        m = (p + self.shift) / (self.R * T) * self.V
        P = m * u
        E = m * (self.cv * T + self.offset + 0.5 * u * u + self.phi) + self.shift * self.V
        self.initial = np.r_[m, P, E, np.zeros(6)]
        self.heat = np.array(spec["heat"]["power_w"])

    def primitive(self, y):
        # Wider arithmetic limits cancellation of absolute energy/pressure offsets.
        # Solver states remain float64; this does not change the physical model.
        y = np.asarray(y, dtype=np.longdouble)
        m = y[: self.n]
        P = y[self.n : 2 * self.n]
        E = y[2 * self.n : 3 * self.n]
        rho = m / self.V
        u = P / m
        T = (E / m - 0.5 * u * u - self.phi - self.offset - self.shift / rho) / self.cv
        p = self.R * rho * T - self.shift
        entropy = self.cv * np.log(T / self.Tref) - self.R * np.log(rho / self.rref)
        return m, P, E, rho, u, T, p, entropy

    def rhs(self, t, y, heat):
        n = self.n
        m, P, E, rho, u, T, p, entropy = self.primitive(y)
        flux = np.zeros((n, 3), dtype=np.longdouble)
        source = np.zeros(n, dtype=np.longdouble)

        def state(i, phi):
            factor = np.exp(-(phi - self.phi[i]) / (self.R * T[i]))
            r = rho[i] * factor
            pressure = (p[i] + self.shift) * factor - self.shift
            conserved = np.array(
                [
                    r,
                    r * u[i],
                    r * (self.cv * T[i] + self.offset + 0.5 * u[i] ** 2 + phi) + self.shift,
                ]
            )
            transport = np.array(
                [
                    r * u[i],
                    r * u[i] ** 2 + pressure,
                    r * u[i] * (self.cp * T[i] + self.offset + 0.5 * u[i] ** 2 + phi),
                ]
            )
            return conserved, transport, abs(u[i]) + np.sqrt(self.gamma * self.R * T[i])

        for i in range(n):
            j = (i + 1) % n
            left, fl, sl = state(i, self.face[i])
            right, fr, sr = state(j, self.face[i])
            flux[i] = self.area * (0.5 * (fl + fr) - 0.5 * max(sl, sr) * (right - left))
            pr = (p[i] + self.shift) * np.exp(
                -(self.face[i] - self.phi[i]) / (self.R * T[i])
            ) - self.shift
            pl = (p[i] + self.shift) * np.exp(
                -(self.face[(i - 1) % n] - self.phi[i]) / (self.R * T[i])
            ) - self.shift
            source[i] = self.area * (pr - pl)
        change = np.zeros((n, 3))
        for i in range(n):
            change[i] = flux[(i - 1) % n] - flux[i]
        drag = self.spec["drag"]["rate_per_s"] * P
        D = self.spec["drag"]["rate_per_s"] * P * P / m
        ambient = np.array(self.spec["heat"]["ambient_ua_w_k"]) * (
            np.array(self.spec["heat"]["ambient_temperature_k"]) - T
        )
        external = heat + ambient
        export = self.spec["drag"]["exported_fraction"] * D
        change[:, 1] += source - drag
        change[:, 2] += external - export
        internal_rate = change[:, 2] - u * change[:, 1] + (0.5 * u * u - self.phi) * change[:, 0]
        h = self.cp * T + self.offset
        chemical = h - T * entropy
        entropy_rate = (internal_rate - chemical * change[:, 0]) / T
        ambientT = np.array(self.spec["heat"]["ambient_temperature_k"])
        entropy_external = sum((heat - export) / T + ambient / ambientT)
        generation = sum(entropy_rate) - entropy_external
        return np.asarray(
            np.r_[
                change[:, 0],
                change[:, 1],
                change[:, 2],
                [sum(external), sum(export), entropy_external, generation, sum(source), sum(drag)],
            ],
            dtype=float,
        )

    def integrate(self, horizon, schedule, method, times):
        y = self.initial.astype(np.longdouble)
        heat = self.heat.copy()
        current = 0.0
        events = {t: None for t in times}
        events[horizon] = None
        events.update(dict(schedule))
        rows = []
        evaluations = 0

        def fun(t, z):
            nonlocal evaluations
            evaluations += 1
            if evaluations > 300000:
                raise ValueError("Independent Euler loop RHS exhausted")
            return self.rhs(t, base + z.astype(np.longdouble), heat)

        def jac(t, z):
            matrix = np.zeros((len(z), len(z)))
            for j in range(3 * self.n):
                h = 1e-6 * max(1.0, float(abs(base[j])))
                up = z.copy()
                down = z.copy()
                up[j] += h
                down[j] -= h
                matrix[:, j] = (fun(t, up) - fun(t, down)) / (2 * h)
            return matrix

        for end, command in sorted(events.items()):
            # Integrate increments around absolute m/P/E in extended arithmetic.
            # This avoids quantising tiny equilibrium updates into large E offsets.
            base = y.copy()
            sol = solve_ivp(
                fun,
                (current, end),
                np.zeros(len(y)),
                method=method,
                rtol=2e-11,
                atol=np.r_[
                    np.full(self.n, 1e-12),
                    np.full(self.n, 1e-8),
                    np.full(self.n, 1e-6),
                    [1e-7, 1e-7, 1e-8, 1e-8, 1e-8, 1e-8],
                ],
                max_step=0.005,
                **({"jac": jac} if method == "Radau" else {}),
            )
            if not sol.success or not np.all(np.isfinite(sol.y)):
                raise ValueError("Independent Euler loop failed")
            y = base + sol.y[:, -1].astype(np.longdouble)
            current = end
            if command is not None:
                heat = np.array([command[f"c{i}"] for i in range(self.n)])
            m, P, E, r, u, T, p, s = self.primitive(y)
            rows.append(
                dict(
                    time_s=end,
                    masses_kg=m.astype(float).tolist(),
                    velocities_m_s=u.astype(float).tolist(),
                    temperatures_k=T.astype(float).tolist(),
                    pressures_pa=p.astype(float).tolist(),
                    ledgers=y[-6:].astype(float).tolist(),
                )
            )
        return rows
