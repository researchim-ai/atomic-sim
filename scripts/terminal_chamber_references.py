"""Independent absolute m/P/E terminal-volume and Euler-channel assembly."""

from copy import copy, deepcopy

import numpy as np
from scipy.integrate import solve_ivp

from scripts.open_channel_references import IndependentOpen
from scripts.open_channel_references import request as open_request


def request(*, bias=0.0, velocity=0.0, height=0.0, shift=1e5, heat=0.0, ambient=False):
    r = open_request(velocity=velocity, bias=bias, height=height, shift=shift)
    chambers = []
    for side in ["left", "right"]:
        b = r["boundaries"][side]
        chambers.append(
            dict(
                volume_m3=0.5,
                initial_pressure_pa=b["pressure_pa"],
                initial_temperature_k=b["temperature_k"],
                initial_velocity_m_s=velocity,
                elevation_m=b["elevation_m"],
                heat_power_w=heat if side == "left" else 0.0,
                ambient_ua_w_k=10.0 if ambient else 0.0,
                ambient_temperature_k=290.0,
            )
        )
    r.pop("boundaries")
    r["format"] = "atomic-conservative-eos-loop"
    return dict(
        format="atomic-terminal-eos-chambers", schema_version=1, channel=r, chambers=chambers
    )


def cases():
    yield "rest", request()
    yield "pressure_forward", request(bias=1000)
    yield "pressure_reverse", request(bias=-1000)
    yield "reflecting_motion", request(velocity=2)
    yield "hydrostatic", request(height=10)
    yield "stiff_heated", request(shift=1e8, heat=500)
    yield "heated_ambient", request(bias=1000, heat=500, ambient=True)
    r = request(bias=1000)
    r["chambers"][0]["volume_m3"] = 0.2
    r["chambers"][1]["volume_m3"] = 0.8
    yield "unequal_volumes", r


class IndependentTerminal:
    def __init__(self, spec):
        self.spec = deepcopy(spec)
        r = deepcopy(spec["channel"])
        r["format"] = "atomic-open-eos-channel"
        r["boundaries"] = {
            side: dict(
                pressure_pa=c["initial_pressure_pa"],
                temperature_k=c["initial_temperature_k"],
                velocity_m_s=c["initial_velocity_m_s"],
                elevation_m=c["elevation_m"],
            )
            for side, c in zip(["left", "right"], spec["chambers"])
        }
        self.channel = IndependentOpen(r)
        e = self.channel.base
        self.start = len(self.channel.initial)
        self.V = np.array([c["volume_m3"] for c in spec["chambers"]])
        self.phi = e.spec["geometry"]["gravity_m_s2"] * np.array(
            [c["elevation_m"] for c in spec["chambers"]]
        )
        T = np.array([c["initial_temperature_k"] for c in spec["chambers"]])
        p = np.array([c["initial_pressure_pa"] for c in spec["chambers"]])
        u = np.array([c["initial_velocity_m_s"] for c in spec["chambers"]])
        m = (p + e.shift) / (e.R * T) * self.V
        E = m * (e.cv * T + e.offset + 0.5 * u * u + self.phi) + e.shift * self.V
        self.initial = np.r_[self.channel.initial, m, m * u, E, np.zeros(4)].astype(np.longdouble)

    def primitive(self, y):
        e = self.channel.base
        m, P, E = y[self.start : self.start + 6].reshape(3, 2)
        rho = m / self.V
        u = P / m
        T = (E / m - 0.5 * u * u - self.phi - e.offset - e.shift / rho) / e.cv
        p = e.R * rho * T - e.shift
        s = e.cv * np.log(T / e.Tref) - e.R * np.log(rho / e.rref)
        return m, P, E, rho, u, T, p, s

    def rhs(self, t, y):
        e = self.channel.base
        m, P, E, rho, u, T, p, s = self.primitive(y)
        view = copy(self.channel)
        view.spec = deepcopy(self.channel.spec)
        for i, side in enumerate(["left", "right"]):
            view.spec["boundaries"][side].update(
                pressure_pa=p[i], temperature_k=T[i], velocity_m_s=u[i]
            )
        channel_rate = view.rhs(t, y[: self.start], view.heat)
        transfer = np.array([channel_rate[-8:-4], channel_rate[-4:]])
        c = np.sqrt(e.gamma * e.R * T)
        walls = []
        for i, sign in enumerate([-1, 1]):
            walls.append(
                e.area * (rho[i] * u[i] * u[i] + p[i] + sign * (abs(u[i]) + c[i]) * rho[i] * u[i])
            )
        specs = self.spec["chambers"]
        ambient = np.array([r["ambient_ua_w_k"] for r in specs]) * (
            np.array([r["ambient_temperature_k"] for r in specs]) - T
        )
        heaters = np.array([r["heat_power_w"] for r in specs])
        Q = heaters + ambient
        dm = -transfer[:, 0]
        dP = np.array([walls[0], -walls[1]]) - transfer[:, 1]
        dE = -transfer[:, 2] + Q
        Udot = dE - u * dP + (0.5 * u * u - self.phi) * dm
        Sdot = (Udot - (e.cp * T + e.offset - T * s) * dm) / T
        SE = sum(heaters / T + ambient / np.array([r["ambient_temperature_k"] for r in specs]))
        SG = sum(Sdot) - SE + sum(transfer[:, 3])
        return np.asarray(
            np.r_[channel_rate, dm, dP, dE, [sum(Q), SE, SG, walls[0] - walls[1]]], dtype=float
        )

    def integrate(self, times, method):
        y = self.initial.copy()
        current = 0.0
        rows = []
        evaluations = 0
        for end in times:
            base = y.copy()

            def fun(t, z):
                nonlocal evaluations
                evaluations += 1
                if evaluations > 300000:
                    raise ValueError("Terminal reference RHS budget")
                return self.rhs(t, base + z.astype(np.longdouble))

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
                raise ValueError("Terminal reference solve failed")
            y = base + sol.y[:, -1].astype(np.longdouble)
            current = end
            m, P, E, r, u, T, p, s = self.primitive(y)
            cm, cP, cE, cr, cu, cT, cp, cs = self.channel.primitive(y[: self.start])
            rows.append(
                dict(
                    channel=dict(
                        masses_kg=cm.astype(float).tolist(),
                        velocities_m_s=cu.astype(float).tolist(),
                        temperatures_k=cT.astype(float).tolist(),
                        pressures_pa=cp.astype(float).tolist(),
                    ),
                    chambers=dict(
                        m=m.astype(float).tolist(),
                        u=u.astype(float).tolist(),
                        T=T.astype(float).tolist(),
                        p=p.astype(float).tolist(),
                    ),
                    ledgers=y[-4:].astype(float).tolist(),
                )
            )
        return rows
