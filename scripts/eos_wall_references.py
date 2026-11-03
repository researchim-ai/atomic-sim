"""Independent absolute-energy method of lines for the two-loop wall assembly."""

from copy import deepcopy

import numpy as np
from scipy.integrate import solve_ivp

from scripts.conservative_loop_references import IndependentLoop
from scripts.conservative_loop_references import request as loop_request


def request(*, uniform=False, gravity=False, stiff=False, ambient=False):
    loops = [
        loop_request(
            velocity=0 if uniform else 5,
            shift=1e8 if stiff else 1e5,
            height=10 if gravity else 0,
            gravity=9.81 if gravity else 0,
        ),
        loop_request(velocity=0 if uniform else -3, shift=1e8 if stiff else 1e5),
    ]
    for r, T in zip(loops, [350.0, 280.0]):
        r["initial"]["temperatures_k"] = [T] * 8
        g = r["geometry"]
        z = np.array(g["elevations_m"])
        shift = r["fluid"]["pressure_shift_pa"]
        r["initial"]["pressures_pa"] = (
            200000.0 + (200000.0 + shift) * np.expm1(-g["gravity_m_s2"] * z / (400 * T))
        ).tolist()
    walls = [
        dict(
            cells=[i, i if uniform else 7 - i],
            capacity_j_k=200.0,
            initial_temperature_k=300.0,
            conductance_w_k=[1000.0, 700.0],
            ambient_ua_w_k=10.0 if ambient else 0.0,
            ambient_temperature_k=290.0,
        )
        for i in (range(8) if uniform else [1, 2, 3])
    ]
    return dict(format="atomic-coupled-eos-wall", schema_version=1, loops=loops, walls=walls)


class IndependentWall:
    def __init__(self, spec):
        self.spec = deepcopy(spec)
        self.loops = [IndependentLoop(r) for r in spec["loops"]]
        self.offset = [0]
        for e in self.loops:
            self.offset.append(self.offset[-1] + len(e.initial))
        self.wall_start = self.offset[-1]
        self.capacity = np.array([w["capacity_j_k"] for w in spec["walls"]])
        self.initial = np.concatenate(
            [
                *[e.initial for e in self.loops],
                self.capacity * np.array([w["initial_temperature_k"] for w in spec["walls"]]),
                np.zeros(5),
            ]
        ).astype(np.longdouble)

    def rhs(self, t, y):
        states = [y[self.offset[i] : self.offset[i + 1]] for i in range(2)]
        points = [e.primitive(z) for e, z in zip(self.loops, states)]
        Tw = y[self.wall_start : -5] / self.capacity
        powers = [e.heat.astype(np.longdouble).copy() for e in self.loops]
        wall = np.zeros(len(self.capacity), dtype=np.longdouble)
        Qf = Sf = Qa = Sa = Sg = np.longdouble(0)
        for j, w in enumerate(self.spec["walls"]):
            ambient = w["ambient_ua_w_k"] * (w["ambient_temperature_k"] - Tw[j])
            wall[j] = ambient
            Qa += ambient
            Sa += ambient / w["ambient_temperature_k"]
            Sg += ambient * (1 / Tw[j] - 1 / w["ambient_temperature_k"])
            for i, cell in enumerate(w["cells"]):
                Tf = points[i][5][cell]
                heat = w["conductance_w_k"][i] * (Tw[j] - Tf)
                powers[i][cell] += heat
                wall[j] -= heat
                Qf += heat
                Sf += heat / Tf
                Sg += heat * (1 / Tf - 1 / Tw[j])
        return np.asarray(
            np.concatenate(
                [
                    *[e.rhs(t, z, p) for e, z, p in zip(self.loops, states, powers)],
                    wall,
                    [Qf, Sf, Qa, Sa, Sg],
                ]
            ),
            dtype=float,
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
                if evaluations > 200000:
                    raise ValueError("Independent wall RHS budget exhausted")
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
                raise ValueError("Independent wall integration failed")
            y = base + sol.y[:, -1].astype(np.longdouble)
            current = end
            row = dict(
                wall_temperatures_k=(y[self.wall_start : -5] / self.capacity)
                .astype(float)
                .tolist(),
                loops=[],
            )
            for i, e in enumerate(self.loops):
                m, P, E, rho, u, T, p, s = e.primitive(y[self.offset[i] : self.offset[i + 1]])
                row["loops"].append(
                    dict(
                        temperatures_k=T.astype(float).tolist(),
                        pressures_pa=p.astype(float).tolist(),
                        masses_kg=m.astype(float).tolist(),
                        velocities_m_s=u.astype(float).tolist(),
                    )
                )
            rows.append(row)
        return rows


def cases():
    for name, options in [
        ("uniform", dict(uniform=True)),
        ("moving", {}),
        ("gravity", dict(gravity=True)),
        ("stiff", dict(stiff=True)),
        ("ambient", dict(ambient=True)),
    ]:
        yield name, request(**options)
    heated = request(gravity=True, ambient=True)
    for i, r in enumerate(heated["loops"]):
        r["drag"] = dict(rate_per_s=0.4, exported_fraction=0.25)
        r["heat"]["power_w"][6] = 1000.0 if i == 0 else -200.0
        r["heat"]["ambient_ua_w_k"][5] = 10.0
    yield "heated_drag", heated
    unequal = request()
    unequal["loops"][1] = loop_request(4, velocity=-3)
    unequal["loops"][1]["initial"]["temperatures_k"] = [280.0] * 4
    for wall in unequal["walls"]:
        wall["cells"][1] %= 4
    yield "unequal_meshes", unequal
