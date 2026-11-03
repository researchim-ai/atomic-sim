"""Independent absolute-state mixed thermal assembly and manufactured cases."""

from copy import copy, deepcopy

import numpy as np
from scipy.integrate import solve_ivp

from scripts.conservative_loop_references import request as loop_request
from scripts.terminal_chamber_references import IndependentTerminal
from scripts.terminal_chamber_references import request as terminal_request

NAMES = ("primary", "secondary")


def request(*, uniform=False, height=0.0, stiff=False, ambient=False, velocity=0.0, unequal=False):
    circuits = {}
    for i, (name, T) in enumerate(zip(NAMES, [350.0, 280.0])):
        n = 8 if unequal and i == 1 else 4
        shift = 1e8 if stiff else 1e5
        h = height if i == 0 else 0.0
        g = 9.81 if h else 0.0
        r = terminal_request(
            bias=0 if uniform else 500,
            velocity=velocity if i == 0 else -velocity,
            shift=shift,
            ambient=ambient,
        )
        r["channel"] = loop_request(n, velocity=velocity if i == 0 else -velocity, shift=shift)
        z = (np.arange(n) + 0.5) * h / n
        r["channel"]["geometry"].update(elevations_m=z.tolist(), gravity_m_s2=g)
        r["channel"]["initial"]["temperatures_k"] = [T] * n
        r["channel"]["initial"]["pressures_pa"] = (
            200000.0 + (200000.0 + shift) * np.expm1(-g * z / (400 * T))
        ).tolist()
        for c, zend, delta in zip(
            r["chambers"], [0.0, h], [0 if uniform else 250, 0 if uniform else -250]
        ):
            c.update(
                volume_m3=0.25,
                initial_temperature_k=T,
                elevation_m=zend,
                initial_pressure_pa=200000.0
                + (200000.0 + shift) * np.expm1(-g * zend / (400 * T))
                + delta,
            )
        circuits[name] = r
    targets = (
        [("channel", i) for i in range(4)] + [("chamber", side) for side in ["left", "right"]]
        if uniform
        else [("channel", 1), ("chamber", "left")]
    )
    walls = []
    for w, (kind, target) in enumerate(targets):
        ports = [
            dict(
                circuit=name,
                component=kind,
                **({"cell": target} if kind == "channel" else {"side": target}),
            )
            for name in NAMES
        ]
        walls.append(
            dict(
                id=f"w{w}",
                ports=ports,
                capacity_j_k=200.0,
                initial_temperature_k=300.0,
                conductance_w_k=[100.0, 70.0],
                heat_power_w=0.0,
                ambient_ua_w_k=5.0 if ambient else 0.0,
                ambient_temperature_k=290.0,
            )
        )
    return dict(
        format="atomic-controlled-eos-assembly", schema_version=1, circuits=circuits, walls=walls
    )


def initial_controls(spec):
    result = {}
    for name, r in spec["circuits"].items():
        for i, v in enumerate(r["channel"]["heat"]["power_w"]):
            result[f"{name}.channel.c{i}"] = float(v)
        for side, c in zip(["left", "right"], r["chambers"]):
            result[f"{name}.chamber.{side}"] = float(c["heat_power_w"])
    for w in spec["walls"]:
        result[f"wall.{w['id']}"] = float(w["heat_power_w"])
    return result


def cases():
    yield "uniform", request(uniform=True), []
    yield "mixed_ports", request(), []
    yield "moving", request(velocity=1.0), []
    yield "gravity", request(height=10.0), []
    yield "stiff", request(stiff=True), []
    yield "ambient", request(ambient=True), []
    yield "unequal_meshes", request(unequal=True), []
    r = request(ambient=True)
    mask = initial_controls(r)
    mask["primary.chamber.right"] = 1000.0
    mask["secondary.channel.c2"] = -200.0
    mask["wall.w0"] = 500.0
    yield "online_command", r, [(0.05, mask)]


class IndependentAssembly:
    def __init__(self, spec):
        self.spec = deepcopy(spec)
        self.circuits = [IndependentTerminal(spec["circuits"][name]) for name in NAMES]
        self.offsets = [0]
        for e in self.circuits:
            self.offsets.append(self.offsets[-1] + len(e.initial))
        self.wall_start = self.offsets[-1]
        self.capacity = np.array([w["capacity_j_k"] for w in spec["walls"]])
        self.initial = np.concatenate(
            [
                *[e.initial for e in self.circuits],
                self.capacity * np.array([w["initial_temperature_k"] for w in spec["walls"]]),
                np.zeros(5),
            ]
        ).astype(np.longdouble)
        self.heaters = initial_controls(spec)

    def rhs(self, t, y, controls):
        local = [y[self.offsets[i] : self.offsets[i + 1]] for i in range(2)]
        tanks = [e.primitive(z) for e, z in zip(self.circuits, local)]
        channels = [e.channel.primitive(z[: e.start]) for e, z in zip(self.circuits, local)]
        Tw = y[self.wall_start : -5] / self.capacity
        channel_heat = [np.zeros(e.channel.n, dtype=np.longdouble) for e in self.circuits]
        tank_heat = [np.zeros(2, dtype=np.longdouble) for _ in self.circuits]
        wall_rate = np.zeros(len(Tw), dtype=np.longdouble)
        Qf = Sf = Qa = Sa = Sg = np.longdouble(0)
        for w, spec in enumerate(self.spec["walls"]):
            ambient = spec["ambient_ua_w_k"] * (spec["ambient_temperature_k"] - Tw[w])
            heater = controls[f"wall.{spec['id']}"]
            wall_rate[w] = heater + ambient
            Qa += heater + ambient
            Sa += heater / Tw[w] + ambient / spec["ambient_temperature_k"]
            Sg += ambient * (1 / Tw[w] - 1 / spec["ambient_temperature_k"])
            for k, port in enumerate(spec["ports"]):
                i = NAMES.index(port["circuit"])
                j = (
                    port["cell"]
                    if port["component"] == "channel"
                    else ["left", "right"].index(port["side"])
                )
                Tf = (channels[i] if port["component"] == "channel" else tanks[i])[5][j]
                heat = spec["conductance_w_k"][k] * (Tw[w] - Tf)
                (channel_heat if port["component"] == "channel" else tank_heat)[i][j] += heat
                wall_rate[w] -= heat
                Qf += heat
                Sf += heat / Tf
                Sg += heat * (1 / Tf - 1 / Tw[w])
        rates = []
        for i, (name, base, z) in enumerate(zip(NAMES, self.circuits, local)):
            view = copy(base)
            view.spec = deepcopy(base.spec)
            view.channel = copy(base.channel)
            view.channel.heat = (
                np.array([controls[f"{name}.channel.c{j}"] for j in range(base.channel.n)])
                + channel_heat[i]
            )
            for j, side in enumerate(["left", "right"]):
                view.spec["chambers"][j]["heat_power_w"] = (
                    controls[f"{name}.chamber.{side}"] + tank_heat[i][j]
                )
            rates.append(view.rhs(t, z))
        return np.asarray(
            np.concatenate([*rates, wall_rate, np.array([Qf, Sf, Qa, Sa, Sg])]), dtype=float
        )

    def integrate(self, times, schedule, method):
        y = self.initial.copy()
        controls = self.heaters.copy()
        current = 0.0
        rows = []
        evaluations = 0
        for end in times:
            for when, mask in schedule:
                if when == current:
                    controls = mask.copy()
            base = y.copy()

            def fun(t, z):
                nonlocal evaluations
                evaluations += 1
                if evaluations > 400000:
                    raise ValueError("Mixed assembly reference RHS budget")
                return self.rhs(t, base + z.astype(np.longdouble), controls)

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
                raise ValueError("Mixed assembly reference failed")
            y = base + sol.y[:, -1].astype(np.longdouble)
            current = end
            row = dict(
                wall_temperatures_k=(y[self.wall_start : -5] / self.capacity)
                .astype(float)
                .tolist(),
                circuits=[],
                wall_ledgers=y[-5:].astype(float).tolist(),
            )
            for i, e in enumerate(self.circuits):
                z = y[self.offsets[i] : self.offsets[i + 1]]
                m, P, E, r, u, T, p, s = e.primitive(z)
                cm, cP, cE, cr, cu, cT, cp, cs = e.channel.primitive(z[: e.start])
                row["circuits"].append(
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
                    )
                )
            rows.append(row)
        return rows
