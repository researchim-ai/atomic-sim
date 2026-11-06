"""Finite HEM vessel benchmarks with prescribed heat, volume and withdrawal.

No nozzle/choking, metastable delay, interfacial rate or empirical heat-transfer
law is asserted. Production stocks are compared to independent pressure/quality
coordinates and exact controlled phase paths.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from atomic_model_plant_extension import water
from atomic_model_workspace.common import digest
from physics.materials import iapws


def raw_saturation(p):
    f, g = iapws.saturation_properties(float(p))
    return f, g


def mix(p, x):
    f, g = raw_saturation(p)
    return np.array(
        [
            (1 - x) * f.v_m3_kg + x * g.v_m3_kg,
            (1 - x) * f.u_j_kg + x * g.u_j_kg,
            (1 - x) * f.s_j_kg_k + x * g.s_j_kg_k,
        ]
    )


def cases():
    p = 1e6
    m = 10.0
    f, g = raw_saturation(p)
    result = []
    for name, start, end in [("boiling", 0.0, 0.2), ("condensation", 0.2, 0.0)]:
        duration = 20.0
        xdot = (end - start) / duration
        result.append(
            dict(
                name=name,
                p0=p,
                x0=start,
                m0=m,
                duration_s=duration,
                V0=m * ((1 - start) * f.v_m3_kg + start * g.v_m3_kg),
                Vdot=m * (g.v_m3_kg - f.v_m3_kg) * xdot,
                heat_w=m * (g.h_j_kg - f.h_j_kg) * xdot,
                withdrawal_kg_s=0.0,
                reference="exact isobaric quality path from independently specified piston and heat",
                x_end=end,
            )
        )
    result.append(
        dict(
            name="flashing",
            p0=p,
            x0=0.0,
            m0=m,
            duration_s=20.0,
            V0=m * f.v_m3_kg,
            Vdot=m * f.v_m3_kg * 0.15,
            heat_w=0.0,
            withdrawal_kg_s=0.0,
            reference="adiabatic piston expansion, independent p/x IVP",
        )
    )
    result.append(
        dict(
            name="limited_blowdown",
            p0=p,
            x0=0.2,
            m0=m,
            duration_s=20.0,
            V0=m * mix(p, 0.2)[0],
            Vdot=0.0,
            heat_w=0.0,
            withdrawal_kg_s=0.1,
            reference="adiabatic well-mixed prescribed mixture withdrawal, no nozzle prediction",
        )
    )
    return result


def identity():
    return dict(
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        production_water_sha256=hashlib.sha256(Path(water.__file__).read_bytes()).hexdigest(),
        if97_sha256=hashlib.sha256(Path(iapws.__file__).read_bytes()).hexdigest(),
    )


def entropy_chain(point, vdot, udot):
    p = point.pressure_pa
    if point.region == 4:
        x = point.quality
        j, _ = water.mixture_jacobian(p, x)
        pdot, xdot = np.linalg.solve(j, np.array([vdot, udot]))
        h = 10.0
        sp = (mix(p + h, x)[2] - mix(p - h, x)[2]) / (2 * h)
        sx = mix(p, 1.0)[2] - mix(p, 0.0)[2]
        return sp * pdot + sx * xdot
    region = point.region
    t = point.temperature_k
    j = water.single_jacobian(p, t, region)
    pdot, tdot = np.linalg.solve(j, [vdot, udot])
    raw = iapws._region1_raw if region == 1 else iapws._region2_raw
    sp = (raw(t, (p + 10.0) * 1e-6)["s"] - raw(t, (p - 10.0) * 1e-6)["s"]) * 1000.0 / 20.0
    st = (raw(t + 1e-4, p * 1e-6)["s"] - raw(t - 1e-4, p * 1e-6)["s"]) * 1000.0 / 2e-4
    return sp * pdot + st * tdot


@dataclass(frozen=True)
class State:
    time_s: float
    stocks: tuple
    calls: tuple


class Vessel:
    def __init__(self, spec):
        if spec not in cases():
            raise ValueError("Only versioned four-case vessel programs admitted")
        self.spec = dict(spec)
        point = water.px(float(spec["p0"]), float(spec["x0"]))
        self.initial_entropy = spec["m0"] * point.entropy_j_kg_k
        self.initial_energy = spec["m0"] * point.internal_energy_j_kg
        self.initial = State(
            0.0, (spec["m0"], self.initial_energy, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), ()
        )
        self.seed = (spec["p0"], point.temperature_k)

    def point(self, t, y):
        V = self.spec["V0"] + self.spec["Vdot"] * t
        if V <= 0 or y[0] <= 0:
            raise ValueError("Nonpositive vessel volume or mass")
        return water.rho_u(float(y[0] / V), float(y[1] / y[0]), *self.seed)

    def rhs(self, t, y):
        p = self.point(t, y)
        q = self.spec["withdrawal_kg_s"]
        heat = self.spec["heat_w"]
        Vdot = self.spec["Vdot"]
        mdot = -q
        udot_total = heat - p.pressure_pa * Vdot - q * p.enthalpy_j_kg
        vdot = (Vdot - mdot / p.density_kg_m3) / y[0]
        udot = (udot_total - p.internal_energy_j_kg * mdot) / y[0]
        sdot_total = y[0] * entropy_chain(p, vdot, udot) + mdot * p.entropy_j_kg_k
        fit = heat / p.temperature_k - q * p.entropy_j_kg_k - sdot_total
        return np.array(
            [
                mdot,
                udot_total,
                p.pressure_pa * Vdot,
                q * p.enthalpy_j_kg,
                heat,
                heat / p.temperature_k,
                q * p.entropy_j_kg_k,
                fit,
            ]
        )

    def advance(self, state, duration, max_step):
        if (
            not np.isfinite([duration, max_step]).all()
            or duration <= 0
            or not 0.01 <= max_step <= 1.0
        ):
            raise ValueError("Invalid vessel interval")
        end = state.time_s + duration
        if end > self.spec["duration_s"] or end <= state.time_s:
            raise ValueError("Vessel horizon exceeded")
        evaluations = 0

        def bounded_rhs(t, y):
            nonlocal evaluations
            evaluations += 1
            if evaluations > 50000:
                raise ValueError("Vessel RHS work budget exhausted")
            return self.rhs(t, y)

        sol = solve_ivp(
            bounded_rhs,
            (state.time_s, end),
            state.stocks,
            method="Radau",
            rtol=2e-10,
            atol=[1e-11, 1e-5, 1e-5, 1e-5, 1e-5, 1e-8, 1e-8, 1e-8],
            max_step=max_step,
        )
        if not sol.success:
            raise ValueError(sol.message)
        final = sol.y[:, -1]
        self.point(end, final)
        return State(
            end, tuple(map(float, final)), state.calls + ((float(duration), float(max_step)),)
        )

    def report(self, state):
        y = np.array(state.stocks)
        p = self.point(state.time_s, y)
        mass = y[0] - self.spec["m0"] + self.spec["withdrawal_kg_s"] * state.time_s
        energy = y[1] - self.initial_energy + y[2] + y[3] - y[4]
        entropy = y[0] * p.entropy_j_kg_k - self.initial_entropy - y[5] + y[6] + y[7]
        return dict(
            time_s=state.time_s,
            pressure_pa=p.pressure_pa,
            quality=p.quality,
            phase=p.phase,
            temperature_k=p.temperature_k,
            mass_kg=y[0],
            mass_defect_kg=mass,
            first_law_defect_j=energy,
            entropy_numerical_defect_j_k=entropy,
            material_gibbs_fit_integral_j_k=y[7],
            boundary_work_j=y[2],
            exported_enthalpy_j=y[3],
        )

    def checkpoint(self, state):
        payload = dict(
            schema_version=1,
            source=identity(),
            spec=self.spec,
            time_s=state.time_s,
            stocks=state.stocks,
            calls=state.calls,
            report=self.report(state),
        )
        return dict(payload=payload, sha256=digest(payload))

    def restore(self, saved):
        if set(saved) != {"payload", "sha256"} or digest(saved["payload"]) != saved["sha256"]:
            raise ValueError("Invalid vessel checkpoint")
        p = saved["payload"]
        if p["source"] != identity() or p["spec"] != self.spec:
            raise ValueError("Vessel source/spec changed")
        state = self.initial
        if not isinstance(p["calls"], (list, tuple)) or len(p["calls"]) > 1000:
            raise ValueError("Vessel replay budget")
        for dt, cap in p["calls"]:
            state = self.advance(state, dt, cap)
        if self.checkpoint(state)["sha256"] != saved["sha256"]:
            raise ValueError("Vessel history replay differs")
        return state


def reference(spec):
    if spec["name"] in ("boiling", "condensation"):
        return dict(
            pressure_pa=spec["p0"],
            quality=spec["x_end"],
            method="exact manufactured isobaric phase path",
        )

    def rhs(t, y):
        p, x = y
        m = spec["m0"] - spec["withdrawal_kg_s"] * t
        v, u, _ = mix(p, x)
        h = u + p * v
        mdot = -spec["withdrawal_kg_s"]
        vdot = (spec["Vdot"] - mdot * v) / m
        udot = (spec["heat_w"] - p * spec["Vdot"] - spec["withdrawal_kg_s"] * h - u * mdot) / m
        left, right = mix(p - 10.0, x), mix(p + 10.0, x)
        dx = mix(p, 1.0) - mix(p, 0.0)
        j = np.column_stack(((right[:2] - left[:2]) / 20.0, dx[:2]))
        return np.linalg.solve(j, [vdot, udot])

    sol = solve_ivp(
        rhs,
        (0, spec["duration_s"]),
        [spec["p0"], spec["x0"]],
        method="DOP853",
        rtol=2e-11,
        atol=[1e-5, 1e-12],
        max_step=0.1,
    )
    if not sol.success:
        raise ValueError("Independent p/x reference failed")
    return dict(
        pressure_pa=float(sol.y[0, -1]),
        quality=float(sol.y[1, -1]),
        method="independent pressure/quality DOP853 with finite-difference property Jacobian",
    )


def qualify():
    rows = []
    for spec in cases():
        ref = reference(spec)
        runs = []
        for cap in [0.4, 0.2, 0.1]:
            model = Vessel(spec)
            mid = model.advance(model.initial, 10.0, cap)
            saved = model.checkpoint(mid)
            final = model.advance(mid, 10.0, cap)
            restored = model.restore(saved)
            resumed = model.advance(restored, 10.0, cap)
            report = model.report(final)
            x = report["quality"] if report["quality"] is not None else 0.0
            if (
                abs(report["pressure_pa"] - ref["pressure_pa"]) > 5.0
                or abs(x - ref["quality"]) > 2e-6
                or abs(report["mass_defect_kg"]) > 1e-9
                or abs(report["first_law_defect_j"]) > 0.02
                or abs(report["entropy_numerical_defect_j_k"]) > 2e-4
                or model.checkpoint(final) != model.checkpoint(resumed)
            ):
                raise ValueError(
                    "Vessel qualification failed:" + str((spec["name"], cap, report, ref))
                )
            runs.append(dict(max_step_s=cap, report=report, restart_equal=True))
        rows.append(dict(case=spec["name"], spec=spec, reference=ref, runs=runs))
    return dict(
        schema_version=1,
        source=identity(),
        cases=rows,
        scope="Numerical four-phenomenon HEM vessel verification; not experimental validation, nozzle/choking or metastable flashing",
        validated=False,
        plant_referenced=False,
    )
