"""Independent total-enthalpy IVP and finite-difference thermodynamics.

Shares only the frozen IF97 EOS and direct phase transport. Does not call the
production p/h inverse, derivatives, friction or steady-channel RHS/stepper.
"""

from math import exp, log

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.special import logsumexp

from atomic_model_hem_qualified import eos
from physics.materials import iapws


def ph(p, h):
    f, g = eos.saturation(float(p))
    if f.enthalpy_j_kg <= h <= g.enthalpy_j_kg:
        return eos.px(float(p), float((h - f.enthalpy_j_kg) / (g.enthalpy_j_kg - f.enthalpy_j_kg)))
    region = 1 if h < f.enthalpy_j_kg else 2
    lo, hi = (300.0, f.temperature_k) if region == 1 else (g.temperature_k, 900.0)
    kernel = iapws._region1_raw if region == 1 else iapws._region2_raw
    t = brentq(lambda t: kernel(t, p * 1e-6)["h"] * 1000 - h, lo, hi, xtol=1e-10)
    return eos._state(float(p), float(t), region)


def solve(request, targets, method="DOP853"):
    G = request["mass_flux_kg_m2_s"]
    D = request["diameter_m"]
    grav = request["gravity_m_s2"]
    rough = request["roughness_m"] / D
    length = request["length_m"]
    profile = request["heat_flux_w_m2"]
    p0 = request["inlet_pressure_pa"]
    h = request["inlet_enthalpy_j_kg"]
    initial = ph(p0, h)
    H0 = h + (G / initial.density_kg_m3) ** 2 / 2
    y = np.array([p0, H0])
    z = 0.0
    evaluations = 0
    observations = []

    def state_at(zz, yy):
        pp, HH = map(float, yy)
        hh = HH - grav * zz
        for _ in range(12):
            state = ph(pp, hh)
            updated = HH - grav * zz - 0.5 * (G / state.density_kg_m3) ** 2
            if abs(updated - hh) < 1e-7:
                return state, updated
            hh = updated
        raise ValueError("Independent total/static enthalpy recovery failed")

    for target in targets:
        while z < target:
            cell = min(int(np.floor(z / length * len(profile) + 1e-12)), len(profile) - 1)
            stop = min(target, length * (cell + 1) / len(profile))
            q = profile[cell]

            def rhs(zz, yy):
                nonlocal evaluations
                evaluations += 1
                if evaluations > 60000:
                    raise ValueError("Reference IVP budget")
                state, hh = state_at(zz, yy)
                pp = float(yy[0])
                v = 1 / state.density_kg_m3
                dp = pp * 1e-7
                dh = 0.01
                vp = (1 / ph(pp + dp, hh).density_kg_m3 - 1 / ph(pp - dp, hh).density_kg_m3) / (
                    2 * dp
                )
                vh = (1 / ph(pp, hh + dh).density_kg_m3 - 1 / ph(pp, hh - dh).density_kg_m3) / (
                    2 * dh
                )
                if state.region == 4:
                    f, g = iapws.saturation_properties(pp)
                    x = state.quality
                    mu = (
                        1 / ((1 - x) / f.viscosity_pa_s + x / g.viscosity_pa_s)
                        if request["viscosity_model"] == "mcadams"
                        else (1 - x) * f.viscosity_pa_s + x * g.viscosity_pa_s
                    )
                else:
                    mu = iapws.viscosity_pa_s(state.density_kg_m3, state.temperature_k)
                re = G * D / mu
                aa = abs(2.457 * log(1 / ((7 / re) ** 0.9 + 0.27 * rough)))
                la = float("-inf") if aa == 0 else 16 * log(aa)
                lb = 16 * log(37530 / re)
                f = 8 * exp(float(logsumexp([12 * log(8 / re), -1.5 * logsumexp([la, lb])])) / 12)
                force = f * G * G * v / (2 * D)
                Hp = 4 * q / (G * D)
                pressure_rate = (
                    (-force - grav / v) * (1 + G * G * v * vh) - G * G * vh * (Hp - grav)
                ) / (1 + G * G * v * vh + G * G * vp)
                return [pressure_rate, Hp]

            solution = solve_ivp(
                rhs, (z, stop), y, method=method, rtol=2e-11, atol=1e-8, max_step=0.05
            )
            if not solution.success:
                raise ValueError("Independent steady IVP failed")
            y, z = solution.y[:, -1], stop
        state, h = state_at(z, y)
        observations.append(
            dict(
                position_m=z,
                pressure_pa=float(y[0]),
                enthalpy_j_kg=h,
                total_enthalpy_with_potential_j_kg=float(y[1]),
                state=state.as_dict(),
            )
        )
    return dict(method=method, evaluations=evaluations, observations=observations)
