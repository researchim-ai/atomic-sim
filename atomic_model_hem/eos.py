"""Strict IF97 conserved-state adapter with phase-specific equilibrium derivatives.

Mixture acoustic derivatives are computed from equilibrium v/u, never from
mass-averaged single-phase cp or sound speeds. Boundary derivatives are refused.
"""

import hashlib
import math
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import scipy

from atomic_model_workspace import common
from physics.materials import iapws

P_MIN, P_MAX = 1e5, 1e7
T_MIN, T_MAX = 300.0, 900.0
RHO_RTOL, U_ATOL = 2e-9, 0.002


def source_identity():
    files = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(Path(__file__).parent.glob("*.py"))
    }
    return dict(
        hem_sources=files,
        if97_sha256=hashlib.sha256(Path(iapws.__file__).read_bytes()).hexdigest(),
        serialization_sha256=hashlib.sha256(Path(common.__file__).read_bytes()).hexdigest(),
        python=platform.python_version(),
        numpy=np.__version__,
        scipy=scipy.__version__,
    )


def number(value, label):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("Invalid " + label)
    return float(value)


@dataclass(frozen=True)
class State:
    pressure_pa: float
    temperature_k: float
    density_kg_m3: float
    internal_energy_j_kg: float
    enthalpy_j_kg: float
    entropy_j_kg_k: float
    quality: float | None
    phase: str
    region: int

    def as_dict(self):
        return asdict(self)


def _admit(state):
    if not all(
        math.isfinite(v)
        for v in [
            state.pressure_pa,
            state.temperature_k,
            state.density_kg_m3,
            state.internal_energy_j_kg,
            state.enthalpy_j_kg,
            state.entropy_j_kg_k,
        ]
    ):
        raise ValueError("Nonfinite equilibrium state")
    if (
        not P_MIN <= state.pressure_pa <= P_MAX
        or not T_MIN <= state.temperature_k <= T_MAX
        or state.density_kg_m3 <= 0
        or state.region not in (1, 2, 4)
    ):
        raise ValueError("State outside restricted IF97 regions/domain")
    return State(
        *map(
            float,
            (
                state.pressure_pa,
                state.temperature_k,
                state.density_kg_m3,
                state.internal_energy_j_kg,
                state.enthalpy_j_kg,
                state.entropy_j_kg_k,
            ),
        ),
        None if state.quality is None else float(state.quality),
        state.phase,
        int(state.region),
    )


def _single(s):
    if s.region not in (1, 2):
        raise ValueError("Unsupported single-phase IF97 region")
    return _admit(
        State(
            s.p_pa,
            s.T_k,
            s.rho_kg_m3,
            s.u_j_kg,
            s.h_j_kg,
            s.s_j_kg_k,
            None,
            "liquid" if s.region == 1 else "vapor",
            s.region,
        )
    )


def pt(pressure_pa, temperature_k):
    p, t = number(pressure_pa, "pressure"), number(temperature_k, "temperature")
    if not P_MIN <= p <= P_MAX or not T_MIN <= t <= T_MAX:
        raise ValueError("p/T outside restricted IF97 domain")
    ts = iapws.saturation_temperature_k(p * 1e-6)
    if abs(t - ts) < 1e-7:
        raise ValueError("Saturation p/T requires an explicit quality")
    return _single(iapws.properties_pt(p, t))


def px(pressure_pa, quality):
    p, x = number(pressure_pa, "pressure"), number(quality, "quality")
    if not P_MIN <= p <= P_MAX or not 0 <= x <= 1:
        raise ValueError("p/x outside restricted equilibrium domain")
    f, g = iapws.saturation_properties(p)
    v = (1 - x) * f.v_m3_kg + x * g.v_m3_kg
    u = (1 - x) * f.u_j_kg + x * g.u_j_kg
    s = (1 - x) * f.s_j_kg_k + x * g.s_j_kg_k
    return _admit(
        State(
            p,
            f.T_k,
            1 / v,
            u,
            u + p * v,
            s,
            x,
            "two_phase" if 0 < x < 1 else "saturated_liquid" if x == 0 else "saturated_vapor",
            4,
        )
    )


def rho_u(density_kg_m3, internal_energy_j_kg, *, pressure_seed_pa=1e6, temperature_seed_k=450.0):
    rho, u = number(density_kg_m3, "density"), number(internal_energy_j_kg, "internal energy")
    p, t = number(pressure_seed_pa, "pressure seed"), number(temperature_seed_k, "temperature seed")
    if rho <= 0 or not P_MIN <= p <= P_MAX or not T_MIN <= t <= T_MAX:
        raise ValueError("Invalid conserved state or inverse seed")
    native = iapws.properties_rho_u(rho, u, p, t, allow_projection=False)
    if native.energy_mismatch_j_kg != 0 or native.volume_mismatch != 0:
        raise ValueError("Projected EOS state refused")
    # IF97's Pa/MPa conversion can put an exact domain endpoint a few ulps
    # outside the interval. Accept an explicit boundary solution only if BOTH
    # conserved coordinates meet the same roundtrip criteria; no p/x clipping.
    if native.region == 4 and not P_MIN <= native.p_pa <= P_MAX:
        boundary = P_MIN if native.p_pa < P_MIN else P_MAX
        liquid, vapor = iapws.saturation_properties(boundary)
        quality = float((u - liquid.u_j_kg) / (vapor.u_j_kg - liquid.u_j_kg))
        if 0 <= quality <= 1:
            edge = px(boundary, quality)
            if (
                abs(edge.density_kg_m3 - rho) <= RHO_RTOL * rho
                and abs(edge.internal_energy_j_kg - u) <= U_ATOL
            ):
                return edge
    state = px(float(native.p_pa), float(native.quality)) if native.region == 4 else _single(native)
    if (
        abs(state.density_kg_m3 - rho) > RHO_RTOL * rho
        or abs(state.internal_energy_j_kg - u) > U_ATOL
    ):
        raise ValueError("Conserved-state inverse roundtrip residual exceeded")
    return state


def derivatives(state):
    """d(p,T)/d(rho,u), equilibrium c² and estimated phase-chart derivative error.

    Central differences of a forward phase chart, checked with a halved step;
    inverse solves are not differenced. No cross-phase derivative is invented.
    """
    _admit(state)
    p = state.pressure_pa
    if not P_MIN + 10 < p < P_MAX - 10:
        raise ValueError("Derivative pressure stencil outside domain")
    if state.region == 4:
        if state.quality is None or not 1e-5 < state.quality < 1 - 1e-5:
            raise ValueError("Two-sided derivative at phase boundary is undefined")
        second = state.quality
        h2 = 1e-4
        forward = px
    else:
        second = state.temperature_k
        h2 = 0.01
        if (
            abs(second - iapws.saturation_temperature_k(p * 1e-6)) < 0.05
            or not T_MIN + 0.02 < second < T_MAX - 0.02
        ):
            raise ValueError("Derivative temperature stencil touches a phase/domain boundary")
        forward = pt
    hp = max(1.0, p * 1e-5)

    def jac(scale):
        a, b = forward(p + hp * scale, second), forward(p - hp * scale, second)
        c, d = forward(p, second + h2 * scale), forward(p, second - h2 * scale)
        return np.array(
            [
                [
                    (1 / a.density_kg_m3 - 1 / b.density_kg_m3) / (2 * hp * scale),
                    (1 / c.density_kg_m3 - 1 / d.density_kg_m3) / (2 * h2 * scale),
                ],
                [
                    (a.internal_energy_j_kg - b.internal_energy_j_kg) / (2 * hp * scale),
                    (c.internal_energy_j_kg - d.internal_energy_j_kg) / (2 * h2 * scale),
                ],
            ]
        )

    j, jh = jac(1.0), jac(0.5)
    error = float(np.max(abs(j - jh) / np.maximum(abs(jh), 1e-20)))
    if not np.isfinite(jh).all() or error > 2e-3:
        raise ValueError("Unresolved phase-chart derivative stencil")
    target = np.array([[-1 / state.density_kg_m3**2, 0.0], [0.0, 1.0]])
    inv = np.linalg.solve(jh, target)
    result = inv
    if state.region == 4:
        dt_dp = (
            iapws.saturation_temperature_k((p + 10) * 1e-6)
            - iapws.saturation_temperature_k((p - 10) * 1e-6)
        ) / 20
        result = np.array([[1.0, 0.0], [dt_dp, 0.0]]) @ inv
    c2 = float(result[0, 0] + result[0, 1] * p / state.density_kg_m3**2)
    if not np.isfinite(result).all() or c2 <= 0:
        raise ValueError("Nonphysical equilibrium derivative")
    return dict(
        pressure_wrt_density=float(result[0, 0]),
        pressure_wrt_energy=float(result[0, 1]),
        temperature_wrt_density=float(result[1, 0]),
        temperature_wrt_energy=float(result[1, 1]),
        equilibrium_sound_speed_squared_m2_s2=c2,
        stencil_relative_difference=error,
        method="phase-chart central differences with step-halving, inverse-function theorem",
    )


def void_fraction(state):
    """Equilibrium vapor volume fraction; quality is a mass fraction."""
    _admit(state)
    if state.region == 1:
        return 0.0
    if state.region == 2:
        return 1.0
    _, vapor = iapws.saturation_properties(state.pressure_pa)
    return float(state.quality * state.density_kg_m3 / vapor.rho_kg_m3)
