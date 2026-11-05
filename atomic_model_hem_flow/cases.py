"""Declared weak-flow and uniform phase-transition qualification inputs."""

import math
from copy import deepcopy

from atomic_model_hem import eos


def uniform(n=4, kind="mixture", velocity=0.0):
    if kind == "liquid":
        point = dict(kind="pt", pressure_pa=3e6, temperature_k=350.0, velocity_m_s=velocity)
    elif kind == "vapor":
        point = dict(kind="pt", pressure_pa=1e6, temperature_k=600.0, velocity_m_s=velocity)
    else:
        point = dict(kind="px", pressure_pa=1e6, quality=0.5, velocity_m_s=velocity)
    return dict(
        schema_version=1,
        geometry=dict(cells=n, length_m=10.0, area_m2=0.01),
        initial=[deepcopy(point) for _ in range(n)],
        heat_power_w=[0.0] * n,
        wave_speed_bound_m_s=2000.0,
        max_step_s=1e-4,
        cfl=0.3,
    )


def acoustic(n=4, amplitude_pa=1.0):
    spec = uniform(n)
    base = eos.px(1e6, 0.5)
    c2 = eos.derivatives(base)["equilibrium_sound_speed_squared_m2_s2"]
    c = math.sqrt(c2)
    k = 2 * math.pi / 10.0
    dx = 10.0 / n
    sinc = math.sin(k * dx / 2) / (k * dx / 2)
    for i in range(n):
        dp = amplitude_pa * sinc * math.cos(k * (i + 0.5) * dx)
        drho = dp / c2
        spec["initial"][i] = dict(
            kind="rho_u",
            density_kg_m3=base.density_kg_m3 + drho,
            internal_energy_j_kg=base.internal_energy_j_kg
            + base.pressure_pa / base.density_kg_m3**2 * drho,
            pressure_seed_pa=base.pressure_pa,
            temperature_seed_k=base.temperature_k,
            velocity_m_s=dp / (base.density_kg_m3 * c),
        )
    return spec


def matrix():
    for name, kind, velocity in [
        ("mixture_rest", "mixture", 0.0),
        ("liquid_transport", "liquid", 1.0),
        ("vapor_transport", "vapor", -1.0),
    ]:
        yield name, uniform(kind=kind, velocity=velocity)
    spec = acoustic(4, 100.0)
    yield "weak_acoustic", spec
    spec = uniform()
    for i, p in enumerate(spec["initial"]):
        p["quality"] = [0.45, 0.5, 0.55, 0.5][i]
    yield "isobaric_quality_contact", spec
    spec = uniform()
    for p in spec["initial"]:
        p["quality"] = 0.999
    spec["heat_power_w"] = [20000.0] * 4
    yield "uniform_evaporation", spec
