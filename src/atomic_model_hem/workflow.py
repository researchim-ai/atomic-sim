"""Executable rigid-vessel demonstration and independent inverse-coordinate checks."""

import shutil
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, root

from atomic_model_hem import eos
from atomic_model_hem.vessel import RigidEquilibriumVessel
from atomic_model_workspace.common import atomic_json, keys, local_file, read
from physics.materials import iapws

# Published IF97 R7-97(2012), Table 5, PDF page 9. Selected in-domain points.
# Attribution: International Association for the Properties of Water and Steam.
OFFICIAL = [
    dict(p=3e6, t=300.0, v=0.00100215168, u=112324.818, h=115331.273, s=392.294792),
    dict(p=3e6, t=500.0, v=0.001202418, u=971934.985, h=975542.239, s=2580.41912),
]
CRITERIA = dict(
    pressure_relative=2e-7,
    temperature_k=2e-5,
    energy_j_kg=0.002,
    density_relative=2e-9,
    official_relative=2e-8,
    event_time_s=1e-6,
)


def specification(segment=20000.0):
    return dict(
        schema_version=1,
        volume_m3=1.0,
        initial=dict(kind="px", pressure_pa=1e6, quality=0.5),
        initial_power_w=20000.0,
        maximum_energy_segment_j_kg=segment,
    )


def inverse_reference(rho, u):
    """Separate phase-chart root. Never calls production conserved inverse."""
    v = 1 / rho

    def mixture(p):
        f, g = iapws.saturation_properties(float(p))
        x = (v - f.v_m3_kg) / (g.v_m3_kg - f.v_m3_kg)
        return float((1 - x) * f.u_j_kg + x * g.u_j_kg - u), float(x), float(f.T_k)

    a, b = mixture(eos.P_MIN)[0], mixture(eos.P_MAX)[0]
    if a * b <= 0:
        p = brentq(lambda p: mixture(p)[0], eos.P_MIN, eos.P_MAX, xtol=1e-5, rtol=1e-13)
        _, x, t = mixture(p)
        if 0 <= x <= 1:
            return dict(pressure_pa=p, temperature_k=t, quality=x)

    def residual(z):
        p, t = float(z[0]) * 1e6, float(z[1])
        if not eos.P_MIN <= p <= eos.P_MAX or not eos.T_MIN <= t <= eos.T_MAX:
            return np.array([1e4, 1e4])
        try:
            s = iapws.properties_pt(p, t)
        except ValueError:
            return np.array([1e4, 1e4])
        return np.array([(s.v_m3_kg - v) / v, (s.u_j_kg - u) / max(abs(u), 1.0)])

    for p in [1.0, 5.0, 0.2, 9.0]:
        for t in [350.0, 650.0, 500.0, 850.0]:
            solved = root(residual, [p, t], tol=1e-10)
            if max(abs(residual(solved.x))) < 1e-10:
                return dict(
                    pressure_pa=float(solved.x[0]) * 1e6,
                    temperature_k=float(solved.x[1]),
                    quality=None,
                )
    raise ValueError("Independent phase-chart inverse failed")


def _call(model, call):
    keys(call, ("duration_s", "command"))
    if call["command"] is None:
        return model.advance(call["duration_s"])
    c = call["command"]
    keys(c, ("power_w", "expected_revision", "controller_state"))
    return model.step(
        c["power_w"],
        call["duration_s"],
        expected_revision=c["expected_revision"],
        controller_state=c["controller_state"],
    )


def execute(spec):
    keys(spec, ("schema_version", "model", "program"))
    if (
        type(spec["schema_version"]) is not int
        or spec["schema_version"] != 1
        or not isinstance(spec["program"], list)
        or not 1 <= len(spec["program"]) <= 16
    ):
        raise ValueError("Invalid vessel program")
    model = RigidEquilibriumVessel(spec["model"])
    observations = [_call(model, call) for call in spec["program"]]
    return dict(
        schema_version=1,
        source=eos.source_identity(),
        observations=observations,
        checkpoint=model.checkpoint(),
        validated=False,
        plant_referenced=False,
    )


def run(spec, folder):
    result = execute(spec)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    try:
        atomic_json(folder / "input.json", spec)
        atomic_json(folder / "result.json", result)
    except BaseException:
        shutil.rmtree(folder)
        raise
    return result


def replay(folder):
    folder = Path(folder)
    if {p.name for p in folder.iterdir()} != {"input.json", "result.json"}:
        raise ValueError("Unexpected vessel artifact inventory")
    saved = read(local_file(folder, "result.json"))
    if saved["source"] != eos.source_identity():
        raise ValueError("Vessel artifact source changed")
    result = execute(read(local_file(folder, "input.json")))
    if result != saved:
        raise ValueError("Vessel full replay differs")
    return result


def demo(folder):
    return run(
        dict(
            schema_version=1,
            model=specification(),
            program=[
                dict(duration_s=600.0, command=None),
                dict(
                    duration_s=600.0,
                    command=dict(
                        power_w=-20000.0, expected_revision=1, controller_state={"tick": 1}
                    ),
                ),
            ],
        ),
        folder,
    )


def qualify():
    source = eos.source_identity()
    published = []
    for ref in OFFICIAL:
        s = eos.pt(ref["p"], ref["t"])
        actual = dict(
            v=1 / s.density_kg_m3, u=s.internal_energy_j_kg, h=s.enthalpy_j_kg, s=s.entropy_j_kg_k
        )
        errors = {k: abs(actual[k] - ref[k]) / abs(ref[k]) for k in actual}
        if max(errors.values()) > CRITERIA["official_relative"]:
            raise ValueError("Published IF97 Table5 criterion")
        published.append(dict(reference=ref, actual=actual, relative_errors=errors))
    grid = []
    for p in [2e5, 5e5, 1e6, 5e6, 9.5e6]:
        for x in [0.001, 0.01, 0.5, 0.9, 0.999]:
            s = eos.px(p, x)
            r = eos.rho_u(
                s.density_kg_m3,
                s.internal_energy_j_kg,
                pressure_seed_pa=p * 0.95,
                temperature_seed_k=s.temperature_k,
            )
            ref = inverse_reference(s.density_kg_m3, s.internal_energy_j_kg)
            error = abs(r.pressure_pa - ref["pressure_pa"]) / p
            if (
                error > CRITERIA["pressure_relative"]
                or abs(r.temperature_k - ref["temperature_k"]) > CRITERIA["temperature_k"]
            ):
                raise ValueError("Equilibrium inverse grid criterion")
            grid.append(
                dict(
                    pressure_pa=p,
                    quality=x,
                    recovered=r.as_dict(),
                    reference=ref,
                    pressure_relative_error=error,
                    derivatives=eos.derivatives(s),
                )
            )
    cases = []
    for kind in ["cycle", "vapor_cooling", "liquid_heating"]:
        trajectories = []
        for segment in [20000.0, 10000.0, 5000.0]:
            request = specification(segment)
            if kind == "vapor_cooling":
                request.update(
                    initial=dict(kind="pt", pressure_pa=3e6, temperature_k=650.0),
                    initial_power_w=-20000.0,
                )
            elif kind == "liquid_heating":
                request.update(
                    initial=dict(kind="pt", pressure_pa=3e6, temperature_k=350.0),
                    initial_power_w=100.0,
                )
            model = RigidEquilibriumVessel(request)
            duration = 600.0 if kind != "liquid_heating" else 100.0
            first = model.advance(duration)
            reference = inverse_reference(
                first["thermodynamic_state"]["density_kg_m3"],
                first["thermodynamic_state"]["internal_energy_j_kg"],
            )
            p_error = (
                abs(first["thermodynamic_state"]["pressure_pa"] - reference["pressure_pa"])
                / reference["pressure_pa"]
            )
            if p_error > CRITERIA["pressure_relative"]:
                raise ValueError("Vessel chart-reference criterion")
            copy = RigidEquilibriumVessel(request)
            copy.restore(model.checkpoint())
            command = dict(
                power_w=-request["initial_power_w"],
                duration_s=duration,
                expected_revision=1,
                controller_state={"tick": 1},
            )
            second = model.step(
                command["power_w"],
                command["duration_s"],
                expected_revision=command["expected_revision"],
                controller_state=command["controller_state"],
            )
            if second != copy.step(
                command["power_w"],
                command["duration_s"],
                expected_revision=command["expected_revision"],
                controller_state=command["controller_state"],
            ):
                raise ValueError("Vessel restart not exact")
            trajectories.append(
                dict(
                    maximum_energy_segment_j_kg=segment,
                    forward=first,
                    reverse=second,
                    independent_reference=reference,
                    pressure_relative_error=p_error,
                    exact_restart=True,
                )
            )
        cases.append(dict(name=kind, trajectories=trajectories))
    if eos.source_identity() != source:
        raise ValueError("HEM qualification source drift")
    return dict(
        schema_version=1,
        source=source,
        criteria=CRITERIA,
        official_table5_checks=published,
        equilibrium_inverse_grid=grid,
        vessel_cases=cases,
        inverse_grid_count=len(grid),
        vessel_trajectories=9,
        interpretation="Exact prescribed heat evolution; energy sampling refinement is EOS admission sensitivity, not temporal/spatial convergence of a flow solver. Reference shares IF97 forward coefficients; inverse algorithms differ. No experimental validation.",
        validated=False,
        plant_referenced=False,
    )
