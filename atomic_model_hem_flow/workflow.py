"""Run/replay and numerical qualification of the periodic HEM Euler family."""

import json
import math
import shutil
from copy import deepcopy
from pathlib import Path

from atomic_model_hem import eos
from atomic_model_hem_flow import cases, reference
from atomic_model_hem_flow.model import PeriodicHEMChannel, signature
from atomic_model_workspace.common import atomic_json, keys, local_file, read

TIMES = [0.0, 0.002, 0.008, 0.018]
CAPS = [1e-4, 5e-5, 2.5e-5]
CASE_CAPS = {"isobaric_quality_contact": [2e-5, 1e-5, 5e-6]}
CRITERIA = dict(
    pressure_pa=0.05, temperature_k=1e-5, density_relative=1e-8, energy_j=0.001, entropy_j_k=1e-5
)


def execute(spec):
    keys(spec, ("schema_version", "model", "program"))
    if (
        type(spec["schema_version"]) is not int
        or spec["schema_version"] != 1
        or not isinstance(spec["program"], list)
        or not 1 <= len(spec["program"]) <= 16
    ):
        raise ValueError("HEM program schema/budget")
    model = PeriodicHEMChannel(spec["model"])
    observations = []
    for call in spec["program"]:
        keys(call, ("duration_s", "command"))
        c = call["command"]
        if c is None:
            result = model.advance(call["duration_s"])
        else:
            keys(c, ("heat_power_w", "expected_revision", "controller_state"))
            result = model.step(
                c["heat_power_w"],
                call["duration_s"],
                expected_revision=c["expected_revision"],
                controller_state=c["controller_state"],
            )
        observations.append(result)
    return dict(
        schema_version=1,
        source=signature(),
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
        raise ValueError("Unexpected HEM artifact inventory")
    saved = read(local_file(folder, "result.json"))
    if saved["source"] != signature():
        raise ValueError("HEM artifact source changed")
    result = execute(read(local_file(folder, "input.json")))
    if result != saved:
        raise ValueError("HEM artifact full replay differs")
    return result


def demo(folder):
    model = dict(cases.matrix())["uniform_evaporation"]
    return run(
        dict(
            schema_version=1,
            model=model,
            program=[
                dict(duration_s=0.018, command=None),
                dict(
                    duration_s=0.018,
                    command=dict(
                        heat_power_w=[-20000.0] * 4,
                        expected_revision=1,
                        controller_state={"tick": 1},
                    ),
                ),
            ],
        ),
        folder,
    )


def spatial():
    result = []
    base = eos.px(1e6, 0.5)
    speed = math.sqrt(eos.derivatives(base)["equilibrium_sound_speed_squared_m2_s2"])
    for n in [4, 8, 16]:
        spec = cases.acoustic(n, 1.0)
        spec["max_step_s"] = 1e-5
        model = PeriodicHEMChannel(spec)
        observation = model.advance(0.002)
        k = 2 * math.pi / 10.0
        dx = 10.0 / n
        sinc = math.sin(k * dx / 2) / (k * dx / 2)
        exact = [1e6 + sinc * math.cos(k * ((i + 0.5) * dx - speed * 0.002)) for i in range(n)]
        error = (
            sum(
                abs(cell["pressure_pa"] - value) for cell, value in zip(observation["cells"], exact)
            )
            / n
        )
        result.append(
            dict(
                cells=n,
                pressure_l1_error_pa=error,
                exact_cell_average_pressure_pa=exact,
                observation=observation,
            )
        )
    if any(
        a["pressure_l1_error_pa"] / b["pressure_l1_error_pa"] < 1.15
        for a, b in zip(result, result[1:])
    ):
        raise ValueError("HEM measured spatial refinement failed")
    return result


def qualify():
    captured = signature()
    results = {}
    for name, spec in cases.matrix():
        refs = {method: reference.solve(spec, TIMES, method) for method in ["DOP853", "Radau"]}
        meshes = []
        for cap in CASE_CAPS.get(name, CAPS):
            request = deepcopy(spec)
            request["max_step_s"] = cap
            model = PeriodicHEMChannel(request)
            observations = [model.snapshot()]
            steps = []
            for a, b in zip(TIMES, TIMES[1:]):
                report = model.advance(b - a)
                steps.append(report["maximum_step_s"])
                observations.append(model.snapshot())
            errors = {}
            for field, label in [
                ("pressure_pa", "pressure_pa"),
                ("temperature_k", "temperature_k"),
            ]:
                error = max(
                    abs(cell[field] - ref[i]["cells"][j][field])
                    for ref in refs.values()
                    for i, o in enumerate(observations)
                    for j, cell in enumerate(o["cells"])
                )
                errors[label] = error
                if error > CRITERIA[label]:
                    raise ValueError("HEM temporal reference criterion " + label)
            errors["density_relative"] = max(
                abs(cell["density_kg_m3"] - ref[i]["cells"][j]["density_kg_m3"])
                / cell["density_kg_m3"]
                for ref in refs.values()
                for i, o in enumerate(observations)
                for j, cell in enumerate(o["cells"])
            )
            errors["energy_j"] = max(
                abs(o["ledgers"]["external_heat_j"] - ref[i]["external_heat_j"])
                for ref in refs.values()
                for i, o in enumerate(observations)
            )
            errors["entropy_j_k"] = max(
                abs(o["ledgers"]["external_heat_entropy_j_k"] - ref[i]["external_heat_entropy_j_k"])
                for ref in refs.values()
                for i, o in enumerate(observations)
            )
            if any(
                errors[k] > CRITERIA[k] for k in ["density_relative", "energy_j", "entropy_j_k"]
            ):
                raise ValueError(
                    "HEM conservative reference criterion: " + name + ": " + str(errors)
                )
            other = PeriodicHEMChannel(request)
            other.restore(json.loads(json.dumps(model.checkpoint())))
            if model.advance(0.001) != other.advance(0.001):
                raise ValueError("HEM restart not exact")
            meshes.append(
                dict(
                    max_step_s=cap,
                    measured_maximum_steps_s=steps,
                    errors=errors,
                    observations=observations,
                    exact_restart=True,
                )
            )
        results[name] = dict(input=spec, meshes=meshes, references=refs)
    refined = spatial()
    if signature() != captured:
        raise ValueError("HEM qualification source drift")
    return dict(
        schema_version=1,
        source=captured,
        criteria=CRITERIA,
        cases=results,
        spatial_refinement=refined,
        case_count=len(results),
        temporal_trajectories=3 * len(results),
        semidiscrete_reference_ivps=2 * len(results),
        scope="Periodic constant-area inviscid IF97 HEM Euler, first-order space and SSPRK3; weak-flow numerical qualification, not validated plant model",
        validated=False,
        plant_referenced=False,
    )
