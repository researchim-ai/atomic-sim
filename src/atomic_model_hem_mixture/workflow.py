"""Source-bound closed graph programs and independent IVP qualification."""

import json
import shutil
import sys
from copy import deepcopy
from pathlib import Path

from atomic_model_hem_mixture import cases, reference
from atomic_model_hem_mixture.model import MixtureHEMNetwork, signature
from atomic_model_workspace.common import atomic_json, keys, local_file, read

CRITERIA = dict(
    pressure_pa=0.1,
    temperature_k=1e-5,
    velocity_m_s=1e-6,
    fluid_total_energy_j=0.001,
    density_relative=1e-8,
    wall_temperature_k=1e-6,
    heat_j=0.001,
    impulse_kg_m_s=1e-6,
    entropy_j_k=1e-5,
)
CAPS = [5e-5, 2.5e-5, 1.25e-5]


def program(name):
    return cases.program(name)


def apply(model, call):
    keys(call, ("duration_s", "command"))
    command = call["command"]
    if command is None:
        return model.advance(call["duration_s"])
    keys(command, ("controls", "expected_revision", "controller_state"))
    return model.step(
        command["controls"],
        call["duration_s"],
        expected_revision=command["expected_revision"],
        controller_state=command["controller_state"],
    )


def execute(spec):
    keys(spec, ("schema_version", "model", "program"))
    if (
        type(spec["schema_version"]) is not int
        or spec["schema_version"] != 1
        or not isinstance(spec["program"], list)
        or not 1 <= len(spec["program"]) <= 16
    ):
        raise ValueError("Graph program schema/budget")
    model = MixtureHEMNetwork(spec["model"])
    observations = [apply(model, c) for c in spec["program"]]
    return dict(
        schema_version=1,
        source=signature(),
        contract=model.contract(),
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
        raise ValueError("Unexpected graph artifact inventory")
    result = read(local_file(folder, "result.json"))
    if result["source"] != signature():
        raise ValueError("Graph artifact source changed")
    actual = execute(read(local_file(folder, "input.json")))
    if actual != result:
        raise ValueError("Graph full program replay differs")
    return actual


def demo(folder):
    name = "heated_phase_exit"
    return run(
        dict(schema_version=1, model=dict(cases.matrix())[name], program=program(name)), folder
    )


def programs():
    return [(name, spec, program(name)) for name, spec in cases.matrix()] + [cases.online()]


def qualify():
    captured = signature()
    results = {}
    for name, spec, calls in programs():
        refs = {}
        for method in ["DOP853", "Radau"]:
            print(f"qualify {name}: {method}", file=sys.stderr, flush=True)
            refs[method] = reference.solve(spec, calls, method)
        print(f"qualify {name}: SSPRK3/restart", file=sys.stderr, flush=True)
        meshes = []
        caps = [2.5e-6, 1.25e-6, 6.25e-7] if name.startswith("turbulent") else CAPS
        if name == "heated_phase_exit":
            caps = [1.25e-5, 6.25e-6, 3.125e-6]
        for cap in caps:
            request = deepcopy(spec)
            request["model"]["max_step_s"] = cap
            model = MixtureHEMNetwork(request)
            observations = [model.snapshot()]
            maximum = []
            minimum = []
            resumed = None
            for i, call in enumerate(calls):
                result = apply(model, call)
                observations.append(model.snapshot())
                maximum.append(result["maximum_step_s"])
                minimum.append(result["minimum_step_s"])
                if resumed is not None and apply(resumed, call) != result:
                    raise ValueError("Graph restart differs")
                if i == 1:
                    resumed = MixtureHEMNetwork(request)
                    resumed.restore(json.loads(json.dumps(model.checkpoint())))
            errors = {
                field: max(
                    abs(c[field] - ref[i]["cells"][j][field])
                    for ref in refs.values()
                    for i, o in enumerate(observations)
                    for j, c in enumerate(o["cells"])
                )
                for field in ["pressure_pa", "temperature_k"]
            }
            errors["velocity_m_s"] = max(
                abs(
                    c["velocity_m_s"]
                    - ref[i]["cells"][j]["momentum_kg_m_s"] / ref[i]["cells"][j]["mass_kg"]
                )
                for ref in refs.values()
                for i, o in enumerate(observations)
                for j, c in enumerate(o["cells"])
            )
            errors["fluid_total_energy_j"] = max(
                abs(c["total_energy_j"] - ref[i]["cells"][j]["total_energy_j"])
                for ref in refs.values()
                for i, o in enumerate(observations)
                for j, c in enumerate(o["cells"])
            )
            errors["density_relative"] = max(
                abs(c["density_kg_m3"] - ref[i]["cells"][j]["density_kg_m3"]) / c["density_kg_m3"]
                for ref in refs.values()
                for i, o in enumerate(observations)
                for j, c in enumerate(o["cells"])
            )
            errors["wall_temperature_k"] = max(
                abs(w["temperature_k"] - ref[i]["wall_temperatures_k"][j])
                for ref in refs.values()
                for i, o in enumerate(observations)
                for j, w in enumerate(o["walls"])
            )
            errors["heat_j"] = max(
                abs(o["global_ledger"][k] - ref[i]["global_integrals"][k])
                for k in ["external_fluid_heat_j", "external_wall_heat_j"]
                for ref in refs.values()
                for i, o in enumerate(observations)
            )
            errors["impulse_kg_m_s"] = max(
                abs(value - ref[i]["local_ledgers"][k][j])
                for k in ["drag_impulse_kg_m_s", "node_support_impulse_kg_m_s"]
                for ref in refs.values()
                for i, o in enumerate(observations)
                for j, value in enumerate(o["local_ledgers"][k])
            )
            errors["entropy_j_k"] = max(
                abs(o["global_ledger"][k] - ref[i]["global_integrals"][k])
                for k in [
                    "external_entropy_j_k",
                    "thermal_entropy_production_j_k",
                    "drag_entropy_production_j_k",
                ]
                for ref in refs.values()
                for i, o in enumerate(observations)
            )
            if any(value > CRITERIA[k] for k, value in errors.items()):
                raise ValueError("Graph reference criterion: " + name + ": " + str(errors))
            meshes.append(
                dict(
                    max_step_s=cap,
                    measured_maximum_steps_s=maximum,
                    measured_minimum_steps_s=minimum,
                    errors=errors,
                    observations=observations,
                    exact_restart=True,
                    checkpoint_sha256=model.checkpoint()["sha256"],
                )
            )
        actual = [max(m["measured_maximum_steps_s"]) for m in meshes]
        if any(a / b < 1.2 for a, b in zip(actual, actual[1:])):
            raise ValueError("Graph steps not distinct")
        results[name] = dict(input=spec, program=calls, meshes=meshes, references=refs)
    from atomic_model_hem import eos
    from atomic_model_hem_mixture.friction import FLAGS, evaluate

    sensitivity = []
    for pressure in [5e5, 1e6, 5e6]:
        for quality in [0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0]:
            state = eos.px(pressure, quality)
            values = {
                model: evaluate(
                    state,
                    50.0 / state.density_kg_m3,
                    0.05,
                    0.0,
                    viscosity_model=model,
                    scope={k: True for k in FLAGS},
                ).pressure_gradient_pa_m
                for model in ["mcadams", "cicchitti"]
            }
            sensitivity.append(
                dict(
                    pressure_pa=pressure,
                    quality=quality,
                    mass_flux_kg_m2_s=50.0,
                    gradients_pa_m=values,
                    ratio=max(values.values()) / min(values.values()),
                )
            )
    if captured != signature():
        raise ValueError("Graph qualification source drift")
    return dict(
        schema_version=1,
        source=captured,
        criteria=CRITERIA,
        cases=results,
        case_count=len(results),
        temporal_trajectories=3 * len(results),
        semidiscrete_reference_ivps=2 * len(results),
        qualification_exclusions=[
            "Moving exact-saturation transient IVPs: not accepted in this protocol; endpoint pointwise laws and stationary endpoint limits are checked separately"
        ],
        model_choice_sensitivity=dict(
            scope="Fixed-state alternatives, not confidence interval or calibrated uncertainty",
            points=sensitivity,
        ),
        scope="Numerical homogeneous no-slip McAdams/Cicchitti plus Darcy graph friction. Independent absolute-stock graph reference, shared IF97. No two-phase experimental calibration, separated flow, boiling heat-transfer or CHF qualification.",
        validated=False,
        plant_referenced=False,
    )


def audit(path):
    saved = read(path)
    if saved.get("source") != signature():
        raise ValueError("Graph campaign source mismatch")
    actual = qualify()
    if saved != actual:
        raise ValueError("Graph campaign recomputation differs")
    return actual
