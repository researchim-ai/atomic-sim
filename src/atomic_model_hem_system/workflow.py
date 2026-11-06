"""Finite-channel/end-volume/wall programs and numerical qualification."""

import json
import shutil
from copy import deepcopy
from pathlib import Path

from atomic_model_hem_system import cases, reference
from atomic_model_hem_system.model import FiniteHEMSystem, signature
from atomic_model_workspace.common import atomic_json, keys, local_file, read

CAPS = [1e-4, 5e-5, 2.5e-5]
CASE_CAPS = {
    "reflected_momentum": [2.5e-5, 1.25e-5, 6.25e-6],
    "wall_driven_evaporation": [2.5e-5, 1.25e-5, 6.25e-6],
}
CRITERIA = dict(
    pressure_pa=0.1,
    temperature_k=1e-5,
    density_relative=1e-8,
    wall_temperature_k=1e-6,
    heat_j=0.001,
    impulse_kg_m_s=1e-6,
    entropy_j_k=1e-5,
)


def apply(model, call):
    keys(call, ("duration_s", "command"))
    c = call["command"]
    if c is None:
        return model.advance(call["duration_s"])
    keys(c, ("controls", "expected_revision", "controller_state"))
    return model.step(
        c["controls"],
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
        raise ValueError("System program schema/budget")
    model = FiniteHEMSystem(spec["model"])
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
        raise ValueError("Unexpected finite-system artifact inventory")
    result = read(local_file(folder, "result.json"))
    if result["source"] != signature():
        raise ValueError("Finite-system artifact source changed")
    actual = execute(read(local_file(folder, "input.json")))
    if result != actual:
        raise ValueError("Finite-system full program replay differs")
    return actual


def demo(folder):
    spec = dict(cases.matrix())["wall_driven_evaporation"]
    return run(dict(schema_version=1, model=spec, program=cases.program(spec)), folder)


def qualify():
    captured = signature()
    results = {}
    for name, spec, program in cases.programs():
        refs = {method: reference.solve(spec, program, method) for method in ["DOP853", "Radau"]}
        meshes = []
        for cap in CASE_CAPS.get(name, CAPS):
            request = deepcopy(spec)
            request["channel"]["max_step_s"] = cap
            model = FiniteHEMSystem(request)
            observations = [model.snapshot()]
            maximum = []
            minimum = []
            resumed = None
            for i, call in enumerate(program):
                result = apply(model, call)
                observations.append(model.snapshot())
                maximum.append(result["maximum_step_s"])
                minimum.append(result["minimum_step_s"])
                if resumed is not None and apply(resumed, call) != result:
                    raise ValueError("Finite-system restart not exact")
                if i == 1:
                    resumed = FiniteHEMSystem(request)
                    resumed.restore(json.loads(json.dumps(model.checkpoint())))
            errors = {}
            for field, label in [
                ("pressure_pa", "pressure_pa"),
                ("temperature_k", "temperature_k"),
            ]:
                errors[label] = max(
                    abs(cell[field] - ref[i]["cells"][j][field])
                    for ref in refs.values()
                    for i, o in enumerate(observations)
                    for j, cell in enumerate(o["cells"])
                )
            errors["density_relative"] = max(
                abs(cell["density_kg_m3"] - ref[i]["cells"][j]["density_kg_m3"])
                / cell["density_kg_m3"]
                for ref in refs.values()
                for i, o in enumerate(observations)
                for j, cell in enumerate(o["cells"])
            )
            errors["wall_temperature_k"] = max(
                abs(wall["temperature_k"] - ref[i]["wall_temperatures_k"][j])
                for ref in refs.values()
                for i, o in enumerate(observations)
                for j, wall in enumerate(o["walls"])
            )
            errors["heat_j"] = max(
                abs(o["global_ledger"][key] - ref[i]["global_integrals"][key])
                for key in ["external_fluid_heat_j", "external_wall_heat_j"]
                for ref in refs.values()
                for i, o in enumerate(observations)
            )
            errors["impulse_kg_m_s"] = max(
                abs(
                    o["global_ledger"]["endwall_impulse_kg_m_s"]
                    - ref[i]["global_integrals"]["endwall_impulse_kg_m_s"]
                )
                for ref in refs.values()
                for i, o in enumerate(observations)
            )
            errors["entropy_j_k"] = max(
                abs(o["global_ledger"][key] - ref[i]["global_integrals"][key])
                for key in ["external_entropy_j_k", "thermal_entropy_production_j_k"]
                for ref in refs.values()
                for i, o in enumerate(observations)
            )
            if any(value > CRITERIA[key] for key, value in errors.items()):
                raise ValueError("Finite-system reference criterion: " + name + ": " + str(errors))
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
        measured = [max(m["measured_maximum_steps_s"]) for m in meshes]
        if any(a / b < 1.2 for a, b in zip(measured, measured[1:])):
            raise ValueError("Finite-system meshes not measurably distinct")
        results[name] = dict(input=spec, program=program, meshes=meshes, references=refs)
    if signature() != captured:
        raise ValueError("System qualification source drift")
    return dict(
        schema_version=1,
        source=captured,
        criteria=CRITERIA,
        cases=results,
        case_count=len(results),
        temporal_trajectories=3 * len(results),
        semidiscrete_reference_ivps=2 * len(results),
        scope="Numerical same-area finite HEM end cells/channel/caloric-wall assembly, fixed reflecting exterior ends. Shared IF97 EOS; separate absolute-stock reference RHS. No general tank/plenum, experimental/plant or full forced entropy validation.",
        validated=False,
        plant_referenced=False,
    )


def audit(path):
    saved = read(path)
    if saved.get("source") != signature():
        raise ValueError("System campaign source mismatch")
    actual = qualify()
    if saved != actual:
        raise ValueError("System campaign recomputation differs")
    return actual
