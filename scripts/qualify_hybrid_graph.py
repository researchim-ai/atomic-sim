"""Source-bound three-mesh, two-reference acceptance campaign for hybrid graph control."""

import argparse
import json
import shutil
from pathlib import Path

from atomic_model_workspace.common import atomic_json, read
from scripts.hybrid_graph_references import cases, reference
from scripts.hybrid_thermal_graph import HybridThermalGraph
from scripts.hybrid_thermal_graph import signature as model_signature
from scripts.solve_hydraulic_graph import sha

EXTRA = [
    "scripts/qualify_hybrid_graph.py",
    "scripts/hybrid_graph_references.py",
    "scripts/hysteresis_references.py",
    "scripts/state_event_references.py",
    "scripts/nonlinear_graph_references.py",
    "scripts/hydraulic_graph_references.py",
    "scripts/scheduled_graph_references.py",
    "scripts/resistance_feedback_references.py",
    "scripts/thermal_graph_references.py",
]
CRITERIA = dict(
    temperature_k=1e-6,
    pressure_pa=1e-5,
    flow_kg_s=1e-7,
    energy_j=0.1,
    event_time_s=1e-5,
    balance_j=1e-4,
)
CAPS = [2.0, 1.0, 0.5]


def signature():
    return dict(model=model_signature(), qualification={p: sha(p) for p in EXTRA})


def call(model, item):
    command = item["command"]
    if command is None:
        return model.advance(item["duration_s"])
    return model.step(
        command["mode"],
        item["duration_s"],
        expected_revision=command["expected_revision"],
        controller_state=command["controller_state"],
        priority=command["priority"],
    )


def evaluate():
    captured = signature()
    results = {}
    for name, spec, program in cases():
        refs = [reference(spec, program, method) for method in ["Radau", "DOP853"]]
        meshes = []
        for cap in CAPS:
            model = HybridThermalGraph(spec, max_macro_step_s=cap)
            states = []
            segments = []
            resumed = None
            for i, item in enumerate(program):
                result = call(model, item)
                segments.extend(result["segments"])
                states.append(model.snapshot())
                if resumed is not None and call(resumed, item) != result:
                    raise ValueError("Hybrid restart is not exact")
                if i == 2:
                    resumed = HybridThermalGraph(spec, max_macro_step_s=cap)
                    if (
                        resumed.restore(json.loads(json.dumps(model.checkpoint())))
                        != model.snapshot()
                    ):
                        raise ValueError("Hybrid checkpoint reconstruction differs")
            history = model.snapshot()["history"]
            errors = {}
            for key, label in [
                ("temperatures_k", "temperature_k"),
                ("pressures_pa", "pressure_pa"),
                ("flows_kg_s", "flow_kg_s"),
            ]:
                error = max(
                    abs(state["engine"]["state"][key][node] - ref["states"][i][key][node])
                    for ref in refs
                    for i, state in enumerate(states)
                    for node in state["engine"]["state"][key]
                )
                if error > CRITERIA[label]:
                    raise ValueError("Hybrid reference criterion: " + label)
                errors[label] = error
            energy = max(
                abs(
                    state["engine"]["state"]["integrated_external_energy_j"]
                    - ref["states"][i]["integrated_external_energy_j"]
                )
                for ref in refs
                for i, state in enumerate(states)
            )
            if energy > CRITERIA["energy_j"]:
                raise ValueError("Hybrid external energy criterion")
            errors["energy_j"] = energy
            root_error = 0.0
            for ref in refs:
                if [(r["winner"], r["source_mode"], r["target_mode"]) for r in history] != [
                    (r["winner"], r["source_mode"], r["target_mode"]) for r in ref["history"]
                ]:
                    raise ValueError("Independent hybrid event path differs")
                if [s["mode"] for s in states] != [s["mode"] for s in ref["states"]]:
                    raise ValueError("Independent hybrid sampled mode differs")
                root_error = max(
                    [root_error]
                    + [abs(a["time_s"] - b["time_s"]) for a, b in zip(history, ref["history"])]
                )
            if root_error > CRITERIA["event_time_s"]:
                raise ValueError("Hybrid event time criterion")
            errors["event_time_s"] = root_error
            errors["balance_j"] = max(
                abs(s["engine"]["state"]["energy_residual_j"]) for s in states
            )
            if errors["balance_j"] > CRITERIA["balance_j"]:
                raise ValueError("Hybrid conservation criterion")
            maximum = max(s["trial_duration_s"] for s in segments)
            meshes.append(
                dict(
                    max_macro_step_s=cap,
                    maximum_trial_duration_s=maximum,
                    errors=errors,
                    states=states,
                    checkpoint_sha256=model.checkpoint()["sha256"],
                    exact_replay_restart=True,
                )
            )
        measured = [m["maximum_trial_duration_s"] for m in meshes]
        if any(a / b < 1.2 for a, b in zip(measured, measured[1:])):
            raise ValueError("Requested mesh caps did not produce distinct measured meshes")
        results[name] = dict(
            input=spec, program=program, meshes=meshes, independent_references=refs
        )
    if signature() != captured:
        raise ValueError("Hybrid qualification sources changed")
    return dict(
        schema_version=1,
        scope="Fixed-property thermal/hydraulic graph hybrid control, not general equipment or plant validation",
        source=captured,
        criteria=CRITERIA,
        cases=results,
        case_count=len(results),
        trajectory_count=3 * len(results),
        independent_ivp_count=2 * len(results),
        validated=False,
        plant_referenced=False,
    )


def run(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    try:
        result = evaluate()
        atomic_json(folder / "campaign.json", result)
    except BaseException:
        shutil.rmtree(folder)
        raise
    return result


def audit(folder):
    saved = read(Path(folder) / "campaign.json")
    if saved["source"] != signature():
        raise ValueError("Hybrid qualification source drift")
    if saved != evaluate():
        raise ValueError("Hybrid campaign recomputation differs")
    return saved


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "audit"])
    parser.add_argument("folder", type=Path)
    args = parser.parse_args(argv)
    result = run(args.folder) if args.command == "run" else audit(args.folder)
    print(
        {
            k: result[k]
            for k in [
                "case_count",
                "trajectory_count",
                "independent_ivp_count",
                "validated",
                "plant_referenced",
            ]
        }
    )


if __name__ == "__main__":
    main()
