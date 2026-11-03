"""Independent trajectory qualification for bounded two-loop wall assemblies."""

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.coupled_eos_wall import BUDGET, CoupledEOSWall, signature
from scripts.eos_wall_references import IndependentWall, cases
from scripts.solve_hydraulic_graph import sha

CAPS = (0.001, 0.0005, 0.00025)
TIMES = (0.02, 0.05, 0.1, 0.15, 0.2)
FIELDS = {
    "temperatures_k": "reference_temperature_atol_k",
    "pressures_pa": "reference_pressure_atol_pa",
    "masses_kg": "reference_mass_atol_kg",
    "velocities_m_s": "reference_velocity_atol_m_s",
}


def capture():
    return dict(
        runtime=signature(),
        qualification={
            p: sha(p)
            for p in (
                "scripts/eos_wall_references.py",
                "scripts/qualify_eos_wall.py",
                "scripts/conservative_loop_references.py",
                "scripts/storage_graph_references.py",
            )
        },
    )


def qualify():
    identity = capture()
    budget = json.loads(BUDGET.read_text())
    rows = []
    for name, spec in cases():
        refs = [IndependentWall(spec).integrate(TIMES, method) for method in ["DOP853", "Radau"]]
        for cap in CAPS:
            e = CoupledEOSWall(spec, max_step_s=cap)
            errors = dict.fromkeys(FIELDS, 0.0)
            wall_error = energy = entropy = step = stability = 0.0
            previous = 0.0
            restored = None
            for j, end in enumerate(TIMES):
                state = e.advance(end - previous)
                if restored is not None:
                    restored.advance(end - previous)
                    if restored.checkpoint() != e.checkpoint():
                        raise AssertionError("Restart differs")
                if end == 0.1:
                    restored = CoupledEOSWall(spec, max_step_s=cap)
                    restored.restore(json.loads(json.dumps(e.checkpoint())))
                for reference in refs:
                    r = reference[j]
                    wall_error = max(
                        wall_error,
                        float(
                            np.max(
                                abs(
                                    np.array(state["wall_temperatures_k"])
                                    - r["wall_temperatures_k"]
                                )
                            )
                        ),
                    )
                    for i in range(2):
                        for field in FIELDS:
                            errors[field] = max(
                                errors[field],
                                float(
                                    np.max(
                                        abs(
                                            np.array(list(state["loops"][i][field].values()))
                                            - r["loops"][i][field]
                                        )
                                    )
                                ),
                            )
                energy = max(energy, abs(state["energy_residual_j"]))
                entropy = max(entropy, abs(state["entropy_residual_j_k"]))
                step = max(step, state["work"]["maximum_internal_step_s"])
                stability = max(stability, state["work"]["maximum_stability_fraction"])
                previous = end
            for field, limit in FIELDS.items():
                if errors[field] > budget[limit]:
                    raise AssertionError((name, cap, field, errors[field], budget[limit]))
            if wall_error > budget["reference_temperature_atol_k"]:
                raise AssertionError((name, "wall", wall_error))
            rows.append(
                dict(
                    case=name,
                    max_step_s=cap,
                    errors=errors,
                    wall_temperature_error_k=wall_error,
                    energy_residual_j=energy,
                    entropy_residual_j_k=entropy,
                    observed_step_s=step,
                    stability_fraction=stability,
                    restart_exact=True,
                    observations=5,
                )
            )
    if identity != capture():
        raise AssertionError("Qualification sources changed")
    report = dict(
        schema_version=1,
        signature=identity,
        cases=7,
        trajectories=21,
        independent_programs=14,
        independent_ivp_segments=70,
        observations=105,
        horizon_s=0.2,
        rows=rows,
        validated=False,
        plant_referenced=False,
    )
    audit(report)
    return report


def audit(report):
    def require(condition):
        if not condition:
            raise ValueError("Invalid EOS wall evidence")

    require(isinstance(report, dict) and report.get("signature") == capture())
    for key, value in dict(
        schema_version=1,
        cases=7,
        trajectories=21,
        independent_programs=14,
        independent_ivp_segments=70,
        observations=105,
        horizon_s=0.2,
    ).items():
        require(type(report.get(key)) is type(value) and report[key] == value)
    require(report.get("validated") is False and report.get("plant_referenced") is False)
    rows = report.get("rows")
    require(isinstance(rows, list) and len(rows) == 21)
    expected = {(name, cap) for name, _ in cases() for cap in CAPS}
    seen = set()
    steps = {}
    budget = json.loads(BUDGET.read_text())

    def metric(v, limit):
        require(type(v) in (int, float) and np.isfinite(v) and 0 <= v <= limit)

    for row in rows:
        require(isinstance(row, dict) and isinstance(row.get("case"), str))
        cap = row.get("max_step_s")
        metric(cap, max(CAPS))
        pair = (row["case"], cap)
        require(pair in expected and pair not in seen)
        seen.add(pair)
        require(
            row.get("restart_exact") is True
            and type(row.get("observations")) is int
            and row["observations"] == 5
        )
        errors = row.get("errors")
        require(isinstance(errors, dict) and set(errors) == set(FIELDS))
        for field, limit in FIELDS.items():
            metric(errors[field], budget[limit])
        metric(row.get("wall_temperature_error_k"), budget["reference_temperature_atol_k"])
        metric(row.get("stability_fraction"), 1.0)
        metric(row.get("observed_step_s"), cap * (1 + 1e-12))
        require(row["observed_step_s"] > 0)
        # Scaled balance tolerances are enforced on full states in the engine.
        metric(row.get("energy_residual_j"), float("inf"))
        metric(row.get("entropy_residual_j_k"), float("inf"))
        steps[pair] = row["observed_step_s"]
    require(seen == expected)
    for name, _ in cases():
        values = [steps[(name, c)] for c in sorted(CAPS)]
        require(all(b / a >= 1.2 for a, b in zip(values, values[1:])))
    return dict(
        passed=True,
        trajectories=21,
        observations=105,
        validated=False,
        plant_referenced=False,
        scope="Source identity, matrix and reference metrics; not a numerical rerun or authentication",
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["run", "audit"])
    p.add_argument("path")
    a = p.parse_args()
    target = Path(a.path)
    if a.mode == "audit":
        print(json.dumps(audit(json.loads(target.read_text())), sort_keys=True))
    else:
        report = qualify()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(audit(report), sort_keys=True))


if __name__ == "__main__":
    main()
