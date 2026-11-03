"""Source-bound sampled PI qualification against independently integrated EOS."""

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.closed_loop_eos_pi import BUDGET, ClosedLoopEOSPI, signature
from scripts.eos_pi_references import IndependentPI, cases
from scripts.solve_hydraulic_graph import sha

CAPS = (0.001, 0.0005, 0.00025)
TICKS = 10


def capture():
    return dict(
        runtime=signature(),
        qualification={
            p: sha(p)
            for p in (
                "scripts/qualify_eos_pi.py",
                "scripts/eos_pi_references.py",
                "scripts/controlled_assembly_references.py",
                "scripts/terminal_chamber_references.py",
                "scripts/open_channel_references.py",
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
        refs = [IndependentPI(spec).integrate(TICKS, method) for method in ["DOP853", "Radau"]]
        for cap in CAPS:
            e = ClosedLoopEOSPI(spec, max_step_s=cap)
            restored = None
            error = dict(
                temperature_k=0.0, filtered_k=0.0, integral_w=0.0, output_w=0.0, external_heat_j=0.0
            )
            energy = entropy = maxstep = stability = 0.0
            samples = []
            for k in range(TICKS):
                s = e.step(expected_revision=e.revision)
                if restored is not None:
                    restored.step(expected_revision=restored.revision)
                    if restored.checkpoint() != e.checkpoint():
                        raise AssertionError("PI restart differs")
                if k == 4:
                    restored = ClosedLoopEOSPI(spec, max_step_s=cap)
                    restored.restore(json.loads(json.dumps(e.checkpoint())))
                plant = s["plant"]
                record = s["history"][-1]
                for ref in refs:
                    r = ref[k]
                    if r["integral_held"] != record["integral_held"]:
                        raise AssertionError((name, cap, k, "anti-windup decision differs"))
                    for port, T in r["port_temperatures_k"].items():
                        error["temperature_k"] = max(
                            error["temperature_k"], abs(plant["port_temperatures_k"][port] - T)
                        )
                    for key in ["filtered_k", "integral_w", "output_w"]:
                        error[key] = max(error[key], abs(s["controller"][key] - r[key]))
                    error["external_heat_j"] = max(
                        error["external_heat_j"],
                        abs(plant["external_heat_j"] - r["external_heat_j"]),
                    )
                energy = max(energy, abs(plant["energy_residual_j"]))
                entropy = max(entropy, abs(plant["entropy_residual_j_k"]))
                maxstep = max(maxstep, s["work"]["maximum_internal_step_s"])
                stability = max(stability, s["work"]["maximum_stability_fraction"])
                samples.append(
                    dict(
                        time_s=plant["time_s"],
                        truth_k=record["truth_k"],
                        filtered_k=record["filtered_k"],
                        integral_before_w=record["integral_before_w"],
                        integral_w=record["integral_after_w"],
                        output_before_w=record["output_before_w"],
                        output_w=record["output_w"],
                        integral_held=record["integral_held"],
                    )
                )
            limits = dict(
                temperature_k=budget["reference_temperature_atol_k"],
                filtered_k=budget["reference_temperature_atol_k"],
                integral_w=budget["reference_control_atol_w"],
                output_w=budget["reference_control_atol_w"],
                external_heat_j=budget["reference_energy_atol_j"],
            )
            for key, limit in limits.items():
                if error[key] > limit:
                    raise AssertionError((name, cap, key, error[key], limit))
            rows.append(
                dict(
                    case=name,
                    max_step_s=cap,
                    observations=TICKS,
                    errors=error,
                    energy_residual_j=energy,
                    entropy_residual_j_k=entropy,
                    observed_step_s=maxstep,
                    stability_fraction=stability,
                    restart_exact=True,
                    controller_trace=samples,
                )
            )
    if identity != capture():
        raise AssertionError("PI qualification sources changed")
    report = dict(
        schema_version=1,
        signature=identity,
        cases=8,
        trajectories=24,
        independent_programs=16,
        independent_ivp_segments=160,
        observations=240,
        horizon_s=0.1,
        rows=rows,
        validated=False,
        plant_referenced=False,
    )
    audit(report)
    return report


def audit(report):
    def require(condition):
        if not condition:
            raise ValueError("Invalid PI episode evidence")

    def finite(value):
        require(type(value) in (int, float) and np.isfinite(value))

    def metric(value, limit=float("inf")):
        finite(value)
        require(0 <= value <= limit)

    require(isinstance(report, dict) and report.get("signature") == capture())
    for key, value in dict(
        schema_version=1,
        cases=8,
        trajectories=24,
        independent_programs=16,
        independent_ivp_segments=160,
        observations=240,
        horizon_s=0.1,
    ).items():
        require(type(report.get(key)) is type(value) and report[key] == value)
    require(report.get("validated") is False and report.get("plant_referenced") is False)
    rows = report.get("rows")
    require(isinstance(rows, list) and len(rows) == 24)
    policies = {name: spec["policy"] for name, spec in cases()}
    expected = {(name, cap) for name in policies for cap in CAPS}
    seen = set()
    steps = {}
    budget = json.loads(BUDGET.read_text())
    limits = dict(
        temperature_k=budget["reference_temperature_atol_k"],
        filtered_k=budget["reference_temperature_atol_k"],
        integral_w=budget["reference_control_atol_w"],
        output_w=budget["reference_control_atol_w"],
        external_heat_j=budget["reference_energy_atol_j"],
    )
    for row in rows:
        require(isinstance(row, dict) and isinstance(row.get("case"), str))
        cap = row.get("max_step_s")
        metric(cap, max(CAPS))
        pair = (row["case"], cap)
        require(pair in expected and pair not in seen)
        seen.add(pair)
        require(
            type(row.get("observations")) is int
            and row["observations"] == 10
            and row.get("restart_exact") is True
        )
        metric(row.get("observed_step_s"), cap * (1 + 1e-12))
        require(row["observed_step_s"] > 0)
        steps[pair] = row["observed_step_s"]
        metric(row.get("stability_fraction"), 1.0)
        errors = row.get("errors")
        require(isinstance(errors, dict) and set(errors) == set(limits))
        for key, limit in limits.items():
            metric(errors[key], limit)
        metric(row.get("energy_residual_j"))
        metric(row.get("entropy_residual_j_k"))
        trace = row.get("controller_trace")
        require(isinstance(trace, list) and len(trace) == 10)
        p = policies[row["case"]]
        previous = p["initial_output_w"]
        last_i = p["initial_integral_w"]
        for k, r in enumerate(trace):
            require(
                isinstance(r, dict)
                and set(r)
                == {
                    "time_s",
                    "truth_k",
                    "filtered_k",
                    "integral_before_w",
                    "integral_w",
                    "output_before_w",
                    "output_w",
                    "integral_held",
                }
            )
            for key in set(r) - {"integral_held"}:
                finite(r[key])
            require(
                type(r["integral_held"]) is bool and r["time_s"] == (k + 1) * p["sample_period_s"]
            )
            require(r["output_before_w"] == previous and r["integral_before_w"] == last_i)
            require(p["output_min_w"] <= r["output_w"] <= p["output_max_w"])
            require(abs(r["output_w"] - previous) <= p["slew_w_s"] * p["sample_period_s"] + 1e-9)
            if r["integral_held"]:
                require(r["integral_w"] == last_i)
            previous = r["output_w"]
            last_i = r["integral_w"]
    require(seen == expected)
    for name in policies:
        values = [steps[(name, cap)] for cap in sorted(CAPS)]
        require(all(b / a >= 1.2 for a, b in zip(values, values[1:])))
    return dict(
        passed=True,
        trajectories=24,
        observations=240,
        validated=False,
        plant_referenced=False,
        scope="Source/matrix/reference errors and trace power/slew/held-integral invariants; no numerical rerun or authentication",
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["run", "audit"])
    p.add_argument("path")
    args = p.parse_args()
    target = Path(args.path)
    if args.mode == "audit":
        print(json.dumps(audit(json.loads(target.read_text())), sort_keys=True))
    else:
        report = qualify()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(audit(report), sort_keys=True))


if __name__ == "__main__":
    main()
