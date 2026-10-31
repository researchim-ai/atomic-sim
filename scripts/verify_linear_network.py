"""Independent analytic checks of the compiled closed thermal-network fixtures."""

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np

from components.linear_thermal_network import LinearThermalNetwork
from sdk.experiment import _source_fingerprint
from sdk.network_experiment import run_network_experiment, write


def reference(spec, time_s):
    initial = spec["initial_temperature_k"]
    if set(initial) == {"a", "b", "c"}:
        amplitude = 10 * np.exp(-0.01 * time_s)
        return dict(a=300 + amplitude, b=300, c=300 - amplitude)
    capacities = {
        n["id"]: n["parameters"]
        for n in spec["topology"]["nodes"]
        if n["model_id"] == "linear-thermal-capacity"
    }
    ua = next(
        n["parameters"]["ua_w_k"]
        for n in spec["topology"]["nodes"]
        if n["model_id"] == "linear-thermal-conductance"
    )
    c1, c2 = capacities["hot"]["heat_capacity_j_k"], capacities["cold"]["heat_capacity_j_k"]
    p1, p2 = capacities["hot"]["source_w"], capacities["cold"]["source_w"]
    lam = ua * (1 / c1 + 1 / c2)
    asymptote = (p1 / c1 - p2 / c2) / lam
    gap = asymptote + (initial["hot"] - initial["cold"] - asymptote) * np.exp(-lam * time_s)
    energy = c1 * initial["hot"] + c2 * initial["cold"] + (p1 + p2) * time_s
    return dict(hot=(energy + c2 * gap) / (c1 + c2), cold=(energy - c1 * gap) / (c1 + c2))


def qualify(output, budget_path):
    budget = json.loads(budget_path.read_text())
    source = _source_fingerprint()
    output.mkdir()
    write(output / "budget.json", budget)
    base = json.loads(Path("scenarios/ai05_thermal_two_capacity.json").read_text())
    cases = []
    for power in (0, 30):
        for reverse in (False, True):
            spec = deepcopy(base)
            spec["name"] = f"two_capacity_{power}_" + ("reverse" if reverse else "forward")
            spec["topology"]["nodes"][0]["parameters"]["source_w"] = power
            if reverse:
                spec["initial_temperature_k"] = dict(hot=300, cold=330)
            cases.append(spec)
    cases.append(json.loads(Path("scenarios/ai05_thermal_star.json").read_text()))
    entries = []
    for spec in cases:
        spec["duration_s"] = budget["horizon_s"]
        spec["energy_tolerance_j"] = budget["energy_abs_j"]
        folder = output / spec["name"]
        report = run_network_experiment(spec, folder)
        manifest = json.loads((folder / "manifest.json").read_text())
        for name, digest in manifest["artifacts"].items():
            if hashlib.sha256((folder / name).read_bytes()).hexdigest() != digest:
                raise ValueError("Bundle checksum mismatch")
        trajectory = json.loads((folder / "trajectory.json").read_text())
        max_t = 0.0
        max_q = 0.0
        max_r_k = 0.0
        max_r_w = 0.0
        for row in trajectory:
            expected = reference(spec, row["time_s"])
            max_t = max(max_t, *(abs(row["temperature_k"][n] - expected[n]) for n in expected))
            if set(expected) == {"hot", "cold"}:
                ua = next(
                    n["parameters"]["ua_w_k"]
                    for n in spec["topology"]["nodes"]
                    if n["model_id"] == "linear-thermal-conductance"
                )
                q = -ua * (expected["hot"] - expected["cold"])
                max_q = max(max_q, abs(row["variables"]["hot.port.thermal.heat_flow_w"] - q))
            else:
                for n in expected:
                    max_q = max(
                        max_q,
                        abs(
                            row["variables"][n + ".port.thermal.heat_flow_w"]
                            + 10 * (expected[n] - 300)
                        ),
                    )
            assembly = json.loads((folder / "assembly.json").read_text())
            for eq in assembly["equations"]:
                value = abs(row["residuals"][eq["name"]])
                if eq["unit"] == "K":
                    max_r_k = max(max_r_k, value)
                elif eq["unit"] == "W":
                    max_r_w = max(max_r_w, value)
                else:
                    raise ValueError("Undeclared fixture residual dimension")
        payload = json.loads((folder / "checkpoint.json").read_text())
        clone = LinearThermalNetwork.from_snapshot(payload)
        if clone.snapshot() != payload:
            raise ValueError("Checkpoint roundtrip mismatch")
        restarted = clone.advance(budget["restart_extension_s"])
        ref = reference(spec, restarted["time_s"])
        restart_error = max(abs(restarted["temperature_k"][n] - ref[n]) for n in ref)
        checks = dict(
            completed=report["numerical_acceptance"],
            temperature=max_t <= budget["temperature_abs_k"],
            heat_flow=max_q <= budget["heat_flow_abs_w"],
            energy=report["energy_absolute_j"] <= budget["energy_abs_j"],
            temperature_residual=max_r_k <= budget["residual_temperature_abs_k"],
            power_residual=max_r_w <= budget["residual_power_abs_w"],
            restart=restart_error <= budget["temperature_abs_k"]
            and abs(restarted["energy_residual_j"]) <= budget["energy_abs_j"],
        )
        checks = {key: bool(value) for key, value in checks.items()}
        entries.append(
            dict(
                id=spec["name"],
                path=str(folder),
                accepted=all(checks.values()),
                checks=checks,
                max_temperature_error_k=max_t,
                max_heat_flow_error_w=max_q,
                max_temperature_residual_k=max_r_k,
                max_power_residual_w=max_r_w,
                restart_time_s=restarted["time_s"],
                restart_error_k=restart_error,
            )
        )
    if source != _source_fingerprint():
        raise RuntimeError("Source changed during verification")
    result = dict(
        schema_version=1,
        source_sha256=source,
        budget_sha256=hashlib.sha256(budget_path.read_bytes()).hexdigest(),
        cases=entries,
        all_cases_accepted=all(e["accepted"] for e in entries),
        validated=False,
        plant_referenced=False,
        reference="Independent two-capacity forced exponential and three-branch antisymmetric mode",
        scope="Closed constant-coefficient thermal network, not equipment or VVER qualification",
    )
    write(output / "verification.json", result)
    hashes = {
        str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in output.rglob("*")
        if p.is_file()
    }
    write(output / "manifest.json", dict(source_sha256=source, sha256=hashes))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--budget", type=Path, default=Path("validation/ai05_linear_network_budget_v1.json")
    )
    args = parser.parse_args()
    result = qualify(args.output, args.budget)
    print(
        json.dumps(
            dict(all_cases_accepted=result["all_cases_accepted"], cases=len(result["cases"]))
        )
    )
    return 0 if result["all_cases_accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
