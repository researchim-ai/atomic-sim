"""Hash-checked Euler temporal and reflection comparisons, including final state."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from physics.hydraulics.euler_channel import EulerLiquidChannel


def load(folder):
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    for name, digest in manifest["artifacts"].items():
        if hashlib.sha256((folder / name).read_bytes()).hexdigest() != digest:
            raise ValueError("Euler artifact checksum mismatch")
    spec = json.loads((folder / "input.json").read_text())
    report = json.loads((folder / "report.json").read_text())
    if (
        spec["engine"] not in ("conservative-euler-liquid-channel", "contact-euler-liquid-channel")
        or not report["numerical_acceptance"]
        or report["status"] != "completed"
        or report["completed_time_s"] != spec["duration_s"]
    ):
        raise ValueError("Euler experiment not fully accepted")
    with (folder / "trajectory.csv").open() as f:
        rows = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]
    times = [r["time_s"] for r in rows]
    if (
        times[0] != 0
        or times[-1] != spec["duration_s"]
        or any(b <= a for a, b in zip(times, times[1:]))
        or not all(np.isfinite(list(r.values())).all() for r in rows)
    ):
        raise ValueError("Invalid Euler trajectory")
    payload = json.loads((folder / "checkpoint.json").read_text())
    model = EulerLiquidChannel.from_snapshot(payload)
    if (
        model.snapshot() != payload
        or model.time_s != spec["duration_s"]
        or model.config.as_dict() != spec["channel"]
    ):
        raise ValueError("Euler checkpoint mismatch")
    for key, value in model.observation().items():
        if value != rows[-1][key]:
            raise ValueError("Euler CSV/checkpoint observation mismatch")
    measured = dict(
        mass_relative=max(abs(r["mass_residual_kg"]) for r in rows)
        / max(abs(model.initial_mass_kg), 1e-9),
        energy_relative=max(abs(r["energy_residual_j"]) for r in rows)
        / max(abs(model.initial_energy_j), 1.0),
        energy_absolute_j=max(abs(r["energy_residual_j"]) for r in rows),
        momentum_absolute_kg_m_s=max(abs(r["momentum_residual_kg_m_s"]) for r in rows),
    )
    for key, value in measured.items():
        if value != report["balance"][key] or value > spec["budgets"][key]:
            raise ValueError("Euler reported budget mismatch")
    return spec, rows, payload, manifest


def compare(left, right, budget, reflection=False):
    a, b = load(left), load(right)
    if a[3]["source_sha256"] != b[3]["source_sha256"] or a[3]["versions"] != b[3]["versions"]:
        raise ValueError("Euler source/dependency mismatch")
    if a[0]["duration_s"] != budget["horizon_s"] or b[0]["duration_s"] != budget["horizon_s"]:
        raise ValueError("Comparison horizon mismatch")
    expected = dict(a[0]["channel"])
    if reflection:
        for quantity in ("pressure_pa", "temperature_k"):
            expected["left_" + quantity], expected["right_" + quantity] = (
                expected["right_" + quantity],
                expected["left_" + quantity],
            )
        expected["left_velocity_m_s"], expected["right_velocity_m_s"] = (
            -expected["right_velocity_m_s"],
            -expected["left_velocity_m_s"],
        )
        expected["initial_velocity_m_s"] *= -1
        for key in (
            "initial_pressure_profile_pa",
            "initial_temperature_profile_k",
            "initial_wall_temperature_profile_k",
            "initial_velocity_profile_m_s",
        ):
            if key in expected:
                expected[key] = list(reversed(expected[key]))
                if key == "initial_velocity_profile_m_s":
                    expected[key] = [-x for x in expected[key]]
        if a[0]["solver"] != b[0]["solver"]:
            raise ValueError("Reflection requires identical solvers")
    if expected != b[0]["channel"]:
        raise ValueError("Incompatible Euler physics")
    bytime = {r["time_s"]: r for r in b[1]}
    if set(bytime) != {r["time_s"] for r in a[1]}:
        raise ValueError("Comparison sampling mismatch")
    fields = dict(
        pressure_abs_pa=("min_pressure_pa", "max_pressure_pa"),
        temperature_abs_k=(
            "min_temperature_k",
            "max_temperature_k",
            "min_wall_temperature_k",
            "max_wall_temperature_k",
        ),
        energy_abs_j=("stored_energy_j", "kinetic_energy_j"),
    )
    measured = {
        key: max(abs(row[f] - bytime[row["time_s"]][f]) for row in a[1] for f in fs)
        for key, fs in fields.items()
    }
    measured["velocity_abs_m_s"] = max(
        abs(row[f] - (-bytime[row["time_s"]][g] if reflection else bytime[row["time_s"]][f]))
        for row in a[1]
        for f, g in (
            ("min_velocity_m_s", "max_velocity_m_s"),
            ("max_velocity_m_s", "min_velocity_m_s"),
        )
    )
    checks = {key: value <= budget[key] for key, value in measured.items()}
    if reflection:
        n = a[0]["channel"]["cells"]
        y = np.array(a[2]["state"])
        z = np.array(b[2]["state"])
        q = y[: 3 * n].reshape(n, 3)[::-1].copy()
        q[:, 1] *= -1
        tail = y[-3:].copy()
        tail[1] *= -1
        reflected = np.r_[q.ravel(), y[3 * n : 4 * n][::-1], tail]
        scaled = float(max(abs(z - reflected) / np.maximum(abs(reflected), 1.0)))
        measured["final_state_scaled"] = scaled
        checks["final_state_scaled"] = scaled <= budget["final_state_scaled"]
    return dict(
        passed=all(checks.values()),
        runs=[str(left), str(right)],
        checks=checks,
        measured=measured,
        reflection=reflection,
        source_sha256=a[3]["source_sha256"],
        validated=False,
        plant_referenced=False,
        scope="Specific Euler numerical case; no full spatial/acoustic/plant qualification",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("left", type=Path)
    p.add_argument("right", type=Path)
    p.add_argument("--budget", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reflection", action="store_true")
    args = p.parse_args()
    report = compare(args.left, args.right, json.loads(args.budget.read_text()), args.reflection)
    report["budget_sha256"] = hashlib.sha256(args.budget.read_bytes()).hexdigest()
    with args.output.open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(report))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
