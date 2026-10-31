"""Case-scoped nonlinear circulation verification against independent references."""

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np

from physics.hydraulics.circulation_loop import CirculationLoop
from scripts import circulation_references as refs
from sdk.circulation_experiment import run_circulation_experiment
from sdk.experiment import _source_fingerprint
from sdk.network_experiment import write


def cases():
    template = json.loads(Path("scenarios/ai05_circulation_reversal.json").read_text())
    result = []
    for name, head, flow in [
        ("drag_forward", 0, 1),
        ("drag_reverse", 0, -1),
        ("driven_forward", 10000, 0),
        ("driven_reverse", -10000, 0),
        ("reversal_forward", 10000, -5),
        ("reversal_reverse", -10000, 5),
        ("head_switch", 10000, 8),
    ]:
        spec = deepcopy(template)
        spec["name"] = name
        spec["initialization"]["mass_flow_kg_s"] = flow
        for node in spec["model"]["topology"]["nodes"]:
            node["parameters"]["drive_pressure_pa"] = head if node["id"] == "s0" else 0
        if name == "head_switch":
            spec["timeline"] = [dict(t=3, heads_pa=dict(s0=-10000, s1=0, s2=0))]
        result.append((spec, "adiabatic"))
    steady = deepcopy(template)
    steady["name"] = "heated_steady"
    steady["initialization"] = dict(mode="steady")
    steady["duration_s"] = 100
    steady["step_s"] = 5
    for node in steady["model"]["topology"]["nodes"]:
        node["parameters"].update(
            drive_pressure_pa=10000 if node["id"] == "s0" else 0,
            fluid_wall_ua_w_k=100,
            ambient_ua_w_k=20,
            wall_power_w=10,
        )
    result.append((steady, "steady"))
    for direction in (1, -1):
        pulse = deepcopy(steady)
        pulse["name"] = "wall_pulse_" + ("forward" if direction > 0 else "reverse")
        pulse["duration_s"] = 30
        pulse["step_s"] = 0.5
        pulse["initialization"] = dict(
            mode="given",
            mass_flow_kg_s=direction * float(np.sqrt(10000 / 150)),
            fluid_temperature_k=dict(s0=300, s1=300, s2=300),
            wall_temperature_k=dict(s0=302, s1=300, s2=300),
        )
        for node in pulse["model"]["topology"]["nodes"]:
            node["parameters"].update(
                drive_pressure_pa=direction * 10000 if node["id"] == "s0" else 0, wall_power_w=0
            )
        result.append((pulse, "modal"))
    nonuniform = deepcopy(result[4][0])
    nonuniform["name"] = "nonuniform_transport"
    nonuniform["initialization"].update(
        mass_flow_kg_s=-3, fluid_temperature_k=dict(s0=303, s1=300, s2=297)
    )
    for i, node in enumerate(nonuniform["model"]["topology"]["nodes"]):
        node["parameters"].update(length_m=[1, 0.7, 1.3][i], loss_coefficient=[4, 17, 9][i])
    result.append((nonuniform, "adiabatic"))
    return result


def expected(spec, kind, time_s, phase="sample"):
    cfg = spec["model"]
    nodes = cfg["topology"]["nodes"]
    names = [v["id"] for v in nodes]
    params = [v["parameters"] for v in nodes]
    rho = cfg["fluid"]["density_kg_m3"]
    cp = cfg["fluid"]["heat_capacity_j_kg_k"]
    area = cfg["fluid"]["area_m2"]
    lengths = np.array([p["length_m"] for p in params])
    losses = np.array([p["loss_coefficient"] for p in params])
    inertia = float(sum(lengths) / area)
    resistance = float(sum(losses) / (2 * rho * area**2))
    heads = np.array([p["drive_pressure_pa"] for p in params])
    head = float(sum(heads))
    commands = [(c["t"], float(sum(c["heads_pa"].values()))) for c in spec["timeline"]]
    if kind == "steady":
        flow = float(np.sign(head) * np.sqrt(abs(head) / resistance))
        work = head * flow * time_s / rho
        zeros = []
        diss = losses * abs(flow) ** 3 / (2 * rho**2 * area**2)
        wall = np.array(
            [
                cfg["ambient_temperature_k"] + (p["wall_power_w"] + d) / p["ambient_ua_w_k"]
                for p, d in zip(params, diss)
            ]
        )
        fluid = wall + diss / np.array([p["fluid_wall_ua_w_k"] for p in params])
    else:
        initial = spec["initialization"]
        m0 = initial["mass_flow_kg_s"]
        t0 = np.array([initial["fluid_temperature_k"][n] for n in names])
        w0 = np.array([initial["wall_temperature_k"][n] for n in names])
        flow, work, zeros = refs.switched_reference(
            time_s, m0, head, inertia, resistance, rho, commands
        )
        if kind == "modal":
            fluid, wall = refs.modal_temperatures(
                time_s,
                t0,
                w0,
                m0,
                rho,
                cp,
                area,
                lengths[0],
                losses[0],
                params[0]["wall_capacity_j_k"],
                params[0]["fluid_wall_ua_w_k"],
                params[0]["ambient_ua_w_k"],
                cfg["ambient_temperature_k"],
            )
        else:
            fluid = t0.copy()
            current = m0
            stamp = 0.0
            active = head
            for when, new_head in commands + [(time_s, None)]:
                if when > time_s:
                    break
                fluid = refs.adiabatic_temperatures(
                    when - stamp, fluid, current, active, rho, cp, area, lengths, losses
                )
                current, _, _ = refs.flow_and_work(
                    when - stamp, current, active, inertia, resistance, rho
                )
                stamp = when
                if new_head is not None:
                    active = new_head
            if stamp < time_s:
                fluid = refs.adiabatic_temperatures(
                    time_s - stamp, fluid, current, active, rho, cp, area, lengths, losses
                )
            wall = w0
    for command in spec["timeline"]:
        if command["t"] < time_s or (command["t"] == time_s and phase == "head_changed"):
            heads = np.array([command["heads_pa"][n] for n in names])
    dm = (float(sum(heads)) - resistance * flow * abs(flow)) / inertia
    drops = lengths / area * dm + losses / (2 * rho * area**2) * flow * abs(flow) - heads
    pressure = np.r_[
        cfg["reference"]["pressure_pa"], cfg["reference"]["pressure_pa"] - np.cumsum(drops)[:-1]
    ]
    return float(flow), float(work), fluid, wall, pressure, zeros


def load(folder):
    manifest = json.loads((folder / "manifest.json").read_text())
    for name, digest in manifest["artifacts"].items():
        if hashlib.sha256((folder / name).read_bytes()).hexdigest() != digest:
            raise ValueError("Circulation bundle hash mismatch")
    rows = json.loads((folder / "trajectory.json").read_text())
    payload = json.loads((folder / "checkpoint.json").read_text())
    model = CirculationLoop.from_snapshot(payload)
    if model.snapshot() != payload:
        raise ValueError("Circulation checkpoint roundtrip mismatch")
    if any(model.observation()[k] != rows[-1][k] for k in model.observation()):
        raise ValueError("Checkpoint/final row mismatch")
    return rows, model


def assess(spec, kind, folder, budget):
    rows, model = load(folder)
    metrics = dict(
        mass_flow_abs_kg_s=0.0,
        temperature_abs_k=0.0,
        work_abs_j=0.0,
        energy_abs_j=0.0,
        local_energy_abs_w=0.0,
        pressure_cycle_abs_pa=0.0,
        pressure_abs_pa=0.0,
        event_time_abs_s=0.0,
    )
    names = model.names
    for row in rows:
        flow, work, fluid, wall, pressure, _ = expected(spec, kind, row["time_s"], row["phase"])
        metrics["mass_flow_abs_kg_s"] = max(
            metrics["mass_flow_abs_kg_s"], abs(row["mass_flow_kg_s"] - flow)
        )
        metrics["temperature_abs_k"] = max(
            metrics["temperature_abs_k"],
            *(abs(row["fluid_temperature_k"][n] - v) for n, v in zip(names, fluid)),
            *(abs(row["wall_temperature_k"][n] - v) for n, v in zip(names, wall)),
        )
        metrics["work_abs_j"] = max(metrics["work_abs_j"], abs(row["external_head_work_j"] - work))
        metrics["energy_abs_j"] = max(metrics["energy_abs_j"], abs(row["energy_residual_j"]))
        metrics["local_energy_abs_w"] = max(
            metrics["local_energy_abs_w"],
            *(abs(v) for v in row["local_energy_residual_w"].values()),
        )
        metrics["pressure_cycle_abs_pa"] = max(
            metrics["pressure_cycle_abs_pa"], abs(row["pressure_cycle_residual_pa"])
        )
        metrics["pressure_abs_pa"] = max(
            metrics["pressure_abs_pa"],
            *(abs(row["ports"][n + ".left"]["pressure_pa"] - v) for n, v in zip(names, pressure)),
        )
        for i, n in enumerate(names):
            a = row["ports"][n + ".right"]
            b = row["ports"][names[(i + 1) % len(names)] + ".left"]
            if (
                a["mass_flow_kg_s"] + b["mass_flow_kg_s"] != 0
                or a["energy_flow_w"] + b["energy_flow_w"] != 0
                or a["pressure_pa"] != b["pressure_pa"]
            ):
                raise ValueError("Internal port fluxes do not cancel")
    report = json.loads((folder / "report.json").read_text())
    expected_zeros = expected(spec, kind, spec["duration_s"])[-1]
    actual_zeros = [e["t"] for e in report["events"] if e["kind"] == "flow_zero"]
    event_count = len(expected_zeros) == len(actual_zeros)
    if event_count and actual_zeros:
        metrics["event_time_abs_s"] = max(abs(a - b) for a, b in zip(actual_zeros, expected_zeros))
    checks = {k: bool(v <= budget[k]) for k, v in metrics.items()}
    checks.update(completed=bool(report["numerical_acceptance"]), event_count=event_count)
    restart = model.advance(budget["restart_extension_s"], **spec["solver"])
    ref = expected(spec, kind, restart["time_s"])
    restart_error = max(
        *(abs(restart["fluid_temperature_k"][n] - v) for n, v in zip(names, ref[2])),
        *(abs(restart["wall_temperature_k"][n] - v) for n, v in zip(names, ref[3])),
    )
    checks["restart"] = bool(
        restart_error <= budget["temperature_abs_k"]
        and abs(restart["mass_flow_kg_s"] - ref[0]) <= budget["mass_flow_abs_kg_s"]
    )
    return dict(
        path=str(folder),
        accepted=all(checks.values()),
        checks=checks,
        metrics={k: float(v) for k, v in metrics.items()},
        restart_error_k=float(restart_error),
    ), rows


def qualify(output, budget_path):
    output.mkdir()
    budget = json.loads(budget_path.read_text())
    source = _source_fingerprint()
    write(output / "budget.json", budget)
    entries = []
    integrations = 0
    for spec, kind in cases():
        run_circulation_experiment(spec, output / spec["name"])
        base, rows = assess(spec, kind, output / spec["name"], budget)
        integrations += 1
        tight = deepcopy(spec)
        tight["name"] += "_tight"
        tight["solver"].update(rtol=1e-11, max_step_s=0.025, atol_scale=0.01)
        run_circulation_experiment(tight, output / tight["name"])
        fine, finerows = assess(tight, kind, output / tight["name"], budget)
        integrations += 1
        temporal = dict(time_flow_abs_kg_s=0.0, time_temperature_abs_k=0.0)
        if [(r["time_s"], r["phase"]) for r in rows] != [
            (r["time_s"], r["phase"]) for r in finerows
        ]:
            raise ValueError("Temporal sampling mismatch")
        for a, b in zip(rows, finerows):
            temporal["time_flow_abs_kg_s"] = max(
                temporal["time_flow_abs_kg_s"], abs(a["mass_flow_kg_s"] - b["mass_flow_kg_s"])
            )
            temporal["time_temperature_abs_k"] = max(
                temporal["time_temperature_abs_k"],
                *(
                    abs(a[field][n] - b[field][n])
                    for field in ("fluid_temperature_k", "wall_temperature_k")
                    for n in a[field]
                ),
            )
        accepted = (
            base["accepted"]
            and fine["accepted"]
            and all(v <= budget[k] for k, v in temporal.items())
        )
        entries.append(
            dict(
                id=spec["name"], accepted=bool(accepted), base=base, tight=fine, time_error=temporal
            )
        )
        print(json.dumps(dict(case=spec["name"], accepted=bool(accepted))), flush=True)
    for spec, kind in cases()[:2]:
        spec["name"] = "bdf_" + spec["name"]
        spec["solver"].update(method="BDF", rtol=1e-11, atol_scale=0.01)
        run_circulation_experiment(spec, output / spec["name"])
        entry, _ = assess(spec, kind, output / spec["name"], budget)
        integrations += 1
        entries.append(dict(id=spec["name"], **entry))
    if source != _source_fingerprint():
        raise RuntimeError("Source changed during circulation qualification")
    result = dict(
        schema_version=1,
        source_sha256=source,
        budget_sha256=hashlib.sha256(budget_path.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        reference_sha256=hashlib.sha256(Path(refs.__file__).read_bytes()).hexdigest(),
        cases=entries,
        integrations=integrations,
        all_cases_accepted=all(e["accepted"] for e in entries),
        validated=False,
        plant_referenced=False,
        scope="Fixed reduced constant-property single-cycle cases, ideal body head; no actual pump, compressible network or plant validation",
    )
    write(output / "verification.json", result)
    write(
        output / "manifest.json",
        dict(
            source_sha256=source,
            sha256={
                str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in output.rglob("*")
                if p.is_file()
            },
        ),
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--budget", type=Path, default=Path("validation/ai05_circulation_budget_v1.json")
    )
    args = parser.parse_args()
    result = qualify(args.output, args.budget)
    print(
        json.dumps(
            dict(
                all_cases_accepted=result["all_cases_accepted"], integrations=result["integrations"]
            )
        )
    )
    return 0 if result["all_cases_accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
