"""Auditable reduced circulation-loop experiments with transactional head commands."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from core.network import NetworkError, number
from physics.hydraulics.circulation_loop import PASSPORT, CirculationLoop, steady_circulation
from sdk.experiment import _source_fingerprint
from sdk.network_experiment import write


def prepare(spec):
    fields = {
        "schema_version",
        "engine",
        "name",
        "model",
        "initialization",
        "duration_s",
        "step_s",
        "solver",
        "timeline",
    }
    if (
        not isinstance(spec, dict)
        or set(spec) != fields
        or type(spec["schema_version"]) is not int
        or spec["schema_version"] != 1
        or spec["engine"] != PASSPORT.model_id
        or not isinstance(spec["name"], str)
        or not spec["name"].strip()
    ):
        raise NetworkError("Exact circulation experiment schema 1 and engine required")
    model_fields = {"topology", "fluid", "reference", "ambient_temperature_k", "energy_tolerance_j"}
    if not isinstance(spec["model"], dict) or set(spec["model"]) != model_fields:
        raise NetworkError("Unknown or missing circulation model fields")
    init = spec["initialization"]
    if not isinstance(init, dict):
        raise NetworkError("Explicit initialization mode required")
    if init.get("mode") == "steady" and set(init) == {"mode"}:
        model = steady_circulation(**spec["model"])
        normalized_init = dict(mode="steady")
    elif init.get("mode") == "given" and set(init) == {
        "mode",
        "mass_flow_kg_s",
        "fluid_temperature_k",
        "wall_temperature_k",
    }:
        model = CirculationLoop(
            **spec["model"],
            initial_mass_flow_kg_s=init["mass_flow_kg_s"],
            initial_temperature_k=init["fluid_temperature_k"],
            initial_wall_temperature_k=init["wall_temperature_k"],
        )
        payload = model.snapshot()
        normalized_init = dict(
            mode="given",
            mass_flow_kg_s=model.initial_flow,
            fluid_temperature_k=payload["initial_temperature_k"],
            wall_temperature_k=payload["initial_wall_temperature_k"],
        )
    else:
        raise NetworkError("Unknown or mixed initialization mode")
    duration = number(spec["duration_s"], "duration_s", positive=True)
    step = number(spec["step_s"], "step_s", positive=True)
    if duration > 86400 or duration / step > 10000:
        raise NetworkError("Circulation experiment resource limits exceeded")
    solver = spec["solver"]
    if (
        not isinstance(solver, dict)
        or set(solver) != {"method", "rtol", "max_step_s", "atol_scale"}
        or solver["method"] not in ("Radau", "BDF")
    ):
        raise NetworkError("Explicit supported solver policy required")
    solver = {k: (v if k == "method" else number(v, k, positive=True)) for k, v in solver.items()}
    if not 1e-12 <= solver["rtol"] <= 1e-2 or not 1e-6 <= solver["atol_scale"] <= 1:
        raise NetworkError("Solver tolerance outside supported range")
    if not isinstance(spec["timeline"], list) or len(spec["timeline"]) > 10000:
        raise NetworkError("Timeline list required")
    timeline = []
    previous = -1.0
    for command in spec["timeline"]:
        if (
            not isinstance(command, dict)
            or set(command) != {"t", "heads_pa"}
            or not isinstance(command["heads_pa"], dict)
            or set(command["heads_pa"]) != set(model.names)
        ):
            raise NetworkError("Exact head timeline command required")
        stamp = number(command["t"], "command time")
        if not 0 <= stamp <= duration or stamp <= previous:
            raise NetworkError("Timeline must be strictly ordered in the experiment horizon")
        previous = stamp
        timeline.append(
            dict(t=stamp, heads_pa={n: number(command["heads_pa"][n], n) for n in model.names})
        )
    normalized = dict(
        schema_version=1,
        engine=PASSPORT.model_id,
        name=spec["name"],
        model=dict(
            topology=model.topology,
            fluid=model.fluid,
            reference=model.reference,
            ambient_temperature_k=model.ambient_temperature_k,
            energy_tolerance_j=model.energy_tolerance_j,
        ),
        initialization=normalized_init,
        duration_s=duration,
        step_s=step,
        solver=solver,
        timeline=timeline,
    )
    return normalized, model


def run_circulation_experiment(spec, output, *, wall_limit_s=None):
    normalized, model = prepare(spec)
    if wall_limit_s is not None:
        wall_limit_s = number(wall_limit_s, "wall_limit_s", positive=True)
    output = Path(output)
    output.mkdir(parents=True)
    write(output / "input.json", normalized)
    initial = model.observation()
    rate = model.derivatives(0, model._state)
    write(
        output / "initialization.json",
        dict(
            mode=normalized["initialization"]["mode"],
            observation=initial,
            momentum_rate_pa=float(model.inertia_total * rate[0]),
            fluid_energy_rates_w=(model.capacity * rate[1 : model.n + 1]).tolist(),
            wall_energy_rates_w=(
                model.wall_capacity * rate[model.n + 1 : 2 * model.n + 1]
            ).tolist(),
        ),
    )
    started = time.monotonic()
    source = _source_fingerprint()
    status = "completed"
    error = None
    original_derivatives = model.derivatives

    def guarded_derivatives(t, y):
        if wall_limit_s is not None and time.monotonic() - started > wall_limit_s:
            raise NetworkError("Circulation wall deadline exceeded during solver evaluation")
        return original_derivatives(t, y)

    model.derivatives = guarded_derivatives
    rows = [dict(phase="initial", **initial)]
    command_index = 0
    intervals = 0
    model.enable_continuation(normalized["duration_s"])
    sample_grid = {
        min(normalized["duration_s"], k * normalized["step_s"])
        for k in range(1, math.ceil(normalized["duration_s"] / normalized["step_s"]) + 1)
    }
    grid = sorted(sample_grid | {c["t"] for c in normalized["timeline"] if c["t"] > 0})

    def commands():
        nonlocal command_index
        while (
            command_index < len(normalized["timeline"])
            and normalized["timeline"][command_index]["t"] == model.time_s
        ):
            obs = model.set_heads(normalized["timeline"][command_index]["heads_pa"])
            rows.append(dict(phase="head_changed", **obs))
            command_index += 1

    try:
        commands()
        for target in grid:
            if wall_limit_s is not None and time.monotonic() - started > wall_limit_s:
                raise NetworkError("Circulation wall deadline exceeded")
            obs = model.advance(target - model.time_s, **normalized["solver"])
            rows.append(
                dict(phase="sample" if target in sample_grid else "control_boundary", **obs)
            )
            intervals += 1
            commands()
    except (NetworkError, ValueError, RuntimeError, ArithmeticError, np.linalg.LinAlgError) as exc:
        status = "failed"
        error = dict(type=type(exc).__name__, message=str(exc))
    if source != _source_fingerprint():
        status = "failed"
        error = dict(
            type="SourceChanged", message="Runtime source changed during circulation experiment"
        )
    report = dict(
        schema_version=1,
        name=normalized["name"],
        status=status,
        requested_duration_s=normalized["duration_s"],
        completed_time_s=model.time_s,
        accepted_intervals=intervals,
        accepted_commands=command_index,
        energy_absolute_j=max(abs(r["energy_residual_j"]) for r in rows),
        local_energy_absolute_w=max(
            abs(v) for r in rows for v in r["local_energy_residual_w"].values()
        ),
        pressure_cycle_absolute_pa=max(abs(r["pressure_cycle_residual_pa"]) for r in rows),
        wall_time_s=time.monotonic() - started,
        events=model.events,
        error=error,
        numerical_acceptance=status == "completed",
        validated=False,
        plant_referenced=False,
        scope="Reduced constant-property single cycle; ideal head source, no real pump/shaft/phase/acoustics",
    )
    write(output / "trajectory.json", rows)
    write(output / "checkpoint.json", model.snapshot())
    write(output / "report.json", report)
    manifest = dict(
        schema_version=1,
        source_sha256=source,
        passport=asdict(PASSPORT),
        versions={k: importlib.metadata.version(k) for k in ("numpy", "scipy")},
        python=platform.python_version(),
        configuration_sha256=model.configuration_sha256(),
        artifacts={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in output.iterdir()
            if p.is_file()
        },
    )
    write(output / "manifest.json", manifest)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wall-limit-s", type=float)
    args = parser.parse_args(argv)
    spec = json.loads(args.scenario.read_text(encoding="utf-8"))
    report = run_circulation_experiment(spec, args.output, wall_limit_s=args.wall_limit_s)
    print(
        json.dumps(
            dict(status=report["status"], numerical_acceptance=report["numerical_acceptance"])
        )
    )
    return 0 if report["numerical_acceptance"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
