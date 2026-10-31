"""Traceable gas-cushion experiments with algebraic or inertial transfer."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import time
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np

from core.network import NetworkError, number
from physics.hydraulics.gas_storage_pair import CONFIG_FIELDS, PASSPORT, GasStoragePair
from physics.hydraulics.inertial_storage_pair import CONFIG_FIELDS as INERTIAL_FIELDS
from physics.hydraulics.inertial_storage_pair import PASSPORT as INERTIAL_PASSPORT
from physics.hydraulics.inertial_storage_pair import InertialStoragePair
from sdk.experiment import _source_fingerprint
from sdk.network_experiment import write


def prepare(spec):
    inertial = isinstance(spec, dict) and spec.get("engine") == INERTIAL_PASSPORT.model_id
    fields = {
        "schema_version",
        "engine",
        "name",
        "model",
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
        or spec["engine"] not in (PASSPORT.model_id, INERTIAL_PASSPORT.model_id)
        or not isinstance(spec["name"], str)
        or not spec["name"].strip()
    ):
        raise NetworkError("Exact gas storage experiment schema 1 required")
    if not isinstance(spec["model"], dict) or set(spec["model"]) != (
        INERTIAL_FIELDS if inertial else CONFIG_FIELDS
    ):
        raise NetworkError("Exact gas storage model fields required")
    model = (InertialStoragePair if inertial else GasStoragePair)(**spec["model"])
    duration = number(spec["duration_s"], "duration_s", positive=True)
    step = number(spec["step_s"], "step_s", positive=True)
    if duration > 86400 or duration / step > 10000:
        raise NetworkError("Storage experiment resource limits exceeded")
    solver = spec["solver"]
    if (
        not isinstance(solver, dict)
        or set(solver) != {"method", "rtol", "max_step_s", "atol_scale"}
        or solver["method"] not in ("Radau", "BDF")
    ):
        raise NetworkError("Explicit supported storage solver required")
    solver = {k: (v if k == "method" else number(v, k, positive=True)) for k, v in solver.items()}
    if not 1e-12 <= solver["rtol"] <= 1e-2 or not 1e-6 <= solver["atol_scale"] <= 1:
        raise NetworkError("Unsupported storage tolerances")
    if not isinstance(spec["timeline"], list) or len(spec["timeline"]) > 10000:
        raise NetworkError("Storage timeline list required")
    command_key = "resistance_pa_s2_kg2" if inertial else "conductance_m3_s_pa"
    commands = []
    previous = -1.0
    for command in spec["timeline"]:
        if not isinstance(command, dict) or set(command) != {"t", command_key}:
            raise NetworkError("Exact conductance command required")
        t = number(command["t"], "command time")
        c = number(command[command_key], command_key)
        if not previous < t <= duration or t < 0 or c < 0:
            raise NetworkError("Invalid storage command time or conductance")
        commands.append(dict(t=t, **{command_key: c}))
        previous = t
    result = deepcopy(spec)
    result.update(
        model=model.configuration,
        duration_s=duration,
        step_s=step,
        solver=solver,
        timeline=commands,
    )
    return result, model


def run_storage_experiment(spec, output, *, wall_limit_s=None):
    normalized, model = prepare(spec)
    if wall_limit_s is not None:
        wall_limit_s = number(wall_limit_s, "wall_limit_s", positive=True)
    output = Path(output)
    if output.exists():
        raise NetworkError("Storage output must be fresh")
    output.mkdir(parents=True)
    write(output / "input.json", normalized)
    initial = model.observation()
    write(
        output / "initialization.json",
        dict(
            mode="explicit-inventories-and-inertial-flow"
            if isinstance(model, InertialStoragePair)
            else "explicit-inventories-algebraic-flow",
            observation=initial,
        ),
    )
    source = _source_fingerprint()
    started = time.monotonic()
    original = model.derivatives

    def guarded(t, y):
        if wall_limit_s is not None and time.monotonic() - started > wall_limit_s:
            raise NetworkError("Storage wall deadline exceeded")
        return original(t, y)

    model.derivatives = guarded
    rows = [dict(phase="initial", **initial)]
    index = 0
    status = "completed"
    error = None
    horizon, step = normalized["duration_s"], normalized["step_s"]
    model.enable_continuation(horizon)
    samples = {min(horizon, k * step) for k in range(1, math.ceil(horizon / step) + 1)}
    grid = sorted(samples | {c["t"] for c in normalized["timeline"] if c["t"] > 0})

    def commands():
        nonlocal index
        while (
            index < len(normalized["timeline"])
            and normalized["timeline"][index]["t"] == model.time_s
        ):
            command = normalized["timeline"][index]
            if isinstance(model, InertialStoragePair):
                o = model.set_resistance(command["resistance_pa_s2_kg2"])
                phase = "resistance_changed"
            else:
                o = model.set_conductance(command["conductance_m3_s_pa"])
                phase = "conductance_changed"
            rows.append(dict(phase=phase, **o))
            index += 1

    try:
        commands()
        for target in grid:
            o = model.advance(target - model.time_s, **normalized["solver"])
            rows.append(dict(phase="sample" if target in samples else "control_boundary", **o))
            commands()
    except (ValueError, RuntimeError, ArithmeticError, np.linalg.LinAlgError) as exc:
        status = "failed"
        error = dict(type=type(exc).__name__, message=str(exc))
    if source != _source_fingerprint():
        status = "failed"
        error = dict(type="SourceChanged", message="Storage runtime changed during experiment")
    report = dict(
        schema_version=1,
        name=normalized["name"],
        status=status,
        numerical_acceptance=status == "completed",
        requested_duration_s=horizon,
        completed_time_s=model.time_s,
        accepted_commands=index,
        energy_absolute_j=max(abs(r["energy_residual_j"]) for r in rows),
        local_energy_absolute_w=max(
            abs(v) for r in rows for v in r["local_energy_residual_w"].values()
        ),
        events=deepcopy(model.events),
        error=error,
        wall_time_s=time.monotonic() - started,
        validated=False,
        plant_referenced=False,
        scope=(
            "Gas storage and ideal hydraulic inertance; no resolved pipe/contact/steam pressurizer/plant qualification"
            if isinstance(model, InertialStoragePair)
            else "Two constant-density liquid stores and ideal gas cushions; zero-inertia line; no steam pressurizer/plant qualification"
        ),
    )
    write(output / "trajectory.json", rows)
    write(output / "checkpoint.json", model.snapshot())
    write(output / "report.json", report)
    write(
        output / "manifest.json",
        dict(
            schema_version=1,
            source_sha256=source,
            configuration_sha256=model.configuration_sha256(),
            passport=asdict(
                INERTIAL_PASSPORT if isinstance(model, InertialStoragePair) else PASSPORT
            ),
            versions={k: importlib.metadata.version(k) for k in ("numpy", "scipy")},
            artifacts={
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in output.iterdir()
                if p.is_file()
            },
        ),
    )
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("scenario", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--wall-limit-s", type=float)
    args = p.parse_args(argv)
    r = run_storage_experiment(
        json.loads(args.scenario.read_text()), args.output, wall_limit_s=args.wall_limit_s
    )
    print(json.dumps(dict(status=r["status"], numerical_acceptance=r["numerical_acceptance"])))
    return 0 if r["numerical_acceptance"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
