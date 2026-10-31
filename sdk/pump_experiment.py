"""Reproducible displacement-pump experiments with explicit motor torque commands."""

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
from physics.hydraulics.pump_loop import PASSPORT, PumpCirculationLoop
from physics.hydraulics.valved_pump_loop import PASSPORT as VALVE_PASSPORT
from physics.hydraulics.valved_pump_loop import ValvedPumpLoop
from sdk.circulation_experiment import prepare as prepare_loop
from sdk.experiment import _source_fingerprint
from sdk.network_experiment import write


def prepare(spec):
    valved = isinstance(spec, dict) and spec.get("engine") == VALVE_PASSPORT.model_id
    if (
        not isinstance(spec, dict)
        or set(spec)
        != {
            "schema_version",
            "engine",
            "name",
            "loop",
            "pump",
            "initial_omega_rad_s",
            "motor_torque_nm",
            "duration_s",
            "step_s",
            "solver",
            "timeline",
        }
        | ({"valve"} if valved else set())
        or type(spec["schema_version"]) is not int
        or spec["schema_version"] != 1
        or spec["engine"] not in (PASSPORT.model_id, VALVE_PASSPORT.model_id)
    ):
        raise NetworkError("Exact pump experiment schema 1 required")
    if not isinstance(spec["loop"], dict) or set(spec["loop"]) != {"model", "initialization"}:
        raise NetworkError("Explicit initial circulation model required")
    if (
        not isinstance(spec["loop"]["initialization"], dict)
        or spec["loop"]["initialization"].get("mode") != "given"
    ):
        raise NetworkError(
            "Pump experiment requires explicit given initialization; use steady_pump_loop for equilibrium"
        )
    loop_spec = dict(
        schema_version=1,
        engine="incompressible-thermal-circulation-loop",
        name=spec["name"],
        **spec["loop"],
        duration_s=spec["duration_s"],
        step_s=spec["step_s"],
        solver=spec["solver"],
        timeline=[],
    )
    normalized, loop = prepare_loop(loop_spec)
    constructor = ValvedPumpLoop if valved else PumpCirculationLoop
    model = constructor(
        loop.snapshot(),
        spec["pump"],
        spec["initial_omega_rad_s"],
        spec["motor_torque_nm"],
        **({"valve": spec["valve"]} if valved else {}),
    )
    if not isinstance(spec["timeline"], list) or len(spec["timeline"]) > 10000:
        raise NetworkError("Pump command list required")
    commands, previous = [], -1.0
    for c in spec["timeline"]:
        allowed = [{"t", "motor_torque_nm"}] + ([{"t", "valve_opening"}] if valved else [])
        if not isinstance(c, dict) or set(c) not in allowed:
            raise NetworkError("Exact supported pump/valve command required")
        t = number(c["t"], "command time")
        key = "valve_opening" if "valve_opening" in c else "motor_torque_nm"
        value = number(c[key], key)
        if key == "valve_opening" and not 0 <= value <= 1:
            raise NetworkError("Valve command outside [0,1]")
        if not previous < t <= normalized["duration_s"] or t < 0:
            raise NetworkError("Control command times must increase inside horizon")
        commands.append(dict(t=t, **{key: value}))
        previous = t
    result = deepcopy(spec)
    result.update(
        loop={k: normalized[k] for k in ("model", "initialization")},
        pump=model.pump,
        initial_omega_rad_s=model._omega0,
        motor_torque_nm=model._torque0,
        duration_s=normalized["duration_s"],
        step_s=normalized["step_s"],
        solver=normalized["solver"],
        timeline=commands,
    )
    if valved:
        result["valve"] = model.valve
    return result, model


def run_pump_experiment(spec, output, *, wall_limit_s=None):
    normalized, model = prepare(spec)
    if wall_limit_s is not None:
        wall_limit_s = number(wall_limit_s, "wall_limit_s", positive=True)
    output = Path(output)
    if output.exists():
        raise NetworkError("Pump output must be a fresh directory")
    output.mkdir(parents=True)
    write(output / "input.json", normalized)
    started, source = time.monotonic(), _source_fingerprint()
    original = model.derivatives

    def guarded(t, y):
        if wall_limit_s is not None and time.monotonic() - started > wall_limit_s:
            raise NetworkError("Pump experiment wall deadline exceeded")
        return original(t, y)

    model.derivatives = guarded
    rows = [dict(phase="initial", **model.observation())]
    status, error, command_index = "completed", None, 0
    horizon, step = normalized["duration_s"], normalized["step_s"]
    model.enable_continuation(horizon)
    samples = {min(horizon, k * step) for k in range(1, math.ceil(horizon / step) + 1)}
    grid = sorted(samples | {c["t"] for c in normalized["timeline"] if c["t"] > 0})

    def commands():
        nonlocal command_index
        while (
            command_index < len(normalized["timeline"])
            and normalized["timeline"][command_index]["t"] == model.time_s
        ):
            command = normalized["timeline"][command_index]
            if "valve_opening" in command:
                obs = model.set_valve_opening(command["valve_opening"])
                phase = "valve_target_changed"
            else:
                obs = model.set_motor_torque(command["motor_torque_nm"])
                phase = "motor_torque_changed"
            rows.append(dict(phase=phase, **obs))
            command_index += 1

    try:
        commands()
        for target in grid:
            obs = model.advance(target - model.time_s, **normalized["solver"])
            rows.append(dict(phase="sample" if target in samples else "control_boundary", **obs))
            commands()
    except (ValueError, RuntimeError, ArithmeticError, np.linalg.LinAlgError) as exc:
        status, error = "failed", dict(type=type(exc).__name__, message=str(exc))
    if source != _source_fingerprint():
        status, error = (
            "failed",
            dict(type="SourceChanged", message="Runtime changed during pump experiment"),
        )
    report = dict(
        schema_version=1,
        status=status,
        numerical_acceptance=status == "completed",
        requested_duration_s=horizon,
        completed_time_s=model.time_s,
        accepted_commands=command_index,
        energy_absolute_j=max(abs(r["total_energy_residual_j"]) for r in rows),
        shaft_energy_absolute_j=max(abs(r["shaft_energy_residual_j"]) for r in rows),
        events=deepcopy(model.events),
        error=error,
        wall_time_s=time.monotonic() - started,
        validated=False,
        plant_referenced=False,
        scope="Reduced constant-property displacement pump; no centrifugal map/cavitation/plant qualification",
    )
    write(output / "trajectory.json", rows)
    write(output / "checkpoint.json", model.snapshot())
    write(output / "report.json", report)
    write(
        output / "manifest.json",
        dict(
            schema_version=1,
            source_sha256=source,
            passport=asdict(VALVE_PASSPORT if isinstance(model, ValvedPumpLoop) else PASSPORT),
            configuration_sha256=model.configuration_sha256(),
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
    report = run_pump_experiment(
        json.loads(args.scenario.read_text()), args.output, wall_limit_s=args.wall_limit_s
    )
    print(
        json.dumps(
            dict(status=report["status"], numerical_acceptance=report["numerical_acceptance"])
        )
    )
    return 0 if report["numerical_acceptance"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
