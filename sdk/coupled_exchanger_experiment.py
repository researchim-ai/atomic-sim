"""Simultaneous two-cycle pump/exchanger experiments with atomic control and restart."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import time
from copy import deepcopy
from pathlib import Path

from core.network import NetworkError, number
from physics.thermal.coupled_pump_exchanger import (
    PASSPORT,
    CoupledPumpExchanger,
    controls,
    steady_coupled_exchanger,
)
from sdk.experiment import _source_fingerprint
from sdk.network_experiment import write
from sdk.pump_experiment import prepare as prepare_pump


def prepare(spec):
    fields = {
        "schema_version",
        "engine",
        "name",
        "loops",
        "exchanger",
        "initialization",
        "energy_tolerance_j",
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
        or not isinstance(spec["loops"], dict)
        or set(spec["loops"]) != {"a", "b"}
        or spec["initialization"] not in ({"mode": "given"}, {"mode": "steady"})
    ):
        raise NetworkError("Exact coupled exchanger experiment schema 1 required")
    raw_fields = {"loop", "pump", "initial_omega_rad_s", "motor_torque_nm"}
    normalized_loops, initials = {}, {}
    for side in ("a", "b"):
        raw = spec["loops"][side]
        if not isinstance(raw, dict) or set(raw) != raw_fields:
            raise NetworkError("Exact initial pump-cycle definition required")
        pump_spec = dict(
            schema_version=1,
            engine="displacement-pump-circulation-loop",
            name=spec["name"],
            **raw,
            duration_s=spec["duration_s"],
            step_s=spec["step_s"],
            solver=spec["solver"],
            timeline=[],
        )
        normalized, model = prepare_pump(pump_spec)
        normalized_loops[side] = {key: normalized[key] for key in raw_fields}
        initials[side] = model.snapshot()
    constructor = (
        steady_coupled_exchanger
        if spec["initialization"]["mode"] == "steady"
        else CoupledPumpExchanger
    )
    model = constructor(initials, spec["exchanger"], energy_tolerance_j=spec["energy_tolerance_j"])
    timeline = spec["timeline"]
    if not isinstance(timeline, list) or len(timeline) > 10000:
        raise NetworkError("Coupled control timeline resource limit exceeded")
    commands, previous = [], -1.0
    for event in timeline:
        if not isinstance(event, dict) or set(event) != {"t", "controls"}:
            raise NetworkError("Exact two-motor control command required")
        stamp = number(event["t"], "command time")
        if not previous < stamp <= normalized["duration_s"] or stamp < 0:
            raise NetworkError("Coupled commands require increasing in-range times")
        commands.append(dict(t=stamp, controls=controls(event["controls"])))
        previous = stamp
    result = deepcopy(spec)
    result.update(
        loops=normalized_loops,
        duration_s=normalized["duration_s"],
        step_s=normalized["step_s"],
        solver=normalized["solver"],
        timeline=commands,
        energy_tolerance_j=model.energy_tolerance_j,
    )
    return result, model


def restore(spec, payload):
    normalized, fresh = prepare(spec)
    model = CoupledPumpExchanger.from_snapshot(payload)
    if model.configuration != fresh.configuration or model.time_s > normalized["duration_s"]:
        raise NetworkError("Coupled restart differs from scenario configuration/horizon")
    expected = [e for e in normalized["timeline"] if e["t"] <= model.time_s]
    actual = [dict(t=e["t"], controls=e["after"]) for e in model.events]
    if expected != actual and not (model.time_s == 0 and not actual):
        raise NetworkError("Coupled restart differs from executed control prefix")
    return normalized, model


def run_coupled_exchanger_experiment(spec, output, *, checkpoint=None, wall_limit_s=None):
    normalized, model = prepare(spec) if checkpoint is None else restore(spec, checkpoint)
    if wall_limit_s is not None:
        wall_limit_s = number(wall_limit_s, "wall_limit_s", positive=True)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write(output / "input.json", normalized)
    write(output / "initialization.json", model.initialization_report())
    if checkpoint is not None:
        write(output / "restart_input.json", checkpoint)
    started, source = time.monotonic(), _source_fingerprint()
    start_time = model.time_s
    original = model.derivatives

    def guarded(t, y):
        if wall_limit_s is not None and time.monotonic() - started > wall_limit_s:
            raise NetworkError("Coupled experiment wall deadline exceeded")
        return original(t, y)

    model.derivatives = guarded
    rows = [dict(model.observation(), phase="initial" if checkpoint is None else "restart")]
    applied = len(model.events)
    status, error, steps = "completed", None, 0
    try:
        while model.time_s < normalized["duration_s"] or (
            applied < len(normalized["timeline"])
            and normalized["timeline"][applied]["t"] == model.time_s
        ):
            backup = model.snapshot()
            previous_applied = applied
            pending = []
            try:
                if wall_limit_s is not None and time.monotonic() - started > wall_limit_s:
                    raise NetworkError("Coupled experiment wall deadline exceeded")
                index = math.floor(model.time_s / normalized["step_s"]) + 1
                end = min(normalized["duration_s"], index * normalized["step_s"])
                if end <= model.time_s and model.time_s < normalized["duration_s"]:
                    end = min(normalized["duration_s"], (index + 1) * normalized["step_s"])
                if end <= model.time_s and model.time_s < normalized["duration_s"]:
                    raise NetworkError("Coupled output clock cannot advance")
                while (
                    applied < len(normalized["timeline"])
                    and normalized["timeline"][applied]["t"] <= end
                ):
                    event = normalized["timeline"][applied]
                    if event["t"] > model.time_s:
                        model.advance(event["t"] - model.time_s, **normalized["solver"])
                    pending.append(dict(model.observation(), phase="before_control"))
                    model.set_controls(event["controls"])
                    pending.append(dict(model.observation(), phase="after_control"))
                    applied += 1
                if model.time_s < end:
                    model.advance(end - model.time_s, **normalized["solver"])
                pending.append(dict(model.observation(), phase="output"))
            except Exception:
                model = CoupledPumpExchanger.from_snapshot(backup)
                applied = previous_applied
                raise
            rows.extend(pending)
            steps += 1
    except (NetworkError, ValueError, RuntimeError, ArithmeticError) as exc:
        status, error = "failed", dict(type=type(exc).__name__, message=str(exc))
    if source != _source_fingerprint():
        status, error = (
            "failed",
            dict(type="SourceChanged", message="Runtime changed during coupled run"),
        )
    report = dict(
        schema_version=1,
        status=status,
        error=error,
        name=normalized["name"],
        start_time_s=start_time,
        requested_duration_s=normalized["duration_s"],
        completed_time_s=model.time_s,
        accepted_output_steps=steps,
        events=model.events,
        numerical_acceptance=status == "completed",
        total_energy_absolute_j=max(abs(r["total_energy_residual_j"]) for r in rows),
        shared_energy_absolute_j=max(abs(r["shared_energy_residual_j"]) for r in rows),
        reciprocal_heat_absolute_w=max(abs(r["reciprocal_heat_residual_w"]) for r in rows),
        wall_time_s=time.monotonic() - started,
        validated=False,
        plant_referenced=False,
        scope="Two constant-property displacement-pump cycles and shared finite-capacity wall; no whole-plant qualification",
    )
    write(output / "trajectory.json", rows)
    write(output / "checkpoint.json", model.snapshot())
    write(output / "report.json", report)
    write(
        output / "manifest.json",
        dict(
            schema_version=1,
            source_sha256=source,
            passport=PASSPORT.as_dict(),
            python=platform.python_version(),
            versions={p: importlib.metadata.version(p) for p in ("numpy", "scipy")},
            artifacts={
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in output.iterdir()
                if p.is_file()
            },
        ),
    )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--wall-limit-s", type=float)
    args = parser.parse_args(argv)
    report = run_coupled_exchanger_experiment(
        json.loads(args.scenario.read_text()),
        args.output,
        checkpoint=json.loads(args.resume.read_text()) if args.resume else None,
        wall_limit_s=args.wall_limit_s,
    )
    print(
        json.dumps(
            dict(status=report["status"], numerical_acceptance=report["numerical_acceptance"])
        )
    )
    return 0 if report["numerical_acceptance"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
