"""Reproducible piecewise-constant two-stream exchanger campaigns and restart."""

import argparse
import hashlib
import json
import platform
import time
from copy import deepcopy
from pathlib import Path

from core.network import NetworkError, number
from physics.thermal.two_stream_exchanger import PASSPORT, TwoStreamExchanger, controls
from sdk.experiment import _source_fingerprint
from sdk.network_experiment import write


def prepare(spec):
    fields = {
        "schema_version",
        "engine",
        "name",
        "model",
        "controls",
        "initialization",
        "duration_s",
        "step_s",
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
        raise NetworkError("Exact named exchanger experiment schema 1 required")
    duration = number(spec["duration_s"], "duration_s", positive=True)
    step = number(spec["step_s"], "step_s", positive=True)
    if duration > 86400 or duration / step > 10000:
        raise NetworkError("Exchanger output resource limit exceeded")
    timeline = spec["timeline"]
    if not isinstance(timeline, list) or len(timeline) > 10000:
        raise NetworkError("Exchanger timeline limit exceeded")
    commands, previous = [], -1.0
    for event in timeline:
        if not isinstance(event, dict) or set(event) != {"t", "controls"}:
            raise NetworkError("Exact exchanger timeline command required")
        stamp = number(event["t"], "command time")
        if not previous < stamp <= duration or stamp < 0:
            raise NetworkError("Exchanger commands require strictly increasing in-range times")
        commands.append(dict(t=stamp, controls=controls(event["controls"])))
        previous = stamp
    model = TwoStreamExchanger(spec["model"], spec["controls"], spec["initialization"])
    normalized = deepcopy(spec)
    normalized.update(
        model=model.configuration,
        controls=model.boundary_controls,
        duration_s=duration,
        step_s=step,
        timeline=commands,
    )
    return normalized, model


def restore(spec, payload):
    normalized, fresh = prepare(spec)
    model = TwoStreamExchanger.from_snapshot(payload)
    baseline = fresh.snapshot()
    for key in ("configuration", "initialization", "initial_controls"):
        if payload[key] != baseline[key]:
            raise NetworkError("Exchanger restart differs from experiment definition")
    if model.time_s > normalized["duration_s"]:
        raise NetworkError("Exchanger restart exceeds experiment horizon")
    expected = [e for e in normalized["timeline"] if e["t"] <= model.time_s]
    applied = [dict(t=e["t"], controls=e["after"]) for e in model.events]
    # The untouched t=0 initial checkpoint is also a valid starting point.
    if applied != expected and not (model.time_s == 0 and not applied):
        raise NetworkError("Exchanger restart control history differs from timeline")
    return normalized, model


def run_exchanger_experiment(spec, output, *, checkpoint=None, wall_limit_s=None):
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
                    raise NetworkError("Exchanger wall deadline exceeded")
                # The global output grid is preserved when resuming off-grid.
                next_index = int(model.time_s // normalized["step_s"]) + 1
                end = min(normalized["duration_s"], next_index * normalized["step_s"])
                if end <= model.time_s and model.time_s < normalized["duration_s"]:
                    next_index += 1
                    end = min(normalized["duration_s"], next_index * normalized["step_s"])
                if end <= model.time_s and model.time_s < normalized["duration_s"]:
                    raise NetworkError("Exchanger output clock cannot advance")
                while (
                    applied < len(normalized["timeline"])
                    and normalized["timeline"][applied]["t"] <= end
                ):
                    event = normalized["timeline"][applied]
                    if event["t"] > model.time_s:
                        model.advance(event["t"] - model.time_s)
                    pending.append(dict(model.observation(), phase="before_control"))
                    model.set_controls(event["controls"])
                    pending.append(dict(model.observation(), phase="after_control"))
                    applied += 1
                if model.time_s < end:
                    model.advance(end - model.time_s)
                pending.append(dict(model.observation(), phase="output"))
                if wall_limit_s is not None and time.monotonic() - started > wall_limit_s:
                    raise NetworkError("Exchanger wall deadline exceeded")
            except Exception:
                model = TwoStreamExchanger.from_snapshot(backup)
                applied = previous_applied
                raise
            rows.extend(pending)
            steps += 1
    except (NetworkError, ValueError, RuntimeError, ArithmeticError) as exc:
        status, error = "failed", dict(type=type(exc).__name__, message=str(exc))
    if source != _source_fingerprint():
        status, error = (
            "failed",
            dict(type="SourceChanged", message="Runtime changed during experiment"),
        )
    report = dict(
        schema_version=1,
        status=status,
        name=normalized["name"],
        start_time_s=start_time,
        completed_time_s=model.time_s,
        requested_duration_s=normalized["duration_s"],
        accepted_output_steps=steps,
        error=error,
        events=model.events,
        energy_absolute_j=max(abs(r["energy_residual_j"]) for r in rows),
        local_energy_absolute_j=max(
            abs(v) for r in rows for row in r["cell_energy_residual_j"] for v in row
        ),
        port_power_absolute_w=max(abs(r["port_power_residual_w"]) for r in rows),
        numerical_acceptance=status == "completed",
        wall_time_s=time.monotonic() - started,
        validated=False,
        plant_referenced=False,
        scope="Constant-property mixed-cell thermal exchanger; prescribed flows, no hydraulic or plant qualification",
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
    report = run_exchanger_experiment(
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
