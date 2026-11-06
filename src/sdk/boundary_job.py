"""Shared transactional output/control runner for models with physical-limit prefixes."""

import hashlib
import importlib.metadata
import math
import platform
import time
from pathlib import Path

from core.network import NetworkError, number
from core.solver_trace import summarize_solver_trace
from core.status import DomainError
from sdk.experiment import _source_fingerprint
from sdk.network_experiment import write


def run_boundary_job(
    spec,
    output,
    *,
    prepare,
    restore,
    model_class,
    passport,
    initial_artifacts,
    checkpoint=None,
    wall_limit_s=None,
):
    normalized, model = prepare(spec) if checkpoint is None else restore(spec, checkpoint)
    if wall_limit_s is not None:
        wall_limit_s = number(wall_limit_s, "wall_limit_s", positive=True)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write(output / "input.json", normalized)
    for name, value in initial_artifacts(model).items():
        write(output / name, value)
    if checkpoint is not None:
        write(output / "restart_input.json", checkpoint)
    start_time = model.time_s
    rows = [dict(model.observation(), phase="initial" if checkpoint is None else "restart")]
    started, source = time.monotonic(), _source_fingerprint()
    if hasattr(model, "derivatives"):
        original_rhs = model.derivatives

        def guarded(t, y):
            if wall_limit_s is not None and time.monotonic() - started > wall_limit_s:
                raise NetworkError("Boundary job wall deadline exceeded")
            return original_rhs(t, y)

        model.derivatives = guarded
    applied = len([e for e in model.events if e["kind"] == "controls"])
    status, error, boundary, steps = "completed", None, None, 0
    solver_trace = []
    try:
        while model.time_s < normalized["duration_s"] or (
            applied < len(normalized["timeline"])
            and normalized["timeline"][applied]["t"] == model.time_s
        ):
            backup = model.snapshot()
            saved_applied = applied
            pending = []
            pending_trace = []
            try:
                if wall_limit_s is not None and time.monotonic() - started > wall_limit_s:
                    raise NetworkError("Boundary job wall deadline exceeded")
                index = math.floor(model.time_s / normalized["step_s"]) + 1
                end = min(normalized["duration_s"], index * normalized["step_s"])
                if end <= model.time_s and model.time_s < normalized["duration_s"]:
                    end = min(normalized["duration_s"], (index + 1) * normalized["step_s"])
                if end <= model.time_s and model.time_s < normalized["duration_s"]:
                    raise NetworkError("Boundary job output clock cannot advance")
                while (
                    applied < len(normalized["timeline"])
                    and normalized["timeline"][applied]["t"] <= end
                ):
                    event = normalized["timeline"][applied]
                    if event["t"] > model.time_s:
                        advanced = model.advance_until_limit(
                            event["t"] - model.time_s, **normalized.get("solver", {})
                        )
                        if "solver" in advanced:
                            pending_trace.append(advanced["solver"])
                        boundary = advanced["boundary"]
                        if boundary is not None:
                            pending.append(dict(advanced["observation"], phase="domain_exit"))
                            break
                    pending.append(dict(model.observation(), phase="before_control"))
                    model.set_controls(event["controls"])
                    pending.append(dict(model.observation(), phase="after_control"))
                    applied += 1
                if boundary is None and model.time_s < end:
                    advanced = model.advance_until_limit(
                        end - model.time_s, **normalized.get("solver", {})
                    )
                    if "solver" in advanced:
                        pending_trace.append(advanced["solver"])
                    boundary = advanced["boundary"]
                    pending.append(
                        dict(advanced["observation"], phase="domain_exit" if boundary else "output")
                    )
                elif boundary is None:
                    pending.append(dict(model.observation(), phase="output"))
                if pending_trace:
                    summarize_solver_trace(
                        pending_trace, start_time_s=backup["time_s"], end_time_s=model.time_s
                    )
            except Exception:
                model = model_class.from_snapshot(backup)
                applied = saved_applied
                raise
            rows.extend(pending)
            solver_trace.extend(pending_trace)
            steps += 1
            if boundary is not None:
                status = "domain_exit"
                break
    except (NetworkError, DomainError, ValueError, RuntimeError, ArithmeticError) as exc:
        status, error = "failed", dict(type=type(exc).__name__, message=str(exc))
    if source != _source_fingerprint():
        status, error = (
            "failed",
            dict(type="SourceChanged", message="Runtime changed during boundary job"),
        )
    report = dict(
        schema_version=1,
        status=status,
        error=error,
        boundary=boundary,
        name=normalized["name"],
        start_time_s=start_time,
        requested_duration_s=normalized["duration_s"],
        completed_time_s=model.time_s,
        accepted_output_intervals=steps,
        events=model.events,
        numerical_acceptance=status == "completed",
        accepted_prefix=status in ("completed", "domain_exit"),
        energy_absolute_j=max(abs(r["energy_residual_j"]) for r in rows),
        power_absolute_w=max(
            max(abs(r.get("power_residual_w", 0.0)), abs(r.get("fluid_power_residual_w", 0.0)))
            for r in rows
        ),
        wall_time_s=time.monotonic() - started,
        validated=False,
        plant_referenced=False,
        scope=passport.applicability,
        solver_statistics=summarize_solver_trace(
            solver_trace, start_time_s=start_time, end_time_s=model.time_s
        ),
    )
    if "fluid_power_residual_w" in rows[0]:
        report["fluid_power_absolute_w"] = max(abs(r["fluid_power_residual_w"]) for r in rows)
    if "mass_residual_kg" in rows[0]:
        report["mass_absolute_kg"] = max(abs(r["mass_residual_kg"]) for r in rows)
    write(output / "trajectory.json", rows)
    write(output / "solver_trace.json", solver_trace)
    write(output / "checkpoint.json", model.snapshot())
    write(output / "report.json", report)
    write(
        output / "manifest.json",
        dict(
            schema_version=1,
            source_sha256=source,
            passport=passport.as_dict(),
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
