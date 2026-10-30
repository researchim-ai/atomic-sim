"""Воспроизводимый эксперимент открытого изобарического канала, отдельно от ВВЭР."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import time
from dataclasses import asdict, fields
from importlib import metadata
from pathlib import Path

from core.status import DomainError, SolverError, classify_model_error
from physics.hydraulics.isobaric import ChannelConfig, IsobaricChannel
from physics.hydraulics.wall_channel import WallChannel, WallConfig
from sdk.experiment import (
    ExperimentInterrupted,
    _number,
    _source_fingerprint,
    _write_json,
    experiment_intervals,
)


def validate_channel_experiment(spec):
    allowed = {
        "schema_version",
        "name",
        "engine",
        "duration_s",
        "step_s",
        "channel",
        "solver",
        "budgets",
        "timeline",
        "wall",
    }
    if not isinstance(spec, dict) or set(spec) - allowed:
        raise ValueError("Неизвестные поля эксперимента изобарического канала")
    if type(spec.get("schema_version")) is not int or spec["schema_version"] != 1:
        raise ValueError("Требуется schema_version: 1")
    if spec.get("engine") not in ("isobaric-channel", "isobaric-wall-channel"):
        raise ValueError("Требуется engine изобарического канала")
    if not isinstance(spec.get("name"), str) or not spec["name"].strip():
        raise ValueError("Нужно имя эксперимента")
    if not isinstance(spec.get("timeline", []), list) or spec.get("timeline", []):
        raise ValueError("Команды timeline не входят в паспорт изобарического канала")
    raw = spec.get("channel")
    if not isinstance(raw, dict) or set(raw) - {f.name for f in fields(ChannelConfig)}:
        raise ValueError("Неизвестная конфигурация канала")
    try:
        config = ChannelConfig(**raw)
    except TypeError as exc:
        raise ValueError("Неполная конфигурация канала") from exc
    channel = asdict(config)
    channel["initial_enthalpy_j_kg"] = list(config.initial_enthalpy_j_kg)
    wall = None
    if spec["engine"] == "isobaric-wall-channel":
        raw_wall = spec.get("wall")
        if not isinstance(raw_wall, dict) or set(raw_wall) - (
            {f.name for f in fields(WallConfig)} - {"channel"}
        ):
            raise ValueError("Нужна допустимая конфигурация стенки")
        try:
            wall_config = WallConfig(channel=config, **raw_wall)
        except TypeError as exc:
            raise ValueError("Неполная конфигурация стенки") from exc
        wall = asdict(wall_config)
        wall.pop("channel")
        wall["initial_temperature_k"] = list(wall_config.initial_temperature_k)
    elif "wall" in spec:
        raise ValueError("Стенка требует engine: isobaric-wall-channel")
    solver = spec.get("solver", {})
    if not isinstance(solver, dict) or set(solver) - {"rtol", "method", "max_step_s"}:
        raise ValueError("Неизвестные параметры solver канала")
    method = solver.get("method", "Radau")
    if method not in ("BDF", "Radau"):
        raise ValueError("Метод канала должен быть BDF или Radau")
    budgets = spec.get("budgets", {})
    if not isinstance(budgets, dict) or set(budgets) - {"mass_relative", "energy_relative"}:
        raise ValueError("Неизвестные бюджеты канала")
    result = {
        "schema_version": 1,
        "name": spec["name"],
        "engine": spec["engine"],
        "duration_s": _number(spec.get("duration_s"), "duration_s", 1e-9, 86400.0),
        "step_s": _number(spec.get("step_s", 5.0), "step_s", 1e-6, 5.0),
        "timeline": [],
        "channel": channel,
        "solver": {
            "method": method,
            "rtol": _number(solver.get("rtol", 1e-7), "rtol", 1e-12, 1e-2),
            "max_step_s": _number(solver.get("max_step_s", 1.0), "max_step_s", 1e-6, 5.0),
        },
        "budgets": {
            key: _number(budgets.get(key, default), key, 0.0, 1.0)
            for key, default in [("mass_relative", 1e-6), ("energy_relative", 1e-5)]
        },
    }
    if wall is not None:
        result["wall"] = wall
    return result


def run_channel_experiment(spec, output, *, wall_limit_s=None, cancel_file=None):
    normalized = validate_channel_experiment(spec)
    if wall_limit_s is not None:
        wall_limit_s = _number(wall_limit_s, "wall_limit_s", 1e-9, 604800.0)
    config = ChannelConfig(**normalized["channel"])
    if normalized["engine"] == "isobaric-wall-channel":
        channel = WallChannel(WallConfig(channel=config, **normalized["wall"]))
    else:
        channel = IsobaricChannel(config)
    return _run_channel_experiment(
        normalized, channel, output, wall_limit_s=wall_limit_s, cancel_file=cancel_file
    )


def _run_channel_experiment(normalized, channel, output, *, wall_limit_s=None, cancel_file=None):
    if wall_limit_s is not None:
        wall_limit_s = _number(wall_limit_s, "wall_limit_s", 1e-9, 604800.0)
    channel.enable_continuation(normalized["duration_s"])
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=False)
    _write_json(folder / "input.json", normalized)
    _write_json(folder / "checkpoint.json", channel.snapshot())
    manifest = {
        "schema_version": 1,
        "source_sha256": _source_fingerprint(),
        "input_sha256": hashlib.sha256((folder / "input.json").read_bytes()).hexdigest(),
        "passport": channel.passport.as_dict(),
        "python": platform.python_version(),
        "versions": {name: metadata.version(name) for name in ("numpy", "scipy")},
        "solver": normalized["solver"],
        "validated": False,
        "plant_referenced": False,
    }
    _write_json(folder / "manifest.json", manifest)
    start = time.monotonic()
    original = channel.derivatives

    def check_abort():
        if wall_limit_s is not None and time.monotonic() - start >= wall_limit_s:
            raise ExperimentInterrupted("wall_limit")
        if cancel_file is not None and Path(cancel_file).exists():
            raise ExperimentInterrupted("cancelled")

    def guarded(t, y):
        check_abort()
        return original(t, y)

    channel.derivatives = guarded
    extrema = {}

    def record(row, writer):
        writer.writerow(row)
        for key, value in row.items():
            value = float(value)
            bounds = extrema.setdefault(key, {"min": value, "max": value})
            bounds["min"], bounds["max"] = min(bounds["min"], value), max(bounds["max"], value)

    status, error, program_error, steps = "completed", None, None, 0
    off = normalized.get("wall", {}).get("power_off_time_s")
    events = [{"kind": "heater_off", "t": 0.0}] if off == 0 else []
    intervals = dict(normalized)
    if off is not None and 0 < off < normalized["duration_s"]:
        intervals["timeline"] = [{"t": off}]
    with (folder / "trajectory.csv").open("w", newline="") as stream:
        initial = channel.observation()
        writer = csv.DictWriter(stream, fieldnames=list(initial))
        writer.writeheader()
        record(initial, writer)
        try:
            for begin, end, _ in experiment_intervals(intervals):
                check_abort()
                row = channel.advance(end - begin, **normalized["solver"])
                steps += 1
                record(row, writer)
                stream.flush()
                _write_json(folder / "checkpoint.json", channel.snapshot())
                if off is not None and channel.time_s >= off and not events:
                    events.append({"kind": "heater_off", "t": off})
        except ExperimentInterrupted as exc:
            status = exc.reason
        except (DomainError, SolverError) as exc:
            status = "failed"
            error = {
                "class": classify_model_error(exc),
                "type": type(exc).__name__,
                "message": str(exc),
            }
        except KeyboardInterrupt:
            status = "cancelled"
        except Exception as exc:
            status, program_error = "program_error", exc
            error = {"class": "program", "type": type(exc).__name__, "message": str(exc)}
    final = channel.observation()
    max_mass = max(abs(extrema["mass_residual_kg"]["min"]), abs(extrema["mass_residual_kg"]["max"]))
    max_energy = max(
        abs(extrema["energy_residual_j"]["min"]), abs(extrema["energy_residual_j"]["max"])
    )
    balance = {
        "mass_relative": max_mass / max(abs(channel.initial_mass_kg), 1e-9),
        "energy_relative": max_energy / max(abs(channel.initial_energy_j), 1.0),
        "max_mass_residual_kg": max_mass,
        "max_energy_residual_j": max_energy,
        "final_mass_residual_kg": float(final["mass_residual_kg"]),
        "final_energy_residual_j": float(final["energy_residual_j"]),
        "normalization": "maximum sampled absolute residual / initial absolute inventory",
    }
    checks = {key: balance[key] <= limit for key, limit in normalized["budgets"].items()}
    report = {
        "schema_version": 1,
        "name": normalized["name"],
        "status": status,
        "model_id": channel.passport.model_id,
        "requested_duration_s": normalized["duration_s"],
        "completed_time_s": channel.time_s,
        "wall_time_s": time.monotonic() - start,
        "accepted_output_steps": steps,
        "balance": balance,
        "budget_checks": checks,
        "numerical_acceptance": status == "completed" and all(checks.values()),
        "validated": False,
        "plant_referenced": False,
        "error": error,
        "events": events,
        "extrema": extrema,
        "scope": {
            "model_id": channel.passport.model_id,
            "approximations": list(channel.passport.approximations),
        },
    }
    _write_json(folder / "report.json", report)
    (folder / "report.md").write_text(
        "# Channel experiment\n\nStatus: **%s**. Time: %g / %g s.\n\n"
        "Numerical acceptance: %s. Physical validation: not established.\n\n"
        "Component scope and approximations: see manifest passport; not full reactor qualification.\n"
        % (status, channel.time_s, normalized["duration_s"], report["numerical_acceptance"])
    )
    manifest["artifacts"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(folder.iterdir())
        if p.name != "manifest.json"
    }
    _write_json(folder / "manifest.json", manifest)
    if program_error is not None:
        raise program_error
    return report


def main(argv=None):
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wall-limit-s", type=float)
    parser.add_argument("--cancel-file", type=Path)
    args = parser.parse_args(argv)
    try:
        report = run_channel_experiment(
            yaml.safe_load(args.scenario.read_text()),
            args.output,
            wall_limit_s=args.wall_limit_s,
            cancel_file=args.cancel_file,
        )
    except (ValueError, DomainError, OSError, yaml.YAMLError) as exc:
        print(str(exc))
        return 2
    print(
        json.dumps(
            {key: report[key] for key in ("status", "completed_time_s", "numerical_acceptance")}
        )
    )
    return 0 if report["numerical_acceptance"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
