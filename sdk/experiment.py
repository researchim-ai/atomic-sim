"""Воспроизводимый пакет эксперимента: сценарий, траектория, баланс и checkpoint.

Отчёт о завершении не означает физической валидации модели. Лимит wall time
проверяется между шагами и в RHS; для блокирующего native-кода нужен supervisor.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from core.manifest import input_manifest
from core.status import DomainError, ModelLimitError, SolverError, classify_model_error
from plants.ledger import HorizonLedger
from sdk.api import create_session, export_snapshot, validate_commands


class ExperimentInterrupted(SolverError):
    """Контролируемая отмена: интегратор обязан откатить непринятый шаг."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _number(value, name: str, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("%s должен быть числом" % name)
    value = float(value)
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError("%s вне [%g, %g]" % (name, low, high))
    return value


def validate_experiment(spec: dict) -> dict:
    """Проверить весь вход до запуска, включая будущие команды и времена."""
    if not isinstance(spec, dict):
        raise ValueError("Эксперимент должен быть объектом")
    allowed = {
        "name",
        "schema_version",
        "engine",
        "model_id",
        "description",
        "duration_s",
        "step_s",
        "timeline",
        "solver",
        "budgets",
        "domain_boundary_tolerance_s",
    }
    if set(spec) - allowed:
        raise ValueError("Неизвестные поля эксперимента: %s" % sorted(set(spec) - allowed))
    if type(spec.get("schema_version")) is not int or spec["schema_version"] != 1:
        raise ValueError("Требуется schema_version: 1")
    if spec.get("engine") != "vver":
        raise ValueError("Движок эксперимента должен быть vver")
    if spec.get("model_id", "vver-lumped-loop") != "vver-lumped-loop":
        raise ValueError("Эксперимент требует другой паспорт")
    if not isinstance(spec.get("name"), str) or not spec["name"].strip():
        raise ValueError("Нужно непустое имя эксперимента")
    if "description" in spec and not isinstance(spec["description"], str):
        raise ValueError("Описание должно быть строкой")
    duration = _number(spec.get("duration_s"), "duration_s", 1e-9, 86400.0)
    step = _number(spec.get("step_s", 0.5), "step_s", 1e-6, 5.0)
    solver = spec.get("solver", {})
    if not isinstance(solver, dict) or set(solver) - {
        "rtol",
        "atol",
        "first_step_s",
        "continuation",
        "jacobian_relative_step",
        "method",
    }:
        raise ValueError("Неизвестные параметры solver BDF")
    if "continuation" in solver and type(solver["continuation"]) is not bool:
        raise ValueError("solver.continuation должен быть bool")
    validated_solver = {
        key: _number(solver.get(key, default), key, 1e-12, 1e-2)
        for key, default in (("rtol", 1e-5), ("atol", 1e-4))
    }
    if "method" in solver:
        if solver["method"] not in ("BDF", "Radau"):
            raise ValueError("solver.method должен быть BDF или Radau")
        validated_solver["method"] = solver["method"]
    if "first_step_s" in solver:
        validated_solver["first_step_s"] = _number(
            solver["first_step_s"], "first_step_s", 1e-12, 5.0
        )
    if "continuation" in solver:
        validated_solver["continuation"] = solver["continuation"]
    if "jacobian_relative_step" in solver:
        validated_solver["jacobian_relative_step"] = _number(
            solver["jacobian_relative_step"], "jacobian_relative_step", 1e-8, 1e-3
        )
    solver = validated_solver
    budgets = spec.get("budgets", {})
    if not isinstance(budgets, dict) or set(budgets) - {
        "energy_relative",
        "mass_relative",
        "boron_relative",
    }:
        raise ValueError("Неизвестный бюджет ошибки")
    budgets = {key: _number(value, key, 0.0, 1.0) for key, value in budgets.items()}
    timeline = spec.get("timeline", [])
    if not isinstance(timeline, list):
        raise ValueError("timeline должен быть списком")
    events = []
    for event in timeline:
        if not isinstance(event, dict) or "t" not in event:
            raise ValueError("Событие должно содержать t")
        stamp = _number(event["t"], "timeline.t", 0.0, duration)
        if stamp >= duration:
            raise ValueError("Событие должно произойти до конца расчёта")
        commands = validate_commands({key: value for key, value in event.items() if key != "t"})
        if not commands:
            raise ValueError("Событие не содержит команды")
        events.append({"t": stamp, **commands})
    # Стабильная сортировка сохраняет порядок одновременных команд.
    events.sort(key=lambda item: item["t"])
    result = {
        "schema_version": 1,
        "name": spec["name"],
        "engine": "vver",
        "model_id": "vver-lumped-loop",
        "description": spec.get("description", ""),
        "duration_s": duration,
        "step_s": step,
        "timeline": events,
        "solver": solver,
        "budgets": budgets,
    }
    if "domain_boundary_tolerance_s" in spec:
        result["domain_boundary_tolerance_s"] = _number(
            spec["domain_boundary_tolerance_s"], "domain_boundary_tolerance_s", 1e-8, 0.1
        )
    return result


def experiment_intervals(spec: dict):
    """Шаг вывода разрывается на каждом заданном времени команды."""
    clock = 0.0
    cursor = 0
    output_index = 1
    events = spec["timeline"]
    while clock < spec["duration_s"]:
        time_epsilon = 8.0 * math.ulp(max(1.0, abs(clock)))
        while output_index * spec["step_s"] <= clock + time_epsilon:
            output_index += 1
        due = []
        while cursor < len(events) and events[cursor]["t"] <= clock:
            due.append(events[cursor])
            cursor += 1
        output_time = min(output_index * spec["step_s"], spec["duration_s"])
        end = output_time
        if cursor < len(events):
            end = min(end, events[cursor]["t"])
        yield clock, end, due
        clock = end


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _source_fingerprint() -> str:
    root = Path(__file__).resolve().parent.parent
    digest = hashlib.sha256()
    for package in (
        "sdk",
        "core",
        "plants",
        "physics",
        "components",
        "control",
        "instrumentation",
        "data",
        "scenarios",
    ):
        for path in sorted((root / package).rglob("*")):
            if path.is_file() and path.suffix in (".py", ".json", ".yaml"):
                digest.update(str(path.relative_to(root)).encode())
                digest.update(b"\0")
                digest.update(path.read_bytes())
    return digest.hexdigest()


def run_experiment(
    spec: dict,
    output: str | Path,
    *,
    wall_limit_s: float | None = None,
    cancel_file: str | Path | None = None,
) -> dict:
    """Выполнить расчёт и сохранить пакет даже при физическом/численном отказе."""
    spec = validate_experiment(spec)
    if wall_limit_s is not None:
        wall_limit_s = _number(wall_limit_s, "wall_limit_s", 1e-9, 604800.0)
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=False)
    plant = create_session("vver")
    if spec["solver"].get("continuation", False):
        plant.enable_continuation(spec["duration_s"])
    ledger = HorizonLedger()
    ledger.attach(plant)
    _write_json(folder / "input.json", spec)
    _write_json(folder / "checkpoint.json", export_snapshot(plant))
    versions = {}
    for package in ("numpy", "scipy", "torch", "gymnasium", "PyYAML", "atomic-sim"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not_installed"
    manifest = {
        "schema_version": 1,
        "input_sha256": input_manifest(spec)["sha256"],
        "source_sha256": _source_fingerprint(),
        "versions": versions,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "solver": {"method": "BDF", **spec["solver"]},
        "passport": plant.passport.as_dict(),
        "deterministic_physics": True,
        "stochastic_inputs": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(folder / "manifest.json", manifest)
    started = time.monotonic()
    original_rhs = plant.derivatives

    def guarded_rhs(t, y):
        if wall_limit_s is not None and time.monotonic() - started >= wall_limit_s:
            raise ExperimentInterrupted("wall_limit")
        if cancel_file is not None and Path(cancel_file).exists():
            raise ExperimentInterrupted("cancelled")
        return original_rhs(t, y)

    plant.derivatives = guarded_rhs
    status = "completed"
    error = None
    events_log = []
    model_boundary = None
    extrema = {}
    accepted_steps = 0
    saved = plant.snapshot()

    def record(row, writer):
        writer.writerow(row)
        for key, value in row.items():
            if (
                isinstance(value, (float, int))
                and not isinstance(value, bool)
                and math.isfinite(value)
            ):
                bounds = extrema.setdefault(key, {"min": value, "max": value})
                bounds["min"] = min(bounds["min"], value)
                bounds["max"] = max(bounds["max"], value)

    initial = plant.observation()
    with (folder / "trajectory.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(initial))
        writer.writeheader()
        record(initial, writer)
        try:
            from sdk.api import apply_commands

            for begin, end, due in experiment_intervals(spec):
                if cancel_file is not None and Path(cancel_file).exists():
                    status = "cancelled"
                    break
                if wall_limit_s is not None and time.monotonic() - started >= wall_limit_s:
                    status = "wall_limit"
                    break
                saved = plant.snapshot()
                try:
                    for event in due:
                        apply_commands(
                            plant, {key: value for key, value in event.items() if key != "t"}
                        )
                    solver_options = {
                        key: value for key, value in spec["solver"].items() if key != "continuation"
                    }
                    step_events = None
                    if "domain_boundary_tolerance_s" in spec:
                        from sdk.boundary import advance_until_limit

                        bounded = advance_until_limit(
                            plant,
                            end - begin,
                            time_tolerance_s=spec["domain_boundary_tolerance_s"],
                            **solver_options,
                        )
                        row, model_boundary, step_events = (
                            bounded.observation,
                            bounded.boundary,
                            bounded.events,
                        )
                        if model_boundary is not None:
                            status = "model_limit"
                            error = {
                                "class": "domain",
                                "type": "ModelLimitError",
                                "message": model_boundary["message"],
                                "attempted_interval_s": [begin, end],
                                "attempted_commands": due,
                            }
                    else:
                        row = plant.advance(end - begin, **solver_options)
                except (DomainError, SolverError):
                    plant.restore(saved)
                    raise
                if plant.time_s > saved.time_s:
                    accepted_steps += 1
                    events_log.extend({"kind": "command", **event} for event in due)
                    if step_events is None:
                        events_log.extend(plant.last_motion_events)
                        if plant.last_protection_event_s is not None:
                            events_log.append(
                                {"kind": "protection", "t": plant.last_protection_event_s}
                            )
                    else:
                        events_log.extend(step_events)
                    record(row, writer)
                elif model_boundary is not None:
                    # Команды без принятого продвижения не попадают в checkpoint.
                    plant.restore(saved)
                stream.flush()
                _write_json(folder / "checkpoint.json", export_snapshot(plant))
                if model_boundary is not None:
                    events_log.append(model_boundary)
                    break
        except ExperimentInterrupted as exc:
            status = exc.reason
        except (DomainError, SolverError) as exc:
            status = "failed"
            error = {
                "class": classify_model_error(exc),
                "type": type(exc).__name__,
                "message": str(exc),
                "attempted_interval_s": [begin, end],
                "attempted_commands": due,
            }
            if isinstance(exc, ModelLimitError):
                error["model_limit"] = exc.details()
        except KeyboardInterrupt:
            plant.restore(saved)
            status = "cancelled"
        except Exception as exc:
            # Ошибка программы не считается физическим отказом и повторно поднимается ниже.
            status = "program_error"
            error = {"class": "program", "type": type(exc).__name__, "message": str(exc)}
            program_error = exc

    balance = ledger.report() if accepted_steps else None
    checks = (
        {key: bool(balance[key] <= limit) for key, limit in spec["budgets"].items()}
        if balance
        else {}
    )
    report = {
        "schema_version": 1,
        "status": status,
        "name": spec["name"],
        "requested_duration_s": spec["duration_s"],
        "completed_time_s": plant.time_s,
        "wall_time_s": time.monotonic() - started,
        "accepted_steps": accepted_steps,
        "balance": balance,
        "budget_checks": checks,
        "numerical_acceptance": status == "completed" and bool(checks) and all(checks.values()),
        "validated": False,
        "plant_referenced": False,
        "error": error,
        "events": events_log,
        "model_boundary": model_boundary,
        "extrema": extrema,
    }
    _write_json(folder / "report.json", report)
    (folder / "report.md").write_text(
        "# Experiment: "
        + spec["name"]
        + "\n\n"
        + "Status: **"
        + status
        + "**\n\n"
        + "Completed simulation time: %g / %g s. Accepted steps: %d.\n\n"
        % (plant.time_s, spec["duration_s"], accepted_steps)
        + "Numerical budgets met: %s. Experimental validation: not established.\n\n"
        % report["numerical_acceptance"]
        + "See report.json for balances, events, extrema and failure details. "
        + "checkpoint.json contains the last accepted state.\n",
        encoding="utf-8",
    )
    manifest["artifacts"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(folder.iterdir())
        if path.name != "manifest.json"
    }
    _write_json(folder / "manifest.json", manifest)
    if status == "program_error":
        raise program_error
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run a reproducible VVER experiment bundle")
    parser.add_argument("scenario", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--wall-limit-s", type=float)
    parser.add_argument("--cancel-file", type=Path)
    args = parser.parse_args(argv)
    try:
        import yaml

        spec = yaml.safe_load(args.scenario.read_text(encoding="utf-8"))
        report = run_experiment(
            spec, args.output, wall_limit_s=args.wall_limit_s, cancel_file=args.cancel_file
        )
    except (ValueError, DomainError, OSError, yaml.YAMLError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": report["status"],
                "completed_time_s": report["completed_time_s"],
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["status"] == "completed" and all(report["budget_checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
