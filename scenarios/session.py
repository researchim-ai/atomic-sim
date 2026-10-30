"""Исполнение сценария тем же шагом, что и SDK.

Файл без engine: vver не переписывается в команды сосредоточенной петли.
"""

from __future__ import annotations

import hashlib
import math
from importlib.resources import files
from pathlib import Path

import yaml

from core.status import Applicability, DomainError
from sdk.api import create_session, export_snapshot


def bundled_scenario(name: str) -> dict:
    """Сценарий из поставленного пакета. Имя — файл внутри scenarios, без пути наружу."""
    if not name.endswith(".yaml") or "/" in name or "\\" in name or name.startswith("."):
        raise DomainError("Имя поставленного сценария должно быть файлом yaml этого пакета")
    raw = files("scenarios").joinpath(name).read_bytes()
    return _scenario_from_bytes(raw, "scenarios/%s" % name)


def load_scenario(path: str | Path) -> dict:
    raw = Path(path).read_bytes()
    return _scenario_from_bytes(raw, str(path))


def _scenario_from_bytes(raw: bytes, label: str) -> dict:
    data = yaml.safe_load(raw)
    if not isinstance(data, dict):
        raise DomainError("Сценарий должен быть отображением")
    data["_sha256"] = hashlib.sha256(raw).hexdigest()
    data["_path"] = label
    return data


def step_lengths(duration_s: float, step_s: float) -> list:
    """Шаги, которые в сумме дают ровно длительность. Хвост короче номинального шага не перешагивает конец."""
    if (
        not math.isfinite(duration_s)
        or not math.isfinite(step_s)
        or duration_s <= 0.0
        or step_s <= 0.0
        or step_s > 5.0
    ):
        raise DomainError("Длительность положительна, шаг сценария лежит в (0, 5] с")
    lengths = []
    covered = 0.0
    while covered < duration_s - 1.0e-12:
        span = min(step_s, duration_s - covered)
        lengths.append(span)
        covered += span
    return lengths


def commands_due(timeline, time_s: float) -> dict:
    """Все команды с t <= time. При одном времени побеждает более поздняя запись списка."""
    commands: dict = {}
    allowed = {
        "t",
        "throttle",
        "rod_command",
        "feed_bias_kg_s",
        "boron_ppm",
        "sink_temperature_k",
        "scram",
        "motor_off",
    }
    ordered = sorted(timeline, key=lambda item: float(item["t"]))
    for event in ordered:
        unknown = sorted(set(event) - allowed)
        if unknown:
            raise DomainError("Неизвестный ключ сценария: %s" % ", ".join(unknown))
        if float(event["t"]) > time_s + 1.0e-12:
            break
        for key in (
            "throttle",
            "rod_command",
            "feed_bias_kg_s",
            "boron_ppm",
            "sink_temperature_k",
        ):
            if key in event:
                commands[key] = event[key]
        if event.get("scram"):
            commands["scram"] = True
        if event.get("motor_off"):
            commands["motor_off"] = True
    return commands


def run_scenario(path: str | Path) -> dict:
    from sdk.api import apply_commands
    from sdk.experiment import experiment_intervals, validate_experiment

    data = load_scenario(path)
    if data.get("engine") != "vver":
        raise DomainError(
            "Сценарий %s не объявляет engine: vver и не исполняется этим ядром" % data.get("name"),
            Applicability.UNSUPPORTED,
        )
    if data.get("schema_version") != 1:
        raise DomainError("Сценарий ВВЭР требует schema_version: 1")
    plant = create_session("vver")
    if data.get("model_id") not in (None, plant.passport.model_id):
        raise DomainError("Сценарий требует другую модель: %s" % data.get("model_id"))
    spec = validate_experiment(
        {key: value for key, value in data.items() if not key.startswith("_")}
    )
    if spec["solver"].get("continuation", False):
        plant.enable_continuation(spec["duration_s"])
    solver_options = {key: value for key, value in spec["solver"].items() if key != "continuation"}
    rows = []
    boundary = None
    for begin, end, due in experiment_intervals(spec):
        saved = plant.snapshot()
        for event in due:
            apply_commands(plant, {key: value for key, value in event.items() if key != "t"})
        if "domain_boundary_tolerance_s" in spec:
            from sdk.boundary import advance_until_limit

            result = advance_until_limit(
                plant,
                end - begin,
                time_tolerance_s=spec["domain_boundary_tolerance_s"],
                **solver_options,
            )
            observed, boundary = result.observation, result.boundary
        else:
            observed = plant.advance(end - begin, **solver_options)
        if plant.time_s > begin:
            rows.append(observed)
        elif boundary is not None:
            plant.restore(saved)
        if boundary is not None:
            break
    return {
        "name": data["name"],
        "sha256": data["_sha256"],
        "model_id": plant.passport.model_id,
        "model_version": plant.passport.version,
        "snapshot": export_snapshot(plant),
        "rows": rows,
        "status": "model_limit" if boundary is not None else "completed",
        "model_boundary": boundary,
    }
