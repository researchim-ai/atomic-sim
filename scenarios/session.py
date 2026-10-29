"""Исполнение сценария тем же шагом, что и SDK.

Файл без engine: vver не переписывается в команды сосредоточенной петли.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

from core.status import Applicability, DomainError
from sdk.api import create_session, export_snapshot, step_session


def load_scenario(path: str | Path) -> dict:
    raw = Path(path).read_bytes()
    data = yaml.safe_load(raw)
    if not isinstance(data, dict):
        raise DomainError("Сценарий должен быть отображением")
    data["_sha256"] = hashlib.sha256(raw).hexdigest()
    data["_path"] = str(path)
    return data


def step_lengths(duration_s: float, step_s: float) -> list:
    """Шаги, которые в сумме дают ровно длительность. Хвост короче номинального шага не перешагивает конец."""
    if duration_s <= 0.0 or step_s <= 0.0 or step_s > 5.0:
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
    step_s = float(data.get("step_s", 0.5))
    duration_s = float(data["duration_s"])
    pending = list(data.get("timeline", []))
    rows = []
    time_s = 0.0
    for span in step_lengths(duration_s, step_s):
        commands = commands_due(pending, time_s)
        observed = step_session(plant, span, commands)
        time_s += span
        rows.append(observed)
    return {
        "name": data["name"],
        "sha256": data["_sha256"],
        "model_id": plant.passport.model_id,
        "model_version": plant.passport.version,
        "snapshot": export_snapshot(plant),
        "rows": rows,
    }
