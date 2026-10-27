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


def run_scenario(path: str | Path) -> dict:
    data = load_scenario(path)
    if data.get("engine") != "vver":
        raise DomainError(
            "Сценарий %s не объявляет engine: vver и не исполняется этим ядром" % data.get("name"),
            Applicability.UNSUPPORTED,
        )
    plant = create_session("vver")
    if data.get("model_id") not in (None, plant.passport.model_id):
        raise DomainError("Сценарий требует другую модель: %s" % data.get("model_id"))
    step_s = float(data.get("step_s", 0.5))
    duration_s = float(data["duration_s"])
    pending = sorted(data.get("timeline", []), key=lambda item: float(item["t"]))
    cursor = 0
    commands: dict = {}
    rows = []
    time_s = 0.0
    while time_s < duration_s - 1.0e-12:
        while cursor < len(pending) and float(pending[cursor]["t"]) <= time_s + 1.0e-12:
            event = pending[cursor]
            for key in ("throttle", "rod_command", "feed_bias_kg_s", "boron_ppm"):
                if key in event:
                    commands[key] = event[key]
            if event.get("scram"):
                commands["scram"] = True
            if event.get("motor_off"):
                commands["motor_off"] = True
            cursor += 1
        observed = step_session(plant, step_s, commands)
        time_s += step_s
        rows.append(observed)
    return {
        "name": data["name"],
        "sha256": data["_sha256"],
        "model_id": plant.passport.model_id,
        "model_version": plant.passport.version,
        "snapshot": export_snapshot(plant),
        "rows": rows,
    }
