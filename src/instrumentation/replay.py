"""Снимок процесса: установка, каналы, очередь привода и генератор шума.

Продолжение в другом процессе читает тот же JSON. Чужая схема снимка не исполняется.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from core.status import DomainError
from instrumentation.channels import DelayedActuator, Measurement, Sensor
from sdk.api import create_session, export_snapshot, restore_snapshot, step_session

SCHEMA_VERSION = 1


def rng_payload(rng: np.random.Generator) -> dict:
    raw = rng.bit_generator.state
    if raw.get("bit_generator") != "PCG64":
        raise DomainError("В снимок процесса входит только генератор PCG64")
    return {
        "bit_generator": "PCG64",
        "state": int(raw["state"]["state"]),
        "inc": int(raw["state"]["inc"]),
        "has_uint32": int(raw["has_uint32"]),
        "uinteger": int(raw["uinteger"]),
    }


def rng_from_payload(payload: dict) -> np.random.Generator:
    if payload.get("bit_generator") != "PCG64":
        raise DomainError("В снимок процесса входит только генератор PCG64")
    bit_generator = np.random.PCG64()
    bit_generator.state = {
        "bit_generator": "PCG64",
        "state": {"state": int(payload["state"]), "inc": int(payload["inc"])},
        "has_uint32": int(payload["has_uint32"]),
        "uinteger": int(payload["uinteger"]),
    }
    return np.random.Generator(bit_generator)


def _measurement_payload(measurement: Measurement) -> dict:
    sensor = measurement.sensor
    return {
        "kind": measurement.kind,
        "time_s": measurement.time_s,
        "tau_s": sensor.tau_s,
        "value": sensor.value,
        "bias": sensor.bias,
        "noise_std": sensor.noise_std,
        "failed": sensor.failed,
        "fault": sensor.fault,
        "quality": sensor.quality,
        "quantum": sensor.quantum,
        "drift_per_s": sensor.drift_per_s,
        "sample_period_s": sensor.sample_period_s,
        "held": sensor.held,
        "sample_elapsed_s": sensor.sample_elapsed_s,
    }


def _measurement_from_payload(payload: dict) -> Measurement:
    sensor = Sensor(
        tau_s=float(payload["tau_s"]),
        value=float(payload["value"]),
        bias=float(payload["bias"]),
        noise_std=float(payload["noise_std"]),
        failed=bool(payload["failed"]),
        fault=str(payload.get("fault", "")),
        quality=str(payload["quality"]),
        quantum=float(payload["quantum"]),
        drift_per_s=float(payload["drift_per_s"]),
        sample_period_s=float(payload["sample_period_s"]),
        sample_elapsed_s=float(payload["sample_elapsed_s"]),
    )
    sensor.held = float(payload["held"])
    return Measurement(str(payload["kind"]), sensor, float(payload["time_s"]))


def _actuator_payload(actuator: DelayedActuator) -> dict:
    return {
        "position": actuator.position,
        "speed_limit": actuator.speed_limit,
        "delay_s": actuator.delay_s,
        "minimum": actuator.minimum,
        "maximum": actuator.maximum,
        "time_s": actuator._time_s,
        "held_demand": actuator._held_demand,
        "queue": [[float(time_s), float(value)] for time_s, value in actuator._queue],
    }


def _actuator_from_payload(payload: dict) -> DelayedActuator:
    actuator = DelayedActuator(
        position=float(payload["position"]),
        speed_limit=float(payload["speed_limit"]),
        delay_s=float(payload["delay_s"]),
        minimum=float(payload["minimum"]),
        maximum=float(payload["maximum"]),
    )
    actuator._time_s = float(payload["time_s"])
    actuator._held_demand = float(payload["held_demand"])
    actuator._queue = [(float(time_s), float(value)) for time_s, value in payload["queue"]]
    return actuator


def export_process(
    plant, measurement: Measurement, actuator: DelayedActuator, rng: np.random.Generator
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "plant": export_snapshot(plant),
        "measurement": _measurement_payload(measurement),
        "actuator": _actuator_payload(actuator),
        "rng": rng_payload(rng),
    }


def restore_process(payload: dict):
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise DomainError("Схема снимка процесса не поддерживается")
    plant = create_session("vver")
    restore_snapshot(plant, payload["plant"])
    measurement = _measurement_from_payload(payload["measurement"])
    actuator = _actuator_from_payload(payload["actuator"])
    rng = rng_from_payload(payload["rng"])
    return plant, measurement, actuator, rng


def step_process(plant, measurement, actuator, rng, dt_s: float, actuator_demand: float) -> dict:
    observed = step_session(plant, dt_s)
    reading = measurement.read(observed["thermal_power_w"], dt_s, rng)
    position = actuator.step(actuator_demand, dt_s)
    return {
        "thermal_power_w": observed["thermal_power_w"],
        "rod_insertion": observed["rod_insertion"],
        "protection_timer_s": plant.power_protection.timer_s,
        "reading_value": reading["value"],
        "reading_quality": reading["quality"],
        "actuator_position": position,
        "rng_probe": float(rng.normal()),
    }


def continue_saved(source: str, destination: str, dt_s: float, actuator_demand: float) -> None:
    payload = json.loads(Path(source).read_text(encoding="utf-8"))
    plant, measurement, actuator, rng = restore_process(payload)
    report = step_process(plant, measurement, actuator, rng, dt_s, actuator_demand)
    Path(destination).write_text(json.dumps(report), encoding="utf-8")
