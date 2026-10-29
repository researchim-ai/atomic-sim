"""Снимок процесса продолжается в новом интерпретаторе."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from core.status import DomainError
from instrumentation.channels import DelayedActuator, Measurement, Sensor
from instrumentation.replay import export_process, restore_process, step_process
from sdk.api import create_session, step_session


def _prepared(path: Path):
    plant = create_session("vver")
    plant.power_protection.setpoint = 0.5 * plant.fission_power_nom
    plant.power_protection.delay_s = 5.0
    measurement = Measurement(
        "power", Sensor(tau_s=1.0, value=plant.fission_power_nom, noise_std=1.0e5)
    )
    actuator = DelayedActuator(position=0.2, speed_limit=0.05, delay_s=0.15)
    rng = np.random.default_rng(11)
    step_session(plant, 0.1)
    measurement.read(plant.observation()["thermal_power_w"], 0.1, rng)
    actuator.step(0.8, 0.1)
    payload = export_process(plant, measurement, actuator, rng)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def test_new_process_matches_the_local_continuation(tmp_path):
    source = tmp_path / "process.json"
    payload = _prepared(source)
    destination = tmp_path / "continued.json"
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from instrumentation.replay import continue_saved; "
            "continue_saved(%r, %r, 0.1, 1.0)" % (str(source), str(destination)),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    remote = json.loads(destination.read_text(encoding="utf-8"))
    plant, measurement, actuator, rng = restore_process(payload)
    local = step_process(plant, measurement, actuator, rng, 0.1, 1.0)
    assert remote["thermal_power_w"] == pytest.approx(local["thermal_power_w"], rel=0.0, abs=0.0)
    assert remote["reading_value"] == pytest.approx(local["reading_value"], rel=0.0, abs=0.0)
    assert remote["actuator_position"] == pytest.approx(
        local["actuator_position"], rel=0.0, abs=0.0
    )
    assert remote["rng_probe"] == pytest.approx(local["rng_probe"], rel=0.0, abs=0.0)
    assert remote["protection_timer_s"] == pytest.approx(0.2, abs=1.0e-6)
    assert remote["protection_timer_s"] == pytest.approx(local["protection_timer_s"], abs=0.0)


def test_unknown_process_schema_is_rejected():
    with pytest.raises(DomainError):
        restore_process({"schema_version": 99})
