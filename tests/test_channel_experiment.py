import csv
import hashlib
import json

import pytest

from core.status import DomainError
from physics.hydraulics.isobaric import IsobaricChannel
from sdk.channel_experiment import run_channel_experiment, validate_channel_experiment


def spec():
    return dict(
        schema_version=1,
        name="channel-test",
        engine="isobaric-channel",
        duration_s=0.5,
        step_s=0.25,
        channel=dict(
            cells=2,
            total_volume_m3=0.002,
            pressure_pa=7e6,
            inlet_enthalpy_j_kg=1.1e6,
            inlet_mass_flow_kg_s=1.0,
            heat_w=3e5,
        ),
    )


def test_complete_bundle_hashes_checkpoint_and_normalized_replay(tmp_path):
    folder = tmp_path / "run"
    report = run_channel_experiment(spec(), folder)
    assert report["status"] == "completed"
    assert report["numerical_acceptance"]
    assert not report["validated"]
    normalized = json.loads((folder / "input.json").read_text())
    assert validate_channel_experiment(normalized) == normalized
    manifest = json.loads((folder / "manifest.json").read_text())
    for name, digest in manifest["artifacts"].items():
        assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == digest
    payload = json.loads((folder / "checkpoint.json").read_text())
    assert IsobaricChannel.from_snapshot(payload).snapshot() == payload
    with (folder / "trajectory.csv").open() as stream:
        assert float(list(csv.DictReader(stream))[-1]["time_s"]) == 0.5
    with pytest.raises(FileExistsError):
        run_channel_experiment(spec(), folder)


def test_preflight_rejects_bad_eos_before_creating_output(tmp_path):
    configured = spec()
    configured["channel"]["inlet_enthalpy_j_kg"] = 1e12
    configured["channel"]["initial_enthalpy_j_kg"] = [1.1e6, 1.1e6]
    with pytest.raises(DomainError):
        run_channel_experiment(configured, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_cancelled_run_keeps_initial_checkpoint_without_acceptance(tmp_path):
    cancel = tmp_path / "cancel"
    cancel.touch()
    report = run_channel_experiment(spec(), tmp_path / "run", cancel_file=cancel)
    assert report["status"] == "cancelled"
    assert report["completed_time_s"] == 0.0
    assert not report["numerical_acceptance"]
    assert json.loads((tmp_path / "run/checkpoint.json").read_text())["time_s"] == 0.0


def test_wall_limit_is_not_a_completed_run(tmp_path):
    report = run_channel_experiment(spec(), tmp_path / "run", wall_limit_s=1e-9)
    assert report["status"] == "wall_limit"
    assert not report["numerical_acceptance"]


def test_program_failure_is_recorded_then_reraised(monkeypatch, tmp_path):
    def fail(*args, **kwargs):
        raise RuntimeError("injected bug")

    monkeypatch.setattr(IsobaricChannel, "advance", fail)
    with pytest.raises(RuntimeError, match="injected bug"):
        run_channel_experiment(spec(), tmp_path / "run")
    report = json.loads((tmp_path / "run/report.json").read_text())
    assert report["status"] == "program_error"
    assert report["error"]["class"] == "program"


def test_unknown_commands_are_not_silently_ignored():
    configured = spec()
    configured["timeline"] = [{"t": 0.1, "scram": True}]
    with pytest.raises(ValueError, match="timeline"):
        validate_channel_experiment(configured)
