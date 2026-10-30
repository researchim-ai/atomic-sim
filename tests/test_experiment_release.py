"""Приёмка переносимого эксперимента, точных команд и отказобезопасных снимков."""

import copy
import csv
import hashlib
import json

import numpy as np
import pytest

from core.status import DomainError
from data.catalog import assert_bundled_checksum
from sdk.api import create_session, export_snapshot, restore_snapshot
from sdk.experiment import experiment_intervals, main, run_experiment, validate_experiment


def scenario():
    return {
        "name": "off-grid-event",
        "schema_version": 1,
        "engine": "vver",
        "duration_s": 0.2,
        "step_s": 0.2,
        "timeline": [{"t": 0.07, "rod_command": 0.02}],
    }


def test_event_between_output_times_is_applied_at_requested_time(tmp_path):
    spec = validate_experiment(scenario())
    intervals = list(experiment_intervals(spec))
    assert [(a, b) for a, b, _ in intervals] == [(0.0, 0.07), (0.07, 0.2)]
    configured = scenario()
    configured["budgets"] = {"mass_relative": 1e-6}
    report = run_experiment(configured, tmp_path / "run")
    assert report["status"] == "completed"
    assert report["completed_time_s"] == pytest.approx(0.2)
    assert report["events"][0]["t"] == pytest.approx(0.07)
    assert report["validated"] is False
    assert report["numerical_acceptance"] is True
    assert report["budget_checks"]["mass_relative"] is True
    checkpoint = json.loads((tmp_path / "run/checkpoint.json").read_text())
    assert checkpoint["state"][12] == pytest.approx(0.25 + 0.02 * 0.13, abs=1e-5)
    manifest = json.loads((tmp_path / "run/manifest.json").read_text())
    for name, digest in manifest["artifacts"].items():
        assert hashlib.sha256((tmp_path / "run" / name).read_bytes()).hexdigest() == digest
    rows = list(csv.DictReader((tmp_path / "run/trajectory.csv").open()))
    assert len(rows) == 3
    with pytest.raises(FileExistsError):
        run_experiment(scenario(), tmp_path / "run")


def test_decimal_time_boundary_does_not_create_a_machine_epsilon_step():
    spec = scenario()
    spec.update(duration_s=0.5, step_s=0.1, timeline=[{"t": 0.3, "throttle": 0.5}])
    intervals = list(experiment_intervals(validate_experiment(spec)))
    assert len(intervals) == 5
    assert all(end - begin > 0.09 for begin, end, _ in intervals)
    at_event = [(begin, due) for begin, _end, due in intervals if due]
    assert len(at_event) == 1
    assert at_event[0][0] == 0.3


def test_failed_physical_step_keeps_checkpoint_and_reports_attempt(tmp_path, monkeypatch):
    from plants.vver import VVERPlant

    advance = VVERPlant.advance

    def fail_second(self, dt_s, **kwargs):
        if self.time_s > 0:
            raise DomainError("Injected component boundary")
        return advance(self, dt_s, **kwargs)

    monkeypatch.setattr(VVERPlant, "advance", fail_second)
    report = run_experiment(scenario(), tmp_path / "failed")
    checkpoint = json.loads((tmp_path / "failed/checkpoint.json").read_text())
    assert report["status"] == "failed"
    assert report["error"]["class"] == "domain"
    assert report["completed_time_s"] == pytest.approx(0.07)
    assert checkpoint["time_s"] == pytest.approx(0.07)
    assert checkpoint["rod_command"] == 0.0
    assert report["error"]["attempted_commands"][0]["rod_command"] == 0.02
    assert report["balance"]["steps"] == 1


def test_program_error_is_not_relabelled_domain_error(tmp_path, monkeypatch):
    from plants.vver import VVERPlant

    def broken(*args, **kwargs):
        raise RuntimeError("code defect")

    monkeypatch.setattr(VVERPlant, "advance", broken)
    with pytest.raises(RuntimeError, match="code defect"):
        run_experiment(scenario(), tmp_path / "bug")
    report = json.loads((tmp_path / "bug/report.json").read_text())
    assert report["status"] == "program_error"
    assert report["error"]["class"] == "program"


def test_cancel_and_wall_limit_preserve_initial_state(tmp_path):
    cancel = tmp_path / "cancel"
    cancel.touch()
    report = run_experiment(scenario(), tmp_path / "cancelled", cancel_file=cancel)
    assert report["status"] == "cancelled"
    assert report["accepted_steps"] == 0
    assert report["completed_time_s"] == 0.0
    assert report["balance"] is None
    wall = run_experiment(scenario(), tmp_path / "limited", wall_limit_s=1e-9)
    assert wall["status"] == "wall_limit"


def test_deadline_interrupts_inside_rhs_and_does_not_commit_step(tmp_path, monkeypatch):
    import time

    from plants.vver import VVERPlant

    def delayed_step(self, dt_s, **kwargs):
        time.sleep(0.02)
        return self.derivatives(self.time_s, self.pack())

    monkeypatch.setattr(VVERPlant, "advance", delayed_step)
    report = run_experiment(scenario(), tmp_path / "rhs_timeout", wall_limit_s=0.01)
    assert report["status"] == "wall_limit"
    assert report["accepted_steps"] == 0
    assert report["completed_time_s"] == 0.0
    assert json.loads((tmp_path / "rhs_timeout/checkpoint.json").read_text())["time_s"] == 0.0


@pytest.mark.parametrize(
    "mutation",
    [
        lambda s: s.update(duration_s=float("nan")),
        lambda s: s.update(timeline=[{"t": -1.0, "throttle": 0.0}]),
        lambda s: s.update(timeline=[{"t": 0.1, "scram": "false"}]),
        lambda s: s.update(solver={"atol": -0.1}),
        lambda s: s.update(timeline=[{"t": 0.2, "throttle": 0.0}]),
    ],
)
def test_bad_input_rejected_before_creating_output(tmp_path, mutation):
    spec = scenario()
    mutation(spec)
    with pytest.raises(ValueError):
        run_experiment(spec, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


def test_invalid_restore_is_atomic_and_catalog_works_away_from_repo(tmp_path, monkeypatch):
    plant = create_session()
    origin = export_snapshot(plant)
    corruptions = [
        ("state", [1.0]),
        ("scram_latched", "false"),
        ("time_s", -1.0),
        ("throttle", 4.0),
    ]
    for key, value in corruptions:
        bad = copy.deepcopy(origin)
        bad[key] = value
        with pytest.raises(DomainError):
            restore_snapshot(plant, bad)
        assert export_snapshot(plant) == origin
    bad = copy.deepcopy(origin)
    bad["state"][0] = np.nan
    with pytest.raises(DomainError):
        restore_snapshot(plant, bad)
    assert export_snapshot(plant) == origin
    bad = copy.deepcopy(origin)
    bad["state"][13] = -1.0
    with pytest.raises(DomainError):
        restore_snapshot(plant, bad)
    assert export_snapshot(plant) == origin
    monkeypatch.chdir(tmp_path)
    assert assert_bundled_checksum("project-open-two-group")["bundled"]


def test_cli_bad_yaml_is_error_without_partial_run(tmp_path):
    source = tmp_path / "bad.yaml"
    source.write_text("name: [broken")
    assert main([str(source), "--output", str(tmp_path / "output")]) == 2
    assert not (tmp_path / "output").exists()
