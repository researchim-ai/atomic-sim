"""Приёмка не должна пропускать обрезанные или изменённые данные."""

import csv
import hashlib
import json

import pytest

from scripts.compare_hour_nominal import FIELDS, compare


def bundle(folder, times, status="completed"):
    folder.mkdir()
    (folder / "input.json").write_text(json.dumps({"timeline": [], "duration_s": 3600}))
    (folder / "report.json").write_text(
        json.dumps(
            {
                "status": status,
                "numerical_acceptance": status == "completed",
                "completed_time_s": times[-1],
            }
        )
    )
    columns = ["time_s"] + [field for field, _ in FIELDS.values()]
    with (folder / "trajectory.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for t in times:
            writer.writerow({key: t if key == "time_s" else 10.0 for key in columns})
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir()}
    (folder / "manifest.json").write_text(json.dumps({"artifacts": hashes}))
    return folder


@pytest.fixture
def budgets(tmp_path):
    path = tmp_path / "budget.json"
    path.write_text(
        json.dumps({"horizon_s": 3600, "observable_budgets": {key: 1e-4 for key in FIELDS}})
    )
    return path


def test_complete_common_grid_passes_and_tampering_is_rejected(tmp_path, budgets):
    run = bundle(tmp_path / "a", [0, 1800, 3600])
    reference = bundle(tmp_path / "b", [0, 900, 1800, 2700, 3600])
    result = compare(run, reference, budgets)
    assert result["numerical_hour_accepted"]
    assert result["comparison_times"] == 3
    assert not result["validated"]
    with (run / "trajectory.csv").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="Checksum"):
        compare(run, reference, budgets)


def test_truncated_run_cannot_pass_even_with_matching_trajectory(tmp_path, budgets):
    run = bundle(tmp_path / "a", [0, 100], status="wall_limit")
    reference = bundle(tmp_path / "b", [0, 100, 3600])
    with pytest.raises(ValueError, match="complete requested horizon"):
        compare(run, reference, budgets)


def test_missing_reference_nodes_are_not_silently_interpolated(tmp_path, budgets):
    run = bundle(tmp_path / "a", [0, 1800, 3600])
    reference = bundle(tmp_path / "b", [0, 3600])
    with pytest.raises(ValueError, match="every requested"):
        compare(run, reference, budgets)


def test_command_comparison_requires_explicit_opt_in_and_identical_timeline(tmp_path, budgets):
    run = bundle(tmp_path / "a", [0, 1800, 3600])
    reference = bundle(tmp_path / "b", [0, 1800, 3600])

    def commands(folder, throttle):
        path = folder / "input.json"
        spec = json.loads(path.read_text())
        spec["timeline"] = [{"t": 10.0, "throttle": throttle}]
        path.write_text(json.dumps(spec))
        manifest_path = folder / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["artifacts"]["input.json"] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest))

    commands(run, 0.98)
    commands(reference, 0.98)
    with pytest.raises(ValueError, match="empty command"):
        compare(run, reference, budgets)
    result = compare(run, reference, budgets, allow_commands=True)
    assert result["numerical_comparison_accepted"]
    assert not result["numerical_hour_accepted"]
    commands(reference, 0.99)
    with pytest.raises(ValueError, match="identical command"):
        compare(run, reference, budgets, allow_commands=True)


def test_bounds_and_required_events_are_part_of_acceptance(tmp_path, budgets):
    run = bundle(tmp_path / "a", [0, 1800, 3600])
    reference = bundle(tmp_path / "b", [0, 1800, 3600])
    budget = json.loads(budgets.read_text())
    budget["state_bounds"] = {"fuel_temperature_k": [0, 9]}
    budget["event_expectations"] = [
        {"kind": "rod_stop", "t": 13.75, "time_tolerance_s": 1e-8, "values": {"position": 1.0}}
    ]
    budgets.write_text(json.dumps(budget))
    result = compare(run, reference, budgets)
    assert not result["numerical_comparison_accepted"]
    assert not result["checks"]["bounds_fuel_temperature_k"]
    assert not result["checks"]["event_0_rod_stop"]
    budget["state_bounds"]["fuel_temperature_k"] = [0, 10]
    budgets.write_text(json.dumps(budget))
    for folder in (run, reference):
        path = folder / "report.json"
        report = json.loads(path.read_text())
        report["events"] = [{"kind": "rod_stop", "t": 13.75, "position": 1.0}]
        path.write_text(json.dumps(report))
        manifest_path = folder / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["artifacts"]["report.json"] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest))
    assert compare(run, reference, budgets)["numerical_comparison_accepted"]
