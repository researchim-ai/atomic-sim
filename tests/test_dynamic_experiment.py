import json
from pathlib import Path

import pytest
import yaml

from physics.hydraulics.dynamic_channel import DynamicWallChannel
from sdk.dynamic_experiment import run_dynamic_experiment, validate_dynamic_experiment


def spec():
    s = yaml.safe_load(Path("scenarios/pressure_wall_120s.yaml").read_text())
    s.update(duration_s=0.02, step_s=0.01)
    s["hydraulics"]["cells"] = 1
    return s


def test_bundle_balances_snapshot_and_normalized_replay(tmp_path):
    p = tmp_path / "run"
    r = run_dynamic_experiment(spec(), p)
    assert r["status"] == "completed" and r["numerical_acceptance"]
    assert not r["validated"] and not r["plant_referenced"]
    snap = json.loads((p / "checkpoint.json").read_text())
    assert DynamicWallChannel.from_snapshot(snap).snapshot() == snap
    normalized = json.loads((p / "input.json").read_text())
    assert validate_dynamic_experiment(normalized) == normalized
    assert r["scope"]["model_id"] == "pressure-driven-wall-channel"


@pytest.mark.parametrize(
    "changes", [dict(engine="isobaric-channel"), dict(unknown=0), dict(timeline=[{"t": 0.01}])]
)
def test_invalid_spec_does_not_create_output(tmp_path, changes):
    s = spec()
    s.update(changes)
    with pytest.raises(ValueError):
        run_dynamic_experiment(s, tmp_path / "run")
    assert not (tmp_path / "run").exists()


def test_cancel_preserves_initial_checkpoint(tmp_path):
    marker = tmp_path / "stop"
    marker.touch()
    p = tmp_path / "run"
    r = run_dynamic_experiment(spec(), p, cancel_file=marker)
    assert r["status"] == "cancelled" and r["completed_time_s"] == 0.0
    assert not r["numerical_acceptance"]
    assert json.loads((p / "checkpoint.json").read_text())["time_s"] == 0.0


def test_existing_bundle_is_never_overwritten(tmp_path):
    p = tmp_path / "run"
    p.mkdir()
    (p / "sentinel").write_text("keep")
    with pytest.raises(FileExistsError):
        run_dynamic_experiment(spec(), p)
    assert (p / "sentinel").read_text() == "keep"


def test_comparison_checks_physics_and_every_output_point(tmp_path):
    from scripts.compare_dynamic_runs import compare

    s = spec()
    a, b = tmp_path / "a", tmp_path / "b"
    run_dynamic_experiment(s, a)
    s["step_s"] = 0.005
    s["solver"]["rtol"] = 1e-8
    run_dynamic_experiment(s, b)
    budget = tmp_path / "budget.json"
    budget.write_text(
        json.dumps(
            dict(
                horizon_s=0.02,
                flow_absolute_kg_s=1e-4,
                pressure_absolute_pa=1.0,
                fluid_temperature_absolute_k=0.001,
                wall_temperature_absolute_k=0.001,
                mass_relative=1e-6,
                energy_relative=1e-5,
            )
        )
    )
    result = compare(a, b, budget)
    assert result["passed"] and result["comparison_times"] == 3
    s["hydraulics"]["loss_coefficient"] = 20.0
    changed = tmp_path / "changed"
    run_dynamic_experiment(s, changed)
    with pytest.raises(ValueError, match="identical"):
        compare(a, changed, budget)
