import json

import pytest

from scripts.compare_isobaric_runs import compare
from sdk.channel_experiment import run_channel_experiment


def case(heat=3e5, step=0.25):
    return dict(
        schema_version=1,
        name="comparison",
        engine="isobaric-channel",
        duration_s=0.5,
        step_s=step,
        channel=dict(
            cells=2,
            total_volume_m3=0.002,
            pressure_pa=7e6,
            inlet_enthalpy_j_kg=1.1e6,
            inlet_mass_flow_kg_s=1.0,
            heat_w=heat,
        ),
    )


@pytest.fixture
def budgets(tmp_path):
    path = tmp_path / "budgets.json"
    path.write_text(
        json.dumps(
            {
                "horizon_s": 0.5,
                "outlet_enthalpy_relative": 1e-4,
                "outlet_quality_absolute": 1e-4,
                "stored_mass_relative": 1e-4,
                "mass_relative": 1e-6,
                "energy_relative": 1e-5,
            }
        )
    )
    return path


def test_complete_common_grid_matches_and_does_not_claim_physical_validation(tmp_path, budgets):
    a, b = tmp_path / "a", tmp_path / "b"
    run_channel_experiment(case(), a)
    run_channel_experiment(case(step=0.125), b)
    result = compare(a, b, budgets)
    assert result["passed"]
    assert result["comparison_times"] == 3
    assert not result["validated"]


def test_modified_csv_is_refused(tmp_path, budgets):
    a, b = tmp_path / "a", tmp_path / "b"
    run_channel_experiment(case(), a)
    run_channel_experiment(case(), b)
    with (a / "trajectory.csv").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="Checksum"):
        compare(a, b, budgets)


def test_truncated_run_cannot_pass(tmp_path, budgets):
    a, b = tmp_path / "a", tmp_path / "b"
    cancel = tmp_path / "cancel"
    cancel.touch()
    run_channel_experiment(case(), a, cancel_file=cancel)
    run_channel_experiment(case(), b)
    with pytest.raises(ValueError, match="requested horizon"):
        compare(a, b, budgets)


def test_different_prescribed_heating_is_not_a_solver_comparison(tmp_path, budgets):
    a, b = tmp_path / "a", tmp_path / "b"
    run_channel_experiment(case(), a)
    run_channel_experiment(case(heat=3.1e5), b)
    with pytest.raises(ValueError, match="identical channel"):
        compare(a, b, budgets)


def test_wall_comparison_checks_temperature_and_rejects_changed_capacity(tmp_path, budgets):
    configured = case()
    configured["engine"] = "isobaric-wall-channel"
    configured["wall"] = dict(total_heat_capacity_j_k=8000.0, total_ua_w_k=10000.0)
    limits = json.loads(budgets.read_text())
    limits["wall_temperature_absolute_k"] = 0.001
    budgets.write_text(json.dumps(limits))
    a, b, changed = (tmp_path / name for name in ("a", "b", "changed"))
    run_channel_experiment(configured, a)
    configured["step_s"] = 0.125
    run_channel_experiment(configured, b)
    result = compare(a, b, budgets)
    assert result["passed"]
    assert "wall_temperature_absolute_k" in result["checks"]
    configured["wall"]["total_heat_capacity_j_k"] *= 2
    run_channel_experiment(configured, changed)
    with pytest.raises(ValueError, match="identical"):
        compare(a, changed, budgets)
