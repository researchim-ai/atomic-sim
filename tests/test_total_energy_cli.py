import json

import pytest

from scripts.generate_total_energy_case import main


def test_reverse_wall_generator_writes_actual_reverse_equilibrium(tmp_path):
    target = tmp_path / "case.json"
    main(["--case", "wall_pulse", "--sign", "-1", "--output", str(target)])
    data = json.loads(target.read_text())
    c = data["hydraulics"]
    assert c["initial_flow_kg_s"] < 0
    assert c["left_pressure_pa"] < c["right_pressure_pa"]
    assert data["name"] == "total_energy_wall_pulse_negative"


def test_total_case_generator_never_overwrites_input(tmp_path):
    target = tmp_path / "case.json"
    target.write_text("keep")
    with pytest.raises(SystemExit) as exc:
        main(["--case", "coastdown", "--output", str(target)])
    assert exc.value.code == 2 and target.read_text() == "keep"


@pytest.mark.parametrize("suffix", [".json", ".JSON"])
def test_generated_json_cli_preserves_exponent_numbers_and_runs(tmp_path, suffix):
    from scripts.generate_total_energy_case import make_spec
    from sdk.dynamic_experiment import main as run_cli

    spec = make_spec(sign=-1)
    spec.update(duration_s=0.02, step_s=0.01)
    scenario = tmp_path / ("reverse" + suffix)
    scenario.write_text(json.dumps(spec))
    output = tmp_path / "run"
    assert run_cli([str(scenario), "--output", str(output)]) == 0
    report = json.loads((output / "report.json").read_text())
    assert report["numerical_acceptance"] and report["completed_time_s"] == 0.02
    checkpoint = json.loads((output / "checkpoint.json").read_text())
    assert checkpoint["config"]["initial_flow_kg_s"] < 0
