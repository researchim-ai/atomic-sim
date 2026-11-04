"""Positive conservation control, negative thermal control and exact report replay."""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from atomic_model_reference.bundle import check_catalog, import_reference
from atomic_model_reference.compare import compare_job
from atomic_model_reference.replay import replay_report
from atomic_model_workspace.common import atomic_json, read
from atomic_model_workspace.jobs import create_job, run_job


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=Path(__file__).parent / "eos_reference")
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    out = a.output or Path(tempfile.mkdtemp(prefix="atomic-reference-workflow-"))
    out.mkdir(parents=True, exist_ok=True)
    import_reference(a.input / "reference.json", out / "temperature-data")
    positive = import_reference(a.input / "energy_reference.json", out / "energy-data")
    catalog = check_catalog([out / "temperature-data", out / "energy-data"])
    assert catalog["accepted"]
    create_job(
        dict(
            schema_version=1,
            scenario=dict(
                schema_version=1,
                config=positive["problem"]["config"],
                seed=17,
                split="dev",
                objective=dict(sensor="wall.w0", target_k=300, tolerance_k=1),
            ),
            policy=positive["problem"]["policy"],
            decision_timeout_s=1,
        ),
        out / "job",
    )
    assert run_job(out / "job")["status"] == "completed"
    thermal = compare_job(out / "job", out / "temperature-data", out / "temperature-report")
    energy = compare_job(out / "job", out / "energy-data", out / "energy-report")
    assert not thermal["passed"] and energy["passed"]
    replay = replay_report(out / "temperature-report", out / "temperature-replayed")
    assert replay["replay_exact"] and not replay["comparison_passed"]
    assert replay_report(out / "energy-report", out / "energy-replayed")["comparison_passed"]
    source = out / "leak-input"
    source.mkdir()
    shutil.copy2(out / "temperature-data/data.csv", source / "data.csv")
    spec = read(out / "temperature-data/input.json")
    spec.update(reference_id="relabelled-case", physical_case_id="new-label", role="calibration")
    spec["problem"]["config"]["max_step_s"] = 0.001
    atomic_json(source / "input.json", spec)
    import_reference(source / "input.json", out / "leaked-data")
    try:
        check_catalog([out / "temperature-data", out / "leaked-data"])
    except ValueError:
        leak_rejected = True
    else:
        raise AssertionError("Physical case split leak accepted")
    result = dict(
        schema_version=1,
        positive_numerical_energy_control_passed=True,
        negative_synthetic_thermal_control_failed=True,
        thermal_mae_k=thermal["channels"]["wall_temperature"]["metrics"]["mae"],
        energy_peak_absolute_residual_j=energy["channels"]["energy_residual"]["metrics"][
            "peak_absolute_error"
        ],
        exact_report_replay=True,
        physical_split_leak_rejected=leak_rejected,
        real_experiments_used=False,
        validated=False,
        plant_referenced=False,
        output=str(out),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
