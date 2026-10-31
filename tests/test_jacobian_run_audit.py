import json

import pytest

from scripts.compare_jacobian_runs import compare
from scripts.generate_pipe_pulse import make_spec
from sdk.dynamic_experiment import run_dynamic_experiment


def test_jacobian_audit_rejects_tolerance_or_source_changes(tmp_path):
    s = make_spec(4, scheme="muscl_mc", jacobian_mode="analytic")
    s.update(duration_s=0.02, step_s=0.01)
    a, b, changed = (tmp_path / n for n in ("analytic", "central", "changed"))
    run_dynamic_experiment(s, a)
    s["solver"].pop("jacobian_mode")
    s["solver"]["jacobian_relative_step"] = 1e-5
    run_dynamic_experiment(s, b)
    budget = tmp_path / "budgets.json"
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
    assert compare(a, b, budget)["passed"]
    s["solver"]["rtol"] *= 2
    run_dynamic_experiment(s, changed)
    with pytest.raises(ValueError, match="tolerances"):
        compare(a, changed, budget)
    manifest = b / "manifest.json"
    m = json.loads(manifest.read_text())
    m["source_sha256"] = "0" * 64
    manifest.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="source hashes"):
        compare(a, b, budget)
