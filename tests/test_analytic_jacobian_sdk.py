import json

import pytest

from core.status import DomainError
from physics.hydraulics.dynamic_channel import DynamicConfig, DynamicWallChannel
from scripts.generate_pipe_pulse import make_spec
from sdk.dynamic_experiment import run_dynamic_experiment


@pytest.mark.parametrize("method", ["BDF", "Radau"])
def test_instantiable_analytic_mode_and_snapshot_restart(method, tmp_path):
    s = make_spec(4, scheme="muscl_mc", jacobian_mode="analytic")
    s.update(duration_s=0.02, step_s=0.01)
    s["solver"]["method"] = method
    r = run_dynamic_experiment(s, tmp_path / "run")
    assert r["numerical_acceptance"]
    d = r["solver_diagnostics"]
    assert d["analytic_jacobian_calls"] > 0
    assert d["analytic_jacobian_successes"] == d["analytic_jacobian_calls"]
    assert d["last_analytic_jacobian_nnz"] > 0
    p = json.loads((tmp_path / "run" / "checkpoint.json").read_text())
    m = DynamicWallChannel.from_snapshot(p)
    assert m.snapshot() == p and m.solver_diagnostics()["analytic_jacobian_calls"] == 0
    m.advance(0.001, method=method, jacobian_mode="analytic")
    assert abs(m.observation()["energy_residual_j"]) < 0.001


@pytest.mark.parametrize("mode,step", [("invented", None), ("analytic", 1e-5)])
def test_invalid_jacobian_choice_creates_no_bundle(mode, step, tmp_path):
    s = make_spec(4)
    s["solver"]["jacobian_mode"] = mode
    if step is not None:
        s["solver"]["jacobian_relative_step"] = step
    with pytest.raises(ValueError):
        run_dynamic_experiment(s, tmp_path / "run")
    assert not (tmp_path / "run").exists()


def test_analytic_failure_rolls_back_all_physical_state(monkeypatch):
    s = make_spec(4, jacobian_mode="analytic")
    m = DynamicWallChannel(DynamicConfig(**s["hydraulics"]))
    before = m.snapshot()

    def fail(*args):
        raise DomainError("injected jacobian failure")

    monkeypatch.setattr(m, "analytic_jacobian", fail)
    with pytest.raises(DomainError, match="jacobian failure"):
        m.advance(0.01, jacobian_mode="analytic")
    assert m.snapshot() == before and m._continuation is None
    assert m.solver_diagnostics()["failed_output_intervals"] == 1


def test_mode_change_discards_history_without_time_jump():
    s = make_spec(4, jacobian_mode="analytic")
    m = DynamicWallChannel(DynamicConfig(**s["hydraulics"]))
    m.enable_continuation(0.02)
    m.advance(0.01, jacobian_mode="analytic")
    engine = m._continuation
    m.advance(0.01, jacobian_relative_step=1e-5)
    assert m._continuation is not engine and m.time_s == 0.02
