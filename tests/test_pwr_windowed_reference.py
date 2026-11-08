"""Work-window resets preserve stocks and remain part of exact Session replay."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from atomic_model_controlled_plant.qualification import protocol
from atomic_model_controlled_plant.session import Session
from atomic_model_multiphysics.provenance import sha
from scripts.run_pwr_windowed_reference import reset_history, tail_checks
from scripts.verify_pwr_refinement import validate_window_journal


def test_actual_reset_preserves_state_ledger_and_exact_replay():
    session = Session(maximum_internal_step_s=.5)
    session.advance(.1)
    before = session.checkpoint()
    ledger = session.ledger.checkpoint()
    reset = reset_history(session)
    assert reset["state_sha256_before"] == reset["state_sha256_after"]
    assert session.checkpoint() == before and session.ledger.checkpoint() == ledger
    session.advance(.1)
    assert session.history[-1]["reset_solver_before_step"] is True
    saved = session.checkpoint()
    restored = Session.restore(saved)
    assert restored.checkpoint() == saved
    assert session.advance(.1) == restored.advance(.1)


def journal_fixture():
    history = [dict(reset_solver_before_step=False) for _ in range(1440)]
    trace = [dict(time_s=float(t), state_sha256="a" * 64) for t in range(5, 7201, 5)]
    resets = []
    for t in range(500, 7200, 500):
        resets.append(dict(time_s=float(t), state_sha256_before="a" * 64, state_sha256_after="a" * 64))
        history[t // 5]["reset_solver_before_step"] = True
    execution = dict(driver_sha256=sha("scripts/run_pwr_windowed_reference.py"),
                     reset_interval_s=500., source_unchanged=True, resets=resets)
    return execution, history, trace


def test_complete_reset_journal_is_admitted():
    validate_window_journal(*journal_fixture(), 7200)


@pytest.mark.parametrize("change", ["missing", "state", "trace", "journal", "source", "driver", "interval"])
def test_refuse_unjournalled_or_state_changing_resets(change):
    execution, history, trace = journal_fixture()
    if change == "missing": execution["resets"].pop()
    elif change == "state": execution["resets"][0]["state_sha256_after"] = "b" * 64
    elif change == "trace": trace[99]["state_sha256"] = "b" * 64
    elif change == "journal": history[100]["reset_solver_before_step"] = False
    elif change == "source": execution["source_unchanged"] = False
    elif change == "driver": execution["driver_sha256"] = "b" * 64
    else: execution["reset_interval_s"] = 1000.
    with pytest.raises(ValueError): validate_window_journal(execution, history, trace, 7200)


def test_settling_retains_native_tail_and_rate_criteria():
    plant = SimpleNamespace(time_s=7200., pack=lambda: np.ones(2), derivatives=lambda t, y: np.zeros(2))
    rows = [dict(time_s=t, relative_fission=.5, primary_flow_kg_s=100., feed_flow_kg_s=10.) for t in (7100., 7150., 7200.)]
    checks, _ = tail_checks(plant, rows, protocol())
    assert all(checks.values())
    changed = deepcopy(rows);changed[-1]["feed_flow_kg_s"] = 20.
    assert not tail_checks(plant, changed, protocol())[0]["settled_feed_flow"]
    plant.derivatives = lambda t, y: np.ones(2) * 1e-3
    assert not tail_checks(plant, rows, protocol())[0]["settled_final_rate"]
    with pytest.raises(ValueError, match="tail"):tail_checks(plant, rows[:-1], protocol())
