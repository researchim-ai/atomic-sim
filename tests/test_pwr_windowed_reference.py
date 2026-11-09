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
from scripts.run_pwr_windowed_reference import save_progress
from atomic_model_controlled_plant.session import identity
from atomic_model_multiphysics.provenance import unseal
from atomic_model_training.spec import read_json


def test_progress_checkpoint_restores_exactly_without_accepting_prefix(tmp_path):
    session = Session(maximum_internal_step_s=.5)
    session.advance(.1)
    trace, observations = tmp_path / "trace.jsonl", tmp_path / "observations.jsonl"
    trace.write_text("trace\n")
    observations.write_text("observations\n")
    arguments = dict(source=identity(), driver=sha("scripts/run_pwr_windowed_reference.py"),
                     resets=[], trace=trace, observations=observations)
    path = save_progress(tmp_path, session, **arguments)
    receipt = unseal(read_json(path))
    assert receipt["full_horizon_accepted"] is False
    assert receipt["time_s"] == .1
    assert receipt["trace_sha256"] == sha(trace)
    assert receipt["observations_sha256"] == sha(observations)
    assert receipt["trace_bytes"] == len(trace.read_bytes())
    assert receipt["observations_bytes"] == len(observations.read_bytes())
    restored = Session.restore(receipt["checkpoint"])
    assert restored.state_hash() == receipt["state_sha256"]
    assert restored.advance(.1) == session.advance(.1)


def test_progress_refuses_overwrite_and_changed_source(tmp_path):
    session = Session(maximum_internal_step_s=.5)
    trace = tmp_path / "trace.jsonl"
    trace.write_text("")
    arguments = dict(source=identity(), driver=sha("scripts/run_pwr_windowed_reference.py"),
                     resets=[], trace=trace, observations=trace)
    path = save_progress(tmp_path, session, **arguments)
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        save_progress(tmp_path, session, **arguments)
    assert path.read_bytes() == original
    arguments["source"] = {}
    with pytest.raises(ValueError, match="changed source"):
        save_progress(tmp_path, session, **arguments)
    arguments["source"] = identity()
    arguments["driver"] = "0" * 64
    with pytest.raises(ValueError, match="changed source"):
        save_progress(tmp_path, session, **arguments)


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


@pytest.fixture
def short_campaign(monkeypatch):
    import scripts.run_pwr_windowed_reference as worker
    monkeypatch.setattr(worker, 'RESET_INTERVAL_S', .1)
    monkeypatch.setattr(worker, 'SETTINGS', dict(worker.SETTINGS, output_step_s=.1))
    monkeypatch.setattr(worker, 'CASES', {worker.CASE: dict(duration_s=.3, changes=[])})
    # This short software test does not qualify 7200s settling.
    monkeypatch.setattr(worker, 'tail_checks', lambda *args: ({'fixture_only': True}, {}))
    return worker


def test_resume_actual_short_campaign_has_identical_full_evidence(tmp_path, short_campaign):
    worker = short_campaign
    original, resumed = tmp_path / 'original', tmp_path / 'resumed'
    direct = worker.run(original)
    receipt = original / 'progress-checkpoints' / 'step-000001.json'
    before = {p: p.read_bytes() for p in original.rglob('*') if p.is_file()}
    continued = worker.run(resumed, resume_from=receipt)
    assert direct['failure'] is continued['failure'] is None
    for key in ('final_state', 'checkpoint_sha256', 'trace_sha256', 'observations_sha256',
                'events', 'ledger', 'steps', 'checks', 'full_horizon_accepted'):
        assert continued[key] == direct[key], key
    assert all(p.read_bytes() == content for p, content in before.items())
    assert continued['reference_execution']['resets'] == direct['reference_execution']['resets']
    assert (resumed / 'resume-origin.json').exists()


@pytest.mark.parametrize('damage', ['log', 'truncated', 'seal', 'source', 'solver', 'clock', 'reset', 'commands'])
def test_resume_refuses_damaged_prefix_before_creating_output(tmp_path, short_campaign, damage):
    from atomic_model_training.spec import atomic_json
    from atomic_model_multiphysics.provenance import seal
    worker = short_campaign
    original = tmp_path / 'original'
    worker.run(original)
    receipt = original / 'progress-checkpoints' / 'step-000002.json'
    envelope = read_json(receipt)
    if damage in ('log', 'truncated'):
        log = original / 'trace.jsonl'
        raw = log.read_bytes()
        log.write_bytes(b'!' + raw[1:] if damage == 'log' else raw[:10])
    elif damage == 'seal':
        envelope['sha256'] = '0' * 64
        atomic_json(receipt, envelope)
    else:
        payload = unseal(envelope)
        if damage == 'source': payload['source'] = {}
        elif damage == 'solver': payload['solver']['rtol'] = 1e-3
        elif damage == 'clock': payload['time_s'] = .15
        elif damage == 'reset': payload['resets'] = []
        else:
            cp = unseal(payload['checkpoint'])
            cp['history'][0]['commands'] = {'steam_throttle': .2}
            payload['checkpoint'] = seal(cp)
        atomic_json(receipt, seal(payload))
    output = tmp_path / 'refused'
    with pytest.raises(ValueError): worker.run(output, resume_from=receipt)
    assert not output.exists()
