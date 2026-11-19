from copy import deepcopy
import pytest
from atomic_model_station import PWRPowerSession
from atomic_model_station_hybrid_heat import record_hybrid_window
from atomic_model_station_hybrid_heat.scenarios import prepare_condensate_reopening
from atomic_model_station_hybrid_heat import model


def test_recipe_reaches_closure_without_manual_state_changes():
    session,patch,recipe=prepare_condensate_reopening();saved=session.checkpoint()
    assert not session.condensate_open and session.state[session.feed_index+6]==0.
    assert PWRPowerSession.restore(saved).checkpoint()==saved
    left=deepcopy(session);left._set_controls(patch);left._prepare_interval();left.snapshot()
    assert not left.condensate_open
    assert left.feed_terms['condensate_drive_pa']==pytest.approx(-100.,abs=1e-7)
    assert (left.state==session.state).all()
    assert not recipe['manual_state_projection'] and not recipe['plant_calibrated']


def test_recorded_internal_opening_preserves_native_solver_exactly():
    session,patch,recipe=prepare_condensate_reopening()
    native=deepcopy(session);recorded=model._copy_recorder(session)
    assert recorded.advance(recipe['duration_s'],patch)==native.advance(recipe['duration_s'],patch)
    assert (recorded.state==native.state).all()
    assert [e['kind'] for e in recorded.heat_events]==['condensate_check_opened']
    epoch=recorded.heat_events[0]['time_s']
    assert session.time<epoch<native.time and native.condensate_open
    assert native.state[native.feed_index+6]>0.


def test_reopening_audit_and_restart_are_reproducible_and_readonly():
    session,patch,recipe=prepare_condensate_reopening();saved=session.checkpoint()
    report=record_hybrid_window(session,recipe['duration_s'],patch,subdivisions=16)
    assert report['passed'] and session.checkpoint()==saved
    restored=PWRPowerSession.restore(saved)
    assert record_hybrid_window(restored,recipe['duration_s'],patch,subdivisions=16)==report
    assert [e['kind'] for e in report['events']]==['condensate_check_opened']


def test_zero_pump_head_does_not_fabricate_an_opening():
    session,patch,recipe=prepare_condensate_reopening();patch['condensate_pump_fraction']=0.
    report=record_hybrid_window(session,recipe['duration_s'],patch,subdivisions=8)
    assert report['passed'] and report['events']==[]
