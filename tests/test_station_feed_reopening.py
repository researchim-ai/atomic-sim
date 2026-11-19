from copy import deepcopy
import pytest
from atomic_model_station import PWRPowerSession
from atomic_model_station_hybrid_heat import record_hybrid_window
from atomic_model_station_hybrid_heat.scenarios import prepare_feed_reopening
from atomic_model_station_hybrid_heat import model


def test_feed_recipe_is_replayable_and_command_does_not_open_valve():
    session,patch,recipe=prepare_feed_reopening();saved=session.checkpoint()
    assert not session.plant._feed_open and session.state[session.plant.feed_index]==0.
    assert PWRPowerSession.restore(saved).checkpoint()==saved
    left=deepcopy(session);left._set_controls(patch);left._prepare_interval();left.snapshot()
    assert not left.plant._feed_open
    assert left.plant.feed_terms()[0]==pytest.approx(-10000.,abs=1e-6)
    assert (left.state==session.state).all()
    assert not recipe['manual_state_projection'] and not recipe['plant_calibrated']


def test_feed_internal_opening_preserves_native_physics_exactly():
    session,patch,recipe=prepare_feed_reopening()
    native=deepcopy(session);recorded=model._copy_recorder(session)
    assert recorded.advance(recipe['duration_s'],patch)==native.advance(recipe['duration_s'],patch)
    assert (recorded.state==native.state).all()
    assert [e['kind'] for e in recorded.heat_events]==['feed_check_opened']
    assert session.time<recorded.heat_events[0]['time_s']<native.time
    assert native.plant._feed_open and native.state[native.plant.feed_index]>0.


def test_feed_reopening_audit_is_readonly_and_restarts_exactly():
    session,patch,recipe=prepare_feed_reopening();saved=session.checkpoint()
    report=record_hybrid_window(session,recipe['duration_s'],patch,subdivisions=16)
    assert report['passed'] and session.checkpoint()==saved
    assert record_hybrid_window(PWRPowerSession.restore(saved),recipe['duration_s'],patch,subdivisions=16)==report
    assert [e['kind'] for e in report['events']]==['feed_check_opened']


def test_feed_zero_head_control_has_no_opening():
    session,patch,recipe=prepare_feed_reopening();patch['feed_pump_fraction']=0.
    report=record_hybrid_window(session,recipe['duration_s'],patch,subdivisions=8)
    assert report['passed'] and not report['events']
