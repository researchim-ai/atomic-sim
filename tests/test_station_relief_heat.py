from copy import deepcopy
import pytest
from atomic_model_station import PWRPowerSession,PWRBypassSession,BypassConfig
from atomic_model_station_hybrid_heat import record_hybrid_window
from atomic_model_station_hybrid_heat import model
from atomic_model_station_hybrid_heat.scenarios import prepare_relief_opening


def test_relief_recorder_matches_native_and_keeps_finite_valve_motion():
    session,patch,recipe=prepare_relief_opening();native=deepcopy(session);recorded=model._copy_recorder(session)
    initial=float(native.state[native.bypass_index])
    assert recorded.advance(recipe['duration_s'],patch)==native.advance(recipe['duration_s'],patch)
    assert (recorded.state==native.state).all()
    assert [e['kind'] for e in recorded.heat_events]==['bypass_relief_open']
    event=recorded.heat_events[0]['event']
    assert abs(event['pressure_pa']-native.bypass_config.relief_open_pa)<=.05
    assert event['inventories_projected'] is False
    assert initial<float(native.state[native.bypass_index])<1. and native.relief_latched


def test_relief_audit_is_readonly_and_restarts_exactly():
    session,patch,recipe=prepare_relief_opening();saved=session.checkpoint()
    report=record_hybrid_window(session,recipe['duration_s'],patch,subdivisions=16)
    assert report['passed'] and session.checkpoint()==saved
    restored=PWRBypassSession.restore(saved)
    assert record_hybrid_window(restored,recipe['duration_s'],patch,subdivisions=16)==report
    assert session.time<report['events'][0]['time_s']<session.time+recipe['duration_s']


def test_native_pressure_residual_guard_is_preserved():
    session=PWRBypassSession(bypass=BypassConfig(relief_open_pa=6368300.,relief_close_pa=6368290.),rtol=1e-10)
    recorded=model._copy_recorder(session)
    with pytest.raises(ValueError,match='pressure-root residual'):
        recorded._custom_root('bypass_relief_open',recorded.state,recorded.time)
    assert not recorded.relief_latched and not recorded.heat_events


def test_missing_backend_root_cannot_fall_through_to_condensate_map():
    recorded=model._copy_recorder(PWRPowerSession());before=recorded.condensate_open
    with pytest.raises(ValueError,match='does not implement'):
        recorded._custom_root('bypass_relief_open',recorded.state,recorded.time)
    assert recorded.condensate_open==before and not recorded.heat_events


def test_untripped_control_does_not_fabricate_pressure_relief():
    session=PWRBypassSession(bypass=BypassConfig(relief_open_pa=6368300.,relief_close_pa=6368290.),rtol=1e-10,max_step_s=.01)
    report=record_hybrid_window(session,.02,subdivisions=8)
    assert report['passed'] and not report['events']


def test_longer_trip_window_retains_time_mismatch_refusal():
    session=PWRBypassSession(bypass=BypassConfig(relief_open_pa=6368300.,relief_close_pa=6368290.),rtol=1e-10,max_step_s=.01)
    before=session.checkpoint();report=record_hybrid_window(session,.02,{'turbine_trip':True},subdivisions=16)
    assert not report['passed'] and not report['checks']['root_match']
    assert session.checkpoint()==before
