from copy import deepcopy
import pytest
from atomic_model_station import PWRPowerSession,PWRBypassSession
from atomic_model_station_hybrid_heat import advance_hybrid_schedule
from atomic_model_station_hybrid_heat import model
from atomic_model_station_event_heat import HeatEventError


def session():return PWRPowerSession(rtol=1e-10,max_step_s=.01)


def test_schedule_matches_native_history_and_state_exactly():
    s=session();native=deepcopy(s)
    schedule=[dict(duration_s=.001,patch=None),dict(duration_s=.001,patch={'load_fraction':.99})]
    for row in schedule:native.advance(row['duration_s'],row['patch'])
    proof=advance_hybrid_schedule(s,schedule,subdivisions=(2,4,8))
    assert proof['passed'] and proof['native_state_committed'] and len(proof['windows'])==2
    assert s.checkpoint()==native.checkpoint()


def test_late_command_failure_rolls_back_whole_schedule():
    s=session();before=s.checkpoint()
    with pytest.raises(HeatEventError) as exc:
        advance_hybrid_schedule(s,[dict(duration_s=.001,patch=None),dict(duration_s=.001,patch={'load_fraction':99})],subdivisions=(2,4,8))
    receipt=exc.value.receipt
    assert receipt['window']==1 and len(receipt['accepted_prefix'])==1
    assert not receipt['native_state_committed'] and s.checkpoint()==before


def test_late_audit_failure_keeps_prefix_diagnostic_but_commits_nothing(monkeypatch):
    s=session();before=s.checkpoint();real=model._record
    def fail(left,native,count):
        report=real(left,native,count)
        if left.time>=.001:report['passed']=False
        return report
    monkeypatch.setattr(model,'_record',fail)
    with pytest.raises(HeatEventError) as exc:
        advance_hybrid_schedule(s,[dict(duration_s=.001,patch=None)]*2,subdivisions=(2,4,8))
    assert exc.value.receipt['reason']=='schedule_heat_refused'
    assert len(exc.value.receipt['accepted_prefix'])==1 and s.checkpoint()==before


@pytest.mark.parametrize('schedule',[[],[dict(duration_s=0.,patch=None)],[dict(duration_s=True,patch=None)],[dict(duration_s=.1)],[dict(duration_s=.1,patch=False)]])
def test_invalid_schedule_is_rejected_before_advancing(schedule):
    s=session();before=s.checkpoint()
    with pytest.raises(ValueError):advance_hybrid_schedule(s,schedule)
    assert s.checkpoint()==before


def test_source_change_rolls_back_schedule(monkeypatch):
    s=session();before=s.checkpoint();versions=iter([1,2]);monkeypatch.setattr(model,'identity',lambda:next(versions))
    with pytest.raises(HeatEventError) as exc:advance_hybrid_schedule(s,[dict(duration_s=.001,patch=None)],subdivisions=(2,4,8))
    assert exc.value.receipt['reason']=='source_changed' and s.checkpoint()==before


def test_joint_native_steam_map_preserves_all_journal_entries():
    recorded=model._copy_recorder(PWRBypassSession())
    indices=[recorded.steam_index+3,recorded.steam_index+4]
    recorded.steam_arrivals=[(recorded.time,i,float(recorded.state[i])) for i in indices]
    before=recorded.state.copy()
    recorded._custom_root('steam_valve_arrival',recorded.state,recorded.time)
    packet=recorded.heat_events[0]
    assert packet['joint_native_map'] and [e['index'] for e in packet['journal_events']]==indices
    assert (recorded.state==before).all()


def test_bypass_position_residual_guard_is_preserved():
    recorded=model._copy_recorder(PWRBypassSession());recorded.bypass_arrival=(0.,1.)
    with pytest.raises(ValueError,match='analytical arrival residual'):
        recorded._custom_root('bypass_position_arrival',recorded.state,0.)
    assert not recorded.heat_events


def test_aggregate_budget_failure_rolls_back_even_if_window_flag_is_corrupt(monkeypatch):
    s=session();before=s.checkpoint();real=model._record
    def corrupt(*args):
        report=real(*args)
        report['native_endpoint_accounting']['absolute_defect_sums_j']['heat_closure_j']=1.
        report['native_endpoint_accounting']['normalization_scale_j']=1.
        return report
    monkeypatch.setattr(model,'_record',corrupt)
    with pytest.raises(HeatEventError) as exc:
        advance_hybrid_schedule(s,[dict(duration_s=.001,patch=None)],subdivisions=(2,4,8))
    assert exc.value.receipt['reason']=='schedule_aggregate_budget' and s.checkpoint()==before
