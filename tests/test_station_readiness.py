from copy import deepcopy
import pytest
from atomic_model_station_faults import FaultSession
from atomic_model_station_readiness import ReadinessConfig,schema,capture,assess,prepare_hot_state


def layout():return dict(size=3,names=['primary','drain.mass','external_ledger'],units=['kg','kg','J'],accounting_indices=[2],accounting_reason='test cumulative energy')


def samples(values):
    return [dict(time_s=float(i),state=[1000.,v,float(i)*1e12],energy_relative=0.,mass_relative=0.,boron_relative=0.,
        stored_power_relative=0.,trip=False,goals_met=True) for i,v in enumerate(values)]


def cfg():return ReadinessConfig(window_s=2.,maximum_gap_s=1.)


def test_cumulative_energy_is_excluded_but_reserve_water_drift_is_not():
    stationary=samples([500.,500.,500.]);assert assess(stationary,layout(),cfg())['ready']
    drifting=samples([500.,503.,506.]);result=assess(drifting,layout(),cfg())
    assert not result['ready'] and result['largest_changes'][0]['name']=='drain.mass'


def test_returning_to_same_endpoint_does_not_hide_an_interior_oscillation():
    result=assess(samples([500.,550.,500.]),layout(),cfg())
    assert not result['ready'] and result['maximum_normalized_span']==.1


def test_insufficient_or_sparse_window_is_not_readiness():
    assert assess(samples([500.,500.]),layout(),cfg())['reason']=='insufficient_window'
    s=samples([500.,500.,500.]);s[1]['time_s']=.1
    assert assess(s,layout(),cfg())['reason']=='sparse_window'


@pytest.mark.parametrize('change',[dict(trip=True),dict(goals_met=False),dict(energy_relative=1e-3),dict(stored_power_relative=.1)])
def test_stable_coordinates_do_not_override_goal_or_physical_audit_failures(change):
    s=samples([500.,500.,500.]);s[1].update(change)
    assert not assess(s,layout(),cfg())['ready']


@pytest.mark.parametrize('change',[dict(time_s=float('nan')),dict(state=[1.,2.]),dict(state=[1.,float('inf'),3.]),dict(goals_met=1)])
def test_readiness_refuses_invalid_evidence(change):
    s=samples([500.,500.,500.]);s[1].update(change)
    with pytest.raises(ValueError):assess(s,layout(),cfg())


def test_schema_masks_only_accounting_and_capture_does_not_change_station():
    owner=FaultSession();before=owner.checkpoint();s=schema(owner)
    assert len(s['accounting_indices'])==7
    assert all('drain' not in s['names'][i] for i in s['accounting_indices'])
    record=capture(owner)
    assert len(record['state'])==s['size'] and owner.checkpoint()==before
    assert 'drain.mass' in s['names'] and 'surge.flow' in s['names']


def test_bounded_warmup_preserves_time_and_returns_explicit_unready_checkpoint():
    result=prepare_hot_state(maximum_warmup_s=.03,tick_s=.01,
        config=ReadinessConfig(window_s=.02,maximum_gap_s=.01,relative_span=1e-12))
    assert not result['prepared'] and not result['time_reset'] and not result['pi_reset']
    assert result['completed_time_s']==pytest.approx(.03)
    assert result['checkpoint']['payload']['control']['payload']['state']['plant']['payload']['snapshot']['time_s']==pytest.approx(.03)
    assert result['remaining_supported_time_s']<=result['physical_remaining_s']
    assert result['remaining_supported_time_s']<=result['journal_remaining_s']
    owner=FaultSession(tick_s=.01,history_budget_steps=8192)
    owner.step();before=owner.checkpoint()
    continued=prepare_hot_state(maximum_warmup_s=.04,tick_s=.01,initial_session=owner,
        config=ReadinessConfig(window_s=.02,maximum_gap_s=.01,relative_span=1e-12))
    assert owner.checkpoint()==before
    assert continued['completed_time_s']==pytest.approx(.04)
    assert len(continued['checkpoint']['payload']['history'])==4


@pytest.mark.parametrize('kw',[dict(window_s=0.),dict(relative_span=2.),dict(maximum_gap_s=700.),dict(energy_relative=True)])
def test_invalid_readiness_configuration(kw):
    with pytest.raises(ValueError):ReadinessConfig(**kw)
