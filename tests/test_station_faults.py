import json
from copy import deepcopy
from dataclasses import asdict

import numpy as np
import pytest

from atomic_model_station import PWRBypassSession
from atomic_model_station.faults import PWRFaultSession
from atomic_model_station_faults import Fault, FaultSession, Monitor, MonitorConfig
from atomic_model_station_faults.spec import canonical
from atomic_model_station_signals import Sensor, StationSignals
from atomic_model_station_control import ControlSession
from atomic_model_multiphysics.provenance import seal, unseal


def signals(faults=(), **kw):
    return StationSignals([Sensor('p', 'steam_pressure_pa', **kw)], faults=faults)


def push(owner, tick, value):
    return owner.ingest(tick, {'steam_pressure_pa': value})['measurements']['p']


@pytest.mark.parametrize('changes', [dict(kind='unknown'), dict(start_tick=True), dict(start_tick=-1),
    dict(end_tick=1), dict(value=float('nan')), dict(name='bad-name'), dict(target='missing',kind='pump_off'),
    dict(kind='cooling_loss',target='external_water',value=1.1), dict(kind='failure',value=1.)])
def test_invalid_fault_contract(changes):
    args=dict(name='f',kind='bias',target='p',start_tick=1)
    args.update(changes)
    with pytest.raises(ValueError):Fault(**args)


def test_schedule_canonical_order_half_open_boundaries_and_overlap_refusal():
    a=Fault('a','bias','p',1,3,value=2.)
    b=Fault('b','stuck','p',3,5)
    assert canonical([b,a])==canonical([a,b])
    assert a.active(1) and not a.active(3) and b.active(3)
    with pytest.raises(ValueError):canonical([a,Fault('c','failure','p',2)])
    with pytest.raises(ValueError):canonical([a,a])
    with pytest.raises(ValueError):canonical([a],domain='equipment')


@pytest.mark.parametrize('kind,expected', [('bias',[102.,112.,122.,130.]),('drift',[100.,112.,124.,130.]),('stuck',[100.,100.,100.,130.])])
def test_acquisition_faults_against_independent_ramp_and_recovery(kind,expected):
    s=signals([Fault('f',kind,'p',0,3,value=2. if kind!='stuck' else 0.)])
    assert [push(s,t,100.+10*t)['value'] for t in range(4)]==expected
    restored=StationSignals.restore(json.loads(json.dumps(s.checkpoint(),sort_keys=True)))
    assert restored.checkpoint()==s.checkpoint()


def test_failed_sample_respects_transport_delay_and_does_not_retroactively_erase_queue():
    s=signals([Fault('f','failure','p',1,2)],delay_ticks=2)
    rows=[push(s,t,100.+t) for t in range(5)]
    assert rows[2]['value']==100. and rows[2]['quality']==['good']
    assert rows[3]['value'] is None and rows[3]['quality']==['unavailable']
    assert rows[4]['value']==102. and rows[4]['quality']==['good']
    assert StationSignals.restore(s.checkpoint()).checkpoint()==s.checkpoint()


def test_dropout_ages_old_data_preserves_pending_delivery_and_recovers():
    s=signals([Fault('f','dropout','p',1,4)],delay_ticks=1,stale_after_ticks=1)
    rows=[push(s,t,100.+t) for t in range(6)]
    assert rows[1]['value']==100. and rows[1]['sampled_at_s']==0.
    assert rows[3]['value']==100. and rows[3]['quality']==['stale']
    assert rows[5]['value']==104. and rows[5]['quality']==['good']
    assert StationSignals.restore(s.checkpoint()).checkpoint()==s.checkpoint()


def test_stuck_at_non_sampling_onset_captures_first_actual_sample_only():
    s=signals([Fault('f','stuck','p',1,5)],period_ticks=2)
    for t in range(4):push(s,t,float(t))
    assert s.fault_memory=={'f':2.}
    saved=s.checkpoint();assert StationSignals.restore(saved).checkpoint()==saved
    assert push(s,4,99.)['value']==2.
    assert push(s,5,100.)['value']==2.
    assert push(s,6,101.)['value']==101.


def test_channel_fault_does_not_change_other_channel_noise_or_truth_memory():
    specs=[Sensor('p','steam_pressure_pa',sigma=1.),Sensor('q','steam_pressure_pa',sigma=1.)]
    a=StationSignals(specs,seed=13);b=StationSignals(specs,seed=13,faults=[Fault('f','stuck','p',1)])
    for t in range(6):
        x=a.ingest(t,{'steam_pressure_pa':float(t)});y=b.ingest(t,{'steam_pressure_pa':float(t)})
        assert x['measurements']['q']==y['measurements']['q']
        assert a.channels['p']['analog']==b.channels['p']['analog']
    assert a.rng['p'].bit_generator.state==b.rng['p'].bit_generator.state


def test_fault_memory_or_failure_quality_tampering_is_rejected():
    s=signals([Fault('f','stuck','p',0)])
    push(s,0,1.);p=unseal(s.checkpoint());p['fault_memory']={}
    with pytest.raises(ValueError):StationSignals.restore(seal(p))
    s=signals([Fault('f','failure','p',0)]);push(s,0,1.)
    p=unseal(s.checkpoint());p['channels']['p']['latest'].pop('unavailable')
    with pytest.raises(ValueError):StationSignals.restore(seal(p))


def frame(t,value,ref,*,sample=None,quality=None):
    sample=t if sample is None else sample
    def row(x):return dict(value=x,unit='Pa',sampled_at_s=sample,available_at_s=sample,age_s=t-sample,quality=quality or ['good'])
    return dict(time_s=t,measurements=dict(pressure=row(value),pressure_reference=row(ref)))


def test_monitor_reports_ambiguity_not_ground_truth_and_has_no_constant_healthy_false_alarm():
    m=Monitor()
    for t in range(5):assert m.advance(frame(t,6e6,6e6))['alarms']==[]
    r=m.advance(frame(5,6.2e6,6e6))
    assert r['alarms']==['paired_disagreement'] and not r['uniquely_diagnosed']
    assert 'bias' not in json.dumps(r)


def test_monitor_does_not_treat_re_reading_delayed_sample_as_stuck():
    m=Monitor()
    for t in range(4):assert not m.advance(frame(t,6e6,6e6,sample=0))['alarms']
    assert m.repeat_count==1


def test_monitor_stuck_requires_reference_motion_and_unique_samples():
    m=Monitor(MonitorConfig(disagreement=1e6,moving_reference=10.,stuck_samples=3))
    for t in range(3):r=m.advance(frame(t,6e6,6e6+10*t))
    assert r['alarms']==['unchanging_channel_with_moving_reference']
    before=m.state()
    with pytest.raises(ValueError):m.advance(frame(2,6e6,6e6))
    assert m.state()==before


def test_no_fault_equipment_subclass_preserves_base_states_exactly():
    a=PWRBypassSession(max_step_s=1.);b=PWRFaultSession(max_step_s=1.)
    for _ in range(2):a.advance(1.);b.advance(1.)
    assert np.array_equal(a.state,b.state)


@pytest.mark.parametrize('kind,target,value', [('pump_off','cooling',0.),('pump_off','feed',0.),('pump_off','condensate',0.),
    ('drive_limit','feed',.72),('drive_limit','condensate',.2),('drive_limit','cooling',.6),('cooling_loss','external_water',.5)])
def test_equipment_fault_changes_physical_rhs_preserving_stocks_and_balances(kind,target,value):
    healthy=PWRFaultSession(max_step_s=1.)
    broken=PWRFaultSession(faults=[Fault('loss',kind,target,0,value=value)],max_step_s=1.)
    a=healthy.advance(1.);b=broken.advance(1.)
    assert not np.array_equal(healthy.state,broken.state)
    assert b['controls']==a['controls']  # Requests unchanged, physically applied capacity differs.
    assert b['mass_relative']<1e-10 and b['energy_relative']<1e-6
    effective=b['equipment_fault_truth']['effective_controls']
    if kind=='pump_off':assert effective[target+'_pump_on'] is False
    if kind=='pump_off' and target=='cooling':assert b['cooling']['pump_head_pa']==0.
    if kind=='drive_limit':assert effective[target+'_pump_fraction']==value
    if kind=='drive_limit' and target=='feed':assert b['feed_train']['feed_controller']['delivered_head_pa']<=12e6*value**2
    if kind=='drive_limit' and target=='condensate':assert b['feed_train']['condensate_head_pa']<=2e6*value**2
    if kind=='cooling_loss':assert b['cooling']['external_flow_kg_s']==pytest.approx(.5*a['cooling']['external_flow_kg_s'])


def test_stuck_valve_defeats_both_controller_and_safety_closure_without_mass_projection():
    s=PWRFaultSession(faults=[Fault('jam','bypass_stuck','bypass',1,3)],max_step_s=1.)
    first=s.advance(1.,dict(steam_bypass=.1));position=first['bypass_position']
    assert position==.1
    r=s.advance(1.,dict(steam_bypass=.8,bypass_available=False))
    assert r['bypass_position']==position and r['header_bypass']['flow_kg_s']>0
    assert r['controls']['steam_bypass']==.8
    assert r['equipment_fault_truth']['stuck_positions']=={'jam':.1}
    saved=json.loads(json.dumps(s.checkpoint(),sort_keys=True))
    restored=PWRFaultSession.restore(saved)
    assert restored.advance(1.)==s.advance(1.)
    assert s.advance(1.)['bypass_position']==0.


def test_equipment_event_epoch_is_atomic_and_replay_rejects_forged_capture():
    s=PWRFaultSession(faults=[Fault('jam','bypass_stuck','bypass',0)])
    before=s.checkpoint()
    with pytest.raises(ValueError):s.advance(1.,dict(steam_bypass=True))
    assert s.checkpoint()==before and not s.fault_events
    with pytest.raises(ValueError):s.advance(.5)
    s.advance(1.);p=unseal(s.checkpoint());p['fault_state']['stuck_positions']['jam']=.9
    with pytest.raises(ValueError):PWRFaultSession.restore(seal(p))


def test_causal_detection_reaction_does_not_disclose_scenario_and_restores_sorted_json():
    s=FaultSession([Fault('secret_fault','bias','pressure',1,value=150000.)],react_on_alarm=True)
    first=s.step()
    assert first['detection']['alarms']==['paired_disagreement'] and first['reaction'] is None
    assert not first['receipt']['trip_latched']
    assert 'secret_fault' not in json.dumps(first) and 'start_tick' not in json.dumps(first)
    saved=json.loads(json.dumps(s.checkpoint(),sort_keys=True));restored=FaultSession.restore(saved)
    a=s.step();b=restored.step()
    assert a==b and a['receipt']['trip_latched']
    assert a['reaction']['observed_at_s']==1.
    assert s.checkpoint()==restored.checkpoint()
    assert ControlSession.restore(s.control.checkpoint()).checkpoint()==s.control.checkpoint()


def test_whole_fault_scenario_rolls_back_monitor_queues_schedule_and_rng_on_bad_action():
    s=FaultSession([Fault('f','pump_off','cooling',0)])
    before=s.checkpoint()
    with pytest.raises(ValueError):s.step(dict(feed_flow_fraction=True))
    assert s.checkpoint()==before


def test_monitor_configuration_rejects_incompatible_channel_units():
    with pytest.raises(ValueError,match='same quantity'):
        FaultSession(monitor=MonitorConfig(sensor='feed_level'))


def test_drift_is_in_units_per_second_on_fractional_clock():
    s=StationSignals([Sensor('p','steam_pressure_pa')],tick_s=.1,
        faults=[Fault('drift','drift','p',0,value=20.)])
    assert push(s,0,100.)['value']==100.
    assert push(s,1,100.)['value']==102.
    assert push(s,2,100.)['value']==104.


def test_simultaneous_equipment_events_do_not_depend_on_input_order():
    faults=[Fault('a','pump_off','cooling',0),Fault('b','drive_limit','feed',0,value=.9)]
    a=PWRFaultSession(faults=faults,max_step_s=1.)
    b=PWRFaultSession(faults=list(reversed(faults)),max_step_s=1.)
    assert a.advance(1.)==b.advance(1.)
    assert a.checkpoint()==b.checkpoint()


def test_dropout_before_first_sample_stays_unavailable_and_restarts_without_inventing_data():
    s=signals([Fault('lost','dropout','p',0,3)],delay_ticks=1)
    for t in range(3):assert push(s,t,float(t))['value'] is None
    restored=StationSignals.restore(s.checkpoint())
    for t in range(3,5):assert push(s,t,float(t))==push(restored,t,float(t))
    assert s.operator_frame()['measurements']['p']['value']==3.
