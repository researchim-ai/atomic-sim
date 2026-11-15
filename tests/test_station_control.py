from copy import deepcopy
import json
import math

import pytest
from atomic_model_station_control import PI,PIConfig,ControlSession,ProtectionConfig
from atomic_model_multiphysics.provenance import seal,unseal


def test_pi_proposal_is_pure_and_manual_auto_transfer_is_bumpless():
    pi=PI(PIConfig('x','steam_admission',1.,1.,kp=.2,ki=.1,bias=.5))
    before=pi.state();p=pi.propose(.2,1.,.4,1.,entering=True)
    assert pi.state()==before and p['command']==pytest.approx(.4)
    pi.commit(p,.4,1.,automatic=True)
    assert pi.state()['integral']==pytest.approx(p['base_integral']+.08)


def test_external_reset_prevents_windup_under_saturation_and_override():
    pi=PI(PIConfig('x','steam_admission',1.,1.,kp=1.,ki=.1,tracking_tau_s=1.))
    for _ in range(100):
        p=pi.propose(0.,1.,0.,1.);pi.commit(p,0.,1.,automatic=True)
    assert abs(pi.integral)<1.1
    p=pi.propose(2.,1.,0.,1.)
    assert p['command']==0.


def test_pi_rate_limit_and_tracking_outside_normal_control_bounds():
    pi=PI(PIConfig('x','steam_admission',1.,1.,kp=1.,ki=.5,lower=.2,bias=.2,rate_per_s=.1,tracking_tau_s=1.))
    p=pi.propose(0.,1.,.5,1.,entering=True);pi.commit(p,.5,1.,automatic=True)
    p=pi.propose(0.,1.,.5,1.)
    assert p['command']==pytest.approx(.6)
    pi.commit(p,0.,1.,automatic=False)
    assert not pi.valid_last and pi.integral==pytest.approx(-1.2)


@pytest.mark.parametrize('bad',[dict(raw=float('nan')),dict(command=2.),dict(saturated=1)])
def test_invalid_pi_commit_is_atomic(bad):
    pi=PI(PIConfig('x','steam_admission',1.,1.));p=pi.propose(1.,1.,.5,1.)
    before=pi.state();p.update(bad)
    with pytest.raises(ValueError):pi.commit(p,.5,1.,automatic=True)
    assert pi.state()==before


@pytest.mark.parametrize('changes',[dict(direction=True),dict(ki=-1.),dict(tracking_tau_s=0.),dict(output='unmodeled_level'),dict(lower=.8,upper=.2),dict(scale=float('inf'))])
def test_pi_configuration_domain(changes):
    kwargs=dict(sensor='x',output='steam_admission',target=1.,scale=1.);kwargs.update(changes)
    with pytest.raises(ValueError):PIConfig(**kwargs)


def test_controller_authority_and_entry_tracking_use_real_positions():
    s=ControlSession();pos=s.plant.snapshot()['actuator_status']['rod_position']
    result=s.step({'rod_target':0.},modes={'rod_target':'auto'})
    r=result['receipt'];d=r['decisions']['rod_target']
    assert d['authority']=='controller' and d['selected']==pytest.approx(pos)
    assert r['actual_positions']['rod']==pytest.approx(pos)
    assert set(r['dispatched'])<=set(r['decisions'])


def test_trip_overrides_operator_controller_and_interlock_without_repeated_actions():
    s=ControlSession(protection=ProtectionConfig(overspeed_fraction=.999))
    initial=s.plant.snapshot()['actuator_status']['rod_position']
    r=s.step({'rod_target':0.,'steam_admission':1.,'steam_inlet_isolated':True},modes={'steam_admission':'auto'})['receipt']
    d=r['decisions']['steam_admission']
    assert d['authority']=='protection' and set(d['candidates'])>={'operator','controller','interlock','protection'}
    assert d['selected']==0. and r['trip_latched']
    assert initial<r['actual_positions']['rod']<1.
    for _ in range(2):
        r=s.step({'scram':False,'turbine_trip':False,'rod_target':0.,'steam_admission':1.},reset=True)['receipt']
        assert not r['dispatched'] and r['trip_latched']
    events=s.plant.snapshot()['events']
    assert sum(e['kind']=='scram_latched' for e in events)==1
    assert sum(e['kind']=='turbine_trip_latched' for e in events)==1
    assert any(e['kind']=='reset_denied' for e in s.events)


def test_interlock_release_restores_held_operator_intent():
    s=ControlSession()
    r=s.step({'generator_connected':False,'steam_admission':1.})['receipt']
    assert r['decisions']['steam_admission']['authority']=='interlock'
    assert s.manual['steam_admission']==1. and s.plant.controls['steam_admission']==0.
    r=s.step({'generator_connected':True})['receipt']
    assert r['dispatched']['steam_admission']==1.
    assert r['decisions']['steam_admission']['reason']=='restored_operator_intent'


def test_protection_runs_without_an_agent_or_automatic_controller():
    s=ControlSession(protection=ProtectionConfig(overspeed_fraction=.999))
    assert s.step()['receipt']['trip_latched']
    assert all(m=='manual' for m in s.modes.values())


def test_measurement_failure_trips_instead_of_using_station_truth():
    from atomic_model_station_signals import StationSignals,Sensor
    bank=StationSignals([Sensor('power','fission_power_w',delay_ticks=2),Sensor('pressure','steam_pressure_pa'),Sensor('speed','turbine_speed_rad_s')])
    loops=[PIConfig('speed','steam_admission',157.,10.,bias=1.)]
    s=ControlSession(instruments=bank,loops=loops)
    r=s.step()['receipt']
    assert r['trip_latched'] and 'measurement_unavailable:power' in r['trip_reasons']


def test_paired_control_checkpoint_restores_actuators_noise_integrals_and_latches():
    s=ControlSession();s.step(modes={k:'auto' for k in s.loops})
    saved=json.loads(json.dumps(s.checkpoint(),sort_keys=True));r=ControlSession.restore(saved)
    for _ in range(2):assert r.step()==s.step()
    assert r.checkpoint()==s.checkpoint()
    p=unseal(saved);p['state']['pi']['rod_target']['integral']+=1.
    with pytest.raises(ValueError):ControlSession.restore(seal(p))


@pytest.mark.parametrize('kwargs',[dict(operator={'steam_admission':2.}),dict(modes={'unknown':'auto'}),dict(modes={'rod_target':'bad'}),dict(targets={'steam_admission':1e12}),dict(reset=1)])
def test_invalid_supervisory_input_rolls_back_every_owner(kwargs):
    s=ControlSession();before=s.checkpoint()
    with pytest.raises(ValueError):s.step(**kwargs)
    assert s.checkpoint()==before


def test_returned_checkpoint_cannot_poison_initial_replay_state():
    s=ControlSession();before=s.checkpoint();copy=s.checkpoint()
    copy['payload']['initial_plant']['payload']['controls']={}
    assert s.checkpoint()==before


def policy_frame(t,*,power=3e9,pressure=6.5e6,speed=157.):
    return dict(time_s=float(t),measurements={name:dict(value=value,unit=unit,sampled_at_s=float(t),available_at_s=float(t),age_s=0.,quality=['good'])
        for name,value,unit in [('power',power,'W'),('pressure',pressure,'Pa'),('speed',speed,'rad/s')]})


def test_synthetic_policy_hysteresis_simultaneity_and_latched_return_conditions():
    from atomic_model_station_control import Supervision
    p=Supervision();normal=dict(grid_available=True,scram=False,turbine_trip=False)
    for t,pressure in enumerate([6.5e6,6.8e6,6.7e6,6.6e6]):
        p.advance(policy_frame(t,pressure=pressure),normal,tick=t,nominal_speed=157.)
        assert p.pressure_active==(t in (1,2))
    p.advance(policy_frame(4,power=4e9,pressure=7.6e6,speed=170.),dict(grid_available=False,scram=True,turbine_trip=True),tick=4,nominal_speed=157.)
    assert p.trip_latched and sum(e['kind']=='protection_latched' for e in p.events)==1
    assert len(p.trip_reasons)==6
    for t in range(5,120):p.advance(policy_frame(t),normal,tick=t,nominal_speed=157.)
    assert p.trip_latched and not p.pressure_active
    assert p.request_reset()['reason']=='new_station_session_required'
    assert sum(e['kind']=='protection_latched' for e in p.events)==1
    assert not Supervision().trip_latched


def test_duplicate_policy_epoch_and_bad_flags_do_not_change_state():
    from atomic_model_station_control import Supervision
    p=Supervision();normal=dict(grid_available=True,scram=False,turbine_trip=False)
    p.advance(policy_frame(0),normal,tick=0,nominal_speed=157.);before=deepcopy(p.__dict__)
    for frame,request,tick in [(policy_frame(0),normal,0),(policy_frame(1),dict(normal,scram=1),1)]:
        with pytest.raises(ValueError):p.advance(frame,request,tick=tick,nominal_speed=157.)
        assert p.__dict__==before


def test_future_or_inconsistent_measurement_is_unavailable_to_protection():
    from atomic_model_station_control import Supervision
    p=Supervision();frame=policy_frame(0);frame['measurements']['power']['available_at_s']=1.
    p.advance(frame,dict(grid_available=True,scram=False,turbine_trip=False),tick=0,nominal_speed=157.)
    assert p.trip_latched and 'measurement_unavailable:power' in p.trip_reasons


@pytest.mark.parametrize('kwargs',[dict(pressure_close_pa=7e6,pressure_open_pa=6.8e6),dict(overspeed_fraction=True),dict(startup_grace_ticks=True)])
def test_protection_configuration_domain(kwargs):
    with pytest.raises(ValueError):ProtectionConfig(**kwargs)


def test_simultaneous_modes_are_independent_of_json_key_order():
    a=ControlSession();b=ControlSession()
    a.step(modes=dict(steam_bypass='auto',rod_target='auto',steam_admission='auto'))
    b.step(modes=dict(rod_target='auto',steam_admission='auto',steam_bypass='auto'))
    assert a.checkpoint()==b.checkpoint()
    saved=json.loads(json.dumps(a.checkpoint(),sort_keys=True))
    restored=ControlSession.restore(saved)
    assert restored.step()==a.step()
