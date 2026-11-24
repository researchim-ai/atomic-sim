from dataclasses import replace
import pytest
from atomic_model_station_env import StationEnv,EnvConfig
from atomic_model_station_energy_eval import DispatchContract,score_dispatch,compare_dispatch,schedule_policy
from atomic_model_station_energy_eval import hold_dispatch_policy


def contract():return DispatchContract((1e8,9.8e7,1.02e8,1e8),(1.,.98,1.02,1.))


def rows(c=None):
    c=c or contract()
    return [dict(start_s=i*.1,end_s=(i+1)*.1,net_export_w=v,trip=False,grid_available=True)
            for i,v in enumerate(c.targets_w)]


def score(records=None,**kwargs):
    c=contract();params=dict(tick_s=.1,execution_completed=True,energy_passed=True,net_export_j=sum(c.targets_w)*.1)
    params.update(kwargs)
    return score_dispatch(c,rows() if records is None else records,**params)


def test_scheduled_energy_and_endpoint_metrics_have_separate_gates():
    r=score();assert r['task_success'] and r['normalized_iae_s']==0.
    assert r['scheduled_energy_j']==r['requested_energy_j']==4e7
    assert r['signed_energy_error_j']==0
    bad=score(net_export_j=3e7)
    assert bad['checks']['tracking_band'] and not bad['checks']['energy_delivery'] and not bad['task_success']
    assert bad['signed_energy_error_j']==-1e7


@pytest.mark.parametrize('kwargs',[dict(net_export_j=None),dict(execution_completed=False),dict(energy_passed=False)])
def test_missing_integral_or_failed_execution_cannot_pass(kwargs):
    assert not score(**kwargs)['task_success']


def test_incomplete_horizon_retains_prefix_without_success():
    r=score(rows()[:2],net_export_j=1.98e7)
    assert not r['task_success'] and not r['checks']['full_horizon']
    assert r['requested_energy_j']==1.98e7 and r['scheduled_energy_j']==4e7


@pytest.mark.parametrize('key,value',[('trip',True),('grid_available',False),('net_export_w',5e7)])
def test_physical_and_tracking_failures_are_retained(key,value):
    r=rows();r[1][key]=value
    assert not score(r)['task_success']


@pytest.mark.parametrize('mutation',['gap','reverse','missing','nan','bool','extra','too_many'])
def test_malformed_trace_refused(mutation):
    r=rows()
    if mutation=='gap':r[1]['start_s']+=.01
    elif mutation=='reverse':r.reverse()
    elif mutation=='missing':r[1].pop('net_export_w')
    elif mutation=='nan':r[1]['net_export_w']=float('nan')
    elif mutation=='bool':r[1]['grid_available']=1
    elif mutation=='extra':r[1]['target_w']=1.
    else:r.append(r[-1])
    with pytest.raises(ValueError):score(r)


@pytest.mark.parametrize('kwargs',[dict(targets_w=[]),dict(load_fractions=(1.,)),dict(targets_w=(True,)*4),
    dict(targets_w=(10**1000,)*4),dict(load_fractions=(.4,)*4),dict(load_fractions=(float('nan'),)*4),
    dict(relative_band=0),dict(relative_band=True),dict(minimum_iae_gain=0),dict(minimum_iae_gain=2)])
def test_invalid_contracts_refused(kwargs):
    with pytest.raises(ValueError):replace(contract(),**kwargs)


def test_reference_policy_uses_only_public_fraction_and_schema():
    assert schedule_policy({}, {'operator':{'load_fraction':{}}}, {'load_fraction':.98})=={'operator':{'load_fraction':.98}}
    with pytest.raises(ValueError):schedule_policy({}, {}, {'load_fraction':.98})


def origin():
    env=StationEnv(EnvConfig(max_steps=4,tick_s=.01,automatic=False));env.reset();return env


def test_policy_factory_failure_keeps_baseline_and_original_unchanged():
    env=origin();before=env.checkpoint()
    def broken():raise RuntimeError('deliberate startup error')
    r=compare_dispatch(env,contract(),broken)
    assert r['candidate']['failure']['phase']=='policy_initialization'
    assert r['baseline']['metrics']['checks']['full_horizon']
    assert not r['comparison_valid'] and r['relative_iae_gain'] is None
    assert env.checkpoint()==before


def test_failed_middle_command_keeps_attempted_and_accepted_prefix():
    env=origin()
    def policy(obs,schema,obj):return {} if obj['index']<2 else {'operator':{'load_fraction':99}}
    r=compare_dispatch(env,contract(),lambda:policy)
    assert r['candidate']['failure']['code']=='invalid_action'
    assert len(r['candidate']['attempted_actions'])==3 and len(r['candidate']['actions'])==2
    assert r['candidate']['metrics']['simulated_s']==.02
    assert r['baseline']['metrics']['checks']['full_horizon'] and not r['task_success']


def test_policy_receives_no_meters_config_or_private_faults():
    env=origin();seen=[]
    def policy(obs,schema,obj):
        assert set(obj)=={'kind','index','target_net_export_w','load_fraction','relative_band','interval_s'}
        assert 'samples' not in obj and 'faults' not in schema and 'config' not in obs
        seen.append(obj['index']);obj['index']=999
        return {}
    r=compare_dispatch(env,contract(),lambda:policy)
    assert seen==[0,1,2,3] and not r['task_success']
    assert r['contract']['targets_w']==contract().targets_w


def test_horizon_and_ids_validated_before_policy_creation():
    env=origin()
    for kwargs in (dict(policy_id=''),dict(baseline_id=' ')):
        with pytest.raises(ValueError):compare_dispatch(env,contract(),lambda:pytest.fail('no call'),**kwargs)
    env.step({})
    with pytest.raises(ValueError,match='horizon'):compare_dispatch(env,contract(),lambda:pytest.fail('no call'))


def test_source_guard_preserves_caller(monkeypatch):
    from atomic_model_station_energy_eval import dispatch
    env=origin();before=env.checkpoint();epochs=iter([1,2])
    monkeypatch.setattr(dispatch,'identity',lambda:next(epochs))
    with pytest.raises(ValueError,match='source changed'):
        compare_dispatch(env,contract(),lambda:hold_dispatch_policy)
    assert env.checkpoint()==before


@pytest.mark.parametrize('gate',['no_trip','grid_connected','full_horizon'])
def test_invalid_baseline_cannot_create_a_gain(monkeypatch,gate):
    from copy import deepcopy
    from atomic_model_station_energy_eval import dispatch
    env=origin()
    c=dict(failure=None,metrics=dict(checks=dict(full_horizon=True,no_trip=True,grid_connected=True,
        tracking_band=True),task_success=True,normalized_iae_s=.01),accounting=dict(passed=True),actions=[{}])
    b=deepcopy(c);b['metrics']['checks'][gate]=False;b['metrics']['checks']['tracking_band']=False
    b['metrics']['normalized_iae_s']=.5
    arms=iter([c,b]);monkeypatch.setattr(dispatch,'_branch',lambda *args:next(arms))
    r=compare_dispatch(env,contract(),lambda:hold_dispatch_policy)
    assert not r['comparison_valid'] and not r['task_success'] and r['relative_iae_gain'] is None


def test_perfect_held_baseline_does_not_produce_artificial_gain(monkeypatch):
    from atomic_model_station_energy_eval import dispatch
    env=origin()
    arm=dict(failure=None,metrics=dict(checks=dict(full_horizon=True,no_trip=True,grid_connected=True,
        tracking_band=True),task_success=True,normalized_iae_s=0.),accounting=dict(passed=True),actions=[{}])
    monkeypatch.setattr(dispatch,'_branch',lambda *args:arm)
    r=compare_dispatch(env,contract(),lambda:hold_dispatch_policy)
    assert r['comparison_valid'] and not r['task_success'] and r['relative_iae_gain'] is None
