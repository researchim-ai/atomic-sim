from copy import deepcopy
import pytest
from atomic_model_station_env import StationEnv,EnvConfig
from atomic_model_station_tasks import RecoveryContract,score_recovery,compare_recovery,hold_policy,restart_flow_policy
from atomic_model_station_tasks import recovery
from atomic_model_station_control.policy import valid_reading


def row(i,value=100.,trip=False):return dict(start_s=i*.1,end_s=(i+1)*.1,value=value,trip=trip)


def contract(**kwargs):return RecoveryContract('feed_flow',100.,steps=3,dwell_s=.2,**kwargs)


@pytest.mark.parametrize('kwargs',[dict(channel='power'),dict(target=True),dict(target=float('nan')),
    dict(steps=True),dict(steps=0),dict(relative_band=0.),dict(dwell_s=0.),dict(minimum_iae_gain=0.)])
def test_invalid_contracts_are_refused(kwargs):
    values=dict(channel='feed_flow',target=100.);values.update(kwargs)
    with pytest.raises(ValueError):RecoveryContract(**values)


def test_recovery_needs_excursion_and_final_sampled_dwell():
    c=contract();rows=[row(0,20.),row(1),row(2)]
    r=score_recovery(c,20.,rows,tick_s=.1,execution_completed=True,energy_passed=True)
    assert r['recovered'] and r['final_in_band_dwell_s']==pytest.approx(.2)
    assert r['normalized_iae_s']==pytest.approx(.08)
    assert not score_recovery(c,100.,rows,tick_s=.1,execution_completed=True,energy_passed=True)['recovered']


@pytest.mark.parametrize('change',['missing','late_excursion','trip','partial','failure','budget'])
def test_invalid_or_incomplete_recovery_never_passes(change):
    rows=[row(0),row(1),row(2)];complete=True;energy=True
    if change=='missing':rows[2]['value']=None
    if change=='late_excursion':rows[2]['value']=20.
    if change=='trip':rows[2]['trip']=True
    if change=='partial':rows=rows[:2]
    if change=='failure':complete=False
    if change=='budget':energy=False
    assert not score_recovery(contract(),20.,rows,tick_s=.1,execution_completed=complete,energy_passed=energy)['recovered']


@pytest.mark.parametrize('rows',[[row(1)], [row(0,float('nan'))], [dict(row(0),trip=1)]])
def test_bad_trace_is_rejected(rows):
    with pytest.raises(ValueError):score_recovery(contract(),20.,rows,tick_s=.1,execution_completed=True,energy_passed=True)


def test_unrepresentable_normalized_error_is_refused_without_clipping():
    c=RecoveryContract('feed_flow',1e-308,steps=1,dwell_s=.1)
    with pytest.raises(ValueError,match='Nonfinite'):
        score_recovery(c,0.,[row(0,1e308)],tick_s=.1,execution_completed=True,energy_passed=True)


def environment():
    env=StationEnv(EnvConfig(max_steps=4,tick_s=.02,automatic=False,rtol=1e-10));env.reset();return env


def test_real_comparison_is_readonly_and_policy_sees_only_public_inputs():
    env=environment();before=env.checkpoint();target=valid_reading(env.observation()[0],'feed_flow','feed_flow_kg_s')
    seen=[]
    def policy(observation,schema,objective):
        assert 'measurements' in observation and 'state' not in observation and 'checkpoint' not in observation
        assert 'operator' in schema and 'fork_sha256' not in objective
        seen.append(observation['time_s']);return {}
    c=RecoveryContract('feed_flow',target,steps=2,dwell_s=.02)
    r=compare_recovery(env,c,lambda:policy)
    assert r['comparison_valid'] and not r['task_success'] and r['relative_iae_gain']==0.
    assert seen==[0.,.02] and env.checkpoint()==before and r['caller_unchanged']
    assert r['candidate']['final_checkpoint']==r['baseline']['final_checkpoint']


def fake_arm(recovered,iae,*,failed=False,actions=None):
    return dict(failure={'reason':'refused'} if failed else None,
        metrics=dict(contract_completed=not failed,measurement_coverage=1.,normalized_iae_s=iae,
            initial_outside_band=True,recovered=recovered),accounting=dict(passed=True),
        actions=[{}] if actions is None else actions)


def compare_fake(monkeypatch,candidate,baseline):
    env=environment();arms=iter([candidate,baseline]);monkeypatch.setattr(recovery,'_branch',lambda *a:next(arms))
    return compare_recovery(env,RecoveryContract('feed_flow',100.,steps=1,dwell_s=.01),lambda:hold_policy)


def test_both_recovered_does_not_credit_natural_recovery(monkeypatch):
    r=compare_fake(monkeypatch,fake_arm(True,.01,actions=[{'operator':{'feed_pump_on':True}}]),fake_arm(True,.5))
    assert r['comparison_valid'] and not r['task_success'] and not r['checks']['baseline_not_recovered']


def test_failed_baseline_cannot_make_candidate_look_successful(monkeypatch):
    r=compare_fake(monkeypatch,fake_arm(True,.01,actions=[{'operator':{'feed_pump_on':True}}]),fake_arm(False,.5,failed=True))
    assert not r['comparison_valid'] and not r['task_success'] and r['relative_iae_gain'] is None


def test_failed_candidate_cannot_publish_a_perfect_gain_from_empty_prefix(monkeypatch):
    r=compare_fake(monkeypatch,fake_arm(False,0.,failed=True),fake_arm(False,.5))
    assert not r['comparison_valid'] and not r['task_success'] and r['relative_iae_gain'] is None


def test_unimproved_or_identical_commands_are_not_credited(monkeypatch):
    r=compare_fake(monkeypatch,fake_arm(True,.49),fake_arm(False,.5))
    assert not r['task_success'] and not r['checks']['relative_iae_gain'] and not r['checks']['commands_differ']


def test_valid_controlled_improvement_passes(monkeypatch):
    r=compare_fake(monkeypatch,fake_arm(True,.1,actions=[{'operator':{'feed_pump_on':True}}]),fake_arm(False,.5))
    assert r['task_success'] and r['relative_iae_gain']==pytest.approx(.8)


def test_policy_failure_is_retained_and_baseline_still_runs():
    env=environment()
    def broken():raise RuntimeError('policy startup failed')
    r=compare_recovery(env,RecoveryContract('feed_flow',100.,steps=1,dwell_s=.01),broken)
    assert r['candidate']['failure']['phase']=='policy_initialization'
    assert r['baseline']['metrics']['contract_completed'] and not r['task_success']


def test_rejected_model_command_is_kept_in_attempted_actions():
    env=environment();action={'operator':{'feed_pump_on':'yes'}}
    r=compare_recovery(env,RecoveryContract('feed_flow',100.,steps=1,dwell_s=.01),lambda:lambda *a:action)
    assert r['candidate']['attempted_actions']==[action] and r['candidate']['actions']==[]
    assert r['candidate']['failure']['phase']=='model' and not r['comparison_valid']


def test_source_change_refuses_result(monkeypatch):
    env=environment();before=env.checkpoint();versions=iter([1,2]);monkeypatch.setattr(recovery,'_identity',lambda:next(versions))
    with pytest.raises(ValueError,match='source changed'):
        compare_recovery(env,RecoveryContract('feed_flow',100.,steps=1,dwell_s=.01),lambda:hold_policy)
    assert env.checkpoint()==before


def test_restart_baseline_uses_no_latent_state_and_holds_when_measurement_missing():
    assert restart_flow_policy({'measurements':{}},{'operator':{'feed_pump_on':{}}},dict(channel='feed_flow',target=100.,relative_band=.1))=={}
