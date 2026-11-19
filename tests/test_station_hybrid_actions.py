from copy import deepcopy
import pytest
from atomic_model_station_env import StationEnv,EnvConfig
from atomic_model_station_event_heat import HeatEventError
from atomic_model_station_hybrid_heat import advance_hybrid_actions
from atomic_model_station_hybrid_heat import rollout


def environment(steps=4):
    env=StationEnv(EnvConfig(max_steps=steps,tick_s=.01,automatic=False,rtol=1e-10))
    env.reset();return env


def test_native_transitions_and_checkpoint_match_at_horizon():
    env=environment(2);native=deepcopy(env);actions=[{}, {'operator':{'load_fraction':.99}}]
    expected=[native.step(a) for a in actions]
    actual,proof=advance_hybrid_actions(env,actions,subdivisions=(2,4,8))
    assert actual==expected and actual[-1][3] and env.checkpoint()==native.checkpoint()
    assert proof['passed'] and proof['native_state_committed']
    assert proof['accounting']['duration_s']==.02
    assert proof['proof_visibility']=='privileged_evaluator'


@pytest.mark.parametrize('actions',[[],{},[{}]*17,[{'operator':{'load_fraction':99}}]])
def test_invalid_sequences_preserve_environment(actions):
    env=environment();before=env.checkpoint()
    with pytest.raises(ValueError):advance_hybrid_actions(env,actions)
    assert env.checkpoint()==before


def test_horizon_is_checked_before_any_execution(monkeypatch):
    env=environment(1);before=env.checkpoint()
    monkeypatch.setattr(rollout,'step_hybrid_heat',lambda *a,**k:pytest.fail('must not run'))
    with pytest.raises(ValueError,match='horizon'):advance_hybrid_actions(env,[{},{}])
    assert env.checkpoint()==before


def test_late_failure_rolls_back_observations_history_and_physics(monkeypatch):
    env=environment();before=env.checkpoint();real=rollout.step_hybrid_heat;calls=[]
    def fail_late(*args,**kwargs):
        calls.append(1)
        if len(calls)==2:raise HeatEventError(dict(reason='injected_solver_refusal'))
        return real(*args,**kwargs)
    monkeypatch.setattr(rollout,'step_hybrid_heat',fail_late)
    with pytest.raises(HeatEventError) as exc:advance_hybrid_actions(env,[{},{}],subdivisions=(2,4,8))
    r=exc.value.receipt
    assert r['action_index']==1 and r['cause']['reason']=='injected_solver_refusal'
    assert len(r['accepted_prefix'])==len(r['accepted_transitions'])==1
    assert not r['native_state_committed'] and not r['prefix_committed_to_caller']
    assert env.checkpoint()==before


def test_early_termination_cannot_silently_drop_remaining_actions(monkeypatch):
    env=environment();before=env.checkpoint();real=rollout.step_hybrid_heat
    def terminal(*args,**kwargs):
        transition,proof=real(*args,**kwargs)
        return (*transition[:2],True,*transition[3:]),proof
    monkeypatch.setattr(rollout,'step_hybrid_heat',terminal)
    with pytest.raises(HeatEventError) as exc:advance_hybrid_actions(env,[{},{}],subdivisions=(2,4,8))
    assert exc.value.receipt['cause']['reason']=='early_episode_end'
    assert env.checkpoint()==before


@pytest.mark.parametrize('value',[float('nan'),float('inf'),-1.,1e20])
def test_aggregate_refuses_invalid_or_excessive_defect_even_with_passed_local_flag(monkeypatch,value):
    env=environment();before=env.checkpoint();real=rollout.step_hybrid_heat
    def corrupt(*args,**kwargs):
        transition,proof=real(*args,**kwargs)
        proof['replays'][-1]['native_endpoint_accounting']['absolute_defect_sums_j']['heat_closure_j']=value
        return transition,proof
    monkeypatch.setattr(rollout,'step_hybrid_heat',corrupt)
    with pytest.raises(HeatEventError) as exc:advance_hybrid_actions(env,[{}],subdivisions=(2,4,8))
    assert exc.value.receipt['reason']=='action_sequence_aggregate_budget' and env.checkpoint()==before


def test_source_change_rolls_back_whole_sequence(monkeypatch):
    env=environment();before=env.checkpoint();versions=iter([1,2])
    monkeypatch.setattr(rollout,'identity',lambda:next(versions))
    with pytest.raises(HeatEventError) as exc:advance_hybrid_actions(env,[{}],subdivisions=(2,4,8))
    assert exc.value.receipt['reason']=='source_changed' and env.checkpoint()==before


def test_checkpoint_continuation_matches_uninterrupted_audit():
    env=environment(3)
    advance_hybrid_actions(env,[{}],subdivisions=(2,4,8))
    restored=StationEnv.restore(env.checkpoint())
    actions=[{'operator':{'load_fraction':.99}},{}]
    a=advance_hybrid_actions(env,actions,subdivisions=(2,4,8))
    b=advance_hybrid_actions(restored,actions,subdivisions=(2,4,8))
    assert a==b and env.checkpoint()==restored.checkpoint()
