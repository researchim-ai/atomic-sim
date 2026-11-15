import io
import json
from copy import deepcopy

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from atomic_model_multiphysics.provenance import seal,unseal
from atomic_model_station_faults import Fault
from atomic_model_station_env import EnvConfig,StationEnv,ActionError,StationGymEnv
from atomic_model_station_env.__main__ import dispatch,serve


def config(**kw):return EnvConfig(**dict(dict(max_steps=3,tick_s=.01,automatic=False),**kw))


@pytest.mark.parametrize('kw',[dict(max_steps=True),dict(max_steps=8193),dict(tick_s=11.),
    dict(terminal_on_trip=1),dict(seed=-1),dict(automatic=1)])
def test_invalid_environment_configuration(kw):
    with pytest.raises(ValueError):config(**kw)


def test_explicit_reset_observation_and_schema_hide_fault_metadata():
    env=StationEnv(config(faults=(Fault('secret_name','bias','pressure',1,value=150000.),)))
    with pytest.raises(ActionError,match='reset'):env.step()
    first,info=env.reset(seed=7)
    assert first['time_s']==0. and info['steps']==0
    schema=env.available_actions()
    assert 'grid_available' not in schema['operator']
    encoded=json.dumps([first,info,schema])
    assert 'secret_name' not in encoded and 'start_tick' not in encoded and 'physical_state' not in encoded
    first['measurements']['power']['value']=-1
    assert env.observation()[0]['measurements']['power']['value']>0


@pytest.mark.parametrize('action',[{'operator':{'rod_target':True}},{'operator':{'load_fraction':1.3}},
    {'operator':{'grid_available':True}},{'faults':[]},{'modes':{'rod_target':'bad'}},
    {'targets':{'rod_target':float('nan')}},{'reset':1},[]])
def test_invalid_action_is_structured_and_preserves_everything(action):
    env=StationEnv(config());env.reset();before=env.checkpoint()
    with pytest.raises(ActionError) as error:env.step(action)
    assert error.value.code=='invalid_action' and env.checkpoint()==before


def test_step_horizon_is_truncation_and_post_horizon_step_is_atomic():
    env=StationEnv(config(max_steps=1));env.reset()
    obs,reward,terminated,truncated,info=env.step()
    assert obs['time_s']==.01 and isinstance(reward,float) and not terminated and truncated
    assert info['truncation_reason']=='declared_step_horizon'
    before=env.checkpoint()
    with pytest.raises(ActionError) as error:env.step()
    assert error.value.code=='episode_finished' and env.checkpoint()==before


def test_trip_termination_is_declared_separately_from_horizon():
    env=StationEnv(config(terminal_on_trip=True));env.reset()
    _,_,terminated,truncated,info=env.step({'operator':{'scram':True}})
    assert terminated and not truncated and info['termination_reason']=='protection_latched'
    continued=StationEnv(config(terminal_on_trip=False));continued.reset()
    assert not continued.step({'operator':{'scram':True}})[2]
    assert continued.step()[4]['trip_latched']


def test_sorted_json_checkpoint_replay_and_wrong_payload_rejection():
    env=StationEnv(config());env.reset(seed=9);env.step()
    saved=json.loads(json.dumps(env.checkpoint(),sort_keys=True));other=StationEnv.restore(saved)
    assert env.step()==other.step() and env.checkpoint()==other.checkpoint()
    p=unseal(saved);p['finished']=True
    with pytest.raises(ValueError):StationEnv.restore(seal(p))


def test_model_assisted_branches_leave_source_unchanged_and_apply_exactly_one_step():
    env=StationEnv(config());env.reset();before=env.checkpoint()
    preview=env.preview([{}, {'operator':{'scram':True}}, {'operator':{'load_fraction':False}}])
    assert env.checkpoint()==before
    p=unseal(preview);assert [r['accepted'] for r in p['candidates']]==[True,True,False]
    preview=json.loads(json.dumps(preview,sort_keys=True))
    result=env.apply(preview,0)
    assert list(result)==unseal(preview)['candidates'][0]['transition']
    assert env.checkpoint()['payload']['info']['steps']==1
    assert 'checkpoint' not in json.dumps(preview) and 'base_sha256' not in json.dumps(preview)
    before=env.checkpoint()
    with pytest.raises(ActionError) as error:env.apply(preview,1)
    assert error.value.code=='stale_branch' and env.checkpoint()==before


def test_forged_branch_transition_or_checkpoint_cannot_commit():
    env=StationEnv(config());env.reset();comparison=env.preview([{}]);before=env.checkpoint()
    p=unseal(deepcopy(comparison));p['candidates'][0]['transition'][1]=100.
    with pytest.raises(ActionError) as error:env.apply(seal(p),0)
    assert error.value.code=='branch_replay_mismatch' and env.checkpoint()==before
    p=unseal(deepcopy(comparison));p['candidates'][0]['action']['operator']={'scram':True}
    with pytest.raises(ActionError):env.apply(seal(p),0)
    assert env.checkpoint()==before


def test_bad_branch_never_applies_and_nonfinite_rejection_is_serializable():
    env=StationEnv(config());env.reset();before=env.checkpoint()
    p=env.preview([{'targets':{'rod_target':float('nan')}}])
    json.dumps(p,allow_nan=False)
    with pytest.raises(ActionError):env.apply(p,0)
    assert env.checkpoint()==before


def test_jsonl_matches_python_and_rejects_privileged_operations_without_progress():
    cfg=config();env=StationEnv(cfg);direct=StationEnv(cfg)
    requests=[{'method':'reset','seed':3},{'method':'step','action':{}},
              {'method':'checkpoint'},{'method':'preview'},{'method':'observe'}]
    source=io.StringIO(''.join(json.dumps(r)+'\n' for r in requests));output=io.StringIO()
    serve(source,output,env)
    rows=[json.loads(s) for s in output.getvalue().splitlines()]
    obs,info=direct.reset(seed=3)
    assert rows[0]['result']==dict(observation=obs,info=info)
    transition=direct.step()
    assert rows[1]['result']['observation']==transition[0]
    assert not rows[2]['ok'] and not rows[3]['ok'] and rows[4]['result']['info']['steps']==1
    assert env.checkpoint()==direct.checkpoint()


def test_jsonl_oversized_and_malformed_lines_are_drained_without_state_change():
    env=StationEnv(config());env.reset();before=env.checkpoint();out=io.StringIO()
    serve(io.StringIO('x'*70000+'\n{bad json}\n{"method":"observe"}\n'),out,env)
    rows=[json.loads(s) for s in out.getvalue().splitlines()]
    assert [r['ok'] for r in rows]==[False,False,True]
    assert rows[0]['error']['code']=='request_too_large' and env.checkpoint()==before


def test_gymnasium_standard_contract_and_same_seed_replay():
    env=StationGymEnv(config(max_steps=4));check_env(env,skip_render_check=True)
    action=dict(setpoints=np.array([0.,0.,0.]),scram=0)
    first,_=env.reset(seed=42);a=env.step(action)
    second,_=env.reset(seed=42);b=env.step(action)
    for key in first:assert np.array_equal(first[key],second[key])
    for key in a[0]:assert np.array_equal(a[0][key],b[0][key])
    assert a[1:]==b[1:]


def test_invalid_gym_action_does_not_change_core_and_quality_mask_is_measured_only():
    env=StationGymEnv(config(faults=(Fault('secret','failure','pressure',1),)))
    env.reset();before=env.station.checkpoint()
    with pytest.raises(ActionError):env.step(dict(setpoints=np.array([2.,0.,0.]),scram=0))
    assert env.station.checkpoint()==before
    obs,*rest=env.step(dict(setpoints=np.zeros(3),scram=0))
    assert env.observation_space.contains(obs) and obs['valid'][5]==0 and obs['values'][5]==0.
    assert 'secret' not in json.dumps(rest)


@pytest.mark.parametrize('size',[0,9])
def test_branch_budget_is_bounded_without_progress(size):
    env=StationEnv(config());env.reset();before=env.checkpoint()
    with pytest.raises(ActionError):env.preview([{}]*size)
    assert env.checkpoint()==before


def test_hidden_future_fault_does_not_change_initial_public_frames_or_plan_identifiers():
    a=StationEnv(config());b=StationEnv(config(faults=(Fault('hidden','bias','pressure',2,value=150000.),)))
    assert a.reset(seed=1)==b.reset(seed=1)
    assert a.available_actions()==b.available_actions()
    assert a.preview([{}])==b.preview([{}])
    assert a.checkpoint()!=b.checkpoint()


def test_restore_or_reset_expires_pending_privileged_plans():
    env=StationEnv(config());env.reset();comparison=env.preview([{}]);saved=env.checkpoint()
    restored=StationEnv.restore(saved)
    with pytest.raises(ActionError,match='live local plan'):restored.apply(comparison,0)
    env.reset()
    with pytest.raises(ActionError):env.apply(comparison,0)


def test_new_preview_expires_previous_plan_even_without_physical_progress():
    env=StationEnv(config());env.reset();old=env.preview([{}]);current=env.preview([{}])
    with pytest.raises(ActionError):env.apply(old,0)
    assert env.apply(current,0)[4]['steps']==1


def test_model_domain_failure_is_structured_and_does_not_commit(monkeypatch):
    from atomic_model_station.faults import PWRFaultSession
    env=StationEnv(config());env.reset();before=env.checkpoint()
    def fail(*args,**kwargs):raise ValueError('Injected physical domain refusal')
    monkeypatch.setattr(PWRFaultSession,'advance',fail)
    with pytest.raises(ActionError) as error:env.step()
    assert error.value.code=='model_domain_rejected' and env.checkpoint()==before
