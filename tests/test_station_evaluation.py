from dataclasses import replace
import pytest

from atomic_model_station_env import EnvConfig
from atomic_model_station_evaluation import EvaluationCase,validate_splits,evaluate,hold_policy


def cases():
    return [EvaluationCase(str(i),EnvConfig(max_steps=i+1,tick_s=.01,automatic=False),
        'object'+str(i),'params'+str(i),'fault'+str(i),'time'+str(i)) for i in range(3)]


def assignment():return {'0':'train','1':'dev','2':'test'}


def test_manifest_is_order_independent_and_copies_assignment():
    rows=cases();a=assignment();manifest=validate_splits(rows,a)
    assert manifest==validate_splits(list(reversed(rows)),dict(reversed(list(a.items()))))
    a['2']='train'
    assert manifest['assignment']['2']=='test'
    assert not manifest['unseen_physical_objects_validated']


@pytest.mark.parametrize('axis',['object_family','parameter_family','fault_family','time_family'])
def test_shared_axis_rejected_across_splits(axis):
    rows=cases();rows[2]=replace(rows[2],**{axis:getattr(rows[0],axis)})
    with pytest.raises(ValueError,match='Cross-split leakage on '+axis):
        validate_splits(rows,assignment())


def test_noise_seed_and_new_labels_do_not_disguise_duplicate_scenario():
    rows=cases();rows[2]=replace(rows[2],config=replace(rows[0].config,seed=999))
    with pytest.raises(ValueError,match='leakage on scenario'):
        validate_splits(rows,assignment())


def test_holdout_axes_must_be_explicit_and_nonempty():
    rows=cases();rows[2]=replace(rows[2],object_family=rows[0].object_family)
    assert validate_splits(rows,assignment(),axes=('parameter_family',))['axes']==['parameter_family']
    with pytest.raises(ValueError):validate_splits(rows,assignment(),axes=())
    with pytest.raises(ValueError):validate_splits(rows,assignment(),axes=('unknown',))


@pytest.mark.parametrize('a',[{'0':'train'},{'0':'train','1':'dev','2':'dev'},
    {'0':'train','1':'dev','2':'test','extra':'test'},{'0':'train','1':'dev','2':'bogus'}])
def test_incomplete_or_invalid_assignments_rejected(a):
    with pytest.raises(ValueError):validate_splits(cases(),a)


def test_duplicate_case_identity_rejected():
    with pytest.raises(ValueError,match='Unique'):
        validate_splits(cases()+[cases()[0]],assignment())


def test_real_station_hold_policy_is_reproducible_and_costs_are_separate():
    first=evaluate(cases(),assignment(),lambda:hold_policy)
    second=evaluate(cases(),assignment(),lambda:hold_policy)
    assert first['summary']['attempted']==first['summary']['completed']==1
    a,b=first['episodes'][0],second['episodes'][0]
    assert a.pop('wall_s')>=0 and b.pop('wall_s')>=0
    assert a==b and a['steps']==3 and a['simulated_s']==pytest.approx(.03)
    assert a['truncation_reason']=='declared_step_horizon'
    assert not first['completion_is_task_success'] and first['complete_denominator']


def test_invalid_agent_action_remains_in_denominator_and_reports_no_fake_time():
    result=evaluate(cases(),assignment(),lambda:lambda obs,schema:{'operator':{'load_fraction':99}})
    assert result['summary']['attempted']==result['summary']['failed']==1
    row=result['episodes'][0]
    assert row['failure']['code']=='invalid_action' and row['steps']==0 and row['simulated_s']==0
    assert row['wall_s']>=0 and result['summary']['completion_fraction']==0


def test_policy_failure_after_one_step_keeps_partial_cost_and_metrics():
    def factory():
        count=0
        def policy(obs,schema):
            nonlocal count
            count+=1
            if count==2:raise RuntimeError('policy crashed')
            return {}
        return policy
    result=evaluate(cases(),assignment(),factory)
    row=result['episodes'][0]
    assert row['status']=='failed' and row['failure']['phase']=='policy'
    assert row['steps']==1 and row['simulated_s']==pytest.approx(.01)
    assert row['mean_tracking_error'] and result['summary']['failed']==1


def test_policy_gets_no_manifest_and_each_episode_has_fresh_memory():
    rows=cases();rows.append(replace(rows[2],name='extra',config=replace(rows[2].config,seed=8)))
    a=assignment();a['extra']='test';calls=[]
    def factory():
        calls.append([])
        memory=calls[-1]
        def policy(obs,schema):
            assert 'config' not in obs and 'scenario_hash' not in schema and 'faults' not in schema
            memory.append(obs['time_s']);obs.clear();schema.clear()
            return {}
        return policy
    result=evaluate(rows,a,factory)
    assert result['summary']['completed']==2 and calls==[[0.,.01,.02],[0.,.01,.02]]


def test_solver_precision_does_not_create_new_physical_scenario():
    rows=cases();rows[2]=replace(rows[2],config=replace(rows[0].config,rtol=1e-8,max_step_s=.05))
    with pytest.raises(ValueError,match='leakage on scenario'):
        validate_splits(rows,assignment())


def test_fault_renaming_does_not_hide_same_physics():
    from atomic_model_station_faults import Fault
    rows=cases();fault=Fault('original','bias','pressure',1,value=1000.)
    rows[0]=replace(rows[0],config=replace(rows[0].config,faults=(fault,)))
    rows[2]=replace(rows[2],config=replace(rows[0].config,faults=(replace(fault,name='alias'),)))
    with pytest.raises(ValueError,match='leakage on scenario'):
        validate_splits(rows,assignment())
