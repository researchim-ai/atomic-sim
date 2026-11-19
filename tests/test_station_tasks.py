from dataclasses import replace
import pytest
from atomic_model_station_tasks import Task,score,run_task,observation_policy,tracking_policy
from atomic_model_station_env import EnvConfig
from atomic_model_station_faults import Fault


def row(start,end,power=3.15e9,alarm=False,trip=False):
    return dict(start_s=start,end_s=end,power_w=power,alarm=alarm,trip=trip)


def test_tracking_measures_error_and_rejects_missing_data():
    task=Task('tracking',.04)
    r=score(task,[row(0,.02),row(.02,.04,power=None)])
    assert not r['task_success'] and r['missing_measurement_s']==.02
    assert r['mean_normalized_error']==0
    assert score(task,[row(0,.04)])['task_success']


def test_recovery_requires_excursion_and_final_dwell():
    task=Task('recovery',.05,dwell_s=.02)
    assert not score(task,[row(0,.05)])['task_success']
    trace=[row(0,.01,power=2e9),row(.01,.03),row(.03,.04,power=2e9),row(.04,.05)]
    r=score(task,trace)
    assert not r['task_success'] and r['recovery_time_s'] is None
    trace=[row(0,.01,power=2e9),row(.01,.03),row(.03,.05)]
    r=score(task,trace)
    assert r['task_success'] and r['recovery_time_s']==.03


def test_missing_sample_resets_recovery_dwell():
    task=Task('recovery',.05,dwell_s=.02)
    r=score(task,[row(0,.01,power=2e9),row(.01,.03),row(.03,.04,power=None),row(.04,.05)])
    assert not r['task_success'] and r['recovery_time_s'] is None


def test_diagnostic_confusion_uses_decision_epoch_and_keeps_missed_events():
    task=Task('diagnosis',.04)
    r=score(task,[row(0,.01,alarm=True),row(.01,.02),row(.02,.03,alarm=True),row(.03,.04)],
            truth_intervals=((.01,.03),))
    assert r['false_positive_s']==pytest.approx(.01)
    assert r['false_negative_s']==pytest.approx(.01)
    assert r['detection_delays_s']==pytest.approx([.01]) and not r['task_success']
    missed=score(task,[row(0,.01),row(.01,.03),row(.03,.04)],truth_intervals=((.01,.03),))
    assert missed['detection_delays_s']==[None]


def test_perfect_diagnosis_and_negative_control():
    task=Task('diagnosis',.04)
    assert score(task,[row(0,.01),row(.01,.03,alarm=True),row(.03,.04)],truth_intervals=((.01,.03),))['task_success']
    assert score(task,[row(0,.04)])['task_success']
    assert not score(task,[row(0,.04,alarm=True)])['task_success']


@pytest.mark.parametrize('records',[[row(.01,.04)],[row(0,.02),row(.01,.04)],[row(0,.05)],
    [row(0,.04,power=float('nan'))],[row(0,.04,alarm=1)]])
def test_invalid_traces_refused(records):
    with pytest.raises(ValueError):score(Task('tracking',.04),records)


def test_partial_horizon_or_execution_failure_cannot_pass():
    t=Task('tracking',.04)
    assert not score(t,[row(0,.02)])['task_success']
    assert not score(t,[row(0,.04)],execution_completed=False)['task_success']
    assert not score(t,[row(0,.04,trip=True)])['task_success']


@pytest.mark.parametrize('kwargs',[dict(kind='economy'),dict(relative_band=float('nan')),
    dict(dwell_s=.1),dict(target_w=True)])
def test_invalid_and_unobservable_tasks_refused(kwargs):
    params=dict(kind='tracking',horizon_s=.04);params.update(kwargs)
    with pytest.raises(ValueError):Task(**params)


def test_unaligned_truth_is_rejected():
    with pytest.raises(ValueError,match='align'):
        score(Task('diagnosis',.04),[row(0,.04)],truth_intervals=((.01,.03),))


def test_real_task_horizon_and_invalid_decision_are_preserved():
    config=EnvConfig(max_steps=3,tick_s=.01,automatic=False)
    task=Task('tracking',.03,relative_band=.2)
    result=run_task(task,config,lambda:tracking_policy,policy_id='tracking/v1')
    assert result['failure'] is None and result['metrics']['execution_completed']
    assert len(result['records'])==3
    bad=run_task(task,config,lambda:lambda *args:{'action':{},'alarm':'yes'})
    assert bad['failure']['phase']=='policy' and not bad['metrics']['task_success']
    assert bad['records']==[] and bad['wall_s']>=0


def test_privileged_truth_is_not_passed_to_policy_and_labels_cannot_be_forged():
    config=EnvConfig(max_steps=4,tick_s=.01,automatic=False,
        faults=(Fault('hidden','bias','pressure',1,value=150000.),))
    task=Task('diagnosis',.04)
    seen=[]
    def factory():
        def policy(obs,schema,objective):
            assert 'truth_intervals' not in objective and 'faults' not in schema
            assert 'hidden' not in str([obs,schema,objective])
            seen.append(obs['time_s'])
            return observation_policy(obs,schema,objective)
        return policy
    r=run_task(task,config,factory)
    assert seen==[0.,.01,.02,.03]
    assert r['truth_intervals']==[[.01,.04]] and r['failure'] is None
    assert r['metrics']['task_success']
    with pytest.raises(ValueError,match='labels must match'):
        run_task(task,config,factory,truth_intervals=())


def test_failed_step_retains_prefix():
    def factory():
        count=0
        def policy(*args):
            nonlocal count
            count+=1
            return dict(action={} if count==1 else {'operator':{'load_fraction':99}},alarm=False)
        return policy
    r=run_task(Task('tracking',.03),EnvConfig(max_steps=3,tick_s=.01,automatic=False),factory)
    assert r['failure']['code']=='invalid_action' and len(r['records'])==1
    assert r['metrics']['simulated_s']==.01 and not r['metrics']['task_success']


def batch():
    from atomic_model_station_evaluation import EvaluationCase
    cases=[EvaluationCase(str(i),EnvConfig(max_steps=i+3,tick_s=.01,automatic=False),
        str(i),str(i),str(i),str(i)) for i in range(3)]
    return cases,{'0':'train','1':'dev','2':'test'},{c.name:Task('tracking',c.config.max_steps*.01) for c in cases}


def test_batch_retains_failed_task_in_denominator():
    from atomic_model_station_tasks import evaluate_tasks
    cases,assignment,tasks=batch()
    r=evaluate_tasks(cases,assignment,tasks,lambda:lambda *args:dict(action={'operator':{'load_fraction':99}},alarm=False))
    assert r['summary']['attempted']==r['summary']['execution_failures']==1
    assert r['summary']['task_success_rate']==0 and r['episodes'][0]['failure']['code']=='invalid_action'


def test_batch_checks_all_task_horizons_before_running_selected_split():
    from atomic_model_station_tasks import evaluate_tasks
    cases,assignment,tasks=batch();tasks['0']=Task('tracking',.06)
    with pytest.raises(ValueError,match='scenario horizon'):
        evaluate_tasks(cases,assignment,tasks,lambda:pytest.fail('Must not execute'))


def test_invalid_task_map_refused():
    from atomic_model_station_tasks import evaluate_tasks
    cases,assignment,tasks=batch();tasks.pop('0')
    with pytest.raises(ValueError,match='Exact task map'):
        evaluate_tasks(cases,assignment,tasks,lambda:observation_policy)
