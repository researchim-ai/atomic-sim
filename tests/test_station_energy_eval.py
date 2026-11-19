from copy import deepcopy
from dataclasses import replace
import pytest
from atomic_model_station_env import StationEnv,EnvConfig
from atomic_model_station_energy_eval import ElectricalTask,read_energy,assess_energy,run_electrical_task


def sample(t,gross,own,export=None,grid=True):
    net=gross-own
    return dict(time_s=t,counters=dict(gross_generated_j=gross,own_needs_j=own,
        net_generated_j=net,net_exported_j=net if export is None else export,backup_used_j=0.),
        powers=dict(gross_electrical_w=100.,own_needs_w=20.,net_generation_w=80.,net_export_w=80. if grid else 0.),
        grid_available=grid,trip=False,energy_relative=0.)


def test_integrated_counters_not_endpoint_power_and_no_double_subtraction():
    a,b=sample(0.,1000.,200.),sample(2.,1300.,250.)
    result=assess_energy([a,b])
    assert result['passed'] and result['energy_delta_j']['net_exported_j']==250
    assert result['average_net_export_w']==125  # endpoint net power is80, not125
    assert result['net_export_kwh']==pytest.approx(250/3.6e6)
    assert result['own_to_gross_ratio']==pytest.approx(50/300)
    assert not result['thermal_efficiency_available']


def test_net_import_is_negative_and_not_clipped():
    a,b=sample(0.,0.,0.),sample(1.,10.,100.)
    result=assess_energy([a,b])
    assert result['passed'] and result['energy_delta_j']['net_exported_j']==-90.
    assert result['own_to_gross_ratio']==10.


def test_disconnected_grid_does_not_export_net_generation():
    a,b=sample(0.,0.,0.,export=0.,grid=False),sample(1.,0.,20.,export=0.,grid=False)
    r=assess_energy([a,b])
    assert r['passed'] and not r['all_sampled_grid_connected']
    assert r['energy_delta_j']['net_generated_j']==-20 and r['energy_delta_j']['net_exported_j']==0
    assert r['own_to_gross_ratio'] is None


@pytest.mark.parametrize('group,key,value,reason',[
 ('counters','net_generated_j',999.,'generation_counter_balance'),
 ('counters','net_exported_j',999.,'grid_counter_balance'),
 ('powers','net_generation_w',99.,'generation_power_balance'),
 ('powers','net_export_w',99.,'grid_power_balance'),
])
def test_corrupted_balances_refused(group,key,value,reason):
    a,b=sample(0.,0.,0.),sample(1.,100.,20.)
    b[group][key]=value
    r=assess_energy([a,b]);assert not r['passed'] and reason in r['reasons']


def test_counter_regression_and_global_budget_failure_retained():
    a,b=sample(0.,100.,20.),sample(1.,90.,19.)
    b['energy_relative']=1e-3
    r=assess_energy([a,b]);assert 'counter_regression' in r['reasons'] and 'physical_energy_budget' in r['reasons']


@pytest.mark.parametrize('mutation',[lambda s:s.update(time_s=float('nan')),
    lambda s:s['powers'].update(net_export_w=float('inf')),
    lambda s:s.update(grid_available=1),lambda s:s['counters'].pop('own_needs_j')])
def test_invalid_sample_is_not_silently_filled(mutation):
    s=sample(0.,0.,0.);mutation(s)
    with pytest.raises(ValueError):assess_energy([s])


def test_nonincreasing_epochs_rejected():
    with pytest.raises(ValueError,match='increasing'):
        assess_energy([sample(0.,0.,0.),sample(0.,100.,20.)])


def fixture():
    config=EnvConfig(max_steps=3,tick_s=.01,automatic=False)
    env=StationEnv(config);obs,_=env.reset()
    task=ElectricalTask(.03,obs['measurements']['power']['value'],1.,1e8)
    return config,env,task


def test_meter_does_not_mutate_live_state_or_expose_extra_physics():
    _,env,_=fixture();before=env.checkpoint();meter=read_energy(env)
    assert env.checkpoint()==before
    assert set(meter)=={'time_s','counters','powers','grid_available','trip','energy_relative'}
    meter['counters']['gross_generated_j']=999
    assert read_energy(env)['counters']['gross_generated_j']==0


def test_real_energy_task_passes_and_budgets_are_independent():
    config,_,task=fixture()
    r=run_electrical_task(task,config,lambda:lambda *args:{})
    assert r['task_success'] and all(r['checks'].values())
    assert len(r['samples'])==4 and r['accounting']['energy_delta_j']['net_exported_j']>0
    assert not r['accounting']['thermal_efficiency_available']
    for impossible,key in [(replace(task,minimum_net_export_j=1e14),'export_budget'),
                           (replace(task,maximum_own_needs_j=0.),'own_needs_budget')]:
        failed=run_electrical_task(impossible,config,lambda:lambda *args:{})
        assert failed['failure'] is None and not failed['task_success'] and not failed['checks'][key]


def test_policy_error_keeps_metered_prefix_and_hides_counters():
    config,_,task=fixture()
    def factory():
        calls=0
        def policy(obs,schema,objective):
            nonlocal calls
            assert 'counters' not in obs and 'net_generated_j' not in str(schema)
            assert set(objective)=={'horizon_s','target_power_w','minimum_net_export_j','maximum_own_needs_j','relative_power_band'}
            calls+=1
            if calls==2:raise RuntimeError('broken agent')
            return {}
        return policy
    r=run_electrical_task(task,config,factory)
    assert not r['task_success'] and r['failure']['phase']=='policy'
    assert r['accounting']['duration_s']==.01 and len(r['samples'])==2


def test_invalid_action_and_trip_cannot_pass():
    config,_,task=fixture()
    r=run_electrical_task(task,config,lambda:lambda *args:{'operator':{'load_fraction':99}})
    assert r['failure']['code']=='invalid_action' and not r['checks']['full_horizon']
    r=run_electrical_task(task,config,lambda:lambda *args:{'operator':{'scram':True}})
    assert not r['task_success'] and not r['checks']['no_trip']


@pytest.mark.parametrize('kwargs',[dict(minimum_net_export_j=-1),dict(maximum_own_needs_j=float('nan')),
    dict(relative_power_band=0),dict(horizon_s=True)])
def test_invalid_task_refused(kwargs):
    args=dict(horizon_s=.03,target_power_w=3e9,minimum_net_export_j=1.,maximum_own_needs_j=1e8);args.update(kwargs)
    with pytest.raises(ValueError):ElectricalTask(**args)


def batch():
    from atomic_model_station_evaluation import EvaluationCase
    cases=[EvaluationCase(str(i),EnvConfig(max_steps=i+3,tick_s=.01,automatic=False),
                          'one_station','same_parameters','no_fault',str(i)) for i in range(3)]
    tasks={c.name:ElectricalTask(c.config.max_steps*.01,3.023463e9,1.,1e8) for c in cases}
    return cases,{'0':'train','1':'dev','2':'test'},tasks


def test_batch_failure_keeps_full_denominator():
    from atomic_model_station_energy_eval import evaluate_electrical_tasks
    cases,assignment,tasks=batch()
    r=evaluate_electrical_tasks(cases,assignment,tasks,lambda:lambda *args:{'operator':{'load_fraction':99}},axes=('time_family',))
    assert r['summary']['attempted']==r['summary']['execution_failures']==1
    assert r['summary']['task_success_rate']==0 and r['summary']['total_metered_s']==0
    assert r['complete_denominator']


def test_all_batch_horizons_checked_before_execution():
    from atomic_model_station_energy_eval import evaluate_electrical_tasks
    cases,assignment,tasks=batch();tasks['0']=replace(tasks['0'],horizon_s=.04)
    with pytest.raises(ValueError,match='scenario horizon'):
        evaluate_electrical_tasks(cases,assignment,tasks,lambda:pytest.fail('not run'),axes=('time_family',))
