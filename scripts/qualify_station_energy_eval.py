"""Integrated electric-energy budget checks, not a plant economic optimum."""
import argparse
from dataclasses import replace
from pathlib import Path
from atomic_model_multiphysics.provenance import seal
from atomic_model_training.spec import atomic_json
from atomic_model_station_env import EnvConfig,StationEnv
from atomic_model_station_control.policy import valid_reading
from atomic_model_station_evaluation import EvaluationCase
from atomic_model_station_energy_eval import ElectricalTask,run_electrical_task,evaluate_electrical_tasks


def run(output):
    out=Path(output).absolute()
    if out.exists() or any(p.is_symlink() for p in (out,*out.parents)):raise ValueError('Fresh ordinary output required')
    out.mkdir(parents=True)
    cases=[EvaluationCase(str(i),EnvConfig(max_steps=i+3,tick_s=.01,automatic=False),
        'research_station','nominal','none','horizon'+str(i)) for i in range(3)]
    assignment={'0':'train','1':'dev','2':'test'};tasks={}
    for case in cases:
        obs,_=StationEnv(case.config).reset();target=valid_reading(obs,'power','fission_power_w')
        tasks[case.name]=ElectricalTask(case.config.max_steps*case.config.tick_s,target,2e7,3e6)
    baseline={split:evaluate_electrical_tasks(cases,assignment,tasks,lambda:lambda *args:{},
        split=split,axes=('time_family',),policy_id='hold/v1') for split in ('train','dev','test')}
    task,config=tasks['0'],cases[0].config
    dhr=run_electrical_task(task,config,lambda:lambda *args:{'operator':{'dhr_requested':True}},policy_id='DHR-meter/control')
    controls={
        'unreachable_export':run_electrical_task(replace(task,minimum_net_export_j=1e14),config,lambda:lambda *args:{}),
        'zero_aux_budget':run_electrical_task(replace(task,maximum_own_needs_j=0.),config,lambda:lambda *args:{}),
        'invalid_action':run_electrical_task(task,config,lambda:lambda *args:{'operator':{'load_fraction':99}}),
        'trip':run_electrical_task(task,config,lambda:lambda *args:{'operator':{'scram':True}}),
    }
    hold=baseline['train']['episodes'][0]
    checks=dict(baselines_pass=all(r['summary']['task_successes']==1 for r in baseline.values()),
        dhr_accounted=dhr['checks']['energy_accounting'] and dhr['accounting']['energy_delta_j']['own_needs_j']>hold['accounting']['energy_delta_j']['own_needs_j'],
        unreachable_export_refused=not controls['unreachable_export']['checks']['export_budget'],
        zero_aux_refused=not controls['zero_aux_budget']['checks']['own_needs_budget'],
        invalid_action_retained=controls['invalid_action']['failure']['code']=='invalid_action',
        trip_refused=not controls['trip']['checks']['no_trip'],
        import_retained=controls['trip']['accounting']['energy_delta_j']['net_exported_j']<0,
        all_negative_tasks_fail=all(not r['task_success'] for r in controls.values()))
    atomic_json(out/'episodes.json',seal(dict(baseline=baseline,dhr_control=dhr,negative_controls=controls)))
    report=dict(passed=all(checks.values()),checks=checks,baseline_episodes=3,dhr_meter_controls=1,negative_controls=4,
        counters='Privileged ideal-model ODE integrals,not physical instruments',
        electrical_budget_contract_qualified=all(checks.values()),thermal_efficiency_available=False,
        industrial_economic_optimality_claimed=False,object_holdout_validated=False,closed_tasks=[])
    atomic_json(out/'report.json',seal(report))
    if not report['passed']:raise ValueError('Electrical budget qualification failed; evidence retained')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    print(run(parser.parse_args().output))
