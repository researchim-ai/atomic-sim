"""Short measured task contracts and negative controls on the actual station environment."""
import argparse
from pathlib import Path
from atomic_model_multiphysics.provenance import seal
from atomic_model_training.spec import atomic_json
from atomic_model_station_env import EnvConfig,StationEnv
from atomic_model_station_control.policy import valid_reading
from atomic_model_station_faults import Fault
from atomic_model_station_evaluation import EvaluationCase
from atomic_model_station_tasks import Task,evaluate_tasks,observation_policy,tracking_policy


def run(output):
    out=Path(output).absolute()
    if out.exists() or any(p.is_symlink() for p in (out,*out.parents)):raise ValueError('Fresh ordinary output required')
    out.mkdir(parents=True)
    cases=[EvaluationCase('tracking',EnvConfig(max_steps=3,tick_s=.01,automatic=False),
            'research_station','nominal','none','tracking_horizon'),
        EvaluationCase('recovery',EnvConfig(max_steps=5,tick_s=.01,automatic=False,
            faults=(Fault('power_excursion','bias','power',1,end_tick=2,value=-4e8),)),
            'research_station','power_bias','temporary_power_bias','recovery_horizon'),
        EvaluationCase('diagnosis',EnvConfig(max_steps=4,tick_s=.01,automatic=False,
            faults=(Fault('pressure_fault','bias','pressure',1,value=150000.),)),
            'research_station','pressure_bias','persistent_pressure_bias','diagnostic_horizon')]
    assignment={'tracking':'train','recovery':'dev','diagnosis':'test'}
    fixed_tasks={c.name:Task(c.name,c.config.max_steps*c.config.tick_s) for c in cases}
    tasks={}
    for case in cases:
        observation,_=StationEnv(case.config).reset()
        target=valid_reading(observation,'power','fission_power_w')
        if target is None:raise ValueError('Initial measured reference unavailable')
        tasks[case.name]=Task(case.name,case.config.max_steps*case.config.tick_s,target_w=target)
    axes=('parameter_family','fault_family','time_family')
    def baseline(obs,schema,objective):
        return (tracking_policy if objective['kind']=='tracking' else observation_policy)(obs,schema,objective)
    results={split:evaluate_tasks(cases,assignment,tasks,lambda:baseline,split=split,axes=axes,policy_id='measured-reference/v1')
             for split in ('train','dev','test')}
    always=evaluate_tasks(cases,assignment,tasks,lambda:lambda *args:dict(action={},alarm=True),axes=axes,policy_id='always-alarm/control')
    rejected=evaluate_tasks(cases,assignment,tasks,lambda:lambda *args:dict(action={'operator':{'load_fraction':99}},alarm=False),axes=axes,policy_id='invalid-action/control')
    fixed=evaluate_tasks(cases,assignment,fixed_tasks,lambda:baseline,split='train',axes=axes,policy_id='fixed-3.15GW/control')
    checks=dict(fixed_target_refused=fixed['summary']['completed']==1 and fixed['summary']['task_successes']==0,
        reference_contracts=all(r['summary']['task_successes']==1 for r in results.values()),
        false_alarm_refused=always['summary']['completed']==1 and always['summary']['task_successes']==0
            and always['episodes'][0]['metrics']['false_positive_s']>0,
        invalid_action_retained=rejected['summary']['attempted']==rejected['summary']['execution_failures']==1
            and rejected['summary']['task_success_rate']==0,
        recovery_has_excursion=results['dev']['episodes'][0]['metrics']['observed_excursion'])
    atomic_json(out/'episodes.json',seal(dict(reference=results,always_alarm=always,invalid_action=rejected,fixed_target=fixed)))
    report=dict(passed=all(checks.values()),checks=checks,reference_episodes=3,negative_controls=3,
        reference_definition="Maintain initial publicly measured power;original fixed3.15GW failure retained separately",
        economic_objective_supported=False,industrial_tasks_qualified=False,object_holdout_validated=False,
        closed_tasks=[],station_acceptance_changed=False)
    atomic_json(out/'report.json',seal(report))
    if not report['passed']:raise ValueError('Measured task qualification failed; evidence retained')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    print(run(parser.parse_args().output))
