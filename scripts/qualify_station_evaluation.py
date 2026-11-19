"""Short station evaluation plumbing, not an industrial-control benchmark."""
import argparse
from pathlib import Path
from atomic_model_multiphysics.provenance import seal
from atomic_model_training.spec import atomic_json
from atomic_model_station_env import EnvConfig
from atomic_model_station_faults import Fault
from atomic_model_station_evaluation import EvaluationCase,evaluate,hold_policy,validate_splits


def run(output):
    out=Path(output).absolute()
    if out.exists() or any(p.is_symlink() for p in (out,*out.parents)):
        raise ValueError('Fresh ordinary output required')
    out.mkdir(parents=True)
    rows=[EvaluationCase('nominal',EnvConfig(max_steps=3,tick_s=.01,automatic=False),
                         'research_station','nominal','none','early'),
          EvaluationCase('bias',EnvConfig(max_steps=3,tick_s=.02,automatic=False,
              faults=(Fault('bias_event','bias','pressure',1,value=15000.),)),
                         'research_station','bias15000','bias','middle'),
          EvaluationCase('unavailable',EnvConfig(max_steps=3,tick_s=.03,automatic=False,
              faults=(Fault('missing_event','failure','power',2),)),
                         'research_station','sensor_failure','failure','late')]
    assignment={'nominal':'train','bias':'dev','unavailable':'test'}
    axes=('parameter_family','fault_family','time_family')
    manifest=validate_splits(rows,assignment,axes=axes)
    atomic_json(out/'manifest.json',seal(manifest))
    baseline={split:evaluate(rows,assignment,lambda:hold_policy,split=split,axes=axes,
        policy_id='hold/v1') for split in ('train','dev','test')}
    refused=evaluate(rows,assignment,lambda:lambda obs,schema:{'operator':{'load_fraction':99}},
        axes=axes,policy_id='invalid-action/control')
    checks=dict(all_baseline_episodes_accounted=all(r['summary']['attempted']==r['summary']['completed']==1 for r in baseline.values()),
        invalid_action_retained=refused['summary']['attempted']==refused['summary']['failed']==1
            and refused['episodes'][0]['failure']['code']=='invalid_action')
    atomic_json(out/'episodes.json',seal(dict(baseline=baseline,negative_control=refused)))
    report=dict(passed=all(checks.values()),checks=checks,baseline_episodes=3,negative_controls=1,
        split_axes=list(axes),object_holdout_validated=False,
        industrial_tasks_qualified=False,station_acceptance_changed=False,closed_tasks=[])
    atomic_json(out/'report.json',seal(report))
    if not report['passed']:raise ValueError('Station evaluation qualification failed')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    print(run(parser.parse_args().output))
