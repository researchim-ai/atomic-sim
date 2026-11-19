"""Actual controlled feed recovery, natural recovery and failed-comparator controls."""
import argparse
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from atomic_model_station_env import StationEnv,EnvConfig
from atomic_model_station_control.policy import valid_reading
from atomic_model_station_tasks import RecoveryContract,compare_recovery,hold_policy,restart_flow_policy
from atomic_model_station_tasks.recovery import _identity
from atomic_model_multiphysics.provenance import seal,sha
from atomic_model_training.spec import atomic_json


def prepare():
    env=StationEnv(EnvConfig(max_steps=25,tick_s=.05,automatic=False,rtol=1e-10))
    observation,_=env.reset();target=valid_reading(observation,'feed_flow','feed_flow_kg_s')
    if target is None:raise ValueError('Initial measured reference unavailable')
    for _ in range(4):env.step({'operator':{'feed_pump_on':False}})
    return env,RecoveryContract('feed_flow',target,steps=20,dwell_s=.2)


def invalid_policy(observation,schema,objective):return {'operator':{'feed_pump_on':'yes'}}


def run(output):
    out=Path(output).absolute()
    if out.exists() or any(p.is_symlink() for p in (out,*out.parents)):raise ValueError('Fresh ordinary output required')
    source=_identity();driver=sha(__file__);out.mkdir(parents=True);checks={};results={};failures={};restart=None
    env,contract=prepare();before=env.checkpoint()
    natural=deepcopy(env);natural.step({'operator':{'feed_pump_on':True}});natural_before=natural.checkpoint()
    atomic_json(out/'recipe.json',seal(dict(source=source,config=asdict(env.config),contract=asdict(contract),
        preparation_steps=4,preparation_command={'operator':{'feed_pump_on':False}},
        natural_extra_command={'operator':{'feed_pump_on':True}},fork_checkpoint=before,
        natural_fork_checkpoint=natural_before,manual_state_projection=False)))
    cases=[('controlled',env,restart_flow_policy,hold_policy),('no-op',env,hold_policy,hold_policy),
        ('natural',natural,hold_policy,hold_policy),('invalid-policy',env,invalid_policy,hold_policy),
        ('invalid-baseline',env,restart_flow_policy,invalid_policy)]
    for name,origin,candidate,baseline in cases:
        try:
            result=compare_recovery(origin,contract,lambda:candidate,baseline_factory=lambda:baseline,
                policy_id='restart-flow/v1' if candidate is restart_flow_policy else name+'/v1',
                baseline_id='hold/v1' if baseline is hold_policy else 'invalid/control')
            results[name]=result;atomic_json(out/(name+'.json'),seal(result))
        except Exception as error:
            failures[name]=dict(type=type(error).__name__,message=str(error))
            atomic_json(out/(name+'.json'),seal(dict(failure=failures[name],task_success=False)))
    try:
        restored=StationEnv.restore(before)
        restart=compare_recovery(restored,contract,lambda:restart_flow_policy,policy_id='restart-flow/v1')
        atomic_json(out/'restart.json',seal(restart))
        checks['restart_exact']=restart==results.get('controlled')
    except Exception as error:failures['restart']=dict(type=type(error).__name__,message=str(error))
    if len(results)==5:
        checks.update(controlled_success=results['controlled']['task_success'],
            no_op_not_credited=not results['no-op']['task_success'] and results['no-op']['relative_iae_gain']==0.,
            natural_not_credited=not results['natural']['task_success'] and all(results['natural'][arm]['metrics']['recovered'] for arm in ('candidate','baseline')),
            invalid_policy_retained=not results['invalid-policy']['comparison_valid'] and bool(results['invalid-policy']['candidate']['attempted_actions']),
            invalid_baseline_not_credited=not results['invalid-baseline']['task_success'] and not results['invalid-baseline']['comparison_valid'])
    checks['all_attempts_retained']=len(results)+len([k for k in failures if k!='restart'])==5
    checks['originals_unchanged']=env.checkpoint()==before and natural.checkpoint()==natural_before
    checks['source_unchanged']=source==_identity() and driver==sha(__file__)
    report=dict(source=source,driver_sha256=driver,checks=checks,failures=failures,
        passed=not failures and len(checks)==9 and all(checks.values()),attempted=5,
        observed_task_successes=sum(r['task_success'] for r in results.values()),negative_controls=4,
        complete_denominator=True,deliberate_controls_not_population_statistics=True,
        controlled_metrics={arm:results['controlled'][arm]['metrics'] for arm in ('candidate','baseline')} if 'controlled' in results else None,
        controlled_relative_iae_gain=results.get('controlled',{}).get('relative_iae_gain'),
        industrial_utility_validated=False,npp_14_03_accepted=False,closed_tasks=[])
    atomic_json(out/'report.json',seal(report))
    if not report['passed']:raise ValueError('Controlled recovery qualification failed; every attempt retained')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    print(run(parser.parse_args().output)['checks'])
