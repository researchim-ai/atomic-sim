"""Changing electrical dispatch, matched baselines, refinement and restart."""
import argparse
from dataclasses import asdict
import hashlib
from pathlib import Path

from atomic_model_station_env import EnvConfig,StationEnv
from atomic_model_station_energy_eval import read_energy,DispatchContract,compare_dispatch,schedule_policy,hold_dispatch_policy
from atomic_model_station_energy_eval.model import identity
from atomic_model_training.spec import atomic_json
from atomic_model_multiphysics.provenance import seal

FRACTIONS=(1.,.98,.98,1.02,1.02,1.)


def scientific(value):
    if isinstance(value,dict):return {k:scientific(v) for k,v in value.items() if k!='wall_s'}
    if isinstance(value,list):return [scientific(v) for v in value]
    if isinstance(value,tuple):return [scientific(v) for v in value]
    return value


def make_origin(*,fine=False):
    config=EnvConfig(max_steps=12 if fine else 6,tick_s=.05 if fine else .1,
        automatic=False,equilibrium_start=True,max_step_s=.025 if fine else .05,rtol=5e-9 if fine else 2e-8)
    env=StationEnv(config);env.reset();meter=read_energy(env)
    fractions=tuple(f for f in FRACTIONS for _ in range(2 if fine else 1))
    gross=meter['powers']['gross_electrical_w'];own=meter['powers']['own_needs_w']
    return env,DispatchContract(tuple(gross*f-own for f in fractions),fractions)


def invalid_policy(obs,schema,objective):return {'operator':{'load_fraction':99}}


def middle_failure(obs,schema,objective):
    return invalid_policy(obs,schema,objective) if objective['index']==2 else schedule_policy(obs,schema,objective)


def run(output):
    out=Path(output).absolute()
    if out.exists() or any(p.is_symlink() for p in (out,*out.parents)):raise ValueError('Fresh ordinary output required')
    out.mkdir(parents=True);source=identity();driver=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    env,contract=make_origin();fine,fine_contract=make_origin(fine=True);before=env.checkpoint()
    atomic_json(out/'recipe.json',seal(dict(source=source,driver_sha256=driver,
        config=asdict(env.config),contract=asdict(contract),fine_config=asdict(fine.config),fine_contract=asdict(fine_contract),
        target_rule='initial evaluator gross power times public fraction minus initial own-needs power; fixed before policy creation',
        sampled_power_band=.002,integrated_energy_band=.002,
        maximum_common_power_refinement_relative=1e-5,maximum_export_refinement_relative=1e-6,
        public_policy_inputs=['operator observation','available actions','declared current dispatch objective'],
        physical_scope='hot equilibrium, six 0.1s ticks, +/-2% generator-load commands; no long dispatch claim')))
    results={};failures={}
    cases=[('reference',env,contract,schedule_policy,hold_dispatch_policy),
        ('fine',fine,fine_contract,schedule_policy,hold_dispatch_policy),
        ('no-op',env,contract,hold_dispatch_policy,hold_dispatch_policy),
        ('invalid-candidate',env,contract,middle_failure,hold_dispatch_policy),
        ('invalid-baseline',env,contract,schedule_policy,invalid_policy),
        ('tracking-baseline',env,contract,schedule_policy,schedule_policy)]
    for name,origin,objective,candidate,baseline in cases:
        try:
            results[name]=compare_dispatch(origin,objective,lambda p=candidate:p,
                baseline_factory=lambda p=baseline:p,policy_id=name+'/v1',baseline_id=baseline.__name__+'/v1')
            atomic_json(out/(name+'.json'),seal(results[name]))
        except Exception as error:
            failures[name]=dict(type=type(error).__name__,message=str(error))
            atomic_json(out/(name+'.json'),seal(dict(failure=failures[name],task_success=False)))
    restart_exact=False;restart_data=None
    try:
        prefix=StationEnv.restore(before)
        for fraction in FRACTIONS[:2]:prefix.step({'operator':{'load_fraction':fraction}})
        checkpoint=prefix.checkpoint();restored=StationEnv.restore(checkpoint)
        suffix=DispatchContract(contract.targets_w[2:],contract.load_fractions[2:])
        direct=compare_dispatch(prefix,suffix,lambda:schedule_policy,policy_id='restart/v1')
        replay=compare_dispatch(restored,suffix,lambda:schedule_policy,policy_id='restart/v1')
        restart_exact=(scientific(direct)==scientific(replay) and prefix.checkpoint()==checkpoint
            and direct['task_success'] and replay['task_success'])
        restart_data=dict(direct=direct,replayed=replay,exact=restart_exact)
        atomic_json(out/'restart.json',seal(restart_data))
    except Exception as error:failures['restart']=dict(type=type(error).__name__,message=str(error))
    checks=dict(all_attempts_retained=len(results)+len([k for k in failures if k!='restart'])==6,
        source_unchanged=source==identity() and driver==hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        original_unchanged=env.checkpoint()==before,restart_exact=restart_exact)
    power_error=energy_error=None
    if len(results)==6:
        ref=results['reference'];refined=results['fine']
        a=ref['candidate']['samples'];b=refined['candidate']['samples'][::2]
        power_error=max(abs(x['powers']['net_export_w']-y['powers']['net_export_w'])/max(abs(y['powers']['net_export_w']),1.) for x,y in zip(a,b)) if len(a)==len(b) else None
        e=ref['candidate']['accounting']['energy_delta_j']['net_exported_j']
        ef=refined['candidate']['accounting']['energy_delta_j']['net_exported_j']
        energy_error=abs(e-ef)/max(abs(ef),1.)
        checks.update(reference_tracks=ref['task_success'],fine_tracks=refined['task_success'],
            held_baseline_fails=not ref['baseline']['metrics']['task_success'],
            no_op_not_credited=not results['no-op']['task_success'] and results['no-op']['relative_iae_gain']==0.,
            invalid_candidate_keeps_prefix=results['invalid-candidate']['candidate']['failure'] is not None and
                len(results['invalid-candidate']['candidate']['actions'])==2 and
                len(results['invalid-candidate']['candidate']['attempted_actions'])==3,
            invalid_baseline_not_credited=not results['invalid-baseline']['comparison_valid'] and
                results['invalid-baseline']['relative_iae_gain'] is None,
            tracking_baseline_not_credited=not results['tracking-baseline']['task_success'],
            common_grid_refined=power_error is not None and power_error<=1e-5 and energy_error<=1e-6)
    report=dict(source=source,driver_sha256=driver,passed=not failures and len(checks)==12 and all(checks.values()),
        checks=checks,failures=failures,attempted_comparisons=6,restart_comparisons=2,
        maximum_common_power_refinement_relative=power_error,export_refinement_relative=energy_error,
        relative_iae_gain=results.get('reference',{}).get('relative_iae_gain'),
        full_dispatch_cycle_validated=False,industrial_dispatch_validated=False,closed_tasks=[])
    atomic_json(out/'report.json',seal(report))
    atomic_json(out/'scientific.json',seal(scientific(dict(report=report,results=results,restart=restart_data))))
    if not report['passed']:raise ValueError('Dispatch qualification failed; all results retained')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    result=run(parser.parse_args().output)
    print({key:result[key] for key in ('passed','checks','relative_iae_gain')})
