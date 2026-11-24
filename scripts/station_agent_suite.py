"""Four separate native station agent objectives and explicit baseline controls.

No blended reward or claim of industrial economic optimality. Recovery uses
matched forks; electrical objectives use integrated model counters visible to
an evaluator, not hidden-state access by the policy.
"""
import argparse
from dataclasses import asdict
import hashlib
from pathlib import Path

from atomic_model_station_env import EnvConfig, StationEnv
from atomic_model_station_faults import Fault
from atomic_model_station_control.policy import valid_reading
from atomic_model_station_tasks import Task, run_task, observation_policy, tracking_policy
from atomic_model_station_tasks import RecoveryContract,compare_recovery,hold_policy,restart_flow_policy
from atomic_model_station_tasks.model import identity as task_identity
from atomic_model_station_energy_eval import ElectricalTask,run_electrical_task
from atomic_model_station_energy_eval.model import identity as energy_identity
from atomic_model_multiphysics.provenance import seal
from atomic_model_training.spec import atomic_json


def identity():
    return dict(tasks=task_identity(), energy=energy_identity(),
        recipe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def hold_electrical(observation,schema,objective):return {}


def diagnostic_silent(observation,schema,objective):return dict(action={},alarm=False)


def invalid_electrical(observation,schema,objective):return {'operator':{'load_fraction':99}}


def invalid_measured(observation,schema,objective):return dict(action=invalid_electrical(observation,schema,objective),alarm=False)


def dhr_electrical(observation,schema,objective):return {'operator':{'dhr_requested':True}}


def recipe():
    return dict(format='station-four-objectives/v1',source=identity(),
        target_rule='initial valid publicly measured fission power; then fixed for the entire task',
        tasks=dict(tracking=dict(metrics=['normalized_absolute_error_integral','outside_band_s','missing_measurement_s'],
            reference_policy='tracking_policy',baseline='hold measured diagnostics',horizon_s=.03),
            electrical_budget=dict(metrics=['net_exported_j','own_needs_j','power_tracking','energy_budget'],
                reference_policy='hold',baseline='DHR requested; retain its added consumption',horizon_s=.03,
                minimum_net_export_j=2e7,maximum_own_needs_j=3e6),
            diagnosis=dict(metrics=['false_positive_s','false_negative_s','detection_delays_s'],
                reference_policy='pressure redundancy check',baseline='silent',horizon_s=.04),
            recovery=dict(metrics=['normalized_iae_s','recovery_time_s','final_in_band_dwell_s','relative_iae_gain'],
                reference_policy='restart measured feed flow',baseline='hold same checkpoint',horizon_s=1.)),
        fail_closed=['incomplete horizon','invalid action','missing required samples','budget violation'],
        policy_interface='copies of observation, public action schema, declared objective only',
        electrical_meters='privileged evaluator ODE integrals, not physical instruments',
        score_aggregation='separate objectives; no universal ranking or blended reward',
        full_operational_matrix=False,industrial_utility_validated=False)


def run(output):
    out=Path(output).absolute()
    if out.exists() or any(p.is_symlink() for p in (out,*out.parents)):raise ValueError('Fresh ordinary output required')
    out.mkdir(parents=True);spec=recipe();atomic_json(out/'recipe.json',seal(spec));results={}
    cfg=EnvConfig(max_steps=3,tick_s=.01,automatic=False)
    obs,_=StationEnv(cfg).reset();target=valid_reading(obs,'power','fission_power_w')
    if target is None:raise ValueError('Valid initial measured power required')
    tracking=Task('tracking',.03,target_w=target)
    for name,policy in [('reference',tracking_policy),('baseline',observation_policy),('invalid',invalid_measured)]:
        results['tracking/'+name]=run_task(tracking,cfg,lambda p=policy:p,policy_id=name+'/v1')
    electricity=ElectricalTask(.03,target,2e7,3e6)
    for name,policy in [('reference',hold_electrical),('baseline',dhr_electrical),('invalid',invalid_electrical)]:
        results['electrical_budget/'+name]=run_electrical_task(electricity,cfg,lambda p=policy:p,policy_id=name+'/v1')
    diagnostic_cfg=EnvConfig(max_steps=4,tick_s=.01,automatic=False,
        faults=(Fault('pressure_bias','bias','pressure',1,value=150000.),))
    for name,policy in [('reference',observation_policy),('baseline',diagnostic_silent),('invalid',invalid_measured)]:
        results['diagnosis/'+name]=run_task(Task('diagnosis',.04),diagnostic_cfg,
            lambda p=policy:p,policy_id=name+'/v1')
    origin=StationEnv(EnvConfig(max_steps=25,tick_s=.05,automatic=False,rtol=1e-10))
    obs,_=origin.reset();flow=valid_reading(obs,'feed_flow','feed_flow_kg_s')
    if flow is None:raise ValueError('Valid initial measured flow required')
    for _ in range(4):origin.step({'operator':{'feed_pump_on':False}})
    before=origin.checkpoint();recovery=RecoveryContract('feed_flow',flow,steps=20,dwell_s=.2)
    for name,policy in [('reference',restart_flow_policy),('baseline',hold_policy),('invalid',invalid_electrical)]:
        results['recovery/'+name]=compare_recovery(origin,recovery,lambda p=policy:p,
            baseline_factory=lambda:hold_policy,policy_id=name+'/v1')
        atomic_json(out/('recovery-'+name+'.json'),seal(results['recovery/'+name]))
    checks=dict(tracking_reference=results['tracking/reference']['metrics']['task_success'],
        electrical_reference=results['electrical_budget/reference']['task_success'],
        diagnosis_reference=results['diagnosis/reference']['metrics']['task_success'],
        recovery_reference=results['recovery/reference']['task_success'],
        silent_diagnosis_fails=not results['diagnosis/baseline']['metrics']['task_success'],
        no_op_recovery_fails=not results['recovery/baseline']['task_success'],
        dhr_extra_consumption=results['electrical_budget/baseline']['accounting']['energy_delta_j']['own_needs_j']>
            results['electrical_budget/reference']['accounting']['energy_delta_j']['own_needs_j'],
        caller_unchanged=origin.checkpoint()==before,source_unchanged=spec==recipe(),
        all_invalid_retained=results['tracking/invalid']['failure'] is not None and
            results['diagnosis/invalid']['failure'] is not None and
            results['electrical_budget/invalid']['failure'] is not None and
            not results['recovery/invalid']['comparison_valid'])
    # Keep each objective's metric namespace intact, including all baselines.
    report=dict(recipe=spec,checks=checks,passed=all(checks.values()),attempted_contracts=len(results),
        results=results,closed_tasks=[],industrial_utility_validated=False)
    atomic_json(out/'results.json',seal(report))
    def scientific(value):
        if isinstance(value,dict):return {k:scientific(v) for k,v in value.items() if k!='wall_s'}
        if isinstance(value,list):return [scientific(v) for v in value]
        return value
    atomic_json(out/'scientific.json',seal(scientific(report)))
    if not report['passed']:raise ValueError('Four-objective native suite failed; results retained')
    return dict(passed=True,attempted_contracts=len(results),checks=checks)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    print(run(parser.parse_args().output))
