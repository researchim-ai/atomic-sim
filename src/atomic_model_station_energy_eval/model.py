"""Use integrated backend counters, never endpoint-power quadrature or double own-load subtraction."""
from copy import deepcopy
from dataclasses import dataclass,asdict
import hashlib
import math
from pathlib import Path
import time

from atomic_model_station_env import EnvConfig,StationEnv,ActionError
from atomic_model_station_control.policy import valid_reading
from atomic_model_station_tasks.model import identity as task_identity

COUNTERS=('gross_generated_j','own_needs_j','net_generated_j','net_exported_j','backup_used_j')
POWERS=('gross_electrical_w','own_needs_w','net_generation_w','net_export_w')


def identity():
    return dict(tasks=task_identity(),files={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(Path(__file__).parent.glob('*.py'))})


def finite(x):return type(x) in (int,float) and math.isfinite(x)


def tolerance(*values):return 1e-5+1e-10*max(1.,*(abs(v) for v in values))


def read_energy(env):
    """Privileged evaluator only: snapshot a copy, leaving live caches/state untouched."""
    if type(env) is not StationEnv:raise ValueError('StationEnv required')
    env._require()
    snapshot=deepcopy(env._session.control.plant).snapshot()
    result=dict(time_s=float(snapshot['time_s']),counters={k:float(snapshot[k]) for k in COUNTERS},
        powers={k:float(snapshot['powers'][k]) for k in POWERS},
        grid_available=snapshot['electrical']['grid_available'],
        trip=snapshot['actuator_status']['scram_latched'] or snapshot['actuator_status']['turbine_trip_latched'],
        energy_relative=float(snapshot['energy_relative']))
    _sample(result)
    return result


def _sample(sample):
    if type(sample) is not dict or set(sample)!={'time_s','counters','powers','grid_available','trip','energy_relative'}:
        raise ValueError('Complete energy sample required')
    if not finite(sample['time_s']) or sample['time_s']<0 or not finite(sample['energy_relative']) or sample['energy_relative']<0:
        raise ValueError('Finite nonnegative clock/global residual required')
    if type(sample['grid_available']) is not bool or type(sample['trip']) is not bool:
        raise ValueError('Boolean grid/trip flags required')
    for group,keys in [('counters',COUNTERS),('powers',POWERS)]:
        if type(sample[group]) is not dict or set(sample[group])!=set(keys) or any(not finite(v) for v in sample[group].values()):
            raise ValueError('Complete finite energy/power fields required')
    for k in ('gross_generated_j','own_needs_j','backup_used_j'):
        if sample['counters'][k]<0:raise ValueError('Nonnegative accumulated generation/consumption required')
    for k in ('gross_electrical_w','own_needs_w'):
        if sample['powers'][k]<0:raise ValueError('Nonnegative gross/own power required')


def assess_energy(samples):
    if type(samples) not in (list,tuple) or not samples:raise ValueError('Nonempty energy samples required')
    for s in samples:_sample(s)
    reasons=set();maximum_power=maximum_counter=0.
    for i,s in enumerate(samples):
        c,p=s['counters'],s['powers']
        defect=c['gross_generated_j']-c['own_needs_j']-c['net_generated_j']
        maximum_counter=max(maximum_counter,abs(defect))
        if abs(defect)>tolerance(c['gross_generated_j'],c['own_needs_j'],c['net_generated_j']):reasons.add('generation_counter_balance')
        power=p['gross_electrical_w']-p['own_needs_w']-p['net_generation_w']
        maximum_power=max(maximum_power,abs(power))
        if abs(power)>tolerance(*p.values()):reasons.add('generation_power_balance')
        expected=p['net_generation_w'] if s['grid_available'] else 0.
        if abs(p['net_export_w']-expected)>tolerance(*p.values()):reasons.add('grid_power_balance')
        if s['energy_relative']>1e-6:reasons.add('physical_energy_budget')
        if i:
            previous=samples[i-1]
            if s['time_s']<=previous['time_s']:raise ValueError('Strictly increasing energy epochs required')
            for key in ('gross_generated_j','own_needs_j','backup_used_j'):
                if c[key]<previous['counters'][key]-tolerance(c[key],previous['counters'][key]):
                    reasons.add('counter_regression')
            if s['grid_available'] and previous['grid_available']:
                generation=c['net_generated_j']-previous['counters']['net_generated_j']
                export=c['net_exported_j']-previous['counters']['net_exported_j']
                if abs(generation-export)>tolerance(generation,export):reasons.add('grid_counter_balance')
    delta={k:samples[-1]['counters'][k]-samples[0]['counters'][k] for k in COUNTERS}
    elapsed=samples[-1]['time_s']-samples[0]['time_s']
    return dict(passed=not reasons,reasons=sorted(reasons),duration_s=elapsed,energy_delta_j=delta,
        net_export_kwh=delta['net_exported_j']/3.6e6,
        average_net_export_w=delta['net_exported_j']/elapsed if elapsed>0 else None,
        own_to_gross_ratio=delta['own_needs_j']/delta['gross_generated_j'] if delta['gross_generated_j']>0 else None,
        maximum_generation_power_defect_w=maximum_power,maximum_generation_counter_defect_j=maximum_counter,
        all_sampled_grid_connected=all(s['grid_available'] for s in samples),
        any_sampled_trip=any(s['trip'] for s in samples),
        ideal_model_counters=True,thermal_efficiency_available=False)


@dataclass(frozen=True)
class ElectricalTask:
    horizon_s: float
    target_power_w: float
    minimum_net_export_j: float
    maximum_own_needs_j: float
    relative_power_band: float=.02

    def __post_init__(self):
        if any(not finite(v) for v in asdict(self).values()):raise ValueError('Finite electrical task parameters required')
        if not .01<=self.horizon_s<=7200 or not 1e6<=self.target_power_w<=3.6e9:
            raise ValueError('Declared research horizon/power required')
        if not 0<=self.minimum_net_export_j<=1e15 or not 0<=self.maximum_own_needs_j<=1e15 or not 0<self.relative_power_band<=.2:
            raise ValueError('Nonnegative bounded energy budgets and power band required')


def run_electrical_task(task,config,policy_factory,*,policy_id='unversioned'):
    if type(task) is not ElectricalTask or type(config) is not EnvConfig or not callable(policy_factory):
        raise ValueError('ElectricalTask,EnvConfig and policy factory required')
    if type(policy_id) is not str or not policy_id.strip() or len(policy_id)>128:raise ValueError('Bounded policy id required')
    if abs(task.horizon_s-config.max_steps*config.tick_s)>1e-10:raise ValueError('Task and environment horizons must match')
    source=identity();started=time.perf_counter();samples=[];tracking=[];failure=None;phase='initialization'
    try:
        env=StationEnv(config);obs,_=env.reset();samples.append(read_energy(env));phase='policy';policy=policy_factory()
        if not callable(policy):raise TypeError('Callable policy required')
        for _ in range(config.max_steps):
            phase='policy'
            action=policy(deepcopy(obs),deepcopy(env.available_actions()),asdict(task))
            phase='model';obs,_,terminated,truncated,info=env.step(action)
            # Both meters and observed tracking are retained at each accepted boundary.
            phase='meter';samples.append(read_energy(env))
            value=valid_reading(obs,'power','fission_power_w')
            tracking.append(dict(time_s=obs['time_s'],power_w=value,trip=info['trip_latched']))
            if terminated or truncated:break
    except Exception as error:
        failure=dict(phase=phase,type=type(error).__name__,code=error.code if isinstance(error,ActionError) else 'execution_error',message=str(error))
    accounting=assess_energy(samples) if samples else None
    complete=failure is None and accounting is not None and abs(accounting['duration_s']-task.horizon_s)<=1e-10 and len(tracking)==config.max_steps
    checks=dict(full_horizon=bool(complete),energy_accounting=accounting is not None and accounting['passed'],
        grid_connected=accounting is not None and accounting['all_sampled_grid_connected'],
        no_trip=accounting is not None and not accounting['any_sampled_trip'] and not any(r['trip'] for r in tracking),
        power_tracking=bool(tracking) and all(r['power_w'] is not None and abs(r['power_w']-task.target_power_w)/task.target_power_w<=task.relative_power_band for r in tracking),
        export_budget=accounting is not None and accounting['energy_delta_j']['net_exported_j']>=task.minimum_net_export_j,
        own_needs_budget=accounting is not None and accounting['energy_delta_j']['own_needs_j']<=task.maximum_own_needs_j)
    if source!=identity():raise ValueError('Energy evaluator/backend changed during run')
    return dict(task=asdict(task),config=asdict(config),policy_id=policy_id,source=source,
        task_success=all(checks.values()),checks=checks,accounting=accounting,samples=samples,
        tracking=tracking,failure=failure,wall_s=time.perf_counter()-started,
        privileged_evaluator_meters=True,industrial_economic_optimality_claimed=False)


def evaluate_electrical_tasks(cases,assignment,tasks,policy_factory,*,split='test',axes=None,policy_id='unversioned'):
    from atomic_model_station_evaluation import validate_splits
    from atomic_model_station_evaluation.model import AXES,SPLITS
    manifest=validate_splits(cases,assignment,axes=AXES if axes is None else axes)
    if split not in SPLITS or type(tasks) is not dict or set(tasks)!={c.name for c in cases}:
        raise ValueError('Exact electrical task map and split required')
    for case in cases:
        task=tasks[case.name]
        if type(task) is not ElectricalTask or abs(task.horizon_s-case.config.max_steps*case.config.tick_s)>1e-10:
            raise ValueError('Every electrical task must match its scenario horizon')
    source=identity();rows=[]
    for case in sorted(cases,key=lambda c:c.name):
        if assignment[case.name]==split:
            rows.append(dict(case=case.name,**run_electrical_task(tasks[case.name],case.config,policy_factory,policy_id=policy_id)))
    if source!=identity():raise ValueError('Energy evaluator/backend changed during batch')
    return dict(manifest=manifest,source=source,split=split,policy_id=policy_id,episodes=rows,
        summary=dict(attempted=len(rows),task_successes=sum(r['task_success'] for r in rows),
            task_success_rate=sum(r['task_success'] for r in rows)/len(rows),
            execution_failures=sum(r['failure'] is not None for r in rows),
            completed=sum(r['checks']['full_horizon'] for r in rows),
            total_wall_s=sum(r['wall_s'] for r in rows),
            total_metered_s=sum(r['accounting']['duration_s'] if r['accounting'] is not None else 0. for r in rows)),
        complete_denominator=True,industrial_economic_optimality_claimed=False)
