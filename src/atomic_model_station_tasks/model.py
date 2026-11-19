"""Sampled objectives with private diagnostic labels and complete-horizon gates."""
from copy import deepcopy
from dataclasses import dataclass,asdict
import hashlib
import math
from pathlib import Path
import time

from atomic_model_station_env import EnvConfig,StationEnv,ActionError
from atomic_model_station_evaluation.model import identity as evaluation_identity
from atomic_model_station_control.policy import valid_reading


def identity():
    return dict(evaluation=evaluation_identity(),files={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(Path(__file__).parent.glob('*.py'))})


def finite(x):return type(x) in (int,float) and math.isfinite(x)


@dataclass(frozen=True)
class Task:
    kind: str
    horizon_s: float
    target_w: float=3.15e9
    relative_band: float=.02
    dwell_s: float=.02
    diagnostic_threshold_pa: float=100000.

    def __post_init__(self):
        if self.kind not in ('tracking','recovery','diagnosis'):
            raise ValueError('Supported measured task required; economic utility is not observable')
        for name in ('horizon_s','target_w','relative_band','dwell_s','diagnostic_threshold_pa'):
            if not finite(getattr(self,name)):
                raise ValueError('Finite task parameter required')
        if not .01<=self.horizon_s<=7200 or not 1e6<=self.target_w<=3.6e9:
            raise ValueError('Task horizon/target outside supported research bounds')
        if not 0<self.relative_band<=.2 or not 0<self.dwell_s<=self.horizon_s or not 0<self.diagnostic_threshold_pa<=1e8:
            raise ValueError('Bounded task tolerance/dwell/diagnostic threshold required')


def _truth(intervals,horizon):
    if type(intervals) not in (tuple,list):raise ValueError('Private interval list required')
    previous=0.
    for pair in intervals:
        if type(pair) not in (tuple,list) or len(pair)!=2:raise ValueError('Truth interval pair required')
        start,end=pair
        if not finite(start) or not finite(end) or not previous<=start<end<=horizon:
            raise ValueError('Ordered disjoint bounded truth intervals required')
        previous=end
    return tuple(tuple(pair) for pair in intervals)


def score(task,records,*,truth_intervals=(),execution_completed=True):
    if type(task) is not Task or type(records) not in (list,tuple) or type(execution_completed) is not bool:
        raise ValueError('Typed task,records and completion required')
    truth=_truth(truth_intervals,task.horizon_s)
    if task.kind!='diagnosis' and truth:raise ValueError('Diagnostic truth only belongs to diagnostic task')
    elapsed=valid=iae=outside=false_positive=false_negative=dwell=0.
    excursion=False;recovery_time=None;trip=False;first_detection={}
    for row in records:
        start,end=row['start_s'],row['end_s']
        if not finite(start) or not finite(end) or abs(start-elapsed)>1e-10 or not start<end<=task.horizon_s+1e-10:
            raise ValueError('Consecutive bounded trace intervals required')
        if type(row['alarm']) is not bool or type(row['trip']) is not bool:
            raise ValueError('Boolean decision/trip required')
        dt=end-start;elapsed=end;trip=trip or row['trip']
        value=row['power_w']
        if value is not None and not finite(value):raise ValueError('Finite measured power or missing value required')
        if value is not None:
            error=abs(value-task.target_w)/task.target_w
            valid+=dt;iae+=error*dt
            if error<=task.relative_band:
                dwell+=dt
                if excursion and recovery_time is None and dwell+1e-10>=task.dwell_s:
                    recovery_time=end
            else:
                outside+=dt;excursion=True;dwell=0.;recovery_time=None
        else:
            dwell=0.;recovery_time=None
        # Alarm describes the operator frame at start, before applying this action.
        active=[i for i,(a,b) in enumerate(truth) if a<=start<b]
        if any(start<a<end or start<b<end for a,b in truth):
            raise ValueError('Truth transitions must align with decision intervals')
        if row['alarm'] and not active:false_positive+=dt
        if active and not row['alarm']:false_negative+=dt
        if active and row['alarm']:
            first_detection.setdefault(active[0],start-truth[active[0]][0])
    complete=execution_completed and abs(elapsed-task.horizon_s)<=1e-10
    coverage=valid/elapsed if elapsed else 0.
    if task.kind=='diagnosis':
        success=complete and false_positive==0 and false_negative==0
    elif task.kind=='tracking':
        success=complete and not trip and coverage>=1-1e-12 and outside==0
    else:
        success=complete and not trip and coverage>=1-1e-12 and excursion and recovery_time is not None and dwell+1e-10>=task.dwell_s
    return dict(task_success=bool(success),execution_completed=bool(complete),simulated_s=elapsed,
        valid_measurement_s=valid,missing_measurement_s=elapsed-valid,
        normalized_absolute_error_integral=iae,mean_normalized_error=iae/valid if valid else None,
        outside_band_s=outside,trip_observed=bool(trip),observed_excursion=excursion,
        recovery_time_s=recovery_time,final_in_band_dwell_s=dwell,
        false_positive_s=false_positive,false_negative_s=false_negative,
        detection_delays_s=[first_detection.get(i) for i in range(len(truth))],
        sampled_contract_only=True)


def observation_policy(observation,schema,objective):
    """Passive baseline: flag unavailable/disagreeing pressure channels, send no controls."""
    p=valid_reading(observation,'pressure','steam_pressure_pa')
    ref=valid_reading(observation,'pressure_reference','steam_pressure_pa')
    return dict(action={},alarm=p is None or ref is None or abs(p-ref)>objective['diagnostic_threshold_pa'])


def tracking_policy(observation,schema,objective):
    result=observation_policy(observation,schema,objective)
    result['action']={'modes':{'rod_target':'auto'},'targets':{'rod_target':objective['target_w']}}
    return result


def _diagnostic_truth(config):
    horizon=config.max_steps*config.tick_s
    intervals=[]
    for fault in config.faults:
        if fault.domain!='sensor' or fault.target not in ('pressure','pressure_reference'):
            continue
        start=fault.start_tick*config.tick_s
        end=min(horizon,(fault.end_tick*config.tick_s if fault.end_tick is not None else horizon))
        if start<end:intervals.append((start,end))
    merged=[]
    for start,end in sorted(intervals):
        if merged and start<=merged[-1][1]:merged[-1]=(merged[-1][0],max(end,merged[-1][1]))
        else:merged.append((start,end))
    return tuple(merged)


def run_task(task,config,policy_factory,*,truth_intervals=None,policy_id='unversioned'):
    if type(task) is not Task or type(config) is not EnvConfig or not callable(policy_factory):
        raise ValueError('Task,EnvConfig and policy factory required')
    if type(policy_id) is not str or not policy_id.strip() or len(policy_id)>128:raise ValueError('Declared policy id required')
    if abs(task.horizon_s-config.max_steps*config.tick_s)>1e-10:raise ValueError('Task/config horizons must agree')
    expected=_diagnostic_truth(config) if task.kind=='diagnosis' else ()
    truth=expected if truth_intervals is None else _truth(truth_intervals,task.horizon_s)
    if truth!=expected:raise ValueError('Diagnostic labels must match configured pressure sensor faults')
    for pair in truth:
        if any(abs(t/config.tick_s-round(t/config.tick_s))>1e-8 for t in pair):
            raise ValueError('Truth intervals must align with environment ticks')
    source=identity();started=time.perf_counter();records=[];failure=None;phase='initialization'
    try:
        env=StationEnv(config);obs,_=env.reset();phase='policy';policy=policy_factory()
        if not callable(policy):raise TypeError('Callable policy required')
        for _ in range(config.max_steps):
            phase='policy'
            decision=policy(deepcopy(obs),deepcopy(env.available_actions()),asdict(task))
            if type(decision) is not dict or set(decision)!={'action','alarm'} or type(decision['alarm']) is not bool:
                raise ValueError('Exactly action and boolean alarm required')
            phase='model';new,_,terminated,truncated,info=env.step(decision['action'])
            records.append(dict(start_s=obs['time_s'],end_s=new['time_s'],
                power_w=valid_reading(new,'power','fission_power_w'),alarm=decision['alarm'],trip=info['trip_latched']))
            obs=new
            if terminated or truncated:break
    except Exception as error:
        failure=dict(phase=phase,type=type(error).__name__,code=error.code if isinstance(error,ActionError) else 'execution_error',message=str(error))
    result=score(task,records,truth_intervals=truth,execution_completed=failure is None)
    if source!=identity():raise ValueError('Task evaluator/backend source changed')
    return dict(task=asdict(task),config=asdict(config),policy_id=policy_id,source=source,
        truth_intervals=[list(p) for p in truth],records=records,metrics=result,failure=failure,
        wall_s=time.perf_counter()-started,industrial_utility_validated=False)


def evaluate_tasks(cases,assignment,tasks,policy_factory,*,split='test',axes=None,policy_id='unversioned'):
    from atomic_model_station_evaluation import validate_splits
    from atomic_model_station_evaluation.model import AXES,SPLITS
    manifest=validate_splits(cases,assignment,axes=AXES if axes is None else axes)
    if split not in SPLITS or type(tasks) is not dict or set(tasks)!={c.name for c in cases}:
        raise ValueError('Exact task map and declared split required')
    for case in cases:
        task=tasks[case.name]
        if type(task) is not Task or abs(task.horizon_s-case.config.max_steps*case.config.tick_s)>1e-10:
            raise ValueError('Every task must match its scenario horizon')
    source=identity();results=[]
    for case in sorted(cases,key=lambda c:c.name):
        if assignment[case.name]==split:
            results.append(dict(case=case.name,**run_task(tasks[case.name],case.config,policy_factory,policy_id=policy_id)))
    if source!=identity():raise ValueError('Task/backend source changed during batch')
    count=len(results)
    return dict(manifest=manifest,source=source,split=split,policy_id=policy_id,episodes=results,
        summary=dict(attempted=count,completed=sum(r['metrics']['execution_completed'] for r in results),
            execution_failures=sum(r['failure'] is not None for r in results),
            task_successes=sum(r['metrics']['task_success'] for r in results),
            task_success_rate=sum(r['metrics']['task_success'] for r in results)/count,
            total_wall_s=sum(r['wall_s'] for r in results),
            total_simulated_s=sum(r['metrics']['simulated_s'] for r in results)),
        complete_denominator=True,industrial_utility_validated=False)
