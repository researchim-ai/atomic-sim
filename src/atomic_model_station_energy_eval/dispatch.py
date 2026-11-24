"""Scheduled electrical-load objectives and matched native-model comparisons.

Power and energy meters belong to the evaluator. A policy only receives the
operator frame, action schema and the predeclared current dispatch objective.
"""
from copy import deepcopy
from dataclasses import dataclass, asdict
import math
import time

from atomic_model_station_env import StationEnv, ActionError
from .model import read_energy, assess_energy, identity


def _finite(value):
    try:
        return type(value) in (int,float) and math.isfinite(value)
    except OverflowError:
        return False


@dataclass(frozen=True)
class DispatchContract:
    targets_w: tuple
    load_fractions: tuple
    relative_band: float = .002
    minimum_iae_gain: float = .2

    def __post_init__(self):
        if (type(self.targets_w) is not tuple or type(self.load_fractions) is not tuple or
                not 1<=len(self.targets_w)<=128 or len(self.targets_w)!=len(self.load_fractions)):
            raise ValueError('One to 128 immutable paired dispatch targets/fractions required')
        if any(not _finite(v) or not 1e6<=v<=3.6e9 for v in self.targets_w):
            raise ValueError('Finite electrical targets in [1MW,3.6GW] required')
        if any(not _finite(v) or not .5<=v<=1.2 for v in self.load_fractions):
            raise ValueError('Native load fractions in [.5,1.2] required')
        if not _finite(self.relative_band) or not 0<self.relative_band<=.2:
            raise ValueError('Relative dispatch band in (0,.2] required')
        if not _finite(self.minimum_iae_gain) or not 0<self.minimum_iae_gain<=1:
            raise ValueError('Positive bounded comparative IAE gain required')


def score_dispatch(contract, records, *, tick_s, execution_completed, energy_passed, net_export_j=None):
    if type(contract) is not DispatchContract or type(records) not in (list,tuple):
        raise ValueError('Typed dispatch contract and records required')
    if not _finite(tick_s) or not .01<=tick_s<=10:
        raise ValueError('Native bounded tick required')
    if type(execution_completed) is not bool or type(energy_passed) is not bool:
        raise ValueError('Boolean completion/energy gates required')
    if len(records)>len(contract.targets_w):raise ValueError('Too many dispatch intervals')
    if net_export_j is not None and not _finite(net_export_j):raise ValueError('Finite export integral or missing meter required')
    iae=outside=elapsed=requested=peak=0.;trip=False;connected=True
    for i,row in enumerate(records):
        if type(row) is not dict or set(row)!={'start_s','end_s','net_export_w','trip','grid_available'}:
            raise ValueError('Complete electrical dispatch interval required')
        a,b,value=row['start_s'],row['end_s'],row['net_export_w']
        if (not _finite(a) or not _finite(b) or b<=a or abs(a-i*tick_s)>1e-8 or
                abs(b-(i+1)*tick_s)>1e-8 or not _finite(value)):
            raise ValueError('Consecutive native intervals and finite power required')
        if type(row['trip']) is not bool or type(row['grid_available']) is not bool:
            raise ValueError('Boolean trip/grid flags required')
        target=contract.targets_w[i];error=abs(value-target)/target;dt=b-a
        iae+=error*dt;peak=max(peak,error);elapsed=b;requested+=target*dt
        if error>contract.relative_band:outside+=dt
        trip=trip or row['trip'];connected=connected and row['grid_available']
    horizon=len(contract.targets_w)*tick_s
    complete=execution_completed and len(records)==len(contract.targets_w) and abs(elapsed-horizon)<=1e-8
    checks=dict(full_horizon=bool(complete),energy_budget=energy_passed,
        no_trip=not trip,grid_connected=connected,tracking_band=bool(records) and outside==0.,
        energy_delivery=net_export_j is not None and requested>0 and
            abs(net_export_j-requested)/requested<=contract.relative_band)
    return dict(checks=checks,task_success=all(checks.values()),simulated_s=elapsed,
        requested_energy_j=requested,scheduled_energy_j=sum(contract.targets_w)*tick_s,
        delivered_net_export_j=net_export_j,
        signed_energy_error_j=net_export_j-requested if net_export_j is not None else None,
        normalized_iae_s=iae,mean_normalized_error=iae/elapsed if elapsed else None,
        peak_normalized_error=peak,outside_band_s=outside,sampled_endpoint_contract_only=True)


def schedule_policy(observation,schema,objective):
    """Feed-forward reference; the current fraction is public, no hidden meters."""
    if 'load_fraction' not in schema.get('operator',{}):raise ValueError('Load command unavailable')
    return {'operator':{'load_fraction':objective['load_fraction']}}


def hold_dispatch_policy(observation,schema,objective):return {}


def _branch(origin,contract,factory):
    env=deepcopy(origin);records=[];samples=[];actions=[];attempted=[];failure=None
    started=time.perf_counter();phase='initialization';offset=origin.observation()[0]['time_s']
    try:
        obs,_=env.observation();samples.append(read_energy(env));phase='policy_initialization';policy=factory()
        if not callable(policy):raise ValueError('Callable dispatch policy required')
        for i,(target,fraction) in enumerate(zip(contract.targets_w,contract.load_fractions)):
            phase='policy'
            objective=dict(kind='electrical_dispatch',index=i,target_net_export_w=target,
                load_fraction=fraction,relative_band=contract.relative_band,interval_s=env.config.tick_s)
            action=policy(deepcopy(obs),deepcopy(env.available_actions()),objective)
            attempted.append(deepcopy(action));phase='model'
            new,_,terminated,truncated,info=env.step(action)
            actions.append(deepcopy(action));phase='meter';meter=read_energy(env);samples.append(meter)
            records.append(dict(start_s=obs['time_s']-offset,end_s=new['time_s']-offset,
                net_export_w=meter['powers']['net_export_w'],trip=info['trip_latched'] or meter['trip'],
                grid_available=meter['grid_available']))
            obs=new
            if terminated or truncated:break
    except Exception as error:
        failure=dict(phase=phase,type=type(error).__name__,message=str(error),
            code=error.code if isinstance(error,ActionError) else 'execution_error')
    accounting=assess_energy(samples) if samples else None
    metrics=score_dispatch(contract,records,tick_s=env.config.tick_s,
        execution_completed=failure is None,energy_passed=accounting is not None and accounting['passed'],
        net_export_j=accounting['energy_delta_j']['net_exported_j'] if accounting is not None else None)
    return dict(metrics=metrics,accounting=accounting,records=records,samples=samples,
        attempted_actions=attempted,actions=actions,failure=failure,wall_s=time.perf_counter()-started)


def compare_dispatch(origin,contract,policy_factory,*,baseline_factory=lambda:hold_dispatch_policy,
                     policy_id='unversioned',baseline_id='hold/v1'):
    if type(origin) is not StationEnv or type(contract) is not DispatchContract:
        raise ValueError('StationEnv and DispatchContract required')
    origin._require()
    if not callable(policy_factory) or not callable(baseline_factory):raise ValueError('Fresh callable policy factories required')
    if any(type(s) is not str or not s.strip() or len(s)>128 for s in (policy_id,baseline_id)):
        raise ValueError('Bounded declared policy ids required')
    if origin._finished or origin.available_actions()['remaining_steps']<len(contract.targets_w):
        raise ValueError('Sufficient live dispatch horizon required')
    source=identity();saved=origin.checkpoint()
    candidate=_branch(origin,contract,policy_factory)
    baseline=_branch(origin,contract,baseline_factory)
    if saved!=origin.checkpoint():raise ValueError('Caller changed during dispatch comparison')
    if source!=identity():raise ValueError('Dispatch evaluator source changed')
    valid=all(a['failure'] is None and a['metrics']['checks']['full_horizon'] and
        a['metrics']['checks']['no_trip'] and a['metrics']['checks']['grid_connected'] and
        a['accounting'] is not None and a['accounting']['passed'] for a in (candidate,baseline))
    gain=None
    if valid:
        denominator=baseline['metrics']['normalized_iae_s']
        gain=(denominator-candidate['metrics']['normalized_iae_s'])/denominator if denominator>1e-12 else None
    checks=dict(comparison_valid=valid,candidate_task_success=candidate['metrics']['task_success'],
        baseline_does_not_track=not baseline['metrics']['checks']['tracking_band'],
        commands_differ=candidate['actions']!=baseline['actions'],
        relative_iae_gain=gain is not None and gain>=contract.minimum_iae_gain)
    return dict(source=source,contract=asdict(contract),fork_sha256=saved['sha256'],
        fork_time_s=origin.observation()[0]['time_s'],policy_id=policy_id,baseline_id=baseline_id,
        candidate=candidate,baseline=baseline,comparison_valid=valid,relative_iae_gain=gain,
        checks=checks,task_success=all(checks.values()),caller_unchanged=True,
        privileged_evaluator_meters=True,industrial_dispatch_validated=False,
        continuous_time_tracking_proven=False)
