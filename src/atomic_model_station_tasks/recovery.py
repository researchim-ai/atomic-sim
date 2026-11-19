"""Matched-fork recovery evaluation; natural recovery is not credited to the agent."""
from copy import deepcopy
from dataclasses import dataclass,asdict
import math
from atomic_model_station_env import StationEnv
from atomic_model_station_control.policy import valid_reading
from .model import identity as task_identity

CHANNELS={'feed_flow':('feed_flow_kg_s','feed_pump_on'),
          'condensate_flow':('condensate_flow_kg_s','condensate_pump_on')}


def finite(x):return type(x) in (int,float) and math.isfinite(x)


@dataclass(frozen=True)
class RecoveryContract:
    channel:str
    target:float
    steps:int=20
    relative_band:float=.1
    dwell_s:float=.2
    minimum_iae_gain:float=.2

    def __post_init__(self):
        if type(self.channel) is not str or self.channel not in CHANNELS:raise ValueError('Supported measured flow channel required')
        if type(self.steps) is not int or not 1<=self.steps<=128:raise ValueError('Recovery steps in [1,128] required')
        if not finite(self.target) or not 0<self.target<=1e6:raise ValueError('Positive finite measured flow target required')
        if not finite(self.relative_band) or not 0<self.relative_band<=.5:raise ValueError('Relative band in (0,.5] required')
        if not finite(self.dwell_s) or not 0<self.dwell_s<=1280:raise ValueError('Positive bounded dwell required')
        if not finite(self.minimum_iae_gain) or not 0<self.minimum_iae_gain<=1:raise ValueError('Positive relative IAE gain required')


def score_recovery(contract,initial_value,records,*,tick_s,execution_completed,energy_passed):
    if type(contract) is not RecoveryContract or type(records) not in (list,tuple):raise ValueError('Typed recovery contract and records required')
    if not finite(initial_value) or not finite(tick_s) or not .01<=tick_s<=10:raise ValueError('Finite initial measurement and native tick in [.01,10] required')
    if type(execution_completed) is not bool or type(energy_passed) is not bool:raise ValueError('Boolean completion and budget gates required')
    horizon=contract.steps*tick_s
    if contract.dwell_s>horizon+1e-10:raise ValueError('Dwell exceeds recovery horizon')
    if len(records)>contract.steps:raise ValueError('Too many recovery records')
    elapsed=valid=iae=dwell=0.;trip=False;arrival=None
    for i,row in enumerate(records):
        if type(row) is not dict or set(row)!={'start_s','end_s','value','trip'}:raise ValueError('Complete sampled recovery record required')
        a,b=row['start_s'],row['end_s'];value=row['value']
        if (not finite(a) or not finite(b) or b<=a or abs(a-i*tick_s)>1e-8 or abs(b-(i+1)*tick_s)>1e-8):
            raise ValueError('Consecutive native tick intervals required')
        if type(row['trip']) is not bool or (value is not None and not finite(value)):
            raise ValueError('Boolean trip and finite/missing measurement required')
        dt=b-a;elapsed=b;trip=trip or row['trip']
        if value is None:dwell=0.;arrival=None;continue
        valid+=dt;error=abs(value-contract.target)/contract.target;iae+=error*dt
        if not finite(error) or not finite(iae):raise ValueError('Nonfinite normalized recovery error')
        if error<=contract.relative_band:
            dwell+=dt
            if arrival is None and dwell+1e-10>=contract.dwell_s:arrival=b
        else:dwell=0.;arrival=None
    complete=execution_completed and len(records)==contract.steps and abs(elapsed-horizon)<=1e-8
    coverage=valid/horizon
    outside=abs(initial_value-contract.target)/contract.target>contract.relative_band
    recovered=complete and energy_passed and outside and not trip and coverage>=1-1e-10 and arrival is not None
    return dict(recovered=bool(recovered),contract_completed=bool(complete),initial_outside_band=outside,
        measurement_coverage=coverage,normalized_iae_s=iae,final_in_band_dwell_s=dwell,
        recovery_time_s=arrival,trip_observed=trip,energy_passed=energy_passed,
        sampled_contract_only=True)


def hold_policy(observation,schema,objective):return {}


def restart_flow_policy(observation,schema,objective):
    """Transparent baseline using only the chosen measured channel and public action schema."""
    channel=objective['channel'];quantity,pump=CHANNELS[channel]
    value=valid_reading(observation,channel,quantity)
    if value is None:return {}
    if pump not in schema['operator']:raise ValueError('Restart command unavailable')
    return {'operator':{pump:True}} if value<objective['target']*(1-objective['relative_band']) else {}


def _identity():
    from atomic_model_station_energy_eval.model import identity as energy_identity
    return dict(tasks=task_identity(),energy=energy_identity())


def _branch(origin,contract,factory):
    from atomic_model_station_energy_eval.model import read_energy,assess_energy
    branch=deepcopy(origin);observation,_=branch.observation();start=observation['time_s']
    records=[];actions=[];attempted=[];receipts=[];energy=[];failure=None;phase='initial_accounting'
    try:
        energy.append(read_energy(branch));phase='policy_initialization'
        policy=factory()
        if not callable(policy):raise ValueError('Callable recovery policy required')
        for i in range(contract.steps):
            phase='policy'
            action=policy(deepcopy(observation),deepcopy(branch.available_actions()),asdict(contract))
            attempted.append(deepcopy(action))
            phase='model'
            new,_,terminated,truncated,info=branch.step(deepcopy(action))
            actions.append(deepcopy(action));receipts.append(deepcopy(info['receipt']))
            records.append(dict(start_s=observation['time_s']-start,end_s=new['time_s']-start,
                value=valid_reading(new,contract.channel,CHANNELS[contract.channel][0]),trip=info['trip_latched']))
            observation=new;phase='accounting';energy.append(read_energy(branch))
            if terminated or truncated:break
    except Exception as error:failure=dict(phase=phase,type=type(error).__name__,message=str(error))
    accounting=assess_energy(energy) if energy else dict(passed=False,reasons=['energy_unavailable'])
    initial=valid_reading(origin.observation()[0],contract.channel,CHANNELS[contract.channel][0])
    score=score_recovery(contract,initial,records,tick_s=origin.config.tick_s,
        execution_completed=failure is None,energy_passed=accounting['passed'])
    return dict(records=records,actions=actions,attempted_actions=attempted,receipts=receipts,accounting=accounting,
        metrics=score,failure=failure,final_checkpoint=branch.checkpoint())


def compare_recovery(env,contract,policy_factory,*,baseline_factory=lambda:hold_policy,
                     policy_id='unversioned',baseline_id='hold/v1'):
    """Evaluate two isolated copies without advancing caller; reports are privileged."""
    if type(env) is not StationEnv or type(contract) is not RecoveryContract:raise ValueError('StationEnv and RecoveryContract required')
    env._require()
    if not callable(policy_factory) or not callable(baseline_factory):raise ValueError('Fresh policy factories required')
    if any(type(x) is not str or not x.strip() or len(x)>128 for x in (policy_id,baseline_id)):
        raise ValueError('Declared bounded policy identifiers required')
    if env._finished or env.available_actions()['remaining_steps']<contract.steps:raise ValueError('Insufficient live episode horizon')
    if contract.dwell_s>contract.steps*env.config.tick_s+1e-10:raise ValueError('Dwell exceeds recovery horizon')
    initial=valid_reading(env.observation()[0],contract.channel,CHANNELS[contract.channel][0])
    if initial is None:raise ValueError('Valid measured initial recovery value required')
    source=_identity();checkpoint=env.checkpoint();origin=deepcopy(env)
    candidate=_branch(origin,contract,policy_factory)
    baseline=_branch(origin,contract,baseline_factory)
    if env.checkpoint()!=checkpoint:raise ValueError('Caller environment changed during comparison')
    if source!=_identity():raise ValueError('Recovery evaluator/backend source changed')
    arms=(candidate,baseline)
    valid=all(a['failure'] is None and a['metrics']['contract_completed'] and
        a['metrics']['measurement_coverage']>=1-1e-10 and a['accounting']['passed'] for a in arms)
    gain=None
    if valid:
        denominator=max(baseline['metrics']['normalized_iae_s'],1e-12)
        gain=(baseline['metrics']['normalized_iae_s']-candidate['metrics']['normalized_iae_s'])/denominator
        if not finite(gain):raise ValueError('Nonfinite paired recovery gain')
    command_difference=candidate['actions']!=baseline['actions']
    checks=dict(comparison_valid=valid,initial_excursion=candidate['metrics']['initial_outside_band'],
        candidate_recovered=candidate['metrics']['recovered'],baseline_not_recovered=not baseline['metrics']['recovered'],
        relative_iae_gain=valid and gain>=contract.minimum_iae_gain,commands_differ=command_difference)
    return dict(source=source,contract=asdict(contract),fork_sha256=checkpoint['sha256'],
        fork_time_s=origin.observation()[0]['time_s'],initial_measurement=initial,
        policy_id=policy_id,baseline_id=baseline_id,candidate=candidate,baseline=baseline,
        comparison_valid=valid,checks=checks,task_success=all(checks.values()),relative_iae_gain=gain,
        caller_unchanged=True,report_visibility='privileged_evaluator',
        causal_scope='Deterministic matched model fork only; not population or real-plant causal inference',
        industrial_utility_validated=False,closed_tasks=[])
