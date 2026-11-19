"""Leakage checks and complete-denominator station policy evaluation.

Manifests contain privileged scenario labels. Policies receive only copies of
operator observations and the public action schema, never the manifest/config.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
import math
import hashlib
from pathlib import Path
import time

from atomic_model_station_env import EnvConfig, StationEnv, ActionError
from atomic_model_station_env.model import signature
from atomic_model_workspace.common import digest

AXES = ('object_family', 'parameter_family', 'fault_family', 'time_family')
SPLITS = ('train', 'dev', 'test')

def identity():
    return dict(backend=signature(), evaluator={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(Path(__file__).parent.glob('*.py'))})


@dataclass(frozen=True)
class EvaluationCase:
    name: str
    config: EnvConfig
    object_family: str
    parameter_family: str
    fault_family: str
    time_family: str

    def __post_init__(self):
        for field in ('name', *AXES):
            value = getattr(self, field)
            if type(value) is not str or not value.strip() or len(value) > 128:
                raise ValueError('Nonempty bounded case identity required: '+field)
        if type(self.config) is not EnvConfig:
            raise ValueError('EnvConfig required')

    @property
    def scenario_hash(self):
        config = asdict(self.config)
        for name in ('seed','rtol','max_step_s'):
            config.pop(name)  # Noise and solver precision do not define a new physical scenario.
        faults = [{k:v for k,v in fault.items() if k!='name'} for fault in config['faults']]
        config['faults'] = sorted(faults,key=digest)
        return digest(config)


def validate_splits(cases, assignment, *, axes=AXES):
    if type(cases) not in (tuple, list) or not cases or any(type(c) is not EvaluationCase for c in cases):
        raise ValueError('Nonempty EvaluationCase collection required')
    names = [c.name for c in cases]
    if len(set(names)) != len(names):
        raise ValueError('Unique case identities required')
    if type(assignment) is not dict or set(assignment) != set(names) or any(v not in SPLITS for v in assignment.values()):
        raise ValueError('Exact train/dev/test assignment required')
    if type(axes) not in (tuple, list) or not axes or len(set(axes)) != len(axes) or any(a not in AXES for a in axes):
        raise ValueError('Explicit unique known holdout axes required')
    if set(assignment.values()) != set(SPLITS):
        raise ValueError('All three splits must be nonempty')
    owners = {}
    for case in cases:
        for axis, value in [('scenario', case.scenario_hash), *((a, getattr(case, a)) for a in axes)]:
            key = (axis, value)
            previous = owners.setdefault(key, assignment[case.name])
            if previous != assignment[case.name]:
                raise ValueError('Cross-split leakage on '+axis)
    payload = dict(axes=list(axes), assignment=dict(sorted(assignment.items())),
        cases=[dict(name=c.name, config=asdict(c.config), scenario_hash=c.scenario_hash,
                    groups={a:getattr(c,a) for a in AXES}) for c in sorted(cases,key=lambda c:c.name)],
        declared_metadata_only=True, unseen_physical_objects_validated=False)
    return dict(payload, sha256=digest(payload))


def hold_policy(observation, available_actions):
    """No new operator command; existing configured control remains in force."""
    return {}


def evaluate(cases, assignment, policy_factory, *, split='test', axes=AXES, policy_id='unversioned'):
    manifest = validate_splits(cases, assignment, axes=axes)
    if type(policy_id) is not str or not policy_id.strip() or len(policy_id)>128:
        raise ValueError('Bounded declared policy identity required')
    if split not in SPLITS or not callable(policy_factory):
        raise ValueError('Declared split and callable policy factory required')
    source = identity()
    rows = []
    for case in sorted(cases,key=lambda c:c.name):
        if assignment[case.name] != split:
            continue
        started = time.perf_counter()
        steps = 0
        reward_sum = 0.
        elapsed = 0.
        error_integrals = {}
        valid_durations = {}
        missing_duration = 0.
        trip = False
        termination = truncation = None
        status, failure = 'completed', None
        phase = 'initialization'
        try:
            env = StationEnv(case.config)
            observation, info = env.reset()
            phase = 'policy'
            policy = policy_factory()  # Fresh agent memory for each episode; no case labels.
            if not callable(policy):
                raise TypeError('Policy factory must return a callable')
            for _ in range(case.config.max_steps):
                phase = 'policy'
                action = policy(deepcopy(observation), deepcopy(env.available_actions()))
                phase = 'model'
                new, reward, terminated, truncated, info = env.step(action)
                dt = new['time_s'] - observation['time_s']
                if not math.isfinite(dt) or dt <= 0 or not math.isfinite(reward):
                    raise ValueError('Invalid public transition')
                steps += 1; elapsed += dt; reward_sum += reward
                terms = info['reward_terms']
                for name, value in terms['normalized_absolute_errors'].items():
                    error_integrals[name] = error_integrals.get(name,0.) + value*dt
                    valid_durations[name] = valid_durations.get(name,0.) + dt
                missing_duration += len(terms['unavailable_channels'])*dt
                trip = trip or info['trip_latched']
                observation = new
                termination, truncation = info['termination_reason'], info['truncation_reason']
                if terminated or truncated:
                    break
        except Exception as error:
            status = 'failed'
            failure = dict(phase=phase, type=type(error).__name__,
                code=error.code if isinstance(error,ActionError) else 'execution_error', message=str(error))
        rows.append(dict(case=case.name,status=status,failure=failure,steps=steps,
            simulated_s=elapsed,wall_s=time.perf_counter()-started,reward_sum=reward_sum,
            tracking_error_integrals=error_integrals,valid_measurement_s=valid_durations,
            mean_tracking_error={k:error_integrals[k]/valid_durations[k] for k in error_integrals},
            missing_channel_s=missing_duration,trip_observed=bool(trip),
            termination_reason=termination,truncation_reason=truncation))
    if source != identity():
        raise ValueError('Station backend changed during evaluation')
    count = len(rows)
    return dict(manifest=manifest,source=source,split=split,policy_id=policy_id,
        policy_identity_verified=False,episodes=rows,
        summary=dict(attempted=count,completed=sum(r['status']=='completed' for r in rows),
            failed=sum(r['status']=='failed' for r in rows),
            completion_fraction=sum(r['status']=='completed' for r in rows)/count,
            total_wall_s=sum(r['wall_s'] for r in rows),
            total_simulated_s=sum(r['simulated_s'] for r in rows)),
        complete_denominator=True,completion_is_task_success=False,
        measured_tracking_only=True,industrial_utility_validated=False)
