"""Privileged, all-or-nothing auditing of a declared agent action sequence."""
from copy import deepcopy
import math
from atomic_model_station_env import StationEnv
from atomic_model_station_event_heat import HeatEventError
from atomic_model_station_heat_accounting.model import FLUXES
from .model import identity,step_hybrid_heat


def advance_hybrid_actions(env,actions,*,subdivisions=(4,8,16)):
    """Commit every native agent tick, or leave the caller's environment unchanged.

    The diagnostic proof is privileged evaluator data, not an agent observation.
    A terminal transition is allowed only at the last requested action. Controllers
    and sensors retain their native tick; refinement only runs diagnostic copies.
    """
    if type(env) is not StationEnv:raise ValueError('StationEnv required')
    env._require()
    if type(actions) not in (list,tuple) or not 1<=len(actions)<=16:
        raise ValueError('One to sixteen actions required')
    if env.config.faults:raise ValueError('Scheduled faults unsupported')
    if env._finished:raise ValueError('Episode already finished')
    if len(actions)>env.available_actions()['remaining_steps']:
        raise ValueError('Sequence exceeds remaining episode horizon')
    for action in actions:env._action(action)
    source=identity();trial=deepcopy(env);reports=[];transitions=[]
    for index,action in enumerate(deepcopy(actions)):
        try:
            transition,proof=step_hybrid_heat(trial,action,subdivisions=subdivisions)
            if (transition[2] or transition[3]) and index!=len(actions)-1:
                raise HeatEventError(dict(reason='early_episode_end'))
            reports.append(proof);transitions.append(transition)
        except Exception as error:
            raise HeatEventError(dict(reason='action_sequence_refused',action_index=index,
                error_type=type(error).__name__,message=str(error),
                cause=deepcopy(getattr(error,'receipt',None)),accepted_prefix=reports,
                accepted_transitions=transitions,prefix_committed_to_caller=False,
                native_state_committed=False)) from error
    ledgers=[r['replays'][-1]['native_endpoint_accounting'] for r in reports]
    defects={k:math.fsum(a['absolute_defect_sums_j'][k] for a in ledgers)
        for k in ledgers[0]['absolute_defect_sums_j']}
    scale=math.fsum(a['normalization_scale_j'] for a in ledgers)
    integrals={k:math.fsum(a['sampled_integrals_j'][k] for a in ledgers) for k in FLUXES}
    duration=math.fsum(a['duration_s'] for a in ledgers)
    valid=bool(defects) and math.isfinite(scale) and scale>0 and math.isfinite(duration) and duration>0
    valid=valid and all(math.isfinite(v) and v>=0 for v in defects.values()) and all(math.isfinite(v) for v in integrals.values())
    if not valid or max(defects.values())/scale>1e-6:
        raise HeatEventError(dict(reason='action_sequence_aggregate_budget',
            accepted_prefix=reports,prefix_committed_to_caller=False,native_state_committed=False))
    if source!=identity():
        raise HeatEventError(dict(reason='source_changed',native_state_committed=False))
    result=dict(source=source,actions=deepcopy(actions),steps=reports,
        accounting=dict(absolute_defect_sums_j=defects,normalization_scale_j=scale,
            sampled_integrals_j=integrals,duration_s=duration),
        passed=True,native_state_committed=True,controller_substepped=False,
        proof_visibility='privileged_evaluator',closed_tasks=[])
    env.__dict__.update(trial.__dict__)
    return transitions,result
