"""Reconstruct the admitted post-command left endpoint; never smear a jump across a tick."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
from pathlib import Path
import time
import numpy as np
from atomic_model_station_env import StationEnv,EnvConfig
from atomic_model_station_heat_accounting import read_heat,assess_heat
from atomic_model_station_heat_accounting.model import identity as heat_identity


def identity():
    return dict(heat=heat_identity(),files={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(Path(__file__).parent.glob('*.py'))})


class HeatEventError(ValueError):
    def __init__(self,receipt):
        self.receipt=receipt
        super().__init__('Event heat audit refused without committing station state')


def _post_command(before,receipt):
    """Mirror the pinned backend's interval preparation on an isolated copy only."""
    left=deepcopy(before);plant=left._session.control.plant
    initial=plant.state.copy()
    plant._select_faults()
    if receipt['dispatched']:plant._set_controls(deepcopy(receipt['dispatched']))
    plant._prepare_interval()
    if not np.array_equal(initial,plant.state):
        raise HeatEventError(dict(reason='instantaneous_state_projection'))
    return left


def step_heat(env,action=None):
    if type(env) is not StationEnv:raise ValueError('StationEnv required')
    env._require()
    if env.config.faults:raise ValueError('Scheduled faults are outside this command-event qualification')
    source=identity();prior=read_heat(env);before=deepcopy(env);trial=deepcopy(env)
    transition=trial.step(deepcopy(action));receipt=transition[4]['receipt']
    start=prior['energy']['time_s'];end=transition[0]['time_s']
    if abs(receipt['time_s']-start)>1e-10 or abs(receipt['completed_at_s']-end)>1e-10:
        raise HeatEventError(dict(reason='receipt_clock'))
    left=read_heat(_post_command(before,receipt));right=read_heat(trial)
    if left['energy']['time_s']!=start or right['energy']['time_s']!=end:
        raise HeatEventError(dict(reason='meter_clock'))
    if (left['energy']['counters']!=prior['energy']['counters'] or
        left['stored_j']!=prior['stored_j'] or left['boundary_j']!=prior['boundary_j']):
        raise HeatEventError(dict(reason='instantaneous_energy_jump'))
    previous=before._session.control.plant.events;events=trial._session.control.plant.events
    if events[:len(previous)]!=previous:raise HeatEventError(dict(reason='event_history_changed'))
    interior=[]
    for event in events[len(previous):]:
        epoch=event.get('time_s')
        if type(epoch) not in (int,float) or not np.isfinite(epoch):
            raise HeatEventError(dict(reason='unlocated_physical_event'))
        # The right endpoint must also have a known pre-event flux.
        # A snapshot after an endpoint switch is not a left-hand limit.
        if start+1e-10<epoch<=end+1e-10:interior.append(deepcopy(event))
    if interior:raise HeatEventError(dict(reason='unresolved_internal_event',events=interior))
    audit=assess_heat([left,right])
    naive=assess_heat([prior,right])
    proof=dict(prior=prior,left=left,right=right,audit=audit,naive_unsplit_audit=naive,
        dispatched=deepcopy(receipt['dispatched']),events=deepcopy(events[len(previous):]),
        state_continuous_at_command=True,source=source)
    if not audit['passed']:raise HeatEventError(dict(reason='segment_heat_budget',candidate=proof))
    if source!=identity():raise HeatEventError(dict(reason='source_changed'))
    env.__dict__.update(trial.__dict__)
    return transition,proof


def run_event_window(config,commands):
    if type(config) is not EnvConfig or config.faults:raise ValueError('Fault-free EnvConfig required')
    if type(commands) is not dict or any(type(k) is not int or not 0<=k<config.max_steps or type(v) is not dict for k,v in commands.items()):
        raise ValueError('Bounded integer-tick action map required')
    source=identity();started=time.perf_counter();env=StationEnv(config)
    records=[];failure=None
    try:
        env.reset()
        for tick in range(config.max_steps):
            transition,proof=step_heat(env,commands.get(tick,{}));records.append(proof)
            if transition[2] or transition[3]:break
    except Exception as error:
        failure=dict(type=type(error).__name__,message=str(error),receipt=getattr(error,'receipt',None),code=getattr(error,'code',None))
    integrals={key:sum(r['audit']['sampled_integrals_j'][key] for r in records)
        for key in ('nuclear_heat_w','primary_pump_loss_out_w','external_cooling_heat_out_w','external_net_w')}
    complete=failure is None and len(records)==config.max_steps
    # Triangle bounds prevent cancellation of individually accepted segment defects.
    closure_bound=sum(abs(r['audit']['heat_closure_j']) for r in records)
    storage_bound=sum(abs(r['audit']['window_storage_defect_j']) for r in records)
    boundary_bound=sum(abs(r['audit']['boundary_quadrature_defect_j']) for r in records)
    scale=sum(max(1.,abs(r['audit']['sampled_integrals_j']['nuclear_heat_w'])+
        abs(r['audit']['sampled_integrals_j']['primary_pump_loss_out_w']+r['audit']['sampled_integrals_j']['external_cooling_heat_out_w'])+
        abs(r['audit']['electrical']['energy_delta_j']['net_exported_j'])) for r in records)
    aggregate=bool(records) and max(closure_bound,storage_bound,boundary_bound)/scale<=1e-6
    if source!=identity():raise ValueError('Event heat source changed during run')
    return dict(source=source,config=asdict(config),commands=deepcopy(commands),records=records,
        failure=failure,complete=complete,passed=complete and aggregate,
        summed_integrals_j=integrals,absolute_defect_sum_j=dict(heat=closure_bound,storage=storage_bound,boundary=boundary_bound),
        normalization_scale_j=scale,naive_refused_segments=sum(not r['naive_unsplit_audit']['passed'] for r in records),
        wall_s=time.perf_counter()-started,internal_event_subdivision_supported=False,
        stationary_efficiency_certified=False,closed_tasks=[])
