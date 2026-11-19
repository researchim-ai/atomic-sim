"""Refined heat integration with analytic rod-arrival boundaries only."""
from copy import deepcopy
from pathlib import Path
import hashlib
import math
import numpy as np
from atomic_model_station_env import StationEnv
from atomic_model_station.faults import PWRFaultSession
from atomic_model_station_event_heat.model import HeatEventError,_post_command
from atomic_model_station_refined_heat.model import identity as refined_identity
from atomic_model_station_heat_accounting import read_heat,assess_heat
from atomic_model_station_heat_accounting.model import FLUXES


def identity():
    return dict(refined=refined_identity(),files={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(Path(__file__).parent.glob('*.py'))})


def _motion(left):
    p=left._session.control.plant
    return p.rod_drive.trajectory(float(p.state[12]),float(p.controls['rod_target']),
        p.scram_latched,p.time)[1]


def _check_events(previous,events,start,end,motion,*,require_count=True):
    if events[:len(previous)]!=previous:
        raise HeatEventError(dict(reason='event_history_changed'))
    arrivals=[]
    for event in events[len(previous):]:
        epoch=event.get('time_s')
        if type(epoch) not in (int,float) or not math.isfinite(epoch):
            raise HeatEventError(dict(reason='unlocated_physical_event'))
        if abs(epoch-start)<=1e-10 and event.get('kind')!='rod_arrived':continue
        valid=(event.get('kind')=='rod_arrived' and motion['direction']!=0 and
            start<epoch<=end+1e-10 and abs(epoch-motion['arrival_s'])<=1e-10 and
            event.get('position')==motion['target'] and
            event.get('reason')==('scram_stop' if motion['scram_latched'] else 'requested_position'))
        if not valid:raise HeatEventError(dict(reason='unresolved_physical_event',event=deepcopy(event)))
        arrivals.append(event)
    expected=bool(motion['direction'] and start<motion['arrival_s']<=end)
    if len(arrivals)>1 or (require_count and len(arrivals)!=int(expected)):
        raise HeatEventError(dict(reason='rod_event_count',expected=int(expected),actual=len(arrivals)))


def _replay(left,right,native_state,count):
    copy=deepcopy(left);plant=copy._session.control.plant
    if type(plant) is not PWRFaultSession or plant.equipment_faults:
        raise ValueError('Pinned fault-free physical backend required')
    start=plant.time;end=right['energy']['time_s'];events=deepcopy(plant.events)
    motion=_motion(left);arrival=motion['arrival_s']
    targets=[start+(end-start)*(i+1)/count for i in range(count)]
    if motion['direction'] and start<arrival<end:
        targets=[t for t in targets if t==end or abs(t-arrival)>1e-12]+[arrival]
    targets=sorted(targets)
    samples=[read_heat(copy)];rod_samples=[]
    rod_at,_=plant.rod_drive.trajectory(float(plant.state[12]),float(plant.controls['rod_target']),plant.scram_latched,start)
    for target in targets:
        # Only on this diagnostic copy: bypass the fixed fault-clock adapter.
        # Controls stay fixed; no controller/sensor/agent step occurs here.
        super(PWRFaultSession,plant).advance(target-plant.time)
        _check_events(events,plant.events,start,plant.time,motion,require_count=False)
        rod_error=abs(float(plant.state[12])-rod_at(plant.time))
        if rod_error>1e-12:raise HeatEventError(dict(reason='rod_trajectory_mismatch'))
        rod_samples.append(dict(time_s=plant.time,position=float(plant.state[12]),analytic_error=rod_error))
        samples.append(read_heat(copy))
    state_error=float(np.max(abs(plant.state-native_state)/np.maximum(abs(native_state),1.)))
    replay_audit=assess_heat(samples)
    # Tie acceptance to the actual committed candidate's terminal counters/storage.
    tied=deepcopy(samples);tied[-1]=deepcopy(right)
    return dict(subdivisions=count,actual_intervals=len(targets),event_boundaries_s=[arrival] if motion['direction'] and start<arrival<=end else [],
        samples=samples,rod_samples=rod_samples,
        replay_rod_event_count=sum(e.get('kind')=='rod_arrived' for e in plant.events[len(events):]),
        boundary_source='analytic drive law; replay journal may omit roundoff-coincident arrival',
        audit=replay_audit,
        native_endpoint_audit=assess_heat(tied),endpoint_state_relative=state_error,
        endpoint_state_scale='componentwise max(abs(native state),1 in native units)')


def step_piecewise_heat(env,action=None,*,subdivisions=(4,8,16)):
    """Atomic native step, accepted only after three independently restarted replays.

    This is a numerical consistency check, not a rigorous error bound or validation
    against measurements. Only analytic rod arrivals are admitted as internal physical events.
    """
    if type(env) is not StationEnv:raise ValueError('StationEnv required')
    env._require()
    if env.config.faults:raise ValueError('Scheduled faults unsupported')
    if (type(subdivisions) not in (tuple,list) or len(subdivisions)!=3 or
        any(type(n) is not int or not 2<=n<=64 for n in subdivisions) or
        subdivisions[1]!=2*subdivisions[0] or subdivisions[2]!=2*subdivisions[1]):
        raise ValueError('Three doubling subdivision counts in [2,64] required')
    source=identity();prior=read_heat(env);trial=deepcopy(env)
    transition=trial.step(deepcopy(action));receipt=transition[4]['receipt']
    right=read_heat(trial);start=prior['energy']['time_s'];end=right['energy']['time_s']
    if receipt['time_s']!=start or receipt['completed_at_s']!=end or end<=start:
        raise HeatEventError(dict(reason='receipt_clock'))
    left_env=_post_command(env,receipt);left=read_heat(left_env)
    motion=_motion(left_env)
    _check_events(env._session.control.plant.events,trial._session.control.plant.events,start,end,motion)
    if (left['energy']['counters']!=prior['energy']['counters'] or
        left['stored_j']!=prior['stored_j'] or left['boundary_j']!=prior['boundary_j']):
        raise HeatEventError(dict(reason='instantaneous_energy_jump'))
    rows=[_replay(left_env,right,trial._session.control.plant.state,n) for n in subdivisions]
    comparisons=[]
    for a,b in zip(rows,rows[1:]):
        x=a['native_endpoint_audit']['sampled_integrals_j'];y=b['native_endpoint_audit']['sampled_integrals_j']
        comparisons.append({k:abs(x[k]-y[k])/max(1.,abs(y[k]),
            abs(y['nuclear_heat_w']) if k=='external_net_w' else 0.) for k in FLUXES})
    checks=dict(replay_balances=all(r['audit']['passed'] for r in rows),
        native_endpoint_balances=all(r['native_endpoint_audit']['passed'] for r in rows),
        endpoint_agreement=all(r['endpoint_state_relative']<=1e-6 for r in rows),
        refined_integrals=max(comparisons[-1].values())<=2e-6,
        refinement_trend=all(comparisons[1][k]<=max(comparisons[0][k],1e-8) for k in FLUXES))
    proof=dict(source=source,checks=checks,passed=all(checks.values()),replays=rows,
        comparisons=comparisons,prior=prior,left=left,right=right,
        dispatched=deepcopy(receipt['dispatched']),coarse_audit=assess_heat([left,right]),
        naive_unsplit_audit=assess_heat([prior,right]),
        audit=rows[-1]['native_endpoint_audit'],native_state_committed=True,
        controller_substepped=False,isolated_quadrature_error_bound=False,
        motion=motion,supported_internal_events=['rod_arrived'],
        discontinuous_flux_events_supported=False,
        stationary_efficiency_certified=False,closed_tasks=[])
    if not proof['passed']:
        proof['native_state_committed']=False
        raise HeatEventError(dict(reason='refinement_failed',candidate=proof))
    if source!=identity():raise HeatEventError(dict(reason='source_changed'))
    env.__dict__.update(trial.__dict__)
    return transition,proof
