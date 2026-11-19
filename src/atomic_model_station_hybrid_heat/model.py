"""Two-sided check-valve/backup diagnostics; existing scientific sources unchanged."""
from copy import deepcopy
from pathlib import Path
import hashlib
import math
import numpy as np
from atomic_model_station import PWRPowerSession,PWRBypassSession
from atomic_model_station.faults import PWRFaultSession
from atomic_model_station_env import StationEnv
from atomic_model_station_event_heat.model import HeatEventError,_post_command
from atomic_model_station_jump_heat.model import assess_segments,_meter,identity as jump_identity
from atomic_model_station_energy_eval.model import POWERS
from atomic_model_station_heat_accounting.model import FLUXES
from .events import assemble_segments

ROOTS={'backup_empty','feed_closed','feed_opened','condensate_closed','condensate_opened','bypass_relief_open','bypass_relief_close','bypass_position_arrival','steam_valve_arrival'}
KINDS={'backup_exhausted','feed_check_closed','feed_check_opened','condensate_check_closed','condensate_check_opened','bypass_relief_open','bypass_relief_close','bypass_position_arrival','steam_valve_arrived'}


def identity():
    return dict(jump=jump_identity(),files={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(Path(__file__).parent.glob('*.py'))})


def _root_meter(session):
    # A root residual may have a tiny negative flow before the native map. The
    # public full snapshot enforces post-map admissibility; do not project the
    # pre-map state merely to obtain meters. Evaluate its actual RHS on a copy.
    copy=deepcopy(session);copy._rhs(copy.time,copy.state)
    l=copy.ledger_index;i=copy.power_index;y=copy.state;stored=float(copy._stored())
    counters=dict(gross_generated_j=float(y[l+1]),own_needs_j=float(y[l+2]),
        net_generated_j=float(y[l+1]-y[l+2]),net_exported_j=float(y[i+1]),backup_used_j=float(y[i+3]))
    energy=dict(time_s=copy.time,counters=counters,powers={k:float(copy.powers[k]) for k in POWERS},
        grid_available=copy.controls['grid_available'],trip=copy.scram_latched or copy.turbine_tripped,
        energy_relative=abs(stored-copy.origin_energy-float(y[l]))/max(abs(copy.origin_energy),float(y[l+3]),1.))
    return dict(energy=energy,stored_j=stored,boundary_j=float(y[l]),fluxes={k:float(copy.powers[k]) for k in FLUXES})


class _RootRecorder:
    def _custom_root(self,label,endpoint,now):
        if label not in ROOTS:raise ValueError('Unsupported hybrid root: '+label)
        if label.endswith('closed'):
            index=self.plant.feed_index if label.startswith('feed_') else self.feed_index+6
            if abs(float(endpoint[index]))>1e-6:raise ValueError('Check-valve zero-root residual too large')
        before=_root_meter(self);journal_start=len(self.events)
        handled=super()._custom_root(label,endpoint,now)
        if not handled:
            if label not in {'feed_closed','feed_opened','condensate_closed','condensate_opened'}:
                raise ValueError('Backend does not implement hybrid root: '+label)
            # Pinned feed._resume_stop map, moved into this hook on diagnostic copies
            # so both sides can be recorded. Native equivalence is tested exactly.
            name=label.split('_')[0]
            index=self.plant.feed_index if name=='feed' else self.feed_index+6
            if label.endswith('closed'):
                residual=float(endpoint[index])
                if abs(residual)>1e-6:raise ValueError('Check-valve zero-root residual too large')
                inertia=self.plant.feed_inertance if name=='feed' else self.feed_config.condensate_inertance_m_inv
                energy=.5*inertia*residual**2/self.feed_rho
                endpoint[index]=0.;endpoint[self.feed_index+1]+=energy
                if name=='feed':self.plant._feed_open=False
                else:self.condensate_open=False
                self.events.append(dict(time_s=now,kind=name+'_check_closed',root_residual_kg_s=residual,root_roundoff_energy_to_tank_j=energy))
            else:
                if name=='feed':self.plant._feed_open=True
                else:self.condensate_open=True
                self.events.append(dict(time_s=now,kind=name+'_check_opened',reason='drive_zero_crossing'))
        journal=deepcopy(self.events[journal_start:])
        if not journal:raise ValueError('Handled root has no journal entry')
        self.heat_events.append(dict(time_s=now,kind=journal[-1]['kind'],event=journal[-1],
            journal_events=journal,joint_native_map=len(journal)>1,
            before=before,after=_root_meter(self)))
        return True


class _PowerRecorder(_RootRecorder,PWRPowerSession):pass
class _FaultRecorder(_RootRecorder,PWRFaultSession):pass
class _BypassRecorder(_RootRecorder,PWRBypassSession):pass


def _copy_recorder(session):
    cls={PWRPowerSession:_PowerRecorder,PWRFaultSession:_FaultRecorder,PWRBypassSession:_BypassRecorder}.get(type(session))
    if cls is None:raise ValueError('Pinned power, bypass or fault session required')
    if type(session) is PWRFaultSession and (session.equipment_faults or session.active_faults):
        raise ValueError('Scheduled equipment faults unsupported')
    result=cls.__new__(cls);result.__dict__.update(deepcopy(session.__dict__));result.heat_events=[]
    return result


def _advance(copy,dt):
    if isinstance(copy,PWRFaultSession):
        # Fixed fault clock is bypassed only on fault-free diagnostic copies.
        copy._fault_advancing=True
        try:super(PWRFaultSession,copy).advance(dt)
        finally:copy._fault_advancing=False
    else:copy.advance(dt)


def _record(left,native,count):
    copy=_copy_recorder(left);copy._prepare_interval()
    start=copy.time;end=native.time;history=deepcopy(copy.events)
    samples=[_meter(copy)]
    for i in range(count):
        target=start+(end-start)*(i+1)/count
        _advance(copy,target-copy.time)
        samples.append(_meter(copy))
    if not copy.backup_exhausted and abs(float(copy.state[copy.power_index]))<=1e-3:
        raise ValueError('Unresolved terminal backup event')
    assembled=assemble_segments(samples,copy.heat_events)
    segments=assembled['segments']
    for event in copy.events[len(history):]:
        if event['kind'] not in KINDS:raise ValueError('Unsupported replay event: '+event['kind'])
    roots=[]
    for event in native.events:
        if event['time_s']>start:
            if event['kind'] not in KINDS:raise ValueError('Unsupported native event: '+event['kind'])
            roots.append(event)
    recorded_roots=[e for packet in copy.heat_events for e in packet['journal_events']]
    root_match=len(roots)==len(recorded_roots) and all(a['kind']==b['kind'] and abs(a['time_s']-b['time_s'])<=1e-9
        and a.get('index')==b.get('index') and a.get('target')==b.get('target') for a,b in zip(roots,recorded_roots))
    audit=assess_segments(segments);tied=deepcopy(segments);tied[-1][-1]=_meter(native)
    native_audit=assess_segments(tied)
    error=float(np.max(abs(copy.state-native.state)/np.maximum(abs(native.state),1.)))
    checks=dict(segment_balances=audit['passed'],native_endpoint_balances=native_audit['passed'],
        root_match=root_match,endpoint_state=error<=1e-6)
    return dict(subdivisions=count,segments=segments,events=copy.heat_events,event_groups=assembled['groups'],
        simultaneous_group_count=assembled['simultaneous_group_count'],absolute_map_impulses_j=assembled['absolute_map_impulses_j'],accounting=audit,
        native_endpoint_accounting=native_audit,endpoint_state_relative=error,checks=checks,passed=all(checks.values()))


def record_hybrid_window(session,duration_s,patch=None,*,subdivisions=16):
    if type(duration_s) not in (int,float) or not math.isfinite(duration_s) or not 0<duration_s<=1:
        raise ValueError('Duration in (0,1] required')
    if type(subdivisions) is not int or not 2<=subdivisions<=256:raise ValueError('Subdivision count in [2,256] required')
    _copy_recorder(session)
    source=identity();native=deepcopy(session);native.advance(duration_s,deepcopy(patch))
    left=deepcopy(session)
    if type(left) is PWRFaultSession:left._select_faults()
    if patch is not None:left._set_controls(deepcopy(patch))
    left._prepare_interval()
    report=_record(left,native,subdivisions)
    if source!=identity():raise ValueError('Hybrid heat source changed')
    return dict(source=source,**report,original_session_unchanged=True,closed_tasks=[])


def step_hybrid_heat(env,action=None,*,subdivisions=(16,32,64)):
    if type(env) is not StationEnv:raise ValueError('StationEnv required')
    env._require()
    if env.config.faults:raise ValueError('Scheduled faults unsupported')
    if (type(subdivisions) not in (list,tuple) or len(subdivisions)!=3 or
        any(type(n) is not int or not 2<=n<=256 for n in subdivisions) or
        subdivisions[1]!=2*subdivisions[0] or subdivisions[2]!=2*subdivisions[1]):
        raise ValueError('Three doubling subdivision counts in [2,256] required')
    source=identity();before=_meter(env._session.control.plant);trial=deepcopy(env)
    transition=trial.step(deepcopy(action));receipt=transition[4]['receipt']
    left_env=_post_command(env,receipt);left=left_env._session.control.plant;native=trial._session.control.plant
    after=_meter(left)
    if receipt['time_s']!=left.time or receipt['completed_at_s']!=native.time or native.time<=left.time:
        raise HeatEventError(dict(reason='receipt_clock'))
    if any(before[k]!=after[k] for k in ('stored_j','boundary_j')) or before['energy']['counters']!=after['energy']['counters']:
        raise HeatEventError(dict(reason='command_energy_impulse'))
    rows=[_record(left,native,n) for n in subdivisions];comparisons=[]
    for a,b in zip(rows,rows[1:]):
        x=a['accounting']['sampled_integrals_j'];y=b['accounting']['sampled_integrals_j']
        comparisons.append({k:abs(x[k]-y[k])/max(1.,abs(y[k]),abs(y['nuclear_heat_w']) if k=='external_net_w' else 0.) for k in x})
    checks=dict(replays=all(r['passed'] for r in rows),refined_integrals=max(comparisons[-1].values())<=2e-6,
        trend=all(comparisons[1][k]<=max(comparisons[0][k],1e-8) for k in comparisons[0]))
    proof=dict(source=source,replays=rows,comparisons=comparisons,checks=checks,passed=all(checks.values()),
        dispatched=deepcopy(receipt['dispatched']),native_state_committed=False,controller_substepped=False,
        relief_valves_supported=True,relief_pressure_cycle_qualified=False,simultaneous_events_supported=True,simultaneous_physical_scenario_qualified=False,closed_tasks=[])
    if not proof['passed']:raise HeatEventError(dict(reason='hybrid_heat_refused',candidate=proof))
    if source!=identity():raise HeatEventError(dict(reason='source_changed'))
    env.__dict__.update(trial.__dict__);proof['native_state_committed']=True
    return transition,proof


def advance_hybrid_schedule(session,schedule,*,subdivisions=(8,16,32)):
    """Commit a native physical schedule only if every refined window is accepted.

    Schedule boundaries are declared numerical/control boundaries, not an agent
    clock. No partial prefix is committed after a later failure.
    """
    _copy_recorder(session)
    if type(schedule) not in (list,tuple) or not 1<=len(schedule)<=16:
        raise ValueError('One to sixteen schedule windows required')
    for row in schedule:
        if type(row) is not dict or set(row)!={'duration_s','patch'}:
            raise ValueError('Explicit duration_s/patch schedule required')
        dt=row['duration_s']
        if type(dt) not in (int,float) or not math.isfinite(dt) or not 0<dt<=1:
            raise ValueError('Schedule duration in (0,1] required')
        if row['patch'] is not None and type(row['patch']) is not dict:raise ValueError('Command dictionary required')
    if (type(subdivisions) not in (tuple,list) or len(subdivisions)!=3 or
        any(type(n) is not int or not 2<=n<=256 for n in subdivisions) or
        subdivisions[1]!=2*subdivisions[0] or subdivisions[2]!=2*subdivisions[1]):
        raise ValueError('Three doubling subdivision counts required')
    source=identity();native=deepcopy(session);reports=[]
    for index,row in enumerate(schedule):
        try:
            left=deepcopy(native);before=_meter(left)
            if type(left) is PWRFaultSession:left._select_faults()
            if row['patch'] is not None:left._set_controls(deepcopy(row['patch']))
            left._prepare_interval();after=_meter(left)
            if any(before[k]!=after[k] for k in ('stored_j','boundary_j')) or before['energy']['counters']!=after['energy']['counters']:
                raise ValueError('Schedule command changes inventories')
            native.advance(row['duration_s'],deepcopy(row['patch']))
            rows=[_record(left,native,n) for n in subdivisions];comparisons=[]
            for a,b in zip(rows,rows[1:]):
                x=a['accounting']['sampled_integrals_j'];y=b['accounting']['sampled_integrals_j']
                comparisons.append({k:abs(x[k]-y[k])/max(1.,abs(y[k]),abs(y['nuclear_heat_w']) if k=='external_net_w' else 0.) for k in x})
            checks=dict(replays=all(r['passed'] for r in rows),refined_integrals=max(comparisons[-1].values())<=2e-6,
                trend=all(comparisons[1][k]<=max(comparisons[0][k],1e-8) for k in comparisons[0]))
            proof=dict(window=index,start_s=left.time,end_s=native.time,replays=rows,comparisons=comparisons,
                checks=checks,passed=all(checks.values()))
            if not proof['passed']:
                raise HeatEventError(dict(reason='schedule_heat_refused',window=index,candidate=proof,
                    accepted_prefix=reports,native_state_committed=False))
            reports.append(proof)
        except HeatEventError:raise
        except Exception as error:
            raise HeatEventError(dict(reason='schedule_execution_failed',window=index,
                error_type=type(error).__name__,message=str(error),accepted_prefix=reports,native_state_committed=False)) from error
    ledgers=[r['replays'][-1]['native_endpoint_accounting'] for r in reports]
    defects={k:math.fsum(a['absolute_defect_sums_j'][k] for a in ledgers) for k in ledgers[0]['absolute_defect_sums_j']}
    scale=math.fsum(a['normalization_scale_j'] for a in ledgers)
    if max(defects.values())/scale>1e-6:raise HeatEventError(dict(reason='schedule_aggregate_budget',native_state_committed=False))
    accounting=dict(sampled_integrals_j={k:math.fsum(a['sampled_integrals_j'][k] for a in ledgers) for k in FLUXES},
        absolute_defect_sums_j=defects,normalization_scale_j=scale,
        duration_s=math.fsum(a['duration_s'] for a in ledgers))
    if source!=identity():raise HeatEventError(dict(reason='source_changed',native_state_committed=False))
    session.__dict__.update(native.__dict__)
    return dict(source=source,schedule=deepcopy(schedule),windows=reports,accounting=accounting,passed=True,native_state_committed=True,
        qualified_general_pressure_cycle=False,closed_tasks=[])
