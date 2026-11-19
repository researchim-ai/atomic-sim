"""Never interpolate through an event's discontinuous power values."""
from copy import deepcopy
from pathlib import Path
import hashlib
import math
from atomic_model_station import PWRPowerSession
from atomic_model_station_energy_eval.model import COUNTERS,POWERS
from atomic_model_station_heat_accounting import assess_heat
from atomic_model_station_heat_accounting.model import FLUXES
from atomic_model_station_piecewise_heat.model import identity as previous_identity


def identity():
    return dict(previous=previous_identity(),files={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(Path(__file__).parent.glob('*.py'))})


def assess_segments(segments):
    """Audit contiguous, two-sided samples. No energy impulse is supported.

    The same event time occurs as the previous segment's last and next segment's
    first sample. Powers may jump; stored energy and every counter must not.
    """
    if type(segments) not in (list,tuple) or not segments:
        raise ValueError('Nonempty heat segments required')
    if any(type(s) not in (list,tuple) or len(s)<2 for s in segments):
        raise ValueError('Each heat segment needs at least two samples')
    audits=[assess_heat(s) for s in segments]
    reasons=set();boundaries=[]
    for i,a in enumerate(audits):
        if not a['passed']:reasons.add('segment_budget')
        if i:
            before=segments[i-1][-1];after=segments[i][0]
            if before['energy']['time_s']!=after['energy']['time_s']:
                reasons.add('boundary_clock')
            differences={k:after['energy']['counters'][k]-before['energy']['counters'][k] for k in COUNTERS}
            differences.update(stored_j=after['stored_j']-before['stored_j'],boundary_j=after['boundary_j']-before['boundary_j'])
            # Absolute energy tolerance, never scaled by the enormous initial inventory.
            if max(abs(v) for v in differences.values())>1e-3:reasons.add('boundary_energy_impulse')
            boundaries.append(dict(time_s=after['energy']['time_s'],energy_jumps_j=differences,
                flux_jumps_w={k:after['fluxes'][k]-before['fluxes'][k] for k in FLUXES},
                power_jumps_w={k:after['energy']['powers'][k]-before['energy']['powers'][k] for k in POWERS}))
    impulse_bounds={k:math.fsum(abs(b['energy_jumps_j'][k]) for b in boundaries) for k in (*COUNTERS,'stored_j','boundary_j')}
    if max(impulse_bounds.values())>1e-3:reasons.add('accumulated_boundary_impulse')
    integrals={k:math.fsum(a['sampled_integrals_j'][k] for a in audits) for k in FLUXES}
    bounds={k:math.fsum(abs(a[k]) for a in audits) for k in ('heat_closure_j','window_storage_defect_j','boundary_quadrature_defect_j')}
    scale=math.fsum(max(1.,abs(a['sampled_integrals_j']['nuclear_heat_w'])+
        abs(a['sampled_integrals_j']['primary_pump_loss_out_w'])+abs(a['sampled_integrals_j']['external_cooling_heat_out_w'])+
        abs(a['electrical']['energy_delta_j']['net_exported_j'])) for a in audits)
    if max(bounds.values())/scale>1e-6:reasons.add('aggregate_budget')
    return dict(passed=not reasons,reasons=sorted(reasons),segments=audits,boundaries=boundaries,
        sampled_integrals_j=integrals,absolute_defect_sums_j=bounds,normalization_scale_j=scale,
        absolute_boundary_impulse_sums_j=impulse_bounds,
        duration_s=math.fsum(a['electrical']['duration_s'] for a in audits),
        impulse_supported=False,stationary_efficiency_certified=False)


def _meter(session):
    copy=deepcopy(session);s=copy.snapshot()
    energy=dict(time_s=float(s['time_s']),counters={k:float(s[k]) for k in COUNTERS},
        powers={k:float(s['powers'][k]) for k in POWERS},grid_available=s['electrical']['grid_available'],
        trip=s['actuator_status']['scram_latched'] or s['actuator_status']['turbine_trip_latched'],
        energy_relative=float(s['energy_relative']))
    return dict(energy=energy,stored_j=float(copy._stored()),boundary_j=float(copy.state[copy.ledger_index]),
        fluxes={k:float(s['powers'][k]) for k in FLUXES})


class _RecordedPower(PWRPowerSession):
    def _custom_root(self,label,endpoint,now):
        if label!='backup_empty':raise ValueError('Unsupported root in backup heat recorder: '+label)
        before=_meter(self)
        handled=super()._custom_root(label,endpoint,now)
        after=_meter(self)
        self.heat_events.append(dict(kind='backup_exhausted',time_s=now,before=before,after=after,
            event=deepcopy(self.events[-1])))
        return handled


def record_power_window(session,duration_s,patch=None,*,subdivisions=16):
    """Read-only diagnostic replay of a PWRPowerSession (not a StationEnv step).

    Does not replace native solver state and does not qualify other hybrid events.
    Whole-interval native and subdivided endpoints are reported separately.
    """
    if type(session) is not PWRPowerSession:raise ValueError('Exact PWRPowerSession required')
    if type(duration_s) not in (float,int) or not math.isfinite(duration_s) or not 0<duration_s<=1:
        raise ValueError('Duration in (0,1] required')
    if type(subdivisions) is not int or not 2<=subdivisions<=256:
        raise ValueError('Subdivision count in [2,256] required')
    source=identity();native=deepcopy(session);native.advance(duration_s,deepcopy(patch))
    copy=_RecordedPower.__new__(_RecordedPower);copy.__dict__.update(deepcopy(session.__dict__));copy.heat_events=[]
    if patch is not None:copy._set_controls(deepcopy(patch))
    copy._prepare_interval();start=copy.time;history=deepcopy(copy.events)
    segments=[];current=[_meter(copy)];seen=0
    for i in range(subdivisions):
        target=start+duration_s*(i+1)/subdivisions
        copy.advance(target-copy.time)
        for event in copy.heat_events[seen:]:
            gap=event['time_s']-current[-1]['energy']['time_s']
            if abs(gap)<=1e-12 and len(current)>=2:
                # A root may be reported on the next solver call at its left edge.
                # Keep the solver's two-sided root sample instead of a duplicate epoch.
                current[-1]=event['before']
            elif gap>0:current.append(event['before'])
            else:raise ValueError('Event boundary without positive left interval')
            segments.append(current);current=[event['after']]
        seen=len(copy.heat_events)
        sample=_meter(copy)
        if sample['energy']['time_s']>current[-1]['energy']['time_s']:current.append(sample)
    if len(current)<2:raise ValueError('Terminal event needs an explicit right interval')
    if not copy.backup_exhausted and abs(float(copy.state[copy.power_index]))<=1e-3:
        raise ValueError('Unresolved terminal backup event')
    segments.append(current)
    for event in copy.events[len(history):]:
        if event['kind']!='backup_exhausted':raise ValueError('Unsupported physical event: '+event['kind'])
    native_events=native.events[len(session.events):]
    native_roots=[e for e in native_events if e['kind']=='backup_exhausted']
    root_match=(len(native_roots)==len(copy.heat_events) and all(abs(a['time_s']-b['time_s'])<=1e-9 for a,b in zip(native_roots,copy.heat_events)))
    for event in native_events:
        if event['time_s']>start and event['kind']!='backup_exhausted':raise ValueError('Unsupported native event: '+event['kind'])
    audit=assess_segments(segments)
    tied=deepcopy(segments);tied[-1][-1]=_meter(native);native_audit=assess_segments(tied)
    if source!=identity():raise ValueError('Heat recorder source changed')
    return dict(source=source,subdivisions=subdivisions,segments=segments,events=copy.heat_events,
        accounting=audit,native_endpoint_accounting=native_audit,native_root_match=root_match,
        passed=audit['passed'] and native_audit['passed'] and root_match,
        original_session_unchanged=True,station_env_integration=False,
        discontinuous_valves_supported=False,closed_tasks=[])
