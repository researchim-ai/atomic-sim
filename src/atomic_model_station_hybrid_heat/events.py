"""Assemble exact-time event groups without erasing intermediate energy impulses."""
from copy import deepcopy
import math
from atomic_model_station_energy_eval.model import COUNTERS
from atomic_model_station_heat_accounting import assess_heat


def _inventory(sample):
    return dict(sample['energy']['counters'],stored_j=sample['stored_j'],boundary_j=sample['boundary_j'])


def _validate_point(sample):
    audit=assess_heat([sample])
    if set(audit['reasons'])-{'empty_window'}:raise ValueError('Invalid event-side energy sample')


def assemble_segments(samples,events):
    """Exact equality groups event times. Nearby but distinct roots stay distinct.

    Intermediate samples must chain exactly, and cumulative absolute inventory
    changes across all event maps may not exceed the existing 1e-3 J allowance.
    Initial/terminal events still require separate boundary handling.
    """
    if type(samples) not in (list,tuple) or len(samples)<2:raise ValueError('Positive sample window required')
    if type(events) not in (list,tuple):raise ValueError('Ordered event sequence required')
    clocks=[s['energy']['time_s'] for s in samples]
    if any(type(t) not in (int,float) or not math.isfinite(t) for t in clocks) or any(b<=a for a,b in zip(clocks,clocks[1:])):
        raise ValueError('Strict finite sample clock required')
    groups=[];impulses={k:0. for k in (*COUNTERS,'stored_j','boundary_j')}
    for event in events:
        t=event['time_s']
        if type(t) not in (int,float) or not math.isfinite(t) or not clocks[0]<t<clocks[-1]:
            raise ValueError('Event requires positive left and right intervals')
        if groups and t<groups[-1]['time_s']:raise ValueError('Event order changed')
        before,after=event['before'],event['after']
        if before['energy']['time_s']!=t or after['energy']['time_s']!=t:raise ValueError('Event-side clock mismatch')
        _validate_point(before);_validate_point(after)
        a,b=_inventory(before),_inventory(after)
        for key in impulses:impulses[key]+=abs(b[key]-a[key])
        if max(impulses.values())>1e-3:raise ValueError('Accumulated event energy impulse')
        if groups and t==groups[-1]['time_s']:
            group=groups[-1]
            if group['after']!=before:raise ValueError('Discontinuous intermediate event chain')
            group['after']=deepcopy(after);group['events'].append(deepcopy(event))
        else:groups.append(dict(time_s=t,before=deepcopy(before),after=deepcopy(after),events=[deepcopy(event)]))
    segments=[];current=[deepcopy(samples[0])];index=1
    for group in groups:
        t=group['time_s']
        while index<len(samples) and clocks[index]<t:
            current.append(deepcopy(samples[index]));index+=1
        current.append(deepcopy(group['before']));segments.append(current)
        current=[deepcopy(group['after'])]
        if index<len(samples) and clocks[index]==t:index+=1
    current.extend(deepcopy(samples[index:]));segments.append(current)
    return dict(segments=segments,groups=groups,absolute_map_impulses_j=impulses,
        simultaneous_group_count=sum(len(g['events'])>1 for g in groups))
