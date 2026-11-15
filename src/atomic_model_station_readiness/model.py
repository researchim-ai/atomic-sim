"""Privileged whole-inventory readiness checks; never alter physical state."""
from collections import deque
from dataclasses import dataclass
from copy import deepcopy
import math

from atomic_model_station_faults import FaultSession
from atomic_model_station_faults.spec import finite
from atomic_model_station_control.policy import valid_reading


@dataclass(frozen=True)
class ReadinessConfig:
    window_s:float=600.
    relative_span:float=1e-3
    maximum_gap_s:float=5.
    stored_power_relative:float=1e-3
    energy_relative:float=1e-6
    mass_relative:float=1e-10
    boron_relative:float=1e-10

    def __post_init__(self):
        for key in ('window_s','relative_span','maximum_gap_s','stored_power_relative','energy_relative','mass_relative','boron_relative'):
            value=getattr(self,key)
            if not finite(value) or value<=0:raise ValueError('Positive finite readiness parameter: '+key)
        if self.window_s>7200 or self.maximum_gap_s>self.window_s or self.relative_span>1:
            raise ValueError('Readiness window/span domain')


def schema(owner):
    if type(owner) is not FaultSession:raise ValueError('FaultSession readiness owner required')
    p=owner.control.plant;size=len(p.state)
    names=['state['+str(i)+']' for i in range(size)]
    units=['SI' for _ in range(size)]
    def assign(index,labels,dimensions):
        names[index:index+len(labels)]=labels;units[index:index+len(labels)]=dimensions
    for i,name in enumerate(p.plant.network.volume_names()):
        assign(13+3*i,[name+'.mass',name+'.internal_energy',name+'.boron'],['kg','J','kg'])
    start=13+3*len(p.plant.network.volumes)
    for i,name in enumerate(p.plant.network.junction_names()):assign(start+i,[name+'.flow'],['kg/s'])
    assign(p.base_size,['rotor.energy','clad.temperature'],['J','K'])
    assign(p.cooling_index,['cooling.cold_energy','cooling.hot_energy','cooling.flow'],['J','J','kg/s'])
    assign(p.feed_index,['feed.mass','feed.energy','makeup.mass','makeup.energy','drain.mass','drain.energy','condensate.flow'],['kg','J','kg','J','kg','J','kg/s'])
    assign(p.power_index,['backup.energy'],['J'])
    assign(p.core_index,['core['+str(i)+']' for i in range(p.neutron_core.size)],['1']*p.neutron_core.size)
    assign(p.steam_index,['header.mass','header.energy','header.outlet_flow','header.inlet_position','header.outlet_position'],['kg','J','kg/s','1','1'])
    assign(p.bypass_index,['bypass.position'],['1'])
    excluded=sorted([*range(p.ledger_index,p.ledger_index+4),*range(p.power_index+1,p.power_index+4)])
    return dict(size=size,names=names,units=units,accounting_indices=excluded,
        accounting_reason='Only cumulative external/gross/own/throughput and grid/DHR/reserve-use ledgers; no physical stock is excluded')


def goals_met(frame,targets):
    checks=[('power','fission_power_w','rod_target',.02,True),
        ('pressure','steam_pressure_pa','steam_bypass',.02,True),
        ('speed','turbine_speed_rad_s','steam_admission',.01,True),
        ('sg_level','sg_collapsed_level_m','feed_flow_fraction',.02,False),
        ('hotwell_level','hotwell_level_m','condensate_flow_fraction',.005,False),
        ('feed_level','feed_tank_level_m','feed_inventory_fraction',.005,False)]
    for name,quantity,target,tolerance,relative in checks:
        value=valid_reading(frame,name,quantity)
        if value is None or abs(value-targets[target])>tolerance*(max(abs(targets[target]),1.) if relative else 1.):
            return False
    return True


def capture(owner):
    snapshot=owner.control.plant.snapshot();powers=snapshot['powers']
    goals=goals_met(owner.control.instruments.operator_frame(),owner.control.targets)
    return dict(time_s=float(snapshot['time_s']),state=owner.control.plant.state.tolist(),
        energy_relative=float(snapshot['energy_relative']),mass_relative=float(snapshot['mass_relative']),boron_relative=float(snapshot['boron_relative']),
        stored_power_relative=float(abs(powers['external_net_w'])/max(abs(powers['nuclear_heat_w']),1.)),
        trip=bool(owner.control.trip_latched),goals_met=bool(goals))


def assess(samples,layout,config=None):
    c=config if config is not None else ReadinessConfig()
    if type(c) is not ReadinessConfig or type(samples) not in (list,tuple) or len(samples)>8193:
        raise ValueError('Readiness samples/config budget')
    if type(layout) is not dict or set(layout)!={'size','names','units','accounting_indices','accounting_reason'}:
        raise ValueError('Readiness schema fields')
    size=layout['size'];excluded=layout['accounting_indices']
    if type(size) is not int or not 1<=size<=256 or len(layout['names'])!=size or len(layout['units'])!=size:
        raise ValueError('Readiness coordinate schema')
    if type(excluded) is not list or len(set(excluded))!=len(excluded) or any(type(i) is not int or not 0<=i<size for i in excluded) or len(excluded)>=size:
        raise ValueError('Readiness accounting mask')
    last=None
    for sample in samples:
        if type(sample) is not dict or set(sample)!={'time_s','state','energy_relative','mass_relative','boron_relative','stored_power_relative','trip','goals_met'}:
            raise ValueError('Readiness sample fields')
        if not finite(sample['time_s']) or sample['time_s']<0 or (last is not None and sample['time_s']<=last):raise ValueError('Readiness monotone clock')
        last=sample['time_s']
        if type(sample['state']) is not list or len(sample['state'])!=size or any(not finite(v) for v in sample['state']):raise ValueError('Finite complete readiness vector')
        if type(sample['trip']) is not bool or type(sample['goals_met']) is not bool or any(not finite(sample[k]) or sample[k]<0 for k in ('energy_relative','mass_relative','boron_relative','stored_power_relative')):
            raise ValueError('Readiness physical audit fields')
    if len(samples)<3:return dict(ready=False,reason='insufficient_window',sample_count=len(samples))
    end=samples[-1]['time_s'];start=end-c.window_s
    window=[s for s in samples if s['time_s']>=start-1e-9]
    if len(window)<3 or window[0]['time_s']>start+1e-8:
        return dict(ready=False,reason='insufficient_window',sample_count=len(window))
    if any(b['time_s']-a['time_s']>c.maximum_gap_s+1e-8 for a,b in zip(window,window[1:])):
        return dict(ready=False,reason='sparse_window',sample_count=len(window))
    coordinates=[]
    for i in range(size):
        if i in excluded:continue
        values=[s['state'][i] for s in window];low=min(values);high=max(values)
        normalized=(high-low)/max(abs(values[-1]),1.)
        coordinates.append(dict(index=i,name=layout['names'][i],unit=layout['units'][i],minimum=low,
            maximum=high,end=values[-1],normalized_span=normalized))
    coordinates.sort(key=lambda x:(-x['normalized_span'],x['index']))
    maxima={k:max(s[k] for s in window) for k in ('energy_relative','mass_relative','boron_relative','stored_power_relative')}
    physical=all(maxima[k]<=getattr(c,k) for k in maxima) and not any(s['trip'] for s in window) and all(s['goals_met'] for s in window)
    ready=bool(coordinates[0]['normalized_span']<=c.relative_span and physical)
    return dict(ready=ready,reason='ready' if ready else 'physical_drift' if coordinates[0]['normalized_span']>c.relative_span else 'physical_audit_or_trip',
        window_start_s=window[0]['time_s'],window_end_s=end,sample_count=len(window),
        maximum_normalized_span=coordinates[0]['normalized_span'],maximum_audits=maxima,
        largest_changes=coordinates[:8],excluded_accounting_indices=excluded,
        scope='Whole physical-vector span, not only measured setpoint error; normalization max(abs(end),1 SI unit)')


def prepare_hot_state(*,maximum_warmup_s=7200.,tick_s=5.,config=None,required_followup_s=0.,max_step_s=1.,rtol=1e-7,initial_session=None):
    c=config if config is not None else ReadinessConfig(maximum_gap_s=tick_s)
    if type(c) is not ReadinessConfig:raise ValueError('ReadinessConfig required')
    if not finite(maximum_warmup_s) or not 0<maximum_warmup_s<=7200 or not finite(tick_s) or not .01<=tick_s<=10:
        raise ValueError('Hot preparation horizon/clock')
    if not finite(required_followup_s) or required_followup_s<0 or required_followup_s>7200:raise ValueError('Hot follow-up horizon')
    if initial_session is None:
        owner=FaultSession(tick_s=tick_s,history_budget_steps=8192,max_step_s=max_step_s,rtol=rtol)
    else:
        if type(initial_session) is not FaultSession or initial_session.faults or initial_session.control.trip_latched:
            raise ValueError('Healthy FaultSession required for hot continuation')
        if initial_session.control.instruments.tick_s!=tick_s:
            raise ValueError('Hot continuation must preserve the control clock')
        owner=deepcopy(initial_session)
    duration=maximum_warmup_s-owner.control.plant.time
    ticks=round(duration/tick_s)
    if duration<=0 or abs(ticks*tick_s-duration)>1e-8 or ticks>8192 or ticks<1:raise ValueError('Hot preparation step budget')
    layout=schema(owner);window=deque(maxlen=math.ceil(c.window_s/tick_s)+2);window.append(capture(owner))
    verdict=assess(list(window),layout,c);failure=None
    for i in range(ticks):
        try:owner.step(modes={k:'auto' for k in owner.control.loops} if i==0 else None)
        except ValueError as error:failure=str(error);break
        window.append(capture(owner));verdict=assess(list(window),layout,c)
        if verdict['ready']:break
    physical_remaining=max(0.,owner.control.plant._maximum_time_s()-owner.control.plant.time)
    journal_remaining=max(0.,(owner.control.plant._maximum_history_steps()-len(owner.history))*tick_s)
    remaining=min(physical_remaining,journal_remaining)
    return dict(prepared=bool(verdict['ready'] and remaining>0 and remaining>=required_followup_s and failure is None),
        stationary=verdict['ready'],readiness=verdict,failure=failure,
        completed_time_s=owner.control.plant.time,remaining_supported_time_s=remaining,
        required_followup_s=required_followup_s,physical_remaining_s=physical_remaining,journal_remaining_s=journal_remaining,checkpoint=owner.checkpoint(),
        time_reset=False,inventories_projected=False,pi_reset=False,r1_accepted=False)
