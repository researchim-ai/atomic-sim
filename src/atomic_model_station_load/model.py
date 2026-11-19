"""Causal load references and fail-closed transient acceptance; no physical reset."""
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from atomic_model_station_campaign import Episode


LOOPS={'rod_target','steam_bypass','steam_admission','feed_flow_fraction','condensate_flow_fraction','feed_inventory_fraction'}

def finite(x):
    return isinstance(x,Real) and not isinstance(x,bool) and isfinite(x)


@dataclass(frozen=True)
class LoadCycle:
    tick_s: float = 5.
    duration_s: float = 1800.
    points: tuple = ((0.,1.),(450.,.98),(900.,1.02),(1350.,1.))
    settling_window_s: float = 60.

    def __post_init__(self):
        if not finite(self.tick_s) or not .01<=self.tick_s<=60:raise ValueError('Load tick')
        if not finite(self.duration_s) or not self.tick_s<=self.duration_s<=7200:raise ValueError('Load duration')
        ratio=self.duration_s/self.tick_s
        if abs(ratio-round(ratio))>1e-9 or round(ratio)>8192:raise ValueError('Integral bounded load ticks')
        if not finite(self.settling_window_s) or self.settling_window_s<self.tick_s or abs(self.settling_window_s/self.tick_s-round(self.settling_window_s/self.tick_s))>1e-9:raise ValueError('Settling window')
        if type(self.points) not in (list,tuple) or not 2<=len(self.points)<=16:raise ValueError('Load points')
        points=[]
        for item in self.points:
            if type(item) not in (list,tuple) or len(item)!=2:raise ValueError('Load point pair')
            t,f=item
            if not finite(t) or not finite(f) or not 0<=t<self.duration_s or not .95<=f<=1.05:
                raise ValueError('Load point domain')
            if abs(t/self.tick_s-round(t/self.tick_s))>1e-9:raise ValueError('Load epoch must be a tick')
            if points and t<=points[-1][0]:raise ValueError('Strictly increasing load epochs')
            points.append((float(t),float(f)))
        if points[0]!=(0.,1.):raise ValueError('Start from declared nominal')
        ends=[p[0] for p in points[1:]]+[self.duration_s]
        if any(round((end-start)/self.tick_s)<round(self.settling_window_s/self.tick_s) for (start,_),end in zip(points,ends)):
            raise ValueError('Every plateau must contain the settling window')
        object.__setattr__(self,'points',tuple(points))

    @property
    def ticks(self):return round(self.duration_s/self.tick_s)

    def reference(self,tick):
        if type(tick) is not int or not 0<=tick<self.ticks:raise ValueError('Load tick outside program')
        return next(f for t,f in reversed(self.points) if t<=tick*self.tick_s)

    def episode(self,name='load_cycle',*,max_step_s=1.,rtol=1e-7):
        commands=tuple(dict(tick=round(t/self.tick_s),operator=dict(load_fraction=f),targets=dict(rod_target=3.15e9*f)) for t,f in self.points)
        return Episode(name,self.ticks,tick_s=self.tick_s,commands=commands,equilibrium_start=True,
            inventory_deadband_m=.0002,max_step_s=max_step_s,rtol=rtol,
            checkpoint_ticks=tuple(sorted({0,self.ticks//2,self.ticks})),record_every=1,
            tracking_tau_s=max(10.,self.tick_s))


def assess(records,cycle):
    """Records represent completed transitions; reference belongs to START of each tick."""
    if type(cycle) is not LoadCycle:raise ValueError('LoadCycle required')
    reasons=[]
    if len(records)!=cycle.ticks:reasons.append('incomplete_trajectory')
    audits={k:0. for k in ('energy_relative','mass_relative','boron_relative')}
    plateaus=[];healthy=True;commands=True
    for index,row in enumerate(records):
        if index>=cycle.ticks:reasons.append('extra_transition');break
        result=row['result'];receipt=result['receipt'];p=row['physical'];measure=result['observation']['measurements']
        expected=cycle.reference(index)
        if (any(not finite(v) for v in (p['time_s'],receipt['time_s'],receipt['completed_at_s'])) or abs(p['time_s']-(index+1)*cycle.tick_s)>1e-8
            or abs(receipt['time_s']-index*cycle.tick_s)>1e-8
            or abs(receipt['completed_at_s']-p['time_s'])>1e-8):reasons.append('clock_or_cadence')
        commands=commands and receipt['targets']['rod_target']==3.15e9*expected and row['load_fraction']==expected
        healthy=healthy and not receipt['trip_latched'] and all(v=='auto' for v in receipt['modes'].values()) and set(receipt['modes'])==LOOPS
        for k in audits:
            value=p[k]
            if not finite(value) or value<0:reasons.append('invalid_audit')
            else:audits[k]=max(audits[k],value)
        # A missing/invalid sensor must not silently disappear from a settling metric.
        for key in ('power','pressure','speed','sg_level','hotwell_level','feed_level'):
            if key not in measure or not finite(measure[key]['value']) or measure[key].get('quality')!=['good']:
                healthy=False
    ends=[p[0] for p in cycle.points[1:]]+[cycle.duration_s]
    for (start,fraction),end in zip(cycle.points,ends):
        selected=[r for r in records if finite(r['result']['receipt']['time_s']) and round((end-cycle.settling_window_s)/cycle.tick_s)<=round(r['result']['receipt']['time_s']/cycle.tick_s)<round(end/cycle.tick_s)]
        expected_count=round(cycle.settling_window_s/cycle.tick_s)
        errors=dict(power_relative=0.,pressure_relative=0.,speed_relative=0.,sg_level_m=0.,hotwell_level_m=0.,feed_level_m=0.)
        own=True;valid=True
        for row in selected:
            m=row['result']['observation']['measurements'];p=row['physical']['powers']
            for key,target,metric in [('power',3.15e9*fraction,'power_relative'),('pressure',6.3e6,'pressure_relative'),('speed',157.,'speed_relative')]:
                v=m[key]['value']
                if finite(v):errors[metric]=max(errors[metric],abs(v/target-1))
                else:valid=False
            for sensor,output,metric in [('sg_level','feed_flow_fraction','sg_level_m'),('hotwell_level','condensate_flow_fraction','hotwell_level_m'),('feed_level','feed_inventory_fraction','feed_level_m')]:
                value=m[sensor]['value'];target=row['result']['receipt']['targets'][output]
                if finite(value) and finite(target):errors[metric]=max(errors[metric],abs(value-target))
                else:valid=False
            own=own and finite(p['gross_electrical_w']) and finite(p['own_needs_w']) and p['gross_electrical_w']>p['own_needs_w']>=0
        passed=(len(selected)==expected_count and own and valid and errors['power_relative']<=.02 and errors['pressure_relative']<=.02 and errors['speed_relative']<=.01 and errors['sg_level_m']<=.02 and errors['hotwell_level_m']<=.005 and errors['feed_level_m']<=.005)
        plateaus.append(dict(start_s=start,end_s=end,fraction=fraction,samples=len(selected),errors=errors,own_needs_supplied=bool(own),passed=bool(passed)))
    if not commands:reasons.append('reference_not_applied')
    if not healthy:reasons.append('trip_mode_or_measurement')
    if any(v>(1e-6 if k=='energy_relative' else 1e-10) for k,v in audits.items()):reasons.append('physical_budget')
    if not all(p['passed'] for p in plateaus):reasons.append('plateau_tracking')
    return dict(passed=not reasons,reasons=sorted(set(reasons)),maxima=audits,plateaus=plateaus,
        completed_ticks=len(records),duration_s=cycle.duration_s,closed_tasks=[],r1_accepted=False,
        scope='Trajectory and terminal-window tracking only; numerical refinement, native restart and installed repeat are separate gates')


def compare(a,b,*,accounting_indices=()):
    """Compare every co-timed physical coordinate; final agreement alone is insufficient."""
    excluded=set(accounting_indices)
    if not a or len(a)!=len(b):return dict(passed=False,reason='trajectory_length',maximum_relative=None)
    maximum=0.;worst=None
    for tick,(x,y) in enumerate(zip(a,b)):
        if x['physical']['time_s']!=y['physical']['time_s'] or len(x['state'])!=len(y['state']):
            return dict(passed=False,reason='trajectory_layout_or_clock',maximum_relative=None)
        for index,(v,w) in enumerate(zip(x['state'],y['state'])):
            if index in excluded:continue
            if not finite(v) or not finite(w):return dict(passed=False,reason='nonfinite_state',maximum_relative=None)
            delta=abs(v-w)/max(abs(w),1.)
            if delta>maximum:maximum=delta;worst=dict(tick=tick,index=index,time_s=x['physical']['time_s'])
    return dict(passed=maximum<=1e-3,maximum_relative=maximum,worst=worst,reason='compared_whole_trajectory')
