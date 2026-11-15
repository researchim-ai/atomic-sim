"""Sampled first-order instruments with delayed delivery and independent noise.

The analog input is piecewise linear between consecutive supplied truth ticks.
No past sample is synthesized using a later tick. Instruments are initialized
at the first hot-state truth value; no cold instrument startup is claimed.
"""
from copy import deepcopy
from dataclasses import asdict,dataclass
import hashlib
import math
from pathlib import Path
from types import MappingProxyType

import numpy as np
from atomic_model_multiphysics.provenance import seal,unseal
from atomic_model_station_faults.spec import Fault,canonical
from atomic_model_station_faults import spec as fault_spec

CATALOG=MappingProxyType({
    'fission_power_w':(('core','fission_total_w'),'W'),
    'steam_pressure_pa':(('steam_header','pressure_pa'),'Pa'),
    'steam_flow_kg_s':(('steam_flow_kg_s',),'kg/s'),
    'turbine_speed_rad_s':(('turbine_speed_rad_s',),'rad/s'),
    'hotwell_mass_kg':(('hotwell_mass_kg',),'kg'),
    'sg_collapsed_level_m':(('level_readouts','sg_collapsed_level_m'),'m'),
    'feed_tank_level_m':(('level_readouts','feed_tank_level_m'),'m'),
    'hotwell_level_m':(('level_readouts','hotwell_level_m'),'m'),
    'sg_outflow_kg_s':(('level_readouts','sg_outflow_kg_s'),'kg/s'),
    'feed_flow_kg_s':(('level_readouts','feed_flow_kg_s'),'kg/s'),
    'condensate_flow_kg_s':(('level_readouts','condensate_flow_kg_s'),'kg/s'),
})


def signature():
    return dict(files={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in sorted(Path(__file__).parent.glob('*.py'))},numpy=np.__version__,fault_spec_sha256=hashlib.sha256(Path(fault_spec.__file__).read_bytes()).hexdigest())


def finite(value):
    if type(value) not in (int,float):return False
    try:return math.isfinite(value)
    except OverflowError:return False


@dataclass(frozen=True)
class Sensor:
    name:str
    quantity:str
    period_ticks:int=1
    delay_ticks:int=0
    lag_s:float=0.
    sigma:float=0.
    stale_after_ticks:int=10
    measurement_min:float|None=None
    measurement_max:float|None=None

    def __post_init__(self):
        if type(self.name) is not str or not self.name.isidentifier() or len(self.name)>64:raise ValueError('Instrument name')
        if type(self.quantity) is not str or self.quantity not in CATALOG:raise ValueError('Instrument quantity')
        for key,low,high in [('period_ticks',1,600),('delay_ticks',0,600),('stale_after_ticks',0,1000000)]:
            value=getattr(self,key)
            if type(value) is not int or not low<=value<=high:raise ValueError('Instrument tick domain:'+key)
        for key,high in [('lag_s',1e6),('sigma',1e12)]:
            value=getattr(self,key)
            if not finite(value) or not 0<=value<=high:raise ValueError('Instrument parameter:'+key)
        for value in (self.measurement_min,self.measurement_max):
            if value is not None and not finite(value):raise ValueError('Instrument range')
        if self.measurement_min is not None and self.measurement_max is not None and self.measurement_min>=self.measurement_max:
            raise ValueError('Instrument range order')


class StationSignals:
    """Privileged instrument owner; only operator_frame() is agent-facing.

    Feed one complete truth frame per integer tick. This object observes a
    plant; it neither advances the plant nor implements an agent transaction.
    """
    def __init__(self,sensors,*,tick_s=1.,seed=0,faults=()):
        if type(sensors) not in (list,tuple) or not 1<=len(sensors)<=32 or any(type(s) is not Sensor for s in sensors):raise ValueError('Instrument list')
        if len({s.name for s in sensors})!=len(sensors):raise ValueError('Duplicate instrument name')
        if not finite(tick_s) or not .01<=tick_s<=60.:raise ValueError('Instrument clock')
        if type(seed) is not int or not 0<=seed<2**63:raise ValueError('Instrument seed')
        self.faults=canonical(faults,domain='sensor')
        if any(f.target not in {s.name for s in sensors} for f in self.faults):raise ValueError('Unknown faulty instrument')
        self.fault_memory={}
        self.sensors=tuple(sorted(sensors,key=lambda s:s.name));self.tick_s=float(tick_s);self.seed=seed;self.tick=-1
        self.channels={s.name:dict(analog=None,previous_truth=None,queue=[],latest=None) for s in self.sensors}
        self.rng={}
        for sensor in self.sensors:
            material=f'{seed}:{sensor.name}'.encode()
            entropy=int.from_bytes(hashlib.sha256(material).digest()[:16],'big')
            self.rng[sensor.name]=np.random.Generator(np.random.PCG64(entropy))

    def ingest_station(self,snapshot):
        """Trusted adapter: copy only declared quantities from a station snapshot."""
        if type(snapshot) is not dict or not finite(snapshot.get('time_s')):raise ValueError('Station measurement clock')
        if not 0<=snapshot['time_s']<=1000000*self.tick_s:raise ValueError('Station measurement time budget')
        clock=snapshot['time_s']/self.tick_s;tick=round(clock)
        if abs(clock-tick)>1e-8:raise ValueError('Station snapshot is off instrument clock')
        values={}
        try:
            for sensor in self.sensors:
                value=snapshot
                for key in CATALOG[sensor.quantity][0]:value=value[key]
                values[sensor.quantity]=value
        except (KeyError,TypeError):raise ValueError('Missing station truth quantity') from None
        return self.ingest(tick,values)

    def ingest(self,tick,values):
        if type(tick) is not int or tick!=self.tick+1 or tick>1000000:raise ValueError('Consecutive instrument tick required')
        quantities={s.quantity for s in self.sensors}
        if type(values) is not dict or set(values)!=quantities or any(not finite(v) or abs(v)>1e100 for v in values.values()):
            raise ValueError('Complete finite instrument truth required')
        trial=deepcopy(self)
        trial._apply(tick,values)
        self.__dict__.update(trial.__dict__)
        return self.operator_frame()

    def _apply(self,tick,values):
        dt=self.tick_s
        for sensor in self.sensors:
            channel=self.channels[sensor.name];truth=float(values[sensor.quantity])
            if tick==0 or sensor.lag_s==0:
                analog=truth
            else:
                x=dt/sensor.lag_s;one_minus=-math.expm1(-x)
                ramp=dt*(x/2-x*x/6+x**3/24-x**4/120) if x<1e-4 else dt-sensor.lag_s*one_minus
                slope=(truth-channel['previous_truth'])/dt
                analog=channel['analog']+one_minus*(channel['previous_truth']-channel['analog'])+slope*ramp
            if not finite(analog):raise ValueError('Instrument response overflow')
            channel['analog']=float(analog);channel['previous_truth']=truth
            if tick%sensor.period_ticks==0:
                noise=float(self.rng[sensor.name].normal(0.,sensor.sigma)) if sensor.sigma else 0.
                value=analog+noise
                if not finite(value):raise ValueError('Instrument reading overflow')
                item=self._fault_sample(sensor,tick,value)
                if item is not None:channel['queue'].append(item)
            while channel['queue'] and channel['queue'][0]['available_tick']<=tick:
                channel['latest']=channel['queue'].pop(0)
        self.tick=tick

    def _active_fault(self,name,tick):
        return next((f for f in self.faults if f.target==name and f.active(tick)),None)

    def _fault_sample(self,sensor,tick,value):
        fault=self._active_fault(sensor.name,tick)
        if fault is not None:
            if fault.kind=='dropout':return None
            if fault.kind=='bias':value+=fault.value
            elif fault.kind=='drift':value+=fault.value*(tick-fault.start_tick)*self.tick_s
            elif fault.kind=='stuck':value=self.fault_memory.setdefault(fault.name,float(value))
        if not finite(value):raise ValueError('Faulted instrument reading overflow')
        item=dict(sampled_tick=tick,available_tick=tick+sensor.delay_ticks,value=float(value))
        if fault is not None and fault.kind=='failure':item['unavailable']=True
        return item

    def operator_frame(self):
        readings={}
        for sensor in self.sensors:
            latest=self.channels[sensor.name]['latest'];unit=CATALOG[sensor.quantity][1]
            if latest is None:
                readings[sensor.name]=dict(value=None,unit=unit,sampled_at_s=None,available_at_s=None,age_s=None,quality=['initializing'])
                continue
            age=self.tick-latest['sampled_tick'];value=latest['value'];quality=[]
            if latest.get('unavailable',False):quality.append('unavailable')
            if age>sensor.stale_after_ticks:quality.append('stale')
            if ((sensor.measurement_min is not None and value<sensor.measurement_min)
                    or (sensor.measurement_max is not None and value>sensor.measurement_max)):quality.append('out_of_range')
            readings[sensor.name]=dict(value=None if latest.get('unavailable',False) else value,unit=unit,sampled_at_s=latest['sampled_tick']*self.tick_s,
                available_at_s=latest['available_tick']*self.tick_s,age_s=age*self.tick_s,quality=quality or ['good'])
        return dict(time_s=None if self.tick<0 else self.tick*self.tick_s,measurements=readings)

    def checkpoint(self):
        return seal(dict(format='station-instruments/v1',source=signature(),tick_s=self.tick_s,seed=self.seed,
            sensors=[asdict(s) for s in self.sensors],faults=[asdict(f) for f in self.faults],fault_memory=deepcopy(self.fault_memory),tick=self.tick,channels=deepcopy(self.channels),
            rng={name:deepcopy(rng.bit_generator.state) for name,rng in self.rng.items()}))

    @classmethod
    def restore(cls,saved):
        p=unseal(deepcopy(saved))
        expected={'format','source','tick_s','seed','sensors','tick','channels','rng','faults','fault_memory'}
        if type(p) is not dict or set(p)!=expected or p['format']!='station-instruments/v1' or p['source']!=signature():raise ValueError('Instrument source/format')
        if type(p['sensors']) is not list or not 1<=len(p['sensors'])<=32:raise ValueError('Instrument restore budget')
        try:obj=cls([Sensor(**s) for s in p['sensors']],tick_s=p['tick_s'],seed=p['seed'],faults=[Fault(**f) for f in p['faults']])
        except (TypeError,KeyError):raise ValueError('Instrument restore configuration') from None
        tick=p['tick'];names={s.name for s in obj.sensors}
        if type(tick) is not int or not -1<=tick<=1000000 or type(p['channels']) is not dict or type(p['rng']) is not dict or set(p['channels'])!=names or set(p['rng'])!=names:raise ValueError('Instrument state/clock')
        for sensor in obj.sensors:
            state=p['channels'][sensor.name]
            if type(state) is not dict or set(state)!={'analog','previous_truth','queue','latest'}:raise ValueError('Instrument state fields')
            if tick<0:
                if state!=obj.channels[sensor.name]:raise ValueError('Uninitialized instrument state')
            else:
                if any(not finite(state[k]) for k in ('analog','previous_truth')):raise ValueError('Instrument analog state')
                first=max(0,tick-sensor.delay_ticks+1)
                first=((first+sensor.period_ticks-1)//sensor.period_ticks)*sensor.period_ticks
                queued=[t for t in range(first,tick+1,sensor.period_ticks) if not obj._is_dropout(sensor.name,t)]
                queue=state['queue']
                if type(queue) is not list or len(queue)!=len(queued):raise ValueError('Instrument delay queue budget')
                for item,when in zip(queue,queued):obj._validate_fault_reading(item,when,sensor)
                last=((tick-sensor.delay_ticks)//sensor.period_ticks)*sensor.period_ticks if tick>=sensor.delay_ticks else None
                while last is not None and last>=0 and obj._is_dropout(sensor.name,last):
                    fault=obj._active_fault(sensor.name,last)
                    last=((fault.start_tick-1)//sensor.period_ticks)*sensor.period_ticks
                if last is not None and last<0:last=None
                if last is None:
                    if state['latest'] is not None:raise ValueError('Future instrument delivery')
                else:obj._validate_fault_reading(state['latest'],last,sensor)
            rng_state=p['rng'][sensor.name]
            if type(rng_state) is not dict or rng_state.get('bit_generator')!='PCG64':raise ValueError('Instrument noise generator')
            try:obj.rng[sensor.name].bit_generator.state=deepcopy(rng_state)
            except (ValueError,TypeError,KeyError,OverflowError):raise ValueError('Instrument noise state') from None
        periods={s.name:s.period_ticks for s in obj.sensors}
        expected_memory=set()
        for fault in obj.faults:
            period=periods[fault.target]
            first=((fault.start_tick+period-1)//period)*period
            if fault.kind=='stuck' and first<=tick and (fault.end_tick is None or first<fault.end_tick):
                expected_memory.add(fault.name)
        if type(p['fault_memory']) is not dict or set(p['fault_memory'])!=expected_memory or any(not finite(v) for v in p['fault_memory'].values()):raise ValueError('Fault instrument memory')
        obj.fault_memory=deepcopy(p['fault_memory'])
        obj.tick=tick;obj.channels=deepcopy(p['channels'])
        if obj.checkpoint()!=saved:raise ValueError('Noncanonical instrument checkpoint')
        return obj

    def _is_dropout(self,name,tick):
        fault=self._active_fault(name,tick)
        return fault is not None and fault.kind=='dropout'

    def _validate_fault_reading(self,item,when,sensor):
        fault=self._active_fault(sensor.name,when)
        failure=fault is not None and fault.kind=='failure'
        if type(item) is not dict or item.get('unavailable',False)!=failure or ('unavailable' in item and item['unavailable'] is not True):raise ValueError('Instrument failure quality')
        self._validate_reading({k:v for k,v in item.items() if k!='unavailable'},when,sensor.delay_ticks)

    @staticmethod
    def _validate_reading(item,when,delay):
        if (type(item) is not dict or set(item)!={'sampled_tick','available_tick','value'}
                or type(item['sampled_tick']) is not int or type(item['available_tick']) is not int
                or item['sampled_tick']!=when or item['available_tick']!=when+delay or not finite(item['value'])):
            raise ValueError('Instrument delivery state')
