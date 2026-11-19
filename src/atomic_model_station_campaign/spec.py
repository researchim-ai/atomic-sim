"""Frozen, bounded station episodes and local worker resource contracts."""
from copy import deepcopy
from dataclasses import asdict, dataclass
import math

from atomic_model_station_faults.spec import Fault, canonical, finite


@dataclass(frozen=True)
class Limits:
    workers: int = 1
    wall_s: int = 900
    cpu_s: int = 900
    memory_mb: int = 8192
    output_mb: int = 64

    def __post_init__(self):
        for name, low, high in [('workers',1,4),('wall_s',1,14400),('cpu_s',1,14400),
                                ('memory_mb',512,16384),('output_mb',1,256)]:
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError('Campaign resource domain: '+name)


@dataclass(frozen=True)
class Episode:
    name: str
    ticks: int
    tick_s: float = 1.
    seed: int = 0
    automatic: bool = True
    faults: tuple = ()
    commands: tuple = ()
    checkpoint_ticks: tuple = ()
    record_every: int = 1
    expected_rejection: bool = False
    react_on_alarm: bool = False
    max_step_s: float = 1.
    rtol: float = 1e-7
    tracking_tau_s: float = 10.
    equilibrium_start: bool = False
    inventory_deadband_m: float = 0.
    film_model: str = "dittus_boelter"

    def __post_init__(self):
        from atomic_model_station.film_correlations import validate_model
        validate_model(self.film_model)
        if type(self.name) is not str or not self.name.isidentifier() or len(self.name)>64:
            raise ValueError('Episode identifier')
        if type(self.ticks) is not int or not 1<=self.ticks<=8192:
            raise ValueError('Episode step budget')
        if not finite(self.tick_s) or not .01<=self.tick_s<=60 or self.ticks*self.tick_s>7200:
            raise ValueError('Episode physical horizon')
        if type(self.seed) is not int or not 0<=self.seed<2**63:
            raise ValueError('Episode seed')
        if any(type(v) is not bool for v in (self.automatic,self.expected_rejection,self.react_on_alarm,self.equilibrium_start)):
            raise ValueError('Episode boolean options')
        if not finite(self.inventory_deadband_m) or not 0<=self.inventory_deadband_m<=.001:
            raise ValueError('Episode inventory neutral band')
        if not finite(self.max_step_s) or not .01<=self.max_step_s<=1:
            raise ValueError('Episode solver step')
        if not finite(self.rtol) or not 1e-10<=self.rtol<=1e-5:
            raise ValueError('Episode solver tolerance')
        if not finite(self.tracking_tau_s) or not self.tick_s<=self.tracking_tau_s<=1e6:
            raise ValueError('Episode tracking clock')
        if type(self.record_every) is not int or not 1<=self.record_every<=8192 or math.ceil(self.ticks/self.record_every)>1024:
            raise ValueError('Episode recording budget')
        faults=canonical(self.faults)
        if any(f.start_tick>self.ticks for f in faults):
            raise ValueError('Fault outside episode')
        object.__setattr__(self,'faults',faults)
        if type(self.checkpoint_ticks) not in (list,tuple) or len(self.checkpoint_ticks)>16:
            raise ValueError('Episode checkpoint budget')
        checkpoints=tuple(sorted(self.checkpoint_ticks))
        if any(type(t) is not int or not 0<=t<=self.ticks for t in checkpoints) or len(set(checkpoints))!=len(checkpoints):
            raise ValueError('Episode checkpoint epochs')
        object.__setattr__(self,'checkpoint_ticks',checkpoints)
        if type(self.commands) not in (list,tuple) or len(self.commands)>8192:
            raise ValueError('Episode command budget')
        commands=[];seen=set()
        for item in self.commands:
            if type(item) is not dict or not set(item)<= {'tick','operator','modes','targets','reset'} or 'tick' not in item:
                raise ValueError('Episode command fields')
            tick=item['tick']
            if type(tick) is not int or not 0<=tick<self.ticks or tick in seen:
                raise ValueError('Unique in-horizon command epochs required')
            seen.add(tick)
            for name in ('operator','modes','targets'):
                values=item.get(name)
                if values is not None and (type(values) is not dict or len(values)>64
                    or any(type(k) is not str or not k.isidentifier() for k in values)
                    or any(type(v) not in (int,float,bool,str) or (type(v) in (int,float) and not finite(v)) for v in values.values())):
                    raise ValueError('Flat finite episode command')
            if 'reset' in item and type(item['reset']) is not bool:
                raise ValueError('Episode reset request')
            commands.append(deepcopy(item))
        object.__setattr__(self,'commands',tuple(sorted(commands,key=lambda c:c['tick'])))

    def payload(self):
        result=asdict(self)
        result['faults']=[asdict(f) for f in self.faults]
        result['commands']=[deepcopy(c) for c in self.commands]
        result['checkpoint_ticks']=list(self.checkpoint_ticks)
        return result

    @classmethod
    def from_payload(cls,payload):
        if type(payload) is not dict:raise ValueError('Episode object required')
        args=deepcopy(payload)
        args['faults']=tuple(Fault(**f) for f in args.get('faults',[]))
        return cls(**args)
