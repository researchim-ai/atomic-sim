"""Transactional preparation/hold lifecycle; no resetting time or physical stocks."""
from collections import deque
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
import math

from atomic_model_multiphysics.provenance import seal, unseal, sha
from atomic_model_station_faults import FaultSession
from atomic_model_station_faults.model import signature as fault_signature
from atomic_model_station_readiness import ReadinessConfig, schema, capture, assess
from atomic_model_station_readiness import model as readiness_model
from atomic_model_station_inventory import capture as inventory_capture, summarize
from atomic_model_station_inventory import model as inventory_model


def signature():
    return dict(faults=fault_signature(), readiness=sha(readiness_model.__file__),
                inventory=sha(inventory_model.__file__),
                files={p.name:sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))})


def finite(x):
    return type(x) in (int, float) and math.isfinite(x)


def ticks(seconds, tick):
    n=round(seconds/tick)
    if n<1 or abs(n*tick-seconds)>1e-8:
        raise ValueError('Hot lifecycle requires whole positive control ticks')
    return n


@dataclass(frozen=True)
class HotConfig:
    maximum_warmup_s:float=3600.
    hold_s:float=3600.
    tick_s:float=5.
    inventory_deadband_m:float=.0002
    max_step_s:float=1.
    rtol:float=1e-7
    power_target_w:float=3.15e9
    equilibrium_start:bool=False
    film_model:str="dittus_boelter"
    readiness:ReadinessConfig=field(default_factory=ReadinessConfig)

    def __post_init__(self):
        from atomic_model_station.film_correlations import validate_model
        validate_model(self.film_model)
        if type(self.equilibrium_start) is not bool:raise ValueError('Boolean equilibrium initialization')
        if self.equilibrium_start and self.power_target_w!=3.15e9:
            raise ValueError('Hot equilibrium reference currently requires 3.15 GW')
        if any(not finite(getattr(self,k)) for k in ('maximum_warmup_s','hold_s','tick_s',
            'inventory_deadband_m','max_step_s','rtol','power_target_w')):
            raise ValueError('Finite hot lifecycle parameters required')
        if not .01<=self.tick_s<=10 or not .01<=self.max_step_s<=1 or not 1e-10<=self.rtol<=1e-5:
            raise ValueError('Hot lifecycle numerical domain')
        if not 0<=self.inventory_deadband_m<=.001 or not 1e6<=self.power_target_w<=3.6e9:
            raise ValueError('Hot lifecycle reference domain')
        if type(self.readiness) is not ReadinessConfig or self.readiness.maximum_gap_s<self.tick_s:
            raise ValueError('Compatible readiness sampling required')
        if self.readiness.window_s<2*self.tick_s or self.maximum_warmup_s<self.readiness.window_s:
            raise ValueError('Warmup must contain a full readiness window')
        ticks(self.readiness.window_s,self.tick_s)
        total=ticks(self.maximum_warmup_s,self.tick_s)+ticks(self.hold_s,self.tick_s)
        if self.maximum_warmup_s+self.hold_s>7200 or total>8192:
            raise ValueError('Preparation and hold exceed physical or journal budget')

    @classmethod
    def from_payload(cls,payload):
        if type(payload) is not dict or set(payload)!=set(cls.__dataclass_fields__):
            raise ValueError('Hot configuration fields')
        p=deepcopy(payload)
        if type(p['readiness']) is not dict or set(p['readiness'])!=set(ReadinessConfig.__dataclass_fields__):
            raise ValueError('Hot readiness configuration fields')
        p['readiness']=ReadinessConfig(**p['readiness'])
        return cls(**p)


class HotSession:
    def __init__(self,config=None):
        self.config=config if config is not None else HotConfig()
        if type(self.config) is not HotConfig:raise ValueError('HotConfig required')
        c=self.config;self.source=signature()
        self.owner=FaultSession(tick_s=c.tick_s,history_budget_steps=8192,max_step_s=c.max_step_s,
                               rtol=c.rtol,inventory_deadband_m=c.inventory_deadband_m,equilibrium_start=c.equilibrium_start,film_model=c.film_model)
        self.layout=schema(self.owner);self.phase='warmup';self.failure=None
        self.attempts=0;self.warmup_end_s=None
        sample=capture(self.owner);stock=inventory_capture(self.owner)
        self.window=deque([sample],maxlen=ticks(c.readiness.window_s,c.tick_s)+1)
        self.last_readiness=assess(list(self.window),self.layout,c.readiness)
        self.records=[dict(phase_before='initial',sample=sample,inventory=stock,result=None)]

    @property
    def finished(self):return self.phase not in ('warmup','hold')

    def step(self):
        if self.finished:raise ValueError('Hot lifecycle already terminal')
        if self.source!=signature():raise ValueError('Hot lifecycle source changed')
        c=self.config;trial=deepcopy(self.owner);before=self.phase
        try:
            result=trial.step(modes={k:'auto' for k in trial.control.loops} if self.attempts==0 else None,
                              targets={'rod_target':c.power_target_w} if self.attempts==0 else None)
            sample=capture(trial);stock=inventory_capture(trial)
            window=[*self.window,sample][-self.window.maxlen:]
            verdict=assess(window,self.layout,c.readiness)
        except ValueError as error:
            # The accepted owner/prefix stays intact when the next tick rejects.
            self.attempts+=1;self.phase='failed';self.failure='tick_rejected: '+str(error)
            return self.report()
        phase=before;failure=None;warmup_end=self.warmup_end_s
        physical=(sample['energy_relative']<=1e-6 and sample['mass_relative']<=1e-10
                  and sample['boron_relative']<=1e-10
                  and stock['maximum_local_continuity_residual_kg_s']<=1e-9
                  and abs(stock['total_rate_kg_s'])<=1e-9)
        if sample['trip']:
            phase='failed';failure='protection_latched'
        elif not physical:
            phase='failed';failure='physical_audit'
        elif before=='warmup':
            if verdict['ready']:
                physical_left=trial.control.plant._maximum_time_s()-trial.control.plant.time
                journal_left=(trial.control.plant._maximum_history_steps()-len(trial.history))*c.tick_s
                if min(physical_left,journal_left)+1e-8<c.hold_s:
                    phase='failed';failure='insufficient_followup_budget'
                else:
                    phase='hold';warmup_end=sample['time_s']
            elif sample['time_s']>=c.maximum_warmup_s-1e-8:
                phase='unprepared';failure='warmup_exhausted'
        else:
            if not verdict['ready']:
                phase='failed';failure='hold_readiness_lost'
            elif sample['time_s']-warmup_end>=c.hold_s-1e-8:
                phase='complete'
        record=dict(phase_before=before,sample=sample,inventory=stock,result=result)
        self.owner=trial;self.attempts+=1;self.phase=phase;self.failure=failure
        self.warmup_end_s=warmup_end;self.window=deque(window,maxlen=self.window.maxlen)
        self.last_readiness=verdict;self.records.append(record)
        return self.report()

    def report(self):
        c=self.config;time=float(self.owner.control.plant.time)
        canonical=(c.readiness==ReadinessConfig(maximum_gap_s=c.tick_s) and c.power_target_w==3.15e9)
        return dict(finished=self.finished,phase=self.phase,failure=self.failure,attempts=self.attempts,
            accepted_steps=len(self.records)-1,time_s=time,warmup_end_s=self.warmup_end_s,
            hold_elapsed_s=0. if self.warmup_end_s is None else time-self.warmup_end_s,
            hold_passed=self.phase=='complete',canonical_criteria=canonical,
            equilibrium_start=c.equilibrium_start,
            canonical_hold_passed=self.phase=='complete' and canonical and c.hold_s>=3600,
            readiness=deepcopy(self.last_readiness),source_unchanged=self.source==signature(),
            time_reset=False,inventories_projected=False,pi_reset=False,
            stationary_accepted=False,r1_accepted=False,closed_tasks=[],
            scope='Single physical program; even a full hold still needs independent refinement/restart/installed qualification')

    def inventory_summary(self):
        return summarize([r['inventory'] for r in self.records],maximum_gap_s=self.config.tick_s) if len(self.records)>1 else None

    def checkpoint(self):
        return seal(dict(format='station-hot-lifecycle/v1',source=deepcopy(self.source),
            config=asdict(self.config),attempts=self.attempts,owner=deepcopy(self.owner.checkpoint()),
            records=deepcopy(self.records),report=self.report()))

    @classmethod
    def restore(cls,saved):
        p=unseal(deepcopy(saved))
        if type(p) is not dict or set(p)!={'format','source','config','attempts','owner','records','report'}:
            raise ValueError('Hot checkpoint fields')
        if p['format']!='station-hot-lifecycle/v1' or p['source']!=signature():
            raise ValueError('Hot checkpoint source/format')
        config=HotConfig.from_payload(p['config'])
        maximum=ticks(config.maximum_warmup_s,config.tick_s)+ticks(config.hold_s,config.tick_s)
        if type(p['attempts']) is not int or not 0<=p['attempts']<=maximum:
            raise ValueError('Hot checkpoint attempt budget')
        if type(p['records']) is not list or not 1<=len(p['records'])<=maximum+1 or len(p['records']) not in (p['attempts'],p['attempts']+1):
            raise ValueError('Hot checkpoint record budget')
        obj=cls(config)
        for _ in range(p['attempts']):obj.step()
        if obj.checkpoint()!=saved:raise ValueError('Hot lifecycle replay differs')
        return obj
