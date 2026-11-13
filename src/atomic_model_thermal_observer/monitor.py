"""One-step thermal hypothesis envelope, repeat detection and exact replay.

A residual rejects consistency with the chosen hypotheses, not a particular
sensor fault. Noise/model margins are assumptions, not confidence intervals.
"""
from dataclasses import dataclass,asdict
from copy import deepcopy
from itertools import product
import math
import hashlib
from pathlib import Path
import numpy as np
from atomic_model_multiphysics.provenance import seal,unseal
from atomic_model_hem_heated.model import canonical


def signature():return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob('*.py'))}


@dataclass(frozen=True)
class MonitorConfig:
    dt_s:float=.00025
    max_power_w:float=100000.
    capacity_min:float=100.
    capacity_max:float=500.
    conductance_min:float=1000.
    conductance_max:float=15000.
    ambient_ua_min:float=0.
    ambient_ua_max:float=100.
    noise_margin_k:float=.02
    model_margin_k:float=.003
    repeated_samples:int=3
    def __post_init__(self):
        for key,value in asdict(self).items():
            if key=='repeated_samples':continue
            if type(value) not in (int,float) or not math.isfinite(value):raise ValueError('Finite monitor parameter: '+key)
        if not 1e-6<=self.dt_s<=.001 or not 1<=self.max_power_w<=1e6:raise ValueError('Monitor time/power domain')
        if not 1<=self.capacity_min<=self.capacity_max<=1e7:raise ValueError('Capacity hypothesis domain')
        if not 0<self.conductance_min<=self.conductance_max<=1e5:raise ValueError('Conductance hypothesis domain')
        if not 0<=self.ambient_ua_min<=self.ambient_ua_max<=1e5:raise ValueError('Ambient hypothesis domain')
        if not 0<=self.noise_margin_k<=1 or not 0<=self.model_margin_k<=1:raise ValueError('Monitor margin domain')
        if type(self.repeated_samples) is not int or not 2<=self.repeated_samples<=32:raise ValueError('Repeat budget2..32')


def predict(wall,fluid,ambient,power,dt,capacity,conductance,ua):
    """Exact affine wall update for fixed fluid temperature and coefficients."""
    rate=(conductance+ua)/capacity
    equilibrium=(power+conductance*fluid+ua*ambient)/(conductance+ua)
    return wall+(equilibrium-wall)*(-math.expm1(-rate*dt))


class ThermalMonitor:
    def __init__(self,config=None):
        self.config=config or MonitorConfig()
        if type(self.config) is not MonitorConfig:raise ValueError('MonitorConfig required')
        self._history=[];self._previous=None;self._repeats=0;self.alarm=False;self.first_alarm=None;self.last=None

    def observe(self,observation):
        o=np.asarray(observation)
        if o.shape!=(6,) or o.dtype.kind not in 'if' or not np.isfinite(o).all():raise ValueError('Finite observation6 required')
        if len(self._history)>=256:raise ValueError('Monitor sample budget256')
        if not 0<=float(o[4])<=1 or not 0<=float(o[5])<=1:raise ValueError('Power/time fraction outside0..1')
        if not np.all((-200<=o[:4]) & (o[:4]<=600)):raise ValueError('Observation temperature research domain')
        cfg=self.config;index=len(self._history);lower=upper=gap=None
        repeats=self._repeats+1 if self._previous is not None and float(o[0])==self._previous[0] else 1
        reasons=[]
        if self._previous is not None:
            previous=self._previous
            temperatures=[440.+previous[0]-cfg.noise_margin_k,440.+previous[0]+cfg.noise_margin_k]
            fluids=[440.+min(previous[1],float(o[1]))-cfg.noise_margin_k,440.+max(previous[1],float(o[1]))+cfg.noise_margin_k]
            values=[predict(t,f,440.+float(o[3]),float(o[4])*cfg.max_power_w,cfg.dt_s,c,k,ua)
                    for t,f,c,k,ua in product(temperatures,fluids,[cfg.capacity_min,cfg.capacity_max],
                         [cfg.conductance_min,cfg.conductance_max],[cfg.ambient_ua_min,cfg.ambient_ua_max])]
            lower=min(values)-cfg.model_margin_k;upper=max(values)+cfg.model_margin_k
            measured=440.+float(o[0]);gap=max(0.,lower-(measured+cfg.noise_margin_k),(measured-cfg.noise_margin_k)-upper)
            if gap>0:reasons.append('measurement_model_inconsistency')
        if repeats>=cfg.repeated_samples:reasons.append('repeated_reading')
        if reasons and self.first_alarm is None:self.first_alarm=dict(index=index,time_s=index*cfg.dt_s,reasons=reasons.copy())
        self.alarm=self.alarm or bool(reasons)
        self.last=dict(index=index,alarm=self.alarm,reasons=reasons,prediction_lower_k=lower,prediction_upper_k=upper,
            inconsistency_gap_k=gap,repeated_samples=repeats,first_alarm=self.first_alarm,
            interpretation='Sampled parameter hypotheses and assumed margins; no unique fault diagnosis or statistical confidence')
        self._previous=list(map(float,o));self._history.append(self._previous.copy());self._repeats=repeats
        return deepcopy(self.last)

    def checkpoint(self):
        return seal(deepcopy(dict(config=asdict(self.config),source=signature(),history=[r.copy() for r in self._history],
            alarm=self.alarm,first_alarm=self.first_alarm,last=self.last)))

    @classmethod
    def restore(cls,saved):
        p=unseal(deepcopy(saved))
        if p['source']!=signature() or not isinstance(p['history'],list) or len(p['history'])>256:raise ValueError('Monitor source/budget')
        monitor=cls(MonitorConfig(**p['config']))
        for obs in p['history']:monitor.observe(obs)
        if canonical(monitor.checkpoint())!=canonical(saved):raise ValueError('Monitor replay differs')
        return monitor
