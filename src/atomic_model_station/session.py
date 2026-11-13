"""Sampled-domain exact linear stepping with ledger and replay restart."""
from copy import deepcopy
from dataclasses import asdict
import math

import numpy as np
from scipy.linalg import expm
from atomic_model_multiphysics.provenance import seal,unseal
from .model import StationConfig,StationResidual,passport,signature


class StationSession:
    def __init__(self,config=None):
        self.config=config if config is not None else StationConfig()
        self.model=StationResidual(self.config);self._state,self.initialization_residual=self.model.steady()
        self._time=0.;self._export=0.;self._cooling=0.;self._source=0.;self._history=[]
        self._power=self.config.nominal_source_power_w;self._connected=True
    def snapshot(self):
        c=self.config
        return dict(time_s=self._time,state=self._state.tolist(),source_power_w=self._power,generator_connected=self._connected,
            rotor_speed_rad_s=math.sqrt(2*self._state[4]/c.rotor_inertia_kg_m2),
            source_energy_j=self._source,electrical_export_j=self._export,cooling_rejection_j=self._cooling,
            initialization_residual=self.initialization_residual,passport=passport())
    def advance(self,duration_s,*,source_power_w=None,generator_connected=None):
        if type(duration_s) not in (int,float) or not math.isfinite(duration_s) or not 0<duration_s<=60:raise ValueError('Duration0..60s required')
        if len(self._history)>=256 or self._time+duration_s>3600:raise ValueError('Station history/horizon budget')
        power=self._power if source_power_w is None else source_power_w
        connected=self._connected if generator_connected is None else generator_connected
        a,b=self.model.operators(power,connected);c=self.config
        # Trial values are owned; no public state changes until all checks pass.
        origin=self._state.copy()
        matrix=np.zeros((8,8));matrix[:5,:5]=a;matrix[:5,7]=a@origin+b
        g=c.generator_drag_nm_s_rad if connected else 0.
        matrix[5,4]=c.generator_efficiency*2*g/c.rotor_inertia_kg_m2
        matrix[5,7]=matrix[5,4]*origin[4]
        matrix[6,3]=c.cooling_ua_w_k;matrix[6,7]=c.cooling_ua_w_k*(origin[3]-c.ambient_temperature_k)
        count=math.ceil(duration_s/.25);step=duration_s/count;transition=expm(matrix*step)
        trial=np.r_[np.zeros(5),0.,0.,1.]
        for _ in range(count):
            trial=transition@trial;self.model.validate(origin+trial[:5])
        weights=np.array([c.primary_capacity_j_k,c.wall_capacity_j_k,c.secondary_capacity_j_k,c.cold_capacity_j_k,1.])
        delta=math.fsum(float(v) for v in weights*trial[:5])
        source=power*duration_s;defect=math.fsum([delta,-source,float(trial[5]),float(trial[6])])
        scale=max(abs(source)+abs(trial[5])+abs(trial[6]),1.)
        if abs(defect)>1e-5+1e-10*scale:raise ValueError('Station global energy ledger failure')
        self._state=origin+trial[:5];self._time+=duration_s;self._source+=source
        self._export+=float(trial[5]);self._cooling+=float(trial[6]);self._power=power;self._connected=connected
        self._history.append(dict(duration_s=duration_s,source_power_w=power,generator_connected=connected))
        return dict(**self.snapshot(),step_energy_residual_j=defect,domain_check_spacing_s=step)
    def checkpoint(self):
        return seal(deepcopy(dict(config=asdict(self.config),source=signature(),history=self._history,snapshot=self.snapshot())))
    @classmethod
    def restore(cls,saved):
        p=unseal(deepcopy(saved))
        if p['source']!=signature() or type(p['history']) is not list or len(p['history'])>256:raise ValueError('Station source/history mismatch')
        s=cls(StationConfig(**p['config']))
        for row in p['history']:s.advance(**row)
        if s.checkpoint()!=saved:raise ValueError('Station replay differs')
        return s
