"""Synthetic transient diffusion/thermal-cell coupling on different meshes.

Six *neutron precursor* families are owned locally. All fission heat is prompt
in this test problem: no decay-heat inventory, fluids, phase changes or PWR
integration is claimed. The temperature coefficient is an explicit synthetic
linear feedback, not evaluated cross-section data.
"""
from copy import deepcopy
from dataclasses import dataclass,asdict
import hashlib
import math
from pathlib import Path

import numpy as np
import scipy
from atomic_model_station.core import CoreConfig,DiffusionCore
from physics.neutronics.point_kinetics import PointKinetics
from atomic_model_multiphysics.provenance import seal,unseal
from atomic_model_spatial.workflow import identity as spatial_identity
from .conservative import ConservativeMap,vector


def _finite(value):
    try:return type(value) in (int,float) and math.isfinite(value)
    except OverflowError:return False


def identity():
    from . import conservative
    from atomic_model_station import core
    from physics.neutronics import point_kinetics
    return dict(files={name:hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        for name,module in [('exchange',conservative),('core',core),('point_kinetics',point_kinetics)]},
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        spatial=spatial_identity(),numpy=np.__version__,scipy=scipy.__version__)


@dataclass(frozen=True)
class ChannelConfig:
    neutron_cells:int=4
    length_m:float=4.
    channel_faces:tuple=(0.,.6,1.7,4.)
    power_w:float=1e6
    capacity_j_k:float=1e5
    conductance_w_k:float=1e4
    sink_k:float=560.
    alpha_per_k:float=-1e-5
    reactivity:tuple=(1e-4,1e-4,1e-4,1e-4)

    def __post_init__(self):
        if type(self.neutron_cells) is not int or not 2<=self.neutron_cells<=8:
            raise ValueError('Two to eight synthetic neutron cells required')
        if type(self.channel_faces) is not tuple or type(self.reactivity) is not tuple:
            raise ValueError('Immutable geometry and local reactivity required')
        for key,low,high in [('length_m',.1,20.),('power_w',1.,1e8),('capacity_j_k',1e3,1e9),
                ('conductance_w_k',10.,1e7),('sink_k',280.,800.),('alpha_per_k',-1e-4,0.)]:
            v=getattr(self,key)
            if not _finite(v) or not low<=v<=high:
                raise ValueError('Synthetic channel parameter outside bounds: '+key)
        if self.sink_k+self.power_w/self.conductance_w_k>1500:
            raise ValueError('Initial thermal state exceeds test-problem domain')
        if len(self.reactivity)!=self.neutron_cells or any(not _finite(v) or abs(v)>1e-3 for v in self.reactivity):
            raise ValueError('Bounded local synthetic reactivity required')
        ConservativeMap(np.linspace(0.,self.length_m,self.neutron_cells+1),self.channel_faces)


class CoreChannels:
    def __init__(self,config=None):
        self.config=ChannelConfig() if config is None else config
        if type(self.config) is not ChannelConfig:raise ValueError('ChannelConfig required')
        c=self.config
        self.kinetics=PointKinetics.thermal_u235(power_nominal_w=c.power_w)
        self.core=DiffusionCore(CoreConfig(kind='diffusion',cells=c.neutron_cells,length_m=c.length_m),self.kinetics)
        self.exchange=ConservativeMap(self.core.model.faces,c.channel_faces)
        self.n=c.neutron_cells;self.m=len(c.channel_faces)-1;self.k=7*self.n;self.size=self.k+self.m+2
        self.capacity=c.capacity_j_k*self.exchange.target_widths/c.length_m
        self.ua=c.conductance_w_k*self.exchange.target_widths/c.length_m
        self.reference_k=c.sink_k+c.power_w/c.conductance_w_k
        self.scale=np.r_[np.ones(self.k+self.m),c.power_w,c.power_w]

    def initial(self):
        return np.r_[self.core.lift(self.kinetics.initial_state()),np.full(self.m,self.reference_k),0.,0.]

    def admit(self,y):
        a=vector(y,self.size)
        if np.any(a[:self.k]<0) or not np.any(a[:self.n]>0):raise ValueError('Nonnegative owned neutron/precursor populations required')
        if np.any(a[self.k:self.k+self.m]<280) or np.any(a[self.k:self.k+self.m]>1500):raise ValueError('Thermal test-domain bounds exceeded')
        if np.any(a[-2:]<0):raise ValueError('Nonnegative heat integral counters required')
        return a

    def terms(self,y):
        c=self.config;t=y[self.k:self.k+self.m]
        feedback=self.exchange.feedback_temperature(t)
        rho=np.asarray(c.reactivity)+c.alpha_per_k*(feedback-self.reference_k)
        if np.any(abs(rho)>1e-3):raise ValueError('Synthetic local feedback outside declared reactivity domain')
        powers=self.core.fission_by_cell(y[:self.k]);mapped=self.exchange.powers(powers)
        removed=self.ua*(t-c.sink_k)
        if np.any(removed<0):raise ValueError('Heating/cooling-to-sink test branch only')
        return rho,powers,mapped,removed

    def rhs(self,time,y):
        y=self.admit(y);rho,power,heat,removed=self.terms(y)
        neutron=self.core.operator@y[:self.k]
        neutron[:self.n]+=rho*y[:self.n]/self.kinetics.generation_time_s
        return np.r_[neutron,(heat-removed)/self.capacity,float(heat.sum()),float(removed.sum())]

    def jacobian(self,time,y):
        y=self.admit(y);rho,_,_,_=self.terms(y);c=self.config
        jac=np.zeros((self.size,self.size));jac[:self.k,:self.k]=self.core.operator
        jac[np.arange(self.n),np.arange(self.n)]+=rho/self.kinetics.generation_time_s
        jac[:self.n,self.k:self.k+self.m]=(y[:self.n,None]*c.alpha_per_k/self.kinetics.generation_time_s)*self.exchange.feedback_matrix
        heating=self.exchange.power_matrix*(c.power_w*self.core.weights)[None,:]
        jac[self.k:self.k+self.m,:self.n]=heating/self.capacity[:,None]
        jac[self.k:self.k+self.m,self.k:self.k+self.m]=-np.diag(self.ua/self.capacity)
        jac[-2,:self.n]=heating.sum(axis=0);jac[-1,self.k:self.k+self.m]=self.ua
        return jac

    def audit(self,y):
        y=self.admit(y);rho,q,_,_=self.terms(y);rate=self.rhs(0.,y)
        weights=np.r_[self.core.model.width/self.core.speed,np.tile(self.core.model.width,6)]
        expected=float(np.sum(self.core.config.nu_fission_per_m*rho*y[:self.n]*self.core.model.width))
        actual=float(weights@rate[:self.k])
        scale=max(float(abs(weights)@abs(rate[:self.k])),abs(expected),1.)
        stored=float(self.capacity@(y[self.k:self.k+self.m]-self.reference_k))
        energy=stored-y[-2]+y[-1]
        exchange=self.exchange.audit(q,y[self.k:self.k+self.m])
        return dict(**exchange,precursor_inventory_relative_defect=abs(actual-expected)/scale,
            energy_residual_j=float(energy),energy_relative_defect=float(abs(energy)/max(abs(y[-2])+abs(y[-1]),1.)),
            fission_total_w=float(q.sum()),thermal_storage_change_j=stored,
            generated_heat_j=float(y[-2]),removed_heat_j=float(y[-1]),
            local_reactivity=rho.tolist(),local_precursors_owned=True,decay_heat_supported=False,
            station_connected=False,experimental_validation=False)


class ChannelSession:
    """Transactional backward Euler with analytic Newton matrix; no clipping."""
    def __init__(self,config=None):
        self.model=CoreChannels(config);self.state=self.model.initial();self.time=0.;self.history=[]

    def advance(self,duration_s,*,step_s=.01):
        if (not _finite(duration_s) or not 0<duration_s<=1 or
            not _finite(step_s) or not 1e-5<=step_s<=.05 or
            self.time+duration_s>10 or len(self.history)>=256):raise ValueError('Bounded native interval/step/history required')
        trial=deepcopy(self);count=math.ceil(duration_s/step_s)
        if count>10000:raise ValueError('Native step work budget exceeded')
        dt=duration_s/count;iterations=0
        for _ in range(count):
            old=trial.state.copy();y=old.copy();epoch=trial.time+dt
            for iteration in range(16):
                rate=trial.model.rhs(epoch,y);residual=y-old-dt*rate
                norm=float(np.max(abs(residual)/trial.model.scale))
                if norm<=1e-10:break
                jac=np.eye(trial.model.size)-dt*trial.model.jacobian(epoch,y)
                delta=np.linalg.solve(jac,-residual);accepted=False
                for damping in (1.,.5,.25,.125,.0625,.03125,.015625):
                    candidate=y+damping*delta
                    try:
                        nr=float(np.max(abs(candidate-old-dt*trial.model.rhs(epoch,candidate))/trial.model.scale))
                    except (ValueError,RuntimeError):continue
                    if nr<norm:y=candidate;accepted=True;break
                if not accepted:raise ValueError('No admissible Newton step; caller unchanged')
                iterations+=1
            else:raise ValueError('Coupled Newton iteration budget exceeded; caller unchanged')
            audit=trial.model.audit(y)
            if max(audit['power_relative_defect'],audit['pairing_relative_defect'],audit['precursor_inventory_relative_defect'])>1e-10 or audit['energy_relative_defect']>1e-8:
                raise ValueError('Coupled conservation gate failed; caller unchanged')
            trial.state=y;trial.time=epoch
        trial.history.append(dict(duration_s=float(duration_s),step_s=float(step_s)))
        self.__dict__.update(trial.__dict__)
        return dict(time_s=self.time,state=self.state.tolist(),audit=self.model.audit(self.state),
            accepted_steps=count,newton_iterations=iterations,projection_used=False)

    def checkpoint(self):
        config=asdict(self.model.config)
        for key in ('channel_faces','reactivity'):config[key]=list(config[key])
        return seal(dict(format='synthetic-core-channels/v1',source=identity(),config=config,
            time_s=self.time,state=self.state.tolist(),history=deepcopy(self.history)))

    @classmethod
    def restore(cls,saved):
        data=unseal(deepcopy(saved))
        if (type(data) is not dict or set(data)!={'format','source','config','time_s','state','history'} or
                data.get('format')!='synthetic-core-channels/v1' or data.get('source')!=identity()):raise ValueError('Coupled checkpoint source/format mismatch')
        if type(data.get('history')) is not list or len(data['history'])>256:raise ValueError('Coupled checkpoint history budget')
        if type(data['config']) is not dict or set(data['config'])!=set(ChannelConfig.__dataclass_fields__):raise ValueError('Coupled checkpoint configuration fields')
        if any(type(data['config'][key]) not in (list,tuple) for key in ('reactivity','channel_faces')):
            raise ValueError('Coupled checkpoint requires geometry and reactivity vectors')
        spec=dict(data['config']);spec['reactivity']=tuple(spec['reactivity']);spec['channel_faces']=tuple(spec['channel_faces'])
        obj=cls(ChannelConfig(**spec))
        for row in data['history']:
            if type(row) is not dict or set(row)!={'duration_s','step_s'}:raise ValueError('Coupled checkpoint history record')
            obj.advance(**row)
        if obj.checkpoint()!=saved:raise ValueError('Coupled checkpoint replay differs')
        return obj
