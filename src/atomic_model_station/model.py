"""R0 thermal/mechanical assembly: no reactor kinetics, fluids or steam EOS."""
from dataclasses import asdict,dataclass
import hashlib
import math
from pathlib import Path

import numpy as np
import scipy
import components.heat_exchanger as wall_module
import core.residual as residual_module
from core.residual import ResidualBlock

STATE_NAMES=('primary_temperature_k','wall_temperature_k','secondary_temperature_k','cold_temperature_k','rotor_energy_j')


def signature():
    return dict(files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob('*.py'))},
                residual_sha256=hashlib.sha256(Path(residual_module.__file__).read_bytes()).hexdigest(),
                heat_wall_sha256=hashlib.sha256(Path(wall_module.__file__).read_bytes()).hexdigest(),
                numpy=np.__version__,scipy=scipy.__version__)


@dataclass(frozen=True)
class StationConfig:
    primary_capacity_j_k:float=1e6
    wall_capacity_j_k:float=2e5
    secondary_capacity_j_k:float=8e5
    cold_capacity_j_k:float=1e6
    primary_ua_w_k:float=30000.
    secondary_ua_w_k:float=30000.
    engine_ua_w_k:float=4000.
    cooling_ua_w_k:float=40000.
    ambient_temperature_k:float=300.
    engine_efficiency:float=.25
    generator_efficiency:float=.96
    rotor_inertia_kg_m2:float=10000.
    generator_drag_nm_s_rad:float=20.
    bearing_drag_nm_s_rad:float=1.
    nominal_source_power_w:float=1e6
    def __post_init__(self):
        for key,v in asdict(self).items():
            if type(v) not in (int,float) or not math.isfinite(v):raise ValueError('Finite config:'+key)
            lo,hi=(1e3,1e9) if 'capacity' in key else (1.,1e7)
            if 'efficiency' in key:lo,hi=.01,.99
            if key=='ambient_temperature_k':lo,hi=280.,330.
            if key=='nominal_source_power_w':lo,hi=1e4,2e6
            if not lo<=v<=hi:raise ValueError('Config domain:'+key)

    def registry(self):
        units={k:('J/K' if 'capacity' in k else 'W/K' if '_ua_' in k else '1' if 'efficiency' in k
                  else 'K' if 'temperature' in k else 'kg*m^2' if 'inertia' in k
                  else 'N*m*s/rad' if 'drag' in k else 'W') for k in asdict(self)}
        return {k:dict(value=v,unit=units[k],source='Explicit synthetic R0 verification parameter',family='illustrative') for k,v in asdict(self).items()}


def passport():
    return dict(model_id='npp-r0-thermal-mechanical',version='0.1.0',level='R0 prototype',
        scope='Four thermal inventories, prescribed heat source, ideal heat engine, rotor and active electrical load',
        state_names=list(STATE_NAMES),validated=False,plant_referenced=False,
        unsupported=['Reactor kinetics','Fluid mass and momentum','Steam cycle and phase changes','Full electrical network',
                     'Continuous-time domain guarantees','Cold start','Real plant operating limits'])


class StationResidual(ResidualBlock):
    size=5
    def __init__(self,config):
        if type(config) is not StationConfig:raise ValueError('StationConfig required')
        self.config=config
    def operators(self,power,connected):
        if type(power) not in (int,float) or not math.isfinite(power) or not 0<=power<=2e6 or type(connected) is not bool:
            raise ValueError('Declared source/load command domain')
        c=self.config;caps=np.array([c.primary_capacity_j_k,c.wall_capacity_j_k,c.secondary_capacity_j_k,c.cold_capacity_j_k])
        a=np.zeros((5,5));b=np.zeros(5)
        def transfer(i,j,u):
            a[i,i]-=u/caps[i];a[i,j]+=u/caps[i];a[j,i]+=u/caps[j];a[j,j]-=u/caps[j]
        transfer(0,1,c.primary_ua_w_k);transfer(1,2,c.secondary_ua_w_k)
        u=c.engine_ua_w_k;eta=c.engine_efficiency
        a[2,2]-=u/caps[2];a[2,3]+=u/caps[2]
        a[3,2]+=(1-eta)*u/caps[3];a[3,3]-=(1-eta)*u/caps[3]
        a[4,2]+=eta*u;a[4,3]-=eta*u
        g=c.generator_drag_nm_s_rad if connected else 0.;friction=c.bearing_drag_nm_s_rad
        a[4,4]=-2*(g+friction)/c.rotor_inertia_kg_m2
        a[3,4]=2*(friction+(1-c.generator_efficiency)*g)/c.rotor_inertia_kg_m2/caps[3]
        a[3,3]-=c.cooling_ua_w_k/caps[3]
        b[3]=c.cooling_ua_w_k*c.ambient_temperature_k/caps[3];b[0]=power/caps[0]
        return a,b
    def residual(self,t,y,ydot,u):
        a,b=self.operators(u['source_power_w'],u['generator_connected'])
        y=np.asarray(y);ydot=np.asarray(ydot)
        if y.shape!=(5,) or ydot.shape!=(5,) or not np.isfinite(y).all() or not np.isfinite(ydot).all():raise ValueError('Finite state/rate5 required')
        return ydot-a@y-b
    def steady(self):
        c=self.config;p=c.nominal_source_power_w
        k=c.rotor_inertia_kg_m2*c.engine_efficiency*p/(2*(c.generator_drag_nm_s_rad+c.bearing_drag_nm_s_rad))
        export=c.generator_efficiency*2*c.generator_drag_nm_s_rad*k/c.rotor_inertia_kg_m2
        cold=c.ambient_temperature_k+(p-export)/c.cooling_ua_w_k
        secondary=cold+p/c.engine_ua_w_k;wall=secondary+p/c.secondary_ua_w_k;primary=wall+p/c.primary_ua_w_k
        y=np.array([primary,wall,secondary,cold,k]);self.validate(y)
        a,_=self.operators(p,True)
        if np.linalg.matrix_rank(a)!=5:raise ValueError('Rank-deficient stationary assembly')
        scales=np.array([max(p/cap,1.) for cap in (c.primary_capacity_j_k,c.wall_capacity_j_k,c.secondary_capacity_j_k,c.cold_capacity_j_k)]+[p])
        residual=float(np.max(abs(self.residual(0.,y,np.zeros(5),dict(source_power_w=p,generator_connected=True)))/scales))
        if residual>1e-8:raise ValueError('Inconsistent initial state')
        return y,residual
    def validate(self,y):
        c=self.config
        if not np.isfinite(y).all() or np.any(y[:4]<280) or np.any(y[:4]>900) or y[4]<=0:raise ValueError('Station temperature/energy domain')
        omega=math.sqrt(2*y[4]/c.rotor_inertia_kg_m2)
        if not 1<=omega<=400:raise ValueError('Rotor operating-speed domain')
        if y[2]<y[3] or c.engine_efficiency>1-y[3]/y[2]+1e-12:raise ValueError('Heat-engine temperature/efficiency domain')
