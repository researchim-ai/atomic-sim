"""Radial time-dependent reduction of Marensi et al.2021 Eqs.2.6/2.9.

Units: R,2Ub,R/(2Ub). Fixed area means u=1/2,Theta=1/2; u_wall=0,
Theta_wall=1. The uniform pressure adjustment and thermal sink are explicit
Lagrange multipliers. No post-step state projection or clipping is used.
All axial/azimuthal perturbations are excluded: this is NOT turbulent DNS,
not a stability calculation, and not a qualified station heat-transfer law.
"""
from copy import deepcopy
from dataclasses import asdict,dataclass
from math import ceil,isfinite
import numpy as np
from scipy.sparse import bmat,csc_matrix,diags
from scipy.sparse.linalg import spsolve,splu
from core.status import DomainError,ModelLimitError,SolverError
from atomic_model_multiphysics.provenance import seal,unseal

LEDGERS=('wall_heat','sink_heat','pressure_work','buoyancy_work','viscous_loss')


def number(v,low,high,name):
    if type(v) not in (int,float) or not isfinite(v) or not low<=v<=high:
        raise ValueError('Finite bounded '+name+' required')
    return float(v)


@dataclass(frozen=True)
class PipeCase:
    reynolds:float=5300.
    prandtl:float=.7
    buoyancy_c:float=0.
    cells:int=64
    max_step:float=10.

    def __post_init__(self):
        for key,lo,hi in [('reynolds',500.,10000.),('prandtl',.6,2.),('buoyancy_c',0.,10.),('max_step',.01,100.)]:
            number(getattr(self,key),lo,hi,key)
        if type(self.cells) is not int or not 16<=self.cells<=256:raise ValueError('Radial cells16..256 required')


def grid(n):
    h=1/n;faces=np.linspace(0.,1.,n+1);x=(faces[1:]+faces[:-1])/2
    w=(faces[1:]**2-faces[:-1]**2)/2;g=faces[1:-1]/h;d=np.zeros(n)
    d[:-1]-=g;d[1:]-=g;d[-1]-=2/h
    K=diags([g,d,g],[-1,0,1],format='csc');wall=np.zeros(n);wall[-1]=2/h
    return x,w,K,wall


def steady_state(case=None):
    case=case if case is not None else PipeCase()
    if type(case) is not PipeCase:raise ValueError('PipeCase required')
    _,w,K,wall=grid(case.cells);col=csc_matrix(w[:,None]);zero=csc_matrix((1,1))
    heat=bmat([[K,-4*col],[col.T,zero]],format='csc')
    thermal=spsolve(heat,np.r_[-wall,.25]);theta=thermal[:-1]
    momentum=bmat([[K,4*col],[col.T,zero]],format='csc')
    velocity=spsolve(momentum,np.r_[-4*w*(1+case.buoyancy_c*theta),.25])[:-1]
    state=np.r_[velocity,theta]
    if not np.all(np.isfinite(state)):raise SolverError('Nonfinite steady radial state')
    return state.tolist()


class PipeSession:
    def __init__(self,case=None,*,initial_state=None):
        self.config=case if case is not None else PipeCase()
        if type(self.config) is not PipeCase:raise ValueError('PipeCase required')
        self.x,self.w,self.K,self.wall=grid(self.config.cells);self.W=diags(self.w,format='csc')
        self._state=np.array(steady_state(self.config) if initial_state is None else self._read_state(initial_state),dtype=float)
        self._time=0.;self._c=float(self.config.buoyancy_c);self._history=[];self._steps=0
        self._initial_state=self._state.tolist();self._ledger={k:0. for k in LEDGERS}
        self._thermal_initial=float(self.w@self._state[self.config.cells:]);self._kinetic_initial=self._kinetic(self._state)
        self._admit(self._state)

    def _read_state(self,value):
        if type(value) is not list or len(value)!=2*self.config.cells or any(type(v) not in (int,float) or not isfinite(v) for v in value):
            raise ValueError('Finite complete radial state list required')
        return value

    @property
    def time(self):return self._time

    @property
    def state(self):return self._state.copy()

    @property
    def ledger(self):return deepcopy(self._ledger)

    def _kinetic(self,y):return float(.5*self.w@(y[:self.config.cells]**2))

    def _admit(self,y):
        n=self.config.cells
        if len(y)!=2*n or not np.all(np.isfinite(y)):raise DomainError('Nonfinite radial state')
        for values,name in ((y[:n],'velocity'),(y[n:],'temperature')):
            if abs(float(2*self.w@values)-.5)>5e-11:raise DomainError('Prescribed area mean violated: '+name)
        if float(np.min(y[:n]))<=0:
            raise ModelLimitError('Radial benchmark has local reverse flow',criterion='pipe_forward_velocity',node='radial_pipe',observed=float(np.min(y[:n])),limit=0.)
        if float(np.min(y[n:]))<0 or float(np.max(y[n:]))>1:raise DomainError('Selected radial temperature interval[0,1] violated')

    def _factor(self,h,c):
        n=self.config.cells;du=h/self.config.reynolds;dt=du/self.config.prandtl
        col=csc_matrix(self.w[:,None]);empty=csc_matrix((1,1))
        matrix=bmat([[self.W-du*self.K/2,-2*du*c*self.W,-4*du*col,None],
                     [None,self.W-dt*self.K/2,None,4*dt*col],
                     [col.T,None,empty,empty],[None,col.T,empty,empty]],format='csc')
        try:return splu(matrix)
        except RuntimeError as error:raise SolverError('Constrained radial factorization failed') from error

    def advance(self,duration,*,buoyancy_c=None):
        duration=number(duration,.01,2000.,'dimensionless duration')
        c=self._c if buoyancy_c is None else number(buoyancy_c,0.,10.,'buoyancy_c')
        count=ceil(duration/self.config.max_step)
        if count>4096 or self._steps+count>8192 or self._time+duration>10000 or len(self._history)>=256:raise ValueError('Radial work/history/horizon budget exceeded')
        # Work on private arrays; the complete owner is committed only after all substeps pass.
        y=self._state.copy();ledger=deepcopy(self._ledger);h=duration/count;n=self.config.cells
        du=h/self.config.reynolds;dt=du/self.config.prandtl;factor=self._factor(h,c)
        for _ in range(count):
            u,theta=y[:n],y[n:]
            rhs=np.r_[self.W@u+du*(self.K@u)/2+2*du*c*self.w*theta+4*du*self.w,
                      self.W@theta+dt*(self.K@theta)/2+dt*self.wall,.25,.25]
            answer=factor.solve(rhs);new=answer[:2*n]
            if not np.all(np.isfinite(answer)):raise SolverError('Nonfinite radial constrained step')
            residual=self._step_residual(answer,rhs,h,c)
            if residual>1e-10:raise SolverError('Radial constrained linear residual')
            self._admit(new)
            um=(u+new[:n])/2;tm=(theta+new[n:])/2;beta=float(answer[-2]);nu=float(answer[-1])
            increments=dict(wall_heat=dt*float(np.sum(self.K@tm+self.wall)),sink_heat=dt*2*nu,
                pressure_work=du*4*(1+beta)*float(self.w@um),buoyancy_work=du*4*c*float(self.w@(um*tm)),
                viscous_loss=-du*float(um@(self.K@um)))
            if increments['viscous_loss']<0 or nu<0:raise DomainError('Radial dissipative/heating domain')
            for key,value in increments.items():ledger[key]+=value
            self._balances(new,ledger,check=True);y=new
        self._state=y;self._ledger=ledger;self._time+=duration;self._c=c;self._steps+=count
        self._history.append(dict(duration=duration,buoyancy_c=buoyancy_c))
        return self.snapshot()

    def _step_residual(self,answer,rhs,h,c):
        n=self.config.cells;du=h/self.config.reynolds;dt=du/self.config.prandtl
        u,theta=answer[:n],answer[n:2*n];beta,nu=answer[-2:]
        a=self.W@u-du*(self.K@u)/2-2*du*c*self.w*theta-4*du*self.w*beta-rhs[:n]
        b=self.W@theta-dt*(self.K@theta)/2+4*dt*self.w*nu-rhs[n:2*n]
        return max(float(np.max(np.abs(a))),float(np.max(np.abs(b))),abs(float(self.w@u)-.25),abs(float(self.w@theta)-.25))

    def _balances(self,y,ledger,*,check=False):
        heat=float(self.w@y[self.config.cells:])-self._thermal_initial-ledger['wall_heat']+ledger['sink_heat']
        work=self._kinetic(y)-self._kinetic_initial-ledger['pressure_work']-ledger['buoyancy_work']+ledger['viscous_loss']
        scale=max(1.,*(abs(v) for v in ledger.values()))
        if check and (not all(isfinite(v) for v in ledger.values()) or max(abs(heat),abs(work))/scale>1e-10):
            raise SolverError('Radial heat/kinetic ledger failure')
        return dict(thermal_residual=heat,kinetic_residual=work,relative_maximum=max(abs(heat),abs(work))/scale)

    def snapshot(self):
        n=self.config.cells;u=self._state[:n];theta=self._state[n:]
        nu=float(np.sum(self.K@theta+self.wall)/2)
        beta=float(-np.sum(self.K@u)/2-self._c*2*(self.w@theta)-1)
        return dict(time=self.time,time_unit='R/(2Ub)',substeps=self._steps,radius_fraction=self.x.tolist(),velocity=u.tolist(),temperature=theta.tolist(),
            area_mean_velocity=float(2*self.w@u),area_mean_temperature=float(2*self.w@theta),
            mixing_cup_temperature=float(self.w@(u*theta)/(self.w@u)),pressure_beta=beta,
            pressure_coefficient=1+beta,sink_ratio_nu=nu,nu_definition='epsilon/epsilon0, not hD/k',
            buoyancy_c=self._c,ledger=self.ledger,checks=self._balances(self._state,self._ledger),
            model_id='jfm2021-radial-invariant-subspace',station_usable=False,dns_qualified=False,validated=False)

    def checkpoint(self):
        from .provenance import identity
        return seal(dict(format='jfm-radial/v1',source=identity(),case=asdict(self.config),initial_state=deepcopy(self._initial_state),
                         history=deepcopy(self._history),snapshot=self.snapshot()))

    @classmethod
    def restore(cls,saved):
        from .provenance import identity
        p=unseal(deepcopy(saved))
        if type(p) is not dict or set(p)!={'format','source','case','initial_state','history','snapshot'} or p['format']!='jfm-radial/v1' or p['source']!=identity():raise ValueError('Radial checkpoint source/format/fields')
        if type(p['history']) is not list or len(p['history'])>256:raise ValueError('Radial checkpoint history budget')
        if type(p['case']) is not dict or set(p['case'])!={'reynolds','prandtl','buoyancy_c','cells','max_step'}:raise ValueError('Radial checkpoint case fields')
        obj=cls(PipeCase(**p['case']),initial_state=p['initial_state'])
        for command in p['history']:
            if type(command) is not dict or set(command)!={'duration','buoyancy_c'}:raise ValueError('Radial checkpoint command fields')
            obj.advance(**command)
        if obj.checkpoint()!=saved:raise ValueError('Radial native replay differs')
        return obj
