"""Steady radial mean subspace of Chu, Marensi, Willis2025, Eqs6-21.

R,2Ub,T0 units. Velocity-weighted axial sink A*u; A=Re*Pr*a.
This is imposed laminar symmetry even atRe5300, not turbulent heat transfer.
"""
from dataclasses import dataclass,asdict
import math
import numpy as np
from scipy.sparse import bmat,csc_matrix,diags
from scipy.sparse.linalg import spsolve
from core.status import DomainError,ModelLimitError,SolverError


@dataclass(frozen=True)
class HeatedCase:
    boundary:str='delta_temperature'
    buoyancy_c:float=0.
    reynolds:float=5300.
    prandtl:float=.7
    cells:int=64

    def __post_init__(self):
        if self.boundary not in ('delta_temperature','heat_flux'):raise ValueError('Explicit thermal boundary required')
        for key,lo,hi in [('buoyancy_c',0.,10.),('reynolds',500.,10000.),('prandtl',.6,2.)]:
            v=getattr(self,key)
            if type(v) not in (int,float) or not math.isfinite(v) or not lo<=v<=hi:raise ValueError('Finite study bound: '+key)
        if type(self.cells) is not int or not 16<=self.cells<=1024:raise ValueError('Cells16..1024 required')


def grid(n):
    h=1/n;f=np.linspace(0,1,n+1);r=(f[:-1]+f[1:])/2;w=np.diff(f*f)/2;g=f[1:-1]/h
    d=np.zeros(n);d[:-1]-=g;d[1:]-=g;d[-1]-=2/h
    ku=diags([g,d,g],[-1,0,1],format='csc')
    kt=ku.tolil();kt[-1,-1]+=2/h-3/h;kt[-1,-2]+=1/(3*h);kt=kt.tocsc()
    wall=np.zeros(n);wall[-1]=8/(3*h)
    return r,w,ku,kt,wall


def _equations(y,c,boundary,w,ku,kt,wall):
    n=len(w);u,t=y[:n],y[n:2*n];beta,a=y[-2:]
    constraint=float(w@t)-.25 if boundary=='delta_temperature' else (a-8.)
    return np.r_[ku@u+4*w*(1+beta+c*t),kt@t+wall-a*w*u,float(w@u)-.25,constraint]


def _norm(residual,w):
    n=len(w)
    return max(float(np.max(np.abs(residual[:n]/w))),float(np.max(np.abs(residual[n:2*n]/w))),
               float(np.max(np.abs(residual[-2:])))*100)


def _newton(initial,c,boundary,w,ku,kt,wall):
    y=initial.copy();n=len(w);W=diags(w,format='csc');column=csc_matrix(w[:,None]);empty=csc_matrix((1,1))
    for iteration in range(16):
        residual=_equations(y,c,boundary,w,ku,kt,wall);norm=_norm(residual,w)
        u=y[:n];a=y[-1]
        last=[None,column.T,empty,empty] if boundary=='delta_temperature' else [None,None,empty,csc_matrix([[1.]])]
        jac=bmat([[ku,4*c*W,4*column,None],[-a*W,kt,None,csc_matrix((-w*u)[:,None])],
                  [column.T,None,empty,empty],last],format='csc')
        try:delta=spsolve(jac,-residual)
        except RuntimeError as e:raise SolverError('Heated radial Newton factorization') from e
        if not np.all(np.isfinite(delta)):raise SolverError('Nonfinite heated Newton correction')
        if norm<5e-8 and float(np.max(np.abs(delta)))<2e-12 and float(np.max(np.abs(residual[-2:])))<1e-11:
            return y,iteration,norm
        for power in range(12):
            candidate=y+delta/(2**power)
            if _norm(_equations(candidate,c,boundary,w,ku,kt,wall),w)<max(norm,5e-8):
                y=candidate;break
        else:raise SolverError('Heated radial Newton line search')
    raise SolverError('Heated radial Newton work budget')


def solve_heated(case=None):
    case=HeatedCase() if case is None else case
    if type(case) is not HeatedCase:raise ValueError('HeatedCase required')
    n=case.cells;r,w,ku,kt,wall=grid(n)
    a=6. if case.boundary=='delta_temperature' else 8.
    y=np.r_[1-r*r,1-a*(3-4*r*r+r**4)/16,0.,a];trace=[]
    for c in np.linspace(0,case.buoyancy_c,max(1,math.ceil(case.buoyancy_c/.5))+1):
        y,iterations,residual=_newton(y,float(c),case.boundary,w,ku,kt,wall)
        trace.append(dict(c=float(c),iterations=iterations,scaled_residual=residual))
        if np.min(y[:n])<=0:raise ModelLimitError('Heated mean branch reverses locally',criterion='heated_forward_flow',node='heated_pipe',observed=float(np.min(y[:n])),limit=0.)
    u,t=y[:n],y[n:2*n];beta,a=map(float,y[-2:]);q=float(np.sum(kt@t+wall))
    area=float(2*w@t);mix=float(w@(u*t)/(w@u));mass=float(2*w@u)
    work=float(4*(1+beta)*(w@u)+4*case.buoyancy_c*(w@(u*t)));loss=float(-u@(ku@u))
    checks=dict(equation_scaled=residual,mass=abs(mass-.5),heat=abs(q-a*float(w@u)),
        momentum=abs(float(np.sum(ku@u))+2*(1+beta)+4*case.buoyancy_c*float(w@t)),
        kinetic_relative=abs(work-loss)/max(1,abs(work),abs(loss)),
        boundary=abs(area-.5) if case.boundary=='delta_temperature' else abs(q-2))
    if not np.all(np.isfinite(y)) or not 0<a<32 or np.min(t)<-4 or np.max(t)>2 or q<=0 or max(area,mix)>=1 or loss<=0:
        raise DomainError('Heated radial study domain')
    if checks['equation_scaled']>5e-8 or max(v for k,v in checks.items() if k!='equation_scaled')>1e-8:
        raise SolverError('Heated radial balance failure')
    return dict(case=asdict(case),radius=r.tolist(),velocity=u.tolist(),temperature=t.tolist(),
        area_mean_velocity=mass,area_mean_temperature=area,mixing_cup_temperature=mix,
        wall_temperature=1.,wall_gradient=q,pressure_beta=beta,axial_gradient_scaled=a,
        axial_gradient=a/(case.reynolds*case.prandtl),nusselt_area=2*q/(1-area),nusselt_mixing=2*q/(1-mix),
        pressure_and_buoyancy_work=work,viscous_loss=loss,checks=checks,continuation=trace,
        branch='zero-buoyancy-connected mean branch; stability not inferred',
        model_id='chu2025-steady-radial-velocity-weighted-sink',station_usable=False,turbulent=False,validated=False)


def flux_to_delta(result):
    """Same physical steady state expressed in the other thermal normalization."""
    if result['case']['boundary']!='heat_flux':raise ValueError('Heat-flux result required')
    k=2*(1-result['area_mean_temperature']);c=result['case']['buoyancy_c'];cd=k*c
    return dict(buoyancy_c=cd,temperature=(1+(np.array(result['temperature'])-1)/k).tolist(),
                pressure_beta=result['pressure_beta']+c-cd,wall_gradient=result['wall_gradient']/k,
                axial_gradient_scaled=result['axial_gradient_scaled']/k,temperature_scale_ratio=k)
