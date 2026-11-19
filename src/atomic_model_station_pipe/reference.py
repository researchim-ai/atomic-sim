"""Continuum polynomial and independently integrated semidiscrete dynamics.

The exponential reference eliminates controllers in the RATE, then propagates
an affine matrix. The production CN solver instead solves a constrained sparse
saddle-point system at every step. Neither alters a state after integration.
"""
import numpy as np
from scipy.linalg import expm
from .model import PipeCase,grid,number


def exact_profiles(radius,buoyancy_c):
    c=number(buoyancy_c,0.,10.,'buoyancy_c')
    x=np.asarray(radius,dtype=float)
    if x.ndim!=1 or not np.all(np.isfinite(x)) or np.any(x<0) or np.any(x>1):raise ValueError('Finite radii in[0,1] required')
    return dict(velocity=((1-x*x)+c*(x*x/3-x**4/4-1/12)).tolist(),temperature=(x*x).tolist(),
                pressure_beta=-c/3,sink_ratio_nu=1.,area_mean_velocity=.5,area_mean_temperature=.5)


def exponential_reference(case,state,duration,*,buoyancy_c=None):
    if type(case) is not PipeCase:raise ValueError('PipeCase required')
    duration=number(duration,.01,2000.,'dimensionless duration')
    c=case.buoyancy_c if buoyancy_c is None else number(buoyancy_c,0.,10.,'buoyancy_c')
    n=case.cells
    if type(state) is not list or len(state)!=2*n or any(type(v) not in (int,float) or not np.isfinite(v) for v in state):raise ValueError('Complete finite state required')
    _,w,K,wall=grid(n);P=np.eye(n)-np.outer(np.ones(n),w)/sum(w);L=K.toarray()/w[:,None]
    matrix=np.zeros((2*n+1,2*n+1))
    matrix[:n,:n]=P@L/case.reynolds;matrix[:n,n:2*n]=4*c*P/case.reynolds
    matrix[n:2*n,n:2*n]=P@L/(case.reynolds*case.prandtl)
    matrix[n:2*n,-1]=P@(wall/w)/(case.reynolds*case.prandtl)
    result=expm(matrix*duration)@np.r_[state,1.]
    if not np.all(np.isfinite(result)):raise ValueError('Nonfinite exponential reference')
    return result[:2*n].tolist()


def relaxation_scales(case):
    """Diffusive rates on the exact zero-area-mean subspace, not 3-D stability.

    The continuum constrained axisymmetric eigenproblem gives J2(k)=0:
    phi=J0(kr)-J0(k), integral(phi*r dr)=J2(k)/2=0 (up to a nonzero factor).
    The pressure/sink supply the required constant part of each eigenmode.
    """
    from scipy.linalg import eigvalsh,null_space
    from scipy.special import jn_zeros
    if type(case) is not PipeCase:raise ValueError('PipeCase required')
    _,w,K,_=grid(case.cells)
    q=null_space(np.sqrt(w)[None,:]);z=q/np.sqrt(w)[:,None]
    rates=-eigvalsh(z.T@K.toarray()@z)[::-1]
    exact=jn_zeros(2,3)**2
    return dict(discrete_radial_eigenvalues=rates[:3].tolist(),continuum_radial_eigenvalues=exact.tolist(),
        relative_errors=np.abs(rates[:3]/exact-1).tolist(),
        slowest_velocity_time=case.reynolds/float(rates[0]),
        slowest_temperature_time=case.reynolds*case.prandtl/float(rates[0]),
        exact_velocity_time=case.reynolds/float(exact[0]),exact_temperature_time=case.reynolds*case.prandtl/float(exact[0]),
        time_unit='R/(2Ub)',scope='Only constrained axisymmetric diffusive modes; no DNS or turbulent-stability inference',
        station_usable=False,dns_qualified=False)
