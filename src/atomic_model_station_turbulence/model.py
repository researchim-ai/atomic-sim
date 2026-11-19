"""Cess Eq B2, Marensi et al. JFM 919 A17 (2021), in radius/wall units.

Re_tau=u_tau R/nu; Re_b=2 Ub R/nu. No adjustable fit parameters.
nu_t is turbulent/molecular kinematic viscosity, not total viscosity.
Study bounds are guards, not a claim of physical qualification throughout.
"""
import math
import numpy as np
from scipy.integrate import quad
from scipy.linalg import solve_banded
from scipy.optimize import brentq


def _tau(value):
    if isinstance(value, (bool, str)) or not np.isscalar(value):
        raise ValueError('Scalar Re_tau required')
    value=float(value)
    if not math.isfinite(value) or not 100 <= value <= 2000:
        raise ValueError('Study requires 100 <= Re_tau <= 2000')
    return value


def _radius(value):
    r=np.asarray(value,dtype=float)
    if r.size == 0 or r.size > 100000 or not np.all(np.isfinite(r)) or np.any((r<0)|(r>1)):
        raise ValueError('Finite radius/R in [0,1] required')
    return r


def eddy_viscosity(radius,re_tau):
    """nu_t/nu. Rationalized square root preserves the viscous wall limit."""
    t=_tau(re_tau);r=_radius(radius)
    damp=-np.expm1(-(1-r)*t/27.)
    x=(.42*t/3.*(1-r*r)*(1+2*r*r)*damp)**2
    return x/(2*(np.sqrt(1+x)+1))


def _integral(function,a,b):
    value,error=quad(function,a,b,epsabs=2e-11,epsrel=2e-11,limit=150)
    if not math.isfinite(value) or error>2e-9*max(1,abs(value)):
        raise ArithmeticError('Cess quadrature did not converge')
    return value


def bulk_state(re_tau):
    t=_tau(re_tau)
    ub=_integral(lambda r:t*r**3/(1+float(eddy_viscosity(r,t))),0,1)
    return dict(re_tau=t,bulk_velocity_plus=ub,re_bulk=2*t*ub,darcy_friction=8/ub**2)


def at_bulk_reynolds(re_bulk):
    if isinstance(re_bulk,(bool,str)) or not np.isscalar(re_bulk):
        raise ValueError('Scalar bulk Reynolds number required')
    value=float(re_bulk);lo=bulk_state(100)['re_bulk'];hi=bulk_state(2000)['re_bulk']
    if not math.isfinite(value) or not lo<=value<=hi:
        raise ValueError('Bulk Reynolds number outside guarded Cess interval')
    t=brentq(lambda x:bulk_state(x)['re_bulk']-value,100,2000,xtol=1e-10,rtol=2e-13)
    result=bulk_state(t)
    result['relative_bulk_residual']=abs(result['re_bulk']/value-1)
    if result['relative_bulk_residual']>1e-10:raise ArithmeticError('Bulk closure residual')
    return result


def velocity(radius,re_tau):
    """Velocity/u_tau, evaluated by independent adaptive stress integration."""
    t=_tau(re_tau);r=_radius(radius)
    return np.array([_integral(lambda s:t*s/(1+float(eddy_viscosity(s,t))),float(x),1)
                     for x in r.flat]).reshape(r.shape)


def finite_volume(re_tau,cells=256):
    """Annular FV solution of -div(nu_total grad u+)=2 Re_tau.

    Work/dissipation use the dimensionless radial measure r dr (no 2 pi).
    Integrated source, symmetric axis flux and no-slip wall are imposed in
    the linear system; velocity is never rescaled after solving.
    """
    t=_tau(re_tau)
    if type(cells) is not int or not 16<=cells<=4096:raise ValueError('Integer cells in [16,4096] required')
    faces=np.linspace(0,1,cells+1);r=(faces[:-1]+faces[1:])/2;h=1/cells
    w=np.diff(faces**2);g=faces[1:]*(1+eddy_viscosity(faces[1:],t))/h;g[-1]*=2
    diag=g.copy();diag[1:]+=g[:-1]
    ab=np.zeros((3,cells));ab[1]=diag;ab[0,1:]=-g[:-1];ab[2,:-1]=-g[:-1]
    b=t*w;u=solve_banded((1,1),ab,b)
    applied=diag*u;applied[1:]-=g[:-1]*u[:-1];applied[:-1]-=g[:-1]*u[1:]
    residual=float(np.max(np.abs(applied-b))/max(1,float(np.max(np.abs(b)))))
    jumps=np.r_[np.diff(u),-u[-1]]
    dissipation=float(g@(jumps*jumps));work=float(b@u)
    balance=abs(work-dissipation)/max(1,abs(work))
    wall_flux=float(g[-1]*u[-1]);stress_error=abs(wall_flux/t-1)
    if residual>1e-8 or balance>1e-9 or stress_error>1e-8 or np.any(u<=0):
        raise ArithmeticError('FV momentum/energy/wall-stress closure failed')
    ub=float(w@u)
    return dict(radius=r.tolist(),velocity_plus=u.tolist(),bulk_velocity_plus=ub,
                re_tau=t,re_bulk=2*t*ub,darcy_friction=8/ub**2,cells=cells,
                linear_residual=residual,pressure_work=work,dissipation=dissipation,
                work_relative_residual=balance,wall_stress_relative_residual=stress_error)
