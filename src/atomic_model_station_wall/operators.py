"""Analytic spin-vector Laplacian and stress, including the axis limit.

Pointwise forcing must be supplied explicitly. Generalized weak loads do
not determine a unique boundary force and are never substituted for it.
"""
import numpy as np
from numpy.polynomial import Legendre
from atomic_model_station_regular import RegularMode
from atomic_model_station_regular.model import _finite_real
from atomic_model_station_stokes_pressure import PressureField


def velocity_jets(model,q,radius):
    if type(model) is not RegularMode:raise ValueError('RegularMode required')
    r=np.asarray(radius,dtype=float)
    if r.ndim!=1 or not 1<=len(r)<=4096 or not np.all(np.isfinite(r)) or np.any(r<0) or np.any(r>1):
        raise ValueError('Finite evaluation radii in[0,1] required')
    coefficients=model.coefficients(q).reshape(3,model.order);s=r*r;one=Legendre([.5,-.5],domain=[0,1])
    spin=[];derivative=[];laplacian=[]
    for c,ell in zip(coefficients,(model.m+1,abs(model.m-1),model.m)):
        p=one*Legendre(c,domain=[0,1]);first=p.deriv();second=p.deriv(2)
        spin.append(r**ell*p(s))
        derivative.append((ell*r**(ell-1)*p(s) if ell else np.zeros_like(r))+2*r**(ell+1)*first(s))
        laplacian.append(r**ell*(4*(ell+1)*first(s)+4*s*second(s)-(model.alpha*model.k)**2*p(s)))
    def cylindrical(values):
        return np.array([.5*(values[0]+values[1]),(values[0]-values[1])/(2j),values[2]])
    return cylindrical(spin),cylindrical(derivative),cylindrical(laplacian)


def audit_momentum(model,before,after,pressure,*,dt,viscosity,radius,force,tolerance=1e-8):
    """Audit a midpoint Stokes step through r=1, without changing any state."""
    if type(model) is not RegularMode or type(pressure) is not PressureField:
        raise ValueError('Regular velocity model and physical PressureField required')
    if pressure.model.fingerprint!=model.fingerprint:raise ValueError('Velocity and pressure models differ')
    _finite_real(dt,'step',1e-10,1e3);_finite_real(viscosity,'viscosity',0.,1e3)
    _finite_real(tolerance,'momentum tolerance',1e-14,1e-4)
    r=np.asarray(radius,dtype=float)
    if r.ndim!=1 or len(r)<2 or np.any(np.diff(r)<=0) or r[-1]!=1.:
        raise ValueError('Increasing evaluation radii ending exactly at the wall required')
    old,_,_=velocity_jets(model,before,r);new,_,_=velocity_jets(model,after,r)
    mid=(np.asarray(before,dtype=complex)+np.asarray(after,dtype=complex))/2
    velocity,derivative,laplacian=velocity_jets(model,mid,r)
    values,gradient=pressure.evaluate(r)
    f=np.asarray(force,dtype=complex)
    if f.shape!=(3,len(r)) or not np.all(np.isfinite(f)):
        raise ValueError('Explicit finite physical vector force on evaluation radii required')
    acceleration=(new-old)/dt;viscous=viscosity*laplacian
    residual=acceleration+gradient-viscous-f
    scale=max(1.,float(np.max(abs(acceleration))),float(np.max(abs(gradient))),float(np.max(abs(viscous))),float(np.max(abs(f))))
    absolute=float(np.max(abs(residual)));wall=float(np.max(abs(residual[:,-1])))
    relative=absolute/scale;relative_wall=wall/scale
    ur,ut,uz=velocity[:,-1];dur,dut,duz=derivative[:,-1]
    traction=np.array([-values[-1]+2*viscosity*dur,
        viscosity*(dut-ut+1j*model.m*ur),viscosity*(duz+1j*model.alpha*model.k*ur)])
    if not np.all(np.isfinite(residual)) or not np.all(np.isfinite(traction)):
        raise ValueError('Nonfinite pointwise momentum or traction')
    def vector(a):return dict(real=a.real.tolist(),imag=a.imag.tolist())
    return dict(maximum_momentum_residual=absolute,wall_momentum_residual=wall,
        normalized_momentum_residual=relative,normalized_wall_residual=relative_wall,normalization=scale,tolerance=tolerance,
        sampled_momentum_passed=bool(relative<tolerance),wall_momentum_passed=bool(relative_wall<tolerance),
        wall_residual=vector(residual[:,-1]),wall_acceleration=vector(acceleration[:,-1]),wall_pressure_gradient=vector(gradient[:,-1]),
        wall_viscous=vector(viscous[:,-1]),wall_force=vector(f[:,-1]),wall_traction=vector(traction),
        evaluation_points=len(r),pressure_normalization='dimensionless kinematic pressure, density=1',
        nonlinear_advection_included=False,station_usable=False,validated=False,
        scope='Sampled strong linear Stokes momentum; a passing sample is not global dynamic qualification')
