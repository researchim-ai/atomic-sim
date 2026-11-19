"""Physical scalar pressure in the regular Galerkin weak momentum equation.

Recovery is finite-dimensional and weak. It does not qualify strong wall
momentum or provide a nonlinear/thermal pressure model.
"""
from dataclasses import dataclass
import numpy as np
from numpy.polynomial import Legendre
from scipy.linalg import lstsq
from atomic_model_station_regular import RegularMode


def pressure_basis(model,radius):
    if type(model) is not RegularMode:raise ValueError('RegularMode required')
    r=np.asarray(radius,dtype=float)
    if r.ndim!=1 or not 1<=len(r)<=4096 or not np.all(np.isfinite(r)) or np.any(r<0) or np.any(r>1):
        raise ValueError('Finite pressure evaluation radii in[0,1] required')
    p=np.zeros((len(r),model.order+1),complex);g=np.zeros((3,len(r),model.order+1),complex)
    for j in range(model.order+1):
        polynomial=Legendre.basis(j,domain=[0,1]);value=polynomial(r*r)
        p[:,j]=r**model.m*value
        leading=model.m*r**(model.m-1)*value if model.m else np.zeros_like(r)
        g[0,:,j]=leading+2*r**(model.m+1)*polynomial.deriv()(r*r)
        if model.m:g[1,:,j]=1j*model.m*r**(model.m-1)*value
        g[2,:,j]=1j*model.alpha*model.k*p[:,j]
    return p,g


def weak_pressure_map(model):
    p,_=pressure_basis(model,model.radius)
    gram=p.conj().T@(model.weights[:,None]*p)
    # <v,grad p> = -<div v,p>, since all velocity test functions vanish on wall.
    return -model.constraints.conj().T@gram


@dataclass(frozen=True)
class PressureField:
    model: RegularMode
    coefficients: np.ndarray

    def __post_init__(self):
        if type(self.model) is not RegularMode:raise ValueError('RegularMode required')
        c=np.array(self.coefficients,dtype=complex,copy=True)
        if c.shape!=(self.model.order+1,) or not np.all(np.isfinite(c)):raise ValueError('Complete finite pressure coefficients required')
        c.setflags(write=False);object.__setattr__(self,'coefficients',c)

    def evaluate(self,radius):
        p,g=pressure_basis(self.model,radius)
        return p@self.coefficients,np.einsum('cni,i->cn',g,self.coefficients)


def recover_pressure(model,reaction):
    """Recover p from f-M du/dt-nu K u=G p, refusing incompatible reactions."""
    if type(model) is not RegularMode:raise ValueError('RegularMode required')
    r=np.asarray(reaction,dtype=complex)
    if r.shape!=(3*model.order,) or not np.all(np.isfinite(r)):raise ValueError('Complete finite weak momentum reaction required')
    mapping=weak_pressure_map(model);a=mapping;b=r
    mean_mode=model.k==0 and model.m==0
    if mean_mode:
        p,_=pressure_basis(model,model.radius);gauge=2*model.weights@p
        a=np.vstack((mapping,gauge));b=np.r_[r,0.]
    coefficients,_,rank,singular=lstsq(a,b,cond=1e-12,lapack_driver='gelsd')
    if rank!=model.order+1:raise ValueError('Pressure recovery has unresolved non-gauge nullspace')
    error=float(np.linalg.norm(mapping@coefficients-r));relative=error/max(1.,float(np.linalg.norm(r)))
    if not np.all(np.isfinite(coefficients)) or not np.isfinite(relative) or relative>1e-9:
        raise ValueError('Reaction cannot be represented by regular physical pressure')
    coefficients.setflags(write=False);field=PressureField(model,coefficients)
    p,gradient=field.evaluate(model.radius);wall,wall_gradient=field.evaluate([1.])
    mean=float(abs(2*model.weights@p)) if mean_mode else None
    if mean_mode and mean>1e-9:raise ValueError('Mean pressure gauge unresolved')
    receipt=dict(physical_pressure_reconstructed=True,recovery_scope='regular finite-dimensional weak momentum',
        pressure_normalization='dimensionless kinematic pressure, density=1',
        weak_momentum_relative_residual=relative,pressure_rank=int(rank),
        pressure_condition=float(singular[0]/singular[-1]),gauge='zero radial-area mean' if mean_mode else 'unique nonmean Fourier pressure',
        mean_gauge_absolute_residual=mean,pressure_wall_real=float(wall[0].real),pressure_wall_imag=float(wall[0].imag),
        wall_gradient_real=wall_gradient[:,0].real.tolist(),wall_gradient_imag=wall_gradient[:,0].imag.tolist(),
        coefficients_real=coefficients.real.tolist(),coefficients_imag=coefficients.imag.tolist(),
        strong_wall_momentum_qualified=False,nonlinear_pressure_qualified=False,station_usable=False,validated=False)
    return field,receipt


def recover_step_pressure(model,before,after,dt,force,viscosity):
    from atomic_model_station_regular.model import _finite_real
    _finite_real(dt,'pressure step',1e-10,1e3);_finite_real(viscosity,'viscosity',0.,1e3)
    if type(model) is not RegularMode:raise ValueError('RegularMode required')
    first=model.coefficients(before);last=model.coefficients(after)
    f=np.asarray(force,dtype=complex)
    if f.shape!=(3*model.order,) or not np.all(np.isfinite(f)):raise ValueError('Complete finite generalized force required')
    reaction=f-model.mass_full@(last-first)/dt-.5*viscosity*model.stiffness_full@(last+first)
    return recover_pressure(model,reaction)
