"""Separate native-style FD Laplacian and the actual composed div_h grad_h.

Boundary rows are explicit. Interior pressure compatibility does not imply
no-slip preservation, axis regularity, or a qualified time integrator.
"""
from dataclasses import dataclass
import warnings
import numpy as np
from scipy.linalg import solve, LinAlgWarning
from atomic_model_station_spectral.modes import derivative_matrices


@dataclass(frozen=True)
class PressureOperators:
    gradient: np.ndarray
    divergence: np.ndarray
    laplacian_fd: np.ndarray
    laplacian_composed: np.ndarray
    radial_first: np.ndarray
    radial_second: np.ndarray
    radius: np.ndarray
    k: int
    m: int
    alpha: float


def pressure_operators(radius, k, m, alpha):
    r = np.asarray(radius, dtype=float)
    if r.ndim != 1:
        raise ValueError('One-dimensional radial mesh required')
    positive, ghost = derivative_matrices(r)
    if type(k) is not int or type(m) is not int or abs(k)>127 or not 0<=m<=1024:
        raise ValueError('Bounded integer Fourier mode required')
    if type(alpha) not in (int,float) or not np.isfinite(alpha) or alpha<=0:
        raise ValueError('Finite positive axial wave number scale required')
    n=len(r);extended=np.r_[-r[:4][::-1],r];second=np.zeros((n,n))
    scalar_sign=(-1)**m; vector_sign=-scalar_sign
    for i in range(n):
        indices=np.arange(i,min(n+4,i+9));nodes=extended[indices]
        scale=max(abs(nodes-r[i]));x=(nodes-r[i])/scale
        matrix=np.vander(x,N=len(x),increasing=True).T
        rhs=np.zeros(len(x));rhs[2]=2/scale**2
        coefficients=np.linalg.solve(matrix,rhs)
        for index,c in zip(indices,coefficients):
            if index<4:second[i,3-index]+=scalar_sign*c
            else:second[i,index-4]+=c
    ds=(positive+scalar_sign*ghost).toarray()
    dv=(positive+vector_sign*ghost).toarray()+np.diag(1/r)
    theta=np.diag(1j*m/r);axial=np.eye(n)*(1j*alpha*k)
    gradient=np.vstack((ds,theta,axial));div=np.hstack((dv,theta,axial))
    lap=second+ds/r[:,None]-np.diag((m/r)**2+(alpha*k)**2)
    composed=div@gradient
    values=[gradient,div,lap,composed,ds,second,r.copy()]
    for value in values:value.setflags(write=False)
    return PressureOperators(*values,k,m,float(alpha))


def solve_pressure(operators, rhs, *, composed, wall_derivative=0., wall_gauge=0.):
    """Solve Poisson with explicit Neumann wall, or gauge for k=m=0.

    The wall PDE row is replaced, so its divergence is reported separately.
    Nothing is applied to a velocity field or an external integrator here.
    """
    if type(operators) is not PressureOperators or type(composed) is not bool:
        raise ValueError('PressureOperators and explicit operator choice required')
    n=len(operators.radius);b=np.asarray(rhs,dtype=complex)
    if b.shape!=(n,) or not np.all(np.isfinite(b)):
        raise ValueError('Complete finite pressure RHS required')
    for value in (wall_derivative,wall_gauge):
        if not np.isscalar(value) or not np.isfinite(value):
            raise ValueError('Finite scalar wall data required')
    a=np.array(operators.laplacian_composed if composed else operators.laplacian_fd,dtype=complex,copy=True);b=b.copy()
    if operators.k==0 and operators.m==0:
        if wall_derivative!=0:raise ValueError('Mean-mode gauge cannot also prescribe wall derivative')
        a[-1]=0;a[-1,-1]=1;b[-1]=wall_gauge;kind='mean-mode Dirichlet gauge'
    else:
        if wall_gauge!=0:raise ValueError('Nonmean-mode Neumann solve cannot also prescribe wall gauge')
        a[-1]=operators.radial_first[-1];b[-1]=wall_derivative;kind='Neumann wall'
    with warnings.catch_warnings():
        warnings.simplefilter('error',LinAlgWarning)
        try:p=solve(a,b,assume_a='gen',check_finite=True)
        except (np.linalg.LinAlgError,LinAlgWarning) as error:
            raise ValueError('Pressure system singular or ill-conditioned') from error
    residual=float(np.max(abs(a@p-b)))
    if residual>1e-8*max(1.,float(np.max(abs(b)))):
        raise ValueError('Pressure linear residual exceeds tolerance')
    return p,dict(boundary=kind,linear_residual=residual,
        interior_div_grad_residual=float(np.max(abs((operators.laplacian_composed@p-np.asarray(rhs))[:-1]))),
        wall_pde_residual=float(abs((operators.laplacian_composed@p-np.asarray(rhs))[-1])),
        no_slip_preservation_qualified=False,axis_regularity_qualified=False,
        time_integrator_qualified=False,station_usable=False,validated=False)


def manufactured_pressure(radius,m,k,alpha):
    """p=r^m(1-r²)^2 exp(r²), analytic cylindrical Laplacian and derivative."""
    # Reuse bounded public validation, including scalar/vector axis parity.
    op=pressure_operators(radius,k,m,alpha);r=op.radius;s=r*r
    f=(1-s)**2*np.exp(s);first=(s*s-1)*np.exp(s)
    second=(s*s+2*s-1)*np.exp(s)
    p=r**m*f
    dr=r**(m-1)*(m*f+2*s*first)
    lap=r**m*(4*(m+1)*first+4*s*second-(alpha*k)**2*f)
    return p,dr,lap
