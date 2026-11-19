"""Separately versioned extension of the regular polynomial Galerkin factory.

Legacy orders delegate to their unchanged factory. Higher orders retain all
conditioning/coercivity gates and receive a distinct source fingerprint.
"""
from pathlib import Path
import hashlib
import numpy as np
from numpy.polynomial import Legendre
from numpy.polynomial.legendre import leggauss
from scipy.linalg import null_space,cholesky,solve_triangular
from atomic_model_multiphysics.provenance import sha
from atomic_model_station_regular.model import RegularMode,regular_mode,_basis,_finite_real
from atomic_model_station_regular import model as legacy


def refined_mode(k,m,alpha,order=16,quadrature=96):
    if type(k) is not int or abs(k)>5 or type(m) is not int or not 0<=m<=5:raise ValueError('Refined Fourier domain |k|<=5,m=0..5 required')
    if type(order) is not int or not 2<=order<=24 or type(quadrature) is not int or not max(64,2*order+m+4)<=quadrature<=256:raise ValueError('Resolved bounded refined polynomial space required')
    _finite_real(alpha,'alpha',.01,10.)
    if order<=10:return regular_mode(k,m,alpha,order=order,quadrature=quadrature)
    x,w=leggauss(quadrature);r=np.sqrt((x+1)/2);w=w/4
    b,dr=_basis(r,m,order);one=Legendre([.5,-.5],domain=[0,1]);s=Legendre([.5,.5],domain=[0,1])
    c=np.zeros((order+1,3*order),complex)
    for j in range(order):
        p=one*Legendre.basis(j,domain=[0,1])
        polynomials=((m+1)*p+s*p.deriv(),p+s*p.deriv() if m==0 else p.deriv(),1j*alpha*k*p)
        for group,pdiv in enumerate(polynomials):c[:len(pdiv.coef),group*order+j]=pdiv.coef
    z=null_space(c,rcond=1e-12)
    def gram(a):return np.einsum('cni,n,cnj->ij',a.conj(),w,a)
    mass_full=gram(b);theta=np.empty_like(b)
    theta[0]=(1j*m*b[0]-b[1])/r[:,None];theta[1]=(1j*m*b[1]+b[0])/r[:,None];theta[2]=1j*m*b[2]/r[:,None]
    stiffness_full=gram(dr)+gram(theta)+(alpha*k)**2*mass_full
    reduced=z.conj().T@mass_full@z;condition=float(np.linalg.cond(reduced))
    if not np.isfinite(condition) or condition>1e10:raise ValueError('Ill-conditioned refined velocity space')
    lower=cholesky(reduced,lower=True);z=z@solve_triangular(lower.conj().T,np.eye(len(reduced)),lower=False)
    mass=z.conj().T@mass_full@z;stiffness=z.conj().T@stiffness_full@z
    mass=(mass+mass.conj().T)/2;stiffness=(stiffness+stiffness.conj().T)/2
    if np.max(abs(c@z))>1e-8 or np.min(np.linalg.eigvalsh(stiffness))<=0:raise ValueError('Refined constraint/coercivity check failed')
    digest=hashlib.sha256((sha(Path(__file__))+sha(legacy.__file__)).encode());digest.update(repr((k,m,float(alpha),order,quadrature)).encode())
    arrays=[r,w,b,z,c,mass_full,stiffness_full,mass,stiffness]
    for a in arrays:digest.update(a.tobytes());a.setflags(write=False)
    return RegularMode(k,m,float(alpha),order,quadrature,*arrays,digest.hexdigest())
