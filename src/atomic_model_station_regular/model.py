"""Regular no-slip Fourier-Galerkin space and constrained viscous forms.

A linear single-mode Stokes research model, not the native Openpipeflow time
integrator, turbulence closure, or plant solver. The dual constraint reaction
is not identified with a physical collocated pressure field.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import numpy as np
from numpy.polynomial import Legendre
from numpy.polynomial.legendre import leggauss
from scipy.linalg import null_space, cholesky, solve_triangular
from atomic_model_multiphysics.provenance import sha


def _finite_real(value, name, low, high):
    if type(value) not in (int,float) or not np.isfinite(value) or not low<=value<=high:
        raise ValueError('Bounded finite '+name+' required')


def _basis(r,m,order):
    r=np.asarray(r,dtype=float)
    if r.ndim!=1 or not 1<=len(r)<=4096 or not np.all(np.isfinite(r)) or np.any(r<0) or np.any(r>1):
        raise ValueError('Finite evaluation radii in[0,1] required')
    s=r*r;values=np.zeros((3,len(r),3*order),complex);derivative=np.zeros_like(values)
    one=Legendre([.5,-.5],domain=[0,1])
    for group,ell in enumerate((m+1,abs(m-1),m)):
        for j in range(order):
            p=one*Legendre.basis(j,domain=[0,1]);f=r**ell*p(s)
            df=(ell*r**(ell-1)*p(s) if ell else np.zeros_like(r))+2*r**(ell+1)*p.deriv()(s)
            column=group*order+j
            if group<2:
                values[0,:,column]=.5*f;derivative[0,:,column]=.5*df
                factor=-.5j if group==0 else .5j
                values[1,:,column]=factor*f;derivative[1,:,column]=factor*df
            else:values[2,:,column]=f;derivative[2,:,column]=df
    return values,derivative


@dataclass(frozen=True)
class RegularMode:
    k: int
    m: int
    alpha: float
    order: int
    quadrature: int
    radius: np.ndarray
    weights: np.ndarray
    basis: np.ndarray
    transform: np.ndarray
    constraints: np.ndarray
    mass_full: np.ndarray
    stiffness_full: np.ndarray
    mass: np.ndarray
    stiffness: np.ndarray
    fingerprint: str

    @property
    def dimension(self):return self.transform.shape[1]

    @property
    def spec(self):
        return dict(k=self.k,m=self.m,alpha=self.alpha,order=self.order,quadrature=self.quadrature)

    def coefficients(self,q):
        a=np.asarray(q,dtype=complex)
        if a.shape!=(self.dimension,) or not np.all(np.isfinite(a)):raise ValueError('Complete finite reduced velocity coefficients required')
        return self.transform@a

    def evaluate(self,q,radius):
        values,_=_basis(radius,self.m,self.order)
        return np.einsum('cni,i->cn',values,self.coefficients(q))

    def spin_components(self,q,radius):
        # Evaluate ur+i*ut and ur-i*ut directly, avoiding cancellation near r=0.
        r=np.asarray(radius,dtype=float);_basis(r,self.m,self.order)
        coefficients=self.coefficients(q).reshape(3,self.order)
        values=[];one=Legendre([.5,-.5],domain=[0,1])
        for c,ell in zip(coefficients,(self.m+1,abs(self.m-1),self.m)):
            values.append(r**ell*(one*Legendre(c,domain=[0,1]))(r*r))
        return np.array(values)

    def analytic_divergence(self,q,radius):
        r=np.asarray(radius,dtype=float);_basis(r,self.m,self.order)
        polynomial=Legendre(self.constraints@self.coefficients(q),domain=[0,1])
        return r**self.m*polynomial(r*r)

    def load(self,force):
        a=np.asarray(force,dtype=complex)
        if a.shape!=(3,len(self.radius)) or not np.all(np.isfinite(a)):raise ValueError('Finite vector force on model quadrature required')
        return np.einsum('cni,n,cn->i',self.basis.conj(),self.weights,a)

    def fit(self,velocity):
        from scipy.linalg import solve
        return solve(self.mass,self.transform.conj().T@self.load(velocity),assume_a='pos')


def regular_mode(k,m,alpha,order=6,quadrature=64):
    if type(k) is not int or abs(k)>16 or type(m) is not int or not 0<=m<=6:
        raise ValueError('Integer Fourier mode |k|<=16,m=0..6 required')
    if type(order) is not int or not 2<=order<=10 or type(quadrature) is not int or not max(16,2*order+m+4)<=quadrature<=256:
        raise ValueError('Resolved bounded polynomial order and quadrature required')
    _finite_real(alpha,'alpha',.01,10.)
    x,w=leggauss(quadrature);r=np.sqrt((x+1)/2);w=w/4
    b,dr=_basis(r,m,order);one=Legendre([.5,-.5],domain=[0,1]);s=Legendre([.5,.5],domain=[0,1])
    c=np.zeros((order+1,3*order),complex)
    for j in range(order):
        p=one*Legendre.basis(j,domain=[0,1])
        polynomials=((m+1)*p+s*p.deriv(),p+s*p.deriv() if m==0 else p.deriv(),1j*alpha*k*p)
        for group,pdiv in enumerate(polynomials):c[:len(pdiv.coef),group*order+j]=pdiv.coef
    z=null_space(c,rcond=1e-12)
    def gram(a):return np.einsum('cni,n,cnj->ij',a.conj(),w,a)
    mass_full=gram(b)
    theta=np.empty_like(b);theta[0]=(1j*m*b[0]-b[1])/r[:,None]
    theta[1]=(1j*m*b[1]+b[0])/r[:,None];theta[2]=1j*m*b[2]/r[:,None]
    stiffness_full=gram(dr)+gram(theta)+(alpha*k)**2*mass_full
    reduced=z.conj().T@mass_full@z
    if not np.isfinite(np.linalg.cond(reduced)) or np.linalg.cond(reduced)>1e10:raise ValueError('Ill-conditioned regular velocity space')
    lower=cholesky(reduced,lower=True);z=z@solve_triangular(lower.conj().T,np.eye(len(reduced)),lower=False)
    mass=z.conj().T@mass_full@z;stiffness=z.conj().T@stiffness_full@z
    mass=(mass+mass.conj().T)/2;stiffness=(stiffness+stiffness.conj().T)/2
    if np.max(abs(c@z))>1e-8 or np.min(np.linalg.eigvalsh(stiffness))<=0:
        raise ValueError('Regular constrained space failed consistency or coercivity')
    digest=hashlib.sha256(sha(Path(__file__)).encode())
    digest.update(repr((k,m,float(alpha),order,quadrature)).encode())
    arrays=[r,w,b,z,c,mass_full,stiffness_full,mass,stiffness]
    for a in arrays:digest.update(a.tobytes());a.setflags(write=False)
    return RegularMode(k,m,float(alpha),order,quadrature,*arrays,digest.hexdigest())
