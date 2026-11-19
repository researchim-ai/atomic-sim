"""Bounded expanded-band products for the refined reference engine.

Same cylindrical equation as v75, with an explicitly increased121-mode
budget and automatic sufficient padding. Direct convolution is independent.
"""
import numpy as np
from atomic_model_station_advection.kernel import _radius,_key
from atomic_model_station_wall import velocity_jets


def _validate(radius,alpha,velocity,radial):
    r=_radius(radius,alpha)
    if type(velocity) is not dict or type(radial) is not dict or set(velocity)!=set(radial) or not 1<=len(velocity)<=121:raise ValueError('Complete expanded spectrum up to121 signed modes required')
    for key,u in velocity.items():
        _key(key,5);opposite=(-key[0],-key[1])
        if opposite not in velocity:raise ValueError('Expanded Hermitian spectrum required')
        for mapping in (velocity,radial):
            a=np.asarray(mapping[key])
            if a.shape!=(3,len(r)) or not np.iscomplexobj(a) or not np.all(np.isfinite(a)):raise ValueError('Finite expanded modal vectors required')
            if not np.allclose(a,np.asarray(mapping[opposite]).conj(),rtol=1e-11,atol=1e-13):raise ValueError('Expanded conjugate symmetry violated')
    return r


def signed_modes(states,radius,alpha):
    if type(states) not in (list,tuple) or not 1<=len(states)<=61:raise ValueError('Bounded expanded canonical sequence required')
    v={};dr={}
    for model,q in states:
        key=(model.k,model.m);_key(key,5)
        if model.alpha!=alpha or (model.m==0 and model.k<0) or key in v:raise ValueError('Unique canonical expanded modes and common alpha required')
        u,d,_=velocity_jets(model,q,radius);v[key]=u;dr[key]=d
        if key!=(0,0):v[(-key[0],-key[1])]=u.conj();dr[(-key[0],-key[1])]=d.conj()
    _validate(radius,alpha,v,dr);return v,dr


def expanded_fft(radius,alpha,velocity,radial,targets):
    r=_validate(radius,alpha,velocity,radial);keys=sorted(set(targets))
    if not 1<=len(keys)<=441:raise ValueError('Bounded nonlinear targets required')
    for key in keys:_key(key,10)
    nz=2*max(abs(k) for k,m in velocity)+max(abs(k) for k,m in keys)+1
    nt=2*max(abs(m) for k,m in velocity)+max(abs(m) for k,m in keys)+1
    if len(r)*nz*nt>300000:raise ValueError('Expanded FFT working-set budget exceeded')
    def physical(mapping,multiplier):
        a=np.zeros((3,len(r),nz,nt),complex)
        for (k,m),u in mapping.items():a[:,:,k%nz,m%nt]=u*multiplier(k,m)
        return np.fft.ifft2(a,axes=(-2,-1))*(nz*nt)
    u=physical(velocity,lambda k,m:1);d=physical(radial,lambda k,m:1)
    theta=physical(velocity,lambda k,m:1j*m);axial=physical(velocity,lambda k,m:1j*alpha*k);rr=r[:,None,None]
    nonlinear=u[0]*d+u[1]/rr*theta+u[2]*axial
    nonlinear[0]-=u[1]*u[1]/rr;nonlinear[1]+=u[1]*u[0]/rr
    a=np.fft.fft2(nonlinear,axes=(-2,-1))/(nz*nt)
    result={key:a[:,:,key[0]%nz,key[1]%nt].copy() for key in keys}
    if any(not np.all(np.isfinite(u)) for u in result.values()):raise ValueError('Nonfinite expanded product')
    return result


def direct_products(radius,alpha,velocity,radial):
    r=_validate(radius,alpha,velocity,radial)
    keys=set(velocity)|{(a[0]+b[0],a[1]+b[1]) for a in velocity for b in velocity}
    output={key:np.zeros((3,len(r)),complex) for key in sorted(keys)}
    for p,a in velocity.items():
        for q,b in velocity.items():
            value=a[0]*radial[q]+1j*q[1]*a[1]/r*b+1j*alpha*q[0]*a[2]*b
            value[0]-=a[1]*b[1]/r;value[1]+=a[1]*b[0]/r
            output[(p[0]+q[0],p[1]+q[1])]+=value
    if any(not np.all(np.isfinite(u)) for u in output.values()):raise ValueError('Nonfinite direct expanded product')
    return output
