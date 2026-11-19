"""Cylindrical vector u.dot(grad)u, exact mode convolution and padded FFT.

No time stepping or radial resampling. The axis is excluded from pointwise
products; regular analytic fields are sampled on strictly positive radii.
"""
import numpy as np
from atomic_model_station_regular import RegularMode
from atomic_model_station_wall import velocity_jets


def _radius(radius,alpha):
    r=np.asarray(radius,dtype=float)
    if r.ndim!=1 or not 2<=len(r)<=512 or not np.all(np.isfinite(r)) or r[0]<=0 or r[-1]>1 or np.any(np.diff(r)<=0):raise ValueError('Increasing positive evaluation radii up to1 required')
    if type(alpha) not in (int,float) or not np.isfinite(alpha) or not .01<=alpha<=10:raise ValueError('Bounded finite alpha required')
    return r


def _key(key,limit):
    if type(key) is not tuple or len(key)!=2 or any(type(v) is not int or abs(v)>limit for v in key):raise ValueError('Bounded integer(k,m) key required')
    return key


def _inputs(radius,alpha,velocity,radial):
    r=_radius(radius,alpha)
    if type(velocity) is not dict or type(radial) is not dict or not 1<=len(velocity)<=81 or set(velocity)!=set(radial):raise ValueError('Complete bounded velocity/derivative mappings required')
    for key in velocity:
        _key(key,8)
        if (-key[0],-key[1]) not in velocity:raise ValueError('Complete Hermitian signed spectrum required')
        for fields in (velocity,radial):
            a=np.asarray(fields[key]);other=np.asarray(fields[(-key[0],-key[1])])
            if a.shape!=(3,len(r)) or not np.iscomplexobj(a) or not np.all(np.isfinite(a)):raise ValueError('Complete finite complex modal vectors required')
            if not np.allclose(a,other.conj(),rtol=1e-11,atol=1e-13):raise ValueError('Hermitian velocity and derivatives required')
    return r


def signed_spectrum(modes,radius,alpha):
    r=_radius(radius,alpha)
    if type(modes) not in (list,tuple) or not 1<=len(modes)<=41:raise ValueError('Bounded canonical mode sequence required')
    velocity={};radial={}
    for model,q in modes:
        if type(model) is not RegularMode or model.alpha!=alpha:raise ValueError('Common regular-mode alpha required')
        key=(model.k,model.m);_key(key,8)
        if model.m==0 and model.k<0:raise ValueError('Mean azimuthal plane uses nonnegative k only')
        if key in velocity:raise ValueError('Duplicate canonical Fourier mode')
        u,dr,_=velocity_jets(model,q,r)
        velocity[key]=u;radial[key]=dr
        if key!=(0,0):
            velocity[(-key[0],-key[1])]=u.conj();radial[(-key[0],-key[1])]=dr.conj()
    _inputs(r,alpha,velocity,radial)
    return velocity,radial


def _targets(velocity,targets):
    if targets is None:
        result=set(velocity)|{(p[0]+q[0],p[1]+q[1]) for p in velocity for q in velocity}
    else:
        if type(targets) not in (list,tuple,set) or not 1<=len(targets)<=1089:raise ValueError('Bounded target mode collection required')
        for key in targets:_key(key,16)
        result=set(targets)
        if len(result)!=len(targets):raise ValueError('Unique target modes required')
    for key in result:_key(key,16)
    if len(result)>1089:raise ValueError('Nonlinear output budget exceeded')
    return sorted(result)


def convolution_advection(radius,alpha,velocity,radial,targets=None):
    r=_inputs(radius,alpha,velocity,radial);keys=_targets(velocity,targets)
    result={key:np.zeros((3,len(r)),complex) for key in keys}
    for p,a in velocity.items():
        for q,b in velocity.items():
            target=(p[0]+q[0],p[1]+q[1])
            if target not in result:continue
            conv=a[0]*radial[q]+1j*q[1]*a[1]/r*b+1j*alpha*q[0]*a[2]*b
            conv[0]-=a[1]*b[1]/r;conv[1]+=a[1]*b[0]/r
            result[target]+=conv
    if any(not np.all(np.isfinite(v)) for v in result.values()):raise ValueError('Nonfinite nonlinear product')
    return result


def fft_advection(radius,alpha,velocity,radial,targets=None,*,grid=None,allow_aliasing=False):
    r=_inputs(radius,alpha,velocity,radial);keys=_targets(velocity,targets)
    if type(allow_aliasing) is not bool:raise ValueError('Explicit aliasing diagnostic flag required')
    kmax=max(abs(k) for k,m in velocity);mmax=max(abs(m) for k,m in velocity)
    kt=max(abs(k) for k,m in keys);mt=max(abs(m) for k,m in keys)
    required=(2*kmax+kt+1,2*mmax+mt+1)
    if grid is None:grid=required
    if type(grid) not in (tuple,list) or len(grid)!=2 or any(type(v) is not int or not 1<=v<=65 for v in grid):raise ValueError('Bounded integer FFT grid required')
    nz,nt=grid
    if nz<2*kmax+1 or nt<2*mmax+1:raise ValueError('FFT grid cannot represent input spectrum')
    safe=nz>=required[0] and nt>=required[1]
    if not safe and not allow_aliasing:raise ValueError('FFT padding insufficient for requested nonlinear targets')
    if len(r)*nz*nt>300000:raise ValueError('FFT working-set budget exceeded')
    def physical(mapping,multiplier):
        coefficients=np.zeros((3,len(r),nz,nt),complex)
        for (k,m),value in mapping.items():coefficients[:,:,k%nz,m%nt]=value*multiplier(k,m)
        return np.fft.ifft2(coefficients,axes=(-2,-1))*(nz*nt)
    u=physical(velocity,lambda k,m:1);dr=physical(radial,lambda k,m:1)
    theta=physical(velocity,lambda k,m:1j*m);axial=physical(velocity,lambda k,m:1j*alpha*k)
    rr=r[:,None,None]
    nonlinear=u[0]*dr+u[1]/rr*theta+u[2]*axial
    nonlinear[0]-=u[1]*u[1]/rr;nonlinear[1]+=u[1]*u[0]/rr
    transformed=np.fft.fft2(nonlinear,axes=(-2,-1))/(nz*nt)
    result={key:transformed[:,:,key[0]%nz,key[1]%nt].copy() for key in keys}
    if any(not np.all(np.isfinite(v)) for v in result.values()):raise ValueError('Nonfinite nonlinear FFT product')
    return result,dict(grid=[nz,nt],required_grid=list(required),alias_free_for_requested_targets=bool(safe),
        aliasing_explicitly_allowed=allow_aliasing,radial_points=len(r),input_modes=len(velocity),output_modes=len(keys),
        axis_sampled=False,nonlinear_time_integrator_qualified=False,station_usable=False,validated=False)


def nonlinear_power(velocity,advection,weights):
    w=np.asarray(weights,dtype=float)
    if w.ndim!=1 or not np.all(np.isfinite(w)) or np.any(w<=0) or abs(w.sum()-.5)>1e-12:raise ValueError('Positive normalized radial quadrature required')
    if type(velocity) is not dict or type(advection) is not dict or not velocity or not set(velocity)<=set(advection):raise ValueError('Complete retained nonlinear coefficients required')
    power=0.;energy=0.
    for key,value in velocity.items():
        _key(key,8)
        if (-key[0],-key[1]) not in velocity:raise ValueError('Full signed velocity spectrum required for power')
        u=np.asarray(value);n=np.asarray(advection[key])
        if u.shape!=(3,len(w)) or n.shape!=u.shape or not np.all(np.isfinite(u)) or not np.all(np.isfinite(n)):raise ValueError('Finite complete power fields required')
        if not np.allclose(u,np.asarray(velocity[(-key[0],-key[1])]).conj(),rtol=1e-11,atol=1e-13):raise ValueError('Hermitian velocity required for real-field power')
        if not np.allclose(n,np.asarray(advection[(-key[0],-key[1])]).conj(),rtol=1e-11,atol=1e-13):raise ValueError('Hermitian advection required for real-field power')
        power+=float(np.real(np.sum(u.conj()*n*w)));energy+=float(.5*np.sum(abs(u)**2*w))
    if not np.isfinite(power) or not np.isfinite(energy):raise ValueError('Nonfinite nonlinear energy diagnostic')
    return dict(kinetic_energy=energy,advection_power=power,absolute_advection_power=abs(power),
        scope='Full signed real-field spectrum; radial quadrature accuracy must be qualified separately')
