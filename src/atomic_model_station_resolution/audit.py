"""Include omitted Fourier content; never compare coordinates of different bases."""
import numpy as np
from atomic_model_station_regular import RegularMode
from atomic_model_station_regular.model import _finite_real
from atomic_model_station_wall import velocity_jets
from atomic_model_station_advection import signed_spectrum,convolution_advection
from atomic_model_station_stokes_pressure import PressureField


def _models(states):
    if type(states) not in (list,tuple) or not 1<=len(states)<=41:raise ValueError('Bounded canonical velocity states required')
    result={}
    for model,q in states:
        if type(model) is not RegularMode:raise ValueError('RegularMode required')
        key=(model.k,model.m)
        if key in result:raise ValueError('Duplicate mode')
        model.coefficients(q);result[key]=(model,q)
    return result


def compare_states(first,second,radius,weights,*,first_time,second_time):
    for value in (first_time,second_time):_finite_real(value,'comparison epoch',0.,1e12)
    epoch_tolerance=max(1e-12,8*float(np.spacing(max(abs(first_time),abs(second_time)))))
    if abs(first_time-second_time)>epoch_tolerance:raise ValueError('Same comparison epoch required')
    a=_models(first);b=_models(second);alpha=next(iter(a.values()))[0].alpha
    if any(model.alpha!=alpha for model,q in [*a.values(),*b.values()]):raise ValueError('Common Fourier period required')
    va,_=signed_spectrum(first,radius,alpha);vb,_=signed_spectrum(second,radius,alpha)
    r=np.asarray(radius);w=np.asarray(weights,dtype=float)
    if w.shape!=r.shape or not np.all(np.isfinite(w)) or np.any(w<=0) or abs(w.sum()-.5)>1e-12:raise ValueError('Positive normalized independent comparison quadrature required')
    common=sorted(set(va)&set(vb));only_a=sorted(set(va)-set(vb));only_b=sorted(set(vb)-set(va))
    def square(x):return float(np.sum(abs(x)**2*w))
    common_error=sum(square(va[key]-vb[key]) for key in common)
    first_omitted=sum(square(va[key]) for key in only_a);second_omitted=sum(square(vb[key]) for key in only_b)
    total=common_error+first_omitted+second_omitted
    norm_a=sum(square(value) for value in va.values());norm_b=sum(square(value) for value in vb.values())
    if not np.all(np.isfinite([total,norm_a,norm_b])):raise ValueError('Nonfinite comparison norm')
    return dict(total_rms=float(np.sqrt(total)),common_rms=float(np.sqrt(common_error)),
        only_first_rms=float(np.sqrt(first_omitted)),only_second_rms=float(np.sqrt(second_omitted)),
        relative_rms=float(np.sqrt(total)/max(np.sqrt(norm_a),np.sqrt(norm_b),1e-30)),
        first_signed_modes=len(va),second_signed_modes=len(vb),common_signed_modes=len(common),
        first_time=float(first_time),second_time=float(second_time),omitted_modes_included=True,
        scope='Velocity norm on explicit quadrature; same epoch and period, including missing Fourier modes')


def audit_nonlinear_momentum(before,after,pressure,*,dt,viscosity,radius,force=None,tolerance=1e-8):
    _finite_real(dt,'step',1e-10,1e3);_finite_real(viscosity,'viscosity',0.,1e3);_finite_real(tolerance,'tolerance',1e-14,1e-4)
    first=_models(before);last=_models(after)
    if set(first)!=set(last) or type(pressure) is not dict or set(pressure)!=set(first):raise ValueError('Same complete retained velocity and pressure modes required')
    r=np.asarray(radius,dtype=float)
    if r.ndim!=1 or len(r)<2 or r[-1]!=1.:raise ValueError('Positive evaluation grid through exact wall required')
    if force is not None and (type(force) is not dict or set(force)!=set(first)):raise ValueError('Explicit complete retained physical force mapping required')
    midpoint=[]
    for key,(model,q) in first.items():
        other,p=last[key];field=pressure[key]
        if other.fingerprint!=model.fingerprint or type(field) is not PressureField or field.model.fingerprint!=model.fingerprint:raise ValueError('Velocity and pressure bases differ')
        midpoint.append((model,(np.asarray(q)+np.asarray(p))/2))
    alpha=midpoint[0][0].alpha
    velocity,radial=signed_spectrum(midpoint,r,alpha)
    nonlinear=convolution_advection(r,alpha,velocity,radial)
    retained_signed=set(velocity);tail=[value for key,value in nonlinear.items() if key not in retained_signed]
    rows={};maximum=0.;wall_maximum=0.;scale=1.
    for key,(model,q) in first.items():
        old,_,_=velocity_jets(model,q,r);new,_,_=velocity_jets(model,last[key][1],r)
        _,_,lap=velocity_jets(model,(np.asarray(q)+np.asarray(last[key][1]))/2,r)
        _,gradient=pressure[key].evaluate(r)
        f=np.zeros((3,len(r)),complex) if force is None else np.asarray(force[key],dtype=complex)
        if f.shape!=(3,len(r)) or not np.all(np.isfinite(f)):raise ValueError('Finite physical force at all sampled radii required')
        acceleration=(new-old)/dt;viscous=viscosity*lap
        residual=acceleration+nonlinear[key]+gradient-viscous-f
        if not np.all(np.isfinite(residual)):raise ValueError('Nonfinite nonlinear momentum residual')
        absolute=float(np.max(abs(residual)));wall=float(np.max(abs(residual[:,-1])))
        local_scale=max(1.,*(float(np.max(abs(v))) for v in (acceleration,nonlinear[key],gradient,viscous,f)))
        maximum=max(maximum,absolute);wall_maximum=max(wall_maximum,wall);scale=max(scale,local_scale)
        rows[str(key)]=dict(maximum_residual=absolute,wall_residual=wall,
            wall_real=residual[:,-1].real.tolist(),wall_imag=residual[:,-1].imag.tolist())
    return dict(maximum_retained_residual=maximum,maximum_retained_wall_residual=wall_maximum,normalization=scale,
        normalized_retained_residual=maximum/scale,normalized_retained_wall_residual=wall_maximum/scale,
        sampled_retained_momentum_passed=bool(maximum/scale<tolerance),retained_wall_passed=bool(wall_maximum/scale<tolerance),
        unretained_nonlinear_maximum=max((float(np.max(abs(v))) for v in tail),default=0.),
        unretained_product_modes=len(tail),unretained_pressure_reconstructed=False,mode_rows=rows,
        nonlinear_advection_included=True,axis_sampled=False,dt=float(dt),viscosity=float(viscosity),alpha=float(alpha),tolerance=tolerance,station_usable=False,validated=False,
        scope='Sampled retained strong nonlinear momentum; unretained product amplitude is an indicator, not a pressure-balanced error bound')
