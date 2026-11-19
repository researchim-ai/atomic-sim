"""Independent sampled strong balances, retaining Fourier truncation scope."""
import numpy as np

from atomic_model_station_advection.kernel import _radius
from atomic_model_station_refined.products import signed_modes, direct_products
from atomic_model_station_stokes_pressure import PressureField
from atomic_model_station_wall import velocity_jets
from .system import ResolvedThermalSystem, ThermalResult
from .products import scalar_products


def audit_resolved(system, result, *, radius):
    if type(system) is not ResolvedThermalSystem or type(result) is not ThermalResult:
        raise ValueError('ThermalSystem and ThermalResult required')
    if result.report.get('equation_fingerprint') != system.equation_fingerprint:
        raise ValueError('Thermal equations or sources differ')
    r = _radius(radius, system.alpha)
    if r[-1] != 1.:
        raise ValueError('Thermal audit must include the wall')
    groups = ((result.velocity, system.models), (result.velocity_derivatives, system.models),
              (result.temperature, system.scalars), (result.temperature_derivatives, system.scalars))
    for pairs, models in groups:
        if type(pairs) is not list or len(pairs) != len(models):
            raise ValueError('Complete thermal endpoint fields required')
        for pair, model in zip(pairs, models):
            if (type(pair) not in (tuple, list) or len(pair) != 2
                    or type(pair[0]) is not type(model) or pair[0].fingerprint != model.fingerprint):
                raise ValueError('Thermal endpoint basis differs')
    if type(result.pressure) is not dict or set(result.pressure) != set(system.keys):
        raise ValueError('Complete thermal pressure required')
    if any(result.report.get(key) != getattr(system,key) for key in ('viscosity','diffusivity','buoyancy')):
        raise ValueError('Thermal equation parameters differ')
    velocity, radial = signed_modes(result.velocity,r,system.alpha)
    theta,thermal_radial = {},{}
    for model,q in result.temperature:
        key=(model.k,model.m)
        theta[key],thermal_radial[key],_=model.jets(q,r)
        if key!=(0,0):
            opposite=(-key[0],-key[1])
            theta[opposite],thermal_radial[opposite]=theta[key].conj(),thermal_radial[key].conj()
    nonlinear=direct_products(r,system.alpha,velocity,radial)
    transport=scalar_products(r,system.alpha,velocity,theta,thermal_radial,method='direct')
    rows={};maximum_v=maximum_t=wall_v=wall_t=0.
    for (m,q),(_,dv),(s,t),(_,dt) in zip(result.velocity,result.velocity_derivatives,result.temperature,result.temperature_derivatives):
        key=(m.k,m.m);field=result.pressure[key]
        if type(field) is not PressureField or field.model.fingerprint!=m.fingerprint:
            raise ValueError('Thermal pressure basis differs')
        _,gradient=field.evaluate(r)
        _,_,lap_v=velocity_jets(m,q,r)
        _,_,lap_t=s.jets(t,r)
        residual_v=m.evaluate(dv,r)+nonlinear[key]+gradient-system.viscosity*lap_v
        residual_v[2]-=system.buoyancy*theta[key]
        residual_t=s.evaluate(dt,r)+transport[key]-system.diffusivity*lap_t-system.source(key,r)
        if not np.all(np.isfinite(residual_v)) or not np.all(np.isfinite(residual_t)):
            raise ValueError('Nonfinite thermal strong residual')
        mv,mt=float(np.max(abs(residual_v))),float(np.max(abs(residual_t)))
        wv,wt=float(np.max(abs(residual_v[:,-1]))),float(abs(residual_t[-1]))
        maximum_v,maximum_t=max(maximum_v,mv),max(maximum_t,mt)
        wall_v,wall_t=max(wall_v,wv),max(wall_t,wt)
        rows[str(key)]=dict(momentum=mv,thermal=mt,momentum_wall=wv,thermal_wall=wt)
    sm,dt=result.temperature_derivatives[system.mean]
    heat_derivative=float(np.real(system.weights@sm.evaluate(dt,system.radius)))
    sm,t=result.temperature[system.mean]
    wall_out=float(-system.diffusivity*np.real(sm.jets(t,[1.])[1][0]))
    heat_source=float(np.real(system.weights@system.source((0,0),system.radius)))
    return dict(maximum_momentum_residual=maximum_v,maximum_thermal_residual=maximum_t,
        maximum_wall_momentum_residual=wall_v,maximum_wall_thermal_residual=wall_t,
        physical_heat_rate_residual=abs(heat_derivative+wall_out-heat_source),rows=rows,
        direct_scalar_convolution=True,retained_modes_only=True,unretained_pressure_reconstructed=False,
        general_strong_pde_qualified=False,station_usable=False,validated=False)
