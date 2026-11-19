"""A common low-order physical seed fitted into separately chosen spaces."""
from functools import lru_cache
import numpy as np

from atomic_model_station_regular import regular_mode
from atomic_model_station_thermal_resolved import (
    resolved_velocity_mode,resolved_scalar_mode,ResolvedThermalSystem,
)
from .catalog import ThermalCase


@lru_cache(maxsize=8)
def _spaces(band,velocity_order,temperature_order,alpha,quadrature):
    pairs=[]
    for m in range(band+1):
        for k in (range(band+1) if m==0 else range(-band,band+1)):
            shell=max(abs(k),m)
            vp=16 if shell==5 else 24 if shell==4 else velocity_order
            tp=16 if shell==5 else 24 if shell==4 else temperature_order
            pairs.append((resolved_velocity_mode(k,m,alpha,order=vp,quadrature=quadrature),
                          resolved_scalar_mode(k,m,alpha,order=tp,quadrature=quadrature)))
    return tuple(pairs)


def build_thermal_case(case,*,band=4,velocity_order=32,temperature_order=40,quadrature=128):
    if type(case) is not ThermalCase:raise ValueError('ThermalCase required')
    if type(band) is not int or not 1<=band<=5:raise ValueError('Case Fourier band1..5 required')
    if type(velocity_order) is not int or not 3<=velocity_order<=32:
        raise ValueError('Seed-resolving velocity order3..32 required')
    if type(temperature_order) is not int or not max(2,case.radial_power)<=temperature_order<=48:
        raise ValueError('Seed-resolving bounded temperature order required')
    if type(quadrature) is not int or not 128<=quadrature<=256:
        raise ValueError('Case quadrature128..256 required')
    seed={};rng=np.random.default_rng(case.seed)
    for m in range(2):
        for k in (range(2) if m==0 else range(-1,2)):
            model=regular_mode(k,m,case.alpha,order=3,quadrature=96)
            if k==m==0:
                r=model.radius;u=np.zeros((3,len(r)),complex)
                u[1]=case.velocity_amplitude*r*(1-r*r)
                u[2]=case.mean_axial*(1-r*r)**2
                q=model.fit(u)
            else:q=case.velocity_amplitude*(rng.normal(size=model.dimension)+1j*rng.normal(size=model.dimension))
            seed[(k,m)]=(model,q)
    velocity=[];temperature=[]
    for v,t in _spaces(band,velocity_order,temperature_order,float(case.alpha),quadrature):
        key=(v.k,v.m)
        q=v.fit(seed[key][0].evaluate(seed[key][1],v.radius)) if key in seed else np.zeros(v.dimension,complex)
        amplitude=(case.temperature_mean if key==(0,0) else case.temperature_modal*(1+1j*case.phase)) if key in seed else 0.
        theta=amplitude*t.radius**t.m*(1-t.radius**2)**case.radial_power
        velocity.append((v,q));temperature.append((t,t.fit(theta)))
    return ResolvedThermalSystem(velocity,temperature,viscosity=float(case.viscosity),
        diffusivity=float(case.diffusivity),buoyancy=float(case.buoyancy),sources={(0,0):[case.source_mean]})
