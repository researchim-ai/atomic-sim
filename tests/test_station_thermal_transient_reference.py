"""Exact coupled transient with constant source and a cylindrical eigenmode.

J0(lambda*r) has eigenvalue -lambda², lambda a zero of J0. Temperature is
its exponentially decaying mode plus the source's stationary polynomial.
The buoyant velocity mode follows the closed-form forced scalar ODE. Equal
viscosity/diffusivity gives the repeated-root limit b*A*t*exp(-nu*lambda²*t).
The axial fields have zero advection. This reference does not use the solver RHS.
"""
import numpy as np
import pytest
from scipy.special import jv, jn_zeros

from atomic_model_station_thermal_resolved import (
    resolved_velocity_mode,resolved_scalar_mode,ResolvedThermalSystem,
    resolve_thermal,audit_resolved,
)


@pytest.mark.parametrize('nu,kappa,source,buoyancy,mode',[
    (.05,.03,.004,.2,1),(.05,.03,-.004,-.3,1),
    (.04,.04,.004,.2,1),(.04,.04,-.004,-.3,2),
])
@pytest.mark.parametrize('time',[.5,2.])
def test_exact_transient_fields_heat_and_equal_diffusivity_limit(nu,kappa,source,buoyancy,mode,time):
    amplitude=.015
    lam=jn_zeros(0,mode)[-1]
    v=resolved_velocity_mode(0,0,.628319,order=16,quadrature=128)
    t=resolved_scalar_mode(0,0,.628319,order=24,quadrature=128)
    def stationary_u(r):
        return buoyancy*source*(3-4*r*r+r**4)/(64*nu*kappa)
    def stationary_t(r):
        return source*(1-r*r)/(4*kappa)
    u=np.zeros((3,len(v.radius)),complex)
    u[2]=stationary_u(v.radius)
    theta=stationary_t(t.radius)+amplitude*jv(0,lam*t.radius)
    system=ResolvedThermalSystem([(v,v.fit(u))],[(t,t.fit(theta))],
        viscosity=nu,diffusivity=kappa,buoyancy=buoyancy,sources={(0,0):[source]})
    result=resolve_thermal(system,end_time=time,rtol=2e-12,atol=2e-15)
    r=np.linspace(0.,1.,151)
    decay=np.exp(-kappa*lam**2*time)
    if nu==kappa:
        velocity_amplitude=buoyancy*amplitude*time*decay
    else:
        velocity_amplitude=buoyancy*amplitude*(decay-np.exp(-nu*lam**2*time))/((nu-kappa)*lam**2)
    vm,vq=result.velocity[0]
    tm,tq=result.temperature[0]
    numerical_u=vm.evaluate(vq,r)
    expected_u=stationary_u(r)+velocity_amplitude*jv(0,lam*r)
    expected_t=stationary_t(r)+amplitude*decay*jv(0,lam*r)
    assert np.max(np.abs(numerical_u[2]-expected_u))<2e-10
    assert np.max(np.abs(numerical_u[:2]))<1e-12
    assert np.max(np.abs(tm.evaluate(tq,r)-expected_t))<2e-10
    # Independently integrated signed physical heat and wall flux (measure r dr).
    exact_heat=source/(16*kappa)+amplitude*decay*jv(1,lam)/lam
    exact_integrated_flux=source*time/2+amplitude*jv(1,lam)*(1-decay)/lam
    assert result.report['final'][2]==pytest.approx(exact_heat,abs=2e-11)
    assert result.report['integrated_rates'][4]==pytest.approx(source*time/2,abs=1e-13)
    assert result.report['integrated_rates'][5]==pytest.approx(exact_integrated_flux,abs=2e-11)
    audit=audit_resolved(system,result,radius=np.linspace(.005,1,129))
    assert audit['maximum_wall_momentum_residual']<1e-8
    assert audit['maximum_wall_thermal_residual']<1e-8
