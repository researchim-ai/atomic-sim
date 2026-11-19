"""Independent polynomial steady solution of the dimensionaless coupled PDE.

For Q constant, theta=Q(1-r²)/(4*kappa) and
uz=b*Q*(3-4*r²+r⁴)/(64*nu*kappa). Cylindrical differentiation gives
kappa*lap(theta)+Q=0 and nu*lap(uz)+b*theta=0. Advection vanishes
because both fields are independent of z, ur=utheta=0. The wall values
are zero; physical outward heat flux equals Q/2 (radial-area measure).
"""
import numpy as np
import pytest

from atomic_model_station_thermal_resolved import (
    ResolvedThermalSystem, resolved_velocity_mode, resolved_scalar_mode,
    resolve_thermal, audit_resolved,
)


@pytest.mark.parametrize('source,buoyancy', [(.004,.2),(-.004,.2),(.004,-.3),(-.004,-.3),(0.,.2)])
def test_coupled_exact_stationary_solution_and_signed_wall_heat(source, buoyancy):
    nu, kappa = .05, .03
    v = resolved_velocity_mode(0,0,.628319,order=8,quadrature=128)
    t = resolved_scalar_mode(0,0,.628319,order=8,quadrature=128)
    r = v.radius
    velocity = np.zeros((3,len(r)), complex)
    velocity[2] = buoyancy*source*(3-4*r*r+r**4)/(64*nu*kappa)
    temperature = source*(1-t.radius**2)/(4*kappa)
    system = ResolvedThermalSystem([(v,v.fit(velocity))],[(t,t.fit(temperature))],
        viscosity=nu,diffusivity=kappa,buoyancy=buoyancy,sources={(0,0):[source]})
    derivative, rates, _ = system.calculate(system.initial)
    assert np.max(np.abs(derivative)) < 1e-10
    assert rates[4] == pytest.approx(source/2, abs=1e-13)
    assert rates[5] == pytest.approx(source/2, abs=1e-12)
    result = resolve_thermal(system,end_time=.5)
    radii = np.linspace(.01,1,101)
    audit = audit_resolved(system,result,radius=radii)
    assert audit['maximum_wall_momentum_residual'] < 1e-8
    assert audit['maximum_wall_thermal_residual'] < 1e-8
    assert result.report['maximum_physical_heat_budget_residual'] < 1e-10
    # Compare physical fields, not the solver's own derivative or ledger alone.
    vm, vq = result.velocity[0]
    tm, tq = result.temperature[0]
    expected_u = buoyancy*source*(3-4*radii**2+radii**4)/(64*nu*kappa)
    assert np.max(np.abs(vm.evaluate(vq,radii)[2]-expected_u)) < 1e-10
    assert np.max(np.abs(tm.evaluate(tq,radii)-source*(1-radii**2)/(4*kappa))) < 1e-10
