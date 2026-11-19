import numpy as np
import pytest
from scipy.special import jv, jn_zeros

from atomic_model_station_refined import refined_mode
from atomic_model_station_thermal import scalar_mode, ThermalSystem, ThermalResult, evolve_thermal, audit_thermal
from atomic_model_station_thermal.products import scalar_products
from scripts.qualify_station_advection import modes


def mean_system(order=8, *, nu=.05, kappa=.03, buoyancy=.2, source=None):
    v=refined_mode(0,0,.628319,order=order)
    s=scalar_mode(0,0,.628319,order=order)
    root=jn_zeros(0,1)[0]
    return ThermalSystem([(v,np.zeros(v.dimension))],[(s,s.fit(.1*jv(0,root*s.radius)))],
                         viscosity=nu,diffusivity=kappa,buoyancy=buoyancy,sources=source)


@pytest.mark.parametrize('m,k',[(0,0),(1,1),(2,-1)])
def test_scalar_axis_wall_and_diffusion_integration_by_parts(m,k):
    s=scalar_mode(k,m,.628319,order=8)
    q=np.arange(8)*.001
    value,dr,lap=s.jets(q,[0.,.3,1.])
    assert np.all(np.isfinite(np.r_[value,dr,lap])) and abs(value[-1])<1e-12
    np.testing.assert_allclose(s.mass,np.eye(8),atol=1e-10)
    assert min(np.linalg.eigvalsh(s.stiffness))>0
    lap=np.column_stack([s.jets(np.eye(8)[i],s.radius)[2] for i in range(8)])
    np.testing.assert_allclose(s.basis.conj().T@(s.weights[:,None]*lap),-s.stiffness,atol=1e-8)


@pytest.mark.parametrize('change',[dict(order=True),dict(order=17),dict(m=3),dict(k=3),dict(quadrature=32),dict(alpha=0)])
def test_scalar_domain_is_explicit(change):
    args=dict(k=0,m=0,alpha=.628319);args.update(change)
    with pytest.raises(ValueError):scalar_mode(**args)


@pytest.mark.parametrize('nu,kappa',[(.05,.03),(.03,.03)])
def test_transient_thermal_buoyancy_matches_bessel_solution(nu,kappa):
    system=mean_system(nu=nu,kappa=kappa)
    result=evolve_thermal(system)
    s,t=result.temperature[0];m,u=result.velocity[0]
    r=np.linspace(0,1,101);a=jn_zeros(0,1)[0];time=.1
    temperature=.1*np.exp(-kappa*a*a*time)*jv(0,a*r)
    factor=time*np.exp(-nu*a*a*time) if nu==kappa else (np.exp(-kappa*a*a*time)-np.exp(-nu*a*a*time))/((nu-kappa)*a*a)
    velocity=.2*.1*factor*jv(0,a*r)
    np.testing.assert_allclose(s.evaluate(t,r),temperature,atol=1e-9)
    np.testing.assert_allclose(m.evaluate(u,r)[2],velocity,atol=1e-9)
    assert result.report['maximum_physical_heat_budget_residual']<1e-10
    assert result.report['integrated_rates'][1]>0
    assert audit_thermal(system,result,radius=r[1:])['maximum_wall_thermal_residual']<1e-8


def test_stationary_heating_balances_wall_heat_and_buoyant_work():
    m=refined_mode(0,0,.628319,order=4);s=scalar_mode(0,0,.628319,order=4);r=s.radius
    u=np.zeros((3,len(r)),complex);u[2]=.2*.1/(16*.05)*(3-4*r*r+r**4)
    system=ThermalSystem([(m,m.fit(u))],[(s,s.fit(.1*(1-r*r)))],sources={(0,0):[.012]})
    result=evolve_thermal(system)
    np.testing.assert_allclose(result.report['initial'],result.report['final'],atol=1e-12)
    rates=result.report['endpoint_rates']
    assert rates[0]>0 and rates[1]==pytest.approx(rates[0],abs=1e-12)
    assert rates[4]==pytest.approx(.006,abs=1e-12) and rates[5]==pytest.approx(.006,abs=1e-12)
    assert audit_thermal(system,result,radius=np.linspace(.01,1,101))['maximum_thermal_residual']<1e-10


def test_scalar_fft_direct_and_global_variance_cancellation():
    velocity=modes(2,4,96);theta=[]
    rng=np.random.default_rng(80)
    for m,q in velocity:
        s=scalar_mode(m.k,m.m,m.alpha,order=5)
        t=.01*(rng.normal(size=5)+1j*rng.normal(size=5))
        if m.k==m.m==0:t=t.real.astype(complex)
        theta.append((s,t))
    system=ThermalSystem(velocity,theta)
    a,ar,af=system.calculate(system.initial)
    b,br,bf=system.calculate(system.initial,method='direct')
    np.testing.assert_allclose(a,b,atol=1e-11)
    np.testing.assert_allclose(ar,br,atol=1e-12)
    assert abs(af['advection_variance'])<1e-12 and abs(bf['advection_variance'])<1e-12
    for key in af['transport']:np.testing.assert_allclose(af['transport'][key],bf['transport'][key],atol=1e-12)
    assert max(np.max(abs(t)) for t in af['transport'].values())>1e-5


def test_buoyancy_sign_and_no_buoyancy_limit():
    positive=evolve_thermal(mean_system(buoyancy=.2))
    negative=evolve_thermal(mean_system(buoyancy=-.2))
    zero=evolve_thermal(mean_system(buoyancy=0.))
    np.testing.assert_allclose(positive.velocity[0][1],-negative.velocity[0][1],atol=1e-10)
    assert max(abs(zero.velocity[0][1]))<1e-12
    np.testing.assert_allclose(positive.temperature[0][1],zero.temperature[0][1],atol=1e-10)


def test_failed_time_policy_does_not_change_input_state():
    system=mean_system();before=system.initial.copy()
    with pytest.raises(ValueError):evolve_thermal(system,end_time=0)
    np.testing.assert_array_equal(system.initial,before)
    assert not system.initial.flags.writeable


@pytest.mark.parametrize('change',[dict(viscosity=0),dict(diffusivity=-1),dict(buoyancy=float('nan'))])
def test_physical_parameter_limits(change):
    system=mean_system()
    with pytest.raises(ValueError):ThermalSystem(system.velocity,system.temperature,**change)


@pytest.mark.parametrize('source',[{(1,0):[1.]},{(0,0):[1j]},{(0,0):[float('nan')]},{(0,0):[]}])
def test_invalid_source_rejected(source):
    with pytest.raises(ValueError):mean_system(source=source)


def test_mean_temperature_must_be_real_and_keys_must_match():
    system=mean_system();s,t=system.temperature[0]
    with pytest.raises(ValueError):ThermalSystem(system.velocity,[(s,t+1j)])
    wrong=scalar_mode(1,0,.628319)
    with pytest.raises(ValueError):ThermalSystem(system.velocity,[(wrong,t)])


def test_endpoint_audit_rejects_missing_fields_and_different_source():
    system=mean_system();result=evolve_thermal(system,end_time=.01)
    broken=ThermalResult(result.velocity,result.temperature,[],result.temperature_derivatives,result.pressure,result.report)
    with pytest.raises(ValueError):audit_thermal(system,broken,radius=np.linspace(.01,1,31))
    other=mean_system(source={(0,0):[.01]})
    with pytest.raises(ValueError,match='sources'):audit_thermal(other,result,radius=np.linspace(.01,1,31))


def test_bdf_and_direct_dop853_agree():
    system=mean_system(order=6)
    a=evolve_thermal(system,end_time=.02)
    b=evolve_thermal(system,end_time=.02,method='DOP853',products='direct',rtol=2e-12,atol=2e-15)
    for pa,pb in ((a.velocity,b.velocity),(a.temperature,b.temperature)):
        for (_,u),(_,v) in zip(pa,pb):np.testing.assert_allclose(u,v,atol=1e-9)


def test_coarse_heat_error_is_not_hidden_by_variance_budget():
    system=mean_system(order=3);result=evolve_thermal(system)
    assert result.report['maximum_variance_budget_residual']<1e-9
    assert result.report['maximum_physical_heat_budget_residual']>1e-8
    assert not result.report['physical_heat_gate_enforced']


def test_continuation_preserves_solution_within_tolerance():
    system=mean_system(order=6)
    whole=evolve_thermal(system,end_time=.04)
    prefix=evolve_thermal(system,end_time=.02)
    resumed=ThermalSystem(prefix.velocity,prefix.temperature)
    tail=evolve_thermal(resumed,start_time=.02,end_time=.04)
    for pairs_a,pairs_b in ((whole.velocity,tail.velocity),(whole.temperature,tail.temperature)):
        for (_,a),(_,b) in zip(pairs_a,pairs_b):np.testing.assert_allclose(a,b,atol=1e-9)
    assert not tail.report['bit_exact_restart']
