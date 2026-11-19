import numpy as np
import pytest
from scipy.special import jv,jn_zeros
from scipy.integrate import quad
from atomic_model_station_regular import regular_mode
from atomic_model_station_refined import refined_mode,evolve_refined,endpoint_momentum,compare_refined,RefinedResult
from atomic_model_station_refined.products import signed_modes,expanded_fft,direct_products
from atomic_model_station_advection import nonlinear_power,fft_advection
from scripts.qualify_station_advection import modes
from scripts.qualify_station_nonlinear import reference_solution


def test_legacy_factory_path_is_byte_identical():
    a=regular_mode(1,1,.628319,order=6,quadrature=96);b=refined_mode(1,1,.628319,order=6)
    assert a.fingerprint==b.fingerprint
    for name in ('basis','transform','mass','stiffness'):np.testing.assert_array_equal(getattr(a,name),getattr(b,name))


@pytest.mark.parametrize('m,order',[(0,12),(1,16),(2,20),(4,20)])
def test_high_order_space_preserves_constraints_and_coercivity(m,order):
    model=refined_mode(1,m,.628319,order=order);q=np.arange(model.dimension)*1e-5
    assert np.max(abs(model.evaluate(q,[1.])))<1e-10
    assert np.max(abs(model.analytic_divergence(q,np.linspace(0,1,101))))<1e-8
    assert min(np.linalg.eigvalsh(model.stiffness))>0
    np.testing.assert_allclose(model.mass,np.eye(model.dimension),atol=1e-9)


@pytest.mark.parametrize('change',[dict(order=25),dict(order=True),dict(k=6),dict(m=6),dict(quadrature=32),dict(alpha=0)])
def test_refined_domain_is_bounded(change):
    args=dict(k=1,m=1,alpha=.628319,order=12);args.update(change)
    with pytest.raises(ValueError):refined_mode(**args)


def test_conditioning_gate_is_not_relaxed(monkeypatch):
    monkeypatch.setattr(np.linalg,'cond',lambda *args:1e11)
    with pytest.raises(ValueError,match='conditioned'):refined_mode(1,1,.628319,order=12)


def test_expanded_fft_matches_independent_pairs_and_conserves_energy():
    rng=np.random.default_rng(79);state=[]
    for m in range(6):
        for k in (range(6) if m==0 else range(-5,6)):
            model=refined_mode(k,m,.628319,order=3,quadrature=64)
            q=np.zeros(model.dimension,complex) if k==m==0 else .0001*(rng.normal(size=model.dimension)+1j*rng.normal(size=model.dimension))
            state.append((model,q))
    r=state[0][0].radius;w=state[0][0].weights;v,d=signed_modes(state,r,.628319)
    assert len(v)==121
    direct=direct_products(r,.628319,v,d);fast=expanded_fft(r,.628319,v,d,sorted(v))
    for key in fast:np.testing.assert_allclose(fast[key],direct[key],atol=1e-12)
    assert nonlinear_power(v,fast,w)['absolute_advection_power']<1e-11
    with pytest.raises(ValueError):fft_advection(r,.628319,v,d,sorted(v))


def test_real_split_stiff_evolution_matches_independent_adaptive_ode():
    state=[(m,q*10) for m,q in modes(1,3,64)];original=[q.copy() for m,q in state]
    result=evolve_refined(state,end_time=.02)
    exact,_=reference_solution(state,.02,2e-12,2e-14)
    assert np.linalg.norm(np.concatenate([q for m,q in result.modes])-exact)<1e-8
    assert result.report['real_split'] and result.report['maximum_energy_budget_residual']<1e-9
    for (_,q),old in zip(state,original):np.testing.assert_array_equal(q,old)


def test_nonlinear_viscous_bessel_vortex_velocity_and_pressure():
    m=refined_mode(0,0,.628319,order=12);root=jn_zeros(1,1)[0];u=np.zeros((3,len(m.radius)),complex);u[1]=jv(1,root*m.radius)
    result=evolve_refined([(m,m.fit(u))],end_time=.02);q=result.modes[0][1];points=np.array([0.,.2,.5,.8,1.]);amplitude=np.exp(-.05*root**2*.02)
    np.testing.assert_allclose(m.evaluate(q,points)[1],amplitude*jv(1,root*points),atol=1e-9)
    def g(r):return jv(1,root*r)**2/r if r else 0.
    gauge=quad(lambda r:g(r)*(1-r*r),0,1,epsabs=1e-13)[0]
    exact=amplitude**2*np.array([quad(g,0,r,epsabs=1e-13)[0]-gauge for r in points])
    pressure,_=result.pressure[(0,0)].evaluate(points)
    np.testing.assert_allclose(pressure,exact,atol=1e-9)
    assert endpoint_momentum(result,radius=np.linspace(.005,1,101))['maximum_wall_residual']<1e-8


def test_continuation_restarts_within_tolerance_not_claimed_bit_exact():
    state=[(m,q*10) for m,q in modes(1,3,64)]
    whole=evolve_refined(state,end_time=.02,rtol=1e-11,atol=1e-14)
    prefix=evolve_refined(state,end_time=.01,rtol=1e-11,atol=1e-14)
    replay=evolve_refined(prefix.modes,start_time=.01,end_time=.02,rtol=1e-11,atol=1e-14)
    m=state[0][0];difference=compare_refined(whole.modes,replay.modes,m.radius,m.weights,first_time=.02,second_time=.02)
    assert difference['velocity_l2']<1e-8


@pytest.mark.parametrize('change',[dict(end_time=0),dict(viscosity=-1),dict(rtol=0),dict(atol=float('nan'))])
def test_invalid_reference_policy_rejected(change):
    with pytest.raises(ValueError):evolve_refined(modes(0,3,64),**change)


def test_endpoint_missing_derivative_cannot_silently_pass():
    result=evolve_refined(modes(0,3,64),end_time=.001)
    bad=RefinedResult(result.modes,[],result.pressure,result.report)
    with pytest.raises(ValueError):endpoint_momentum(bad,radius=np.linspace(.005,1,31))


def test_expanded_comparison_includes_missing_velocity_modes():
    mean=regular_mode(0,0,.628319);a=(mean,np.zeros(mean.dimension))
    other=regular_mode(0,1,.628319);u=np.zeros((3,len(other.radius)),complex);u[2]=other.radius*(1-other.radius**2)
    result=compare_refined([a],[a,(other,other.fit(u))],mean.radius,mean.weights,first_time=0,second_time=0)
    assert result['velocity_l2']==pytest.approx(np.sqrt(1/12),abs=1e-12) and result['common_l2']<1e-12
