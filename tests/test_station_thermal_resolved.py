import numpy as np
import pytest
from scipy.special import jv, jn_zeros
from types import SimpleNamespace

from atomic_model_station_thermal import scalar_mode, ThermalSystem, ThermalResult, evolve_thermal
from atomic_model_station_refined import refined_mode
from atomic_model_station_thermal_resolved import (
    resolved_scalar_mode, resolved_velocity_mode, ResolvedThermalSystem, resolve_thermal,
    audit_resolved, accept_resolved, ThermalQualityError, initial_wall_compatibility, compare_thermal,
)


def mean_system(order=8):
    m=resolved_velocity_mode(0,0,.628319,order=order,quadrature=128)
    s=resolved_scalar_mode(0,0,.628319,order=order,quadrature=128)
    return ResolvedThermalSystem([(m,np.zeros(m.dimension))],[(s,s.fit(.1*jv(0,jn_zeros(0,1)[0]*s.radius)))])


def test_low_order_factories_preserve_prior_fingerprints():
    a=resolved_scalar_mode(1,1,.628319,order=8,quadrature=128)
    b=scalar_mode(1,1,.628319,order=8,quadrature=128)
    c=resolved_velocity_mode(1,1,.628319,order=12,quadrature=128)
    d=refined_mode(1,1,.628319,order=12,quadrature=128)
    assert a.fingerprint==b.fingerprint and c.fingerprint==d.fingerprint
    np.testing.assert_array_equal(a.stiffness,b.stiffness)
    np.testing.assert_array_equal(c.transform,d.transform)


@pytest.mark.parametrize('m,p',[(0,28),(1,32),(3,28)])
def test_extended_velocity_space_keeps_axis_wall_and_coercivity(m,p):
    model=resolved_velocity_mode(1,m,.628319,order=p,quadrature=128)
    q=np.arange(model.dimension)*1e-5
    assert np.max(abs(model.analytic_divergence(q,np.linspace(0,1,101))))<1e-8
    assert np.max(abs(model.evaluate(q,[1.])))<1e-10
    assert min(np.linalg.eigvalsh(model.stiffness))>0
    np.testing.assert_allclose(model.mass,np.eye(model.dimension),atol=1e-9)


@pytest.mark.parametrize('m,p',[(0,40),(2,48),(4,24)])
def test_extended_scalar_diffusion_integrates_by_parts(m,p):
    s=resolved_scalar_mode(1,m,.628319,order=p,quadrature=128)
    q=np.arange(p)*1e-6
    assert abs(s.evaluate(q,[1.])[0])<1e-12
    assert np.all(np.isfinite(np.concatenate(s.jets(q,[0.,.5,1.]))))
    lap=np.column_stack([s.jets(np.eye(p)[i],s.radius)[2] for i in range(p)])
    np.testing.assert_allclose(s.basis.conj().T@(s.weights[:,None]*lap),-s.stiffness,atol=1e-5)


@pytest.mark.parametrize('change',[dict(order=49),dict(order=True),dict(k=6),dict(m=6),dict(quadrature=64)])
def test_high_scalar_domain_and_resolution_are_bounded(change):
    args=dict(k=1,m=1,alpha=.628319,order=40,quadrature=128);args.update(change)
    with pytest.raises(ValueError):resolved_scalar_mode(**args)


def test_velocity_conditioning_refusal_is_preserved():
    with pytest.raises(ValueError,match='conditioned'):
        resolved_velocity_mode(1,4,.628319,order=32,quadrature=128)
    with pytest.raises(ValueError):resolved_velocity_mode(1,1,.628319,order=33,quadrature=128)
    with pytest.raises(ValueError):refined_mode(1,1,.628319,order=28,quadrature=128)
    with pytest.raises(ValueError):scalar_mode(1,1,.628319,order=40,quadrature=128)


def test_extended_scalar_conditioning_gate_is_not_relaxed(monkeypatch):
    monkeypatch.setattr(np.linalg,'cond',lambda a:1e11)
    with pytest.raises(ValueError,match='conditioned'):
        resolved_scalar_mode(0,0,.628319,order=40,quadrature=128)


def test_old_coupled_dynamics_remain_byte_identical_at_shared_orders():
    system=mean_system(8)
    old=ThermalSystem(system.velocity,system.temperature)
    a=resolve_thermal(system,end_time=.02)
    b=evolve_thermal(old,end_time=.02)
    assert a.report==b.report
    for pa,pb in ((a.velocity,b.velocity),(a.temperature,b.temperature)):
        for (_,u),(_,v) in zip(pa,pb):np.testing.assert_array_equal(u,v)


def test_initial_wall_incompatibility_is_measured_without_projection():
    system=mean_system(8);s,_=system.temperature[0]
    system=ResolvedThermalSystem(system.velocity,[(s,s.fit(.1*(1-s.radius*s.radius)))])
    before=system.initial.copy();proof=initial_wall_compatibility(system)
    assert proof['maximum_temperature_rate_mismatch']==pytest.approx(.012,abs=1e-10)
    assert not proof['compatible'] and not proof['state_modified']
    np.testing.assert_array_equal(system.initial,before)


def test_quality_failure_preserves_candidate_and_reports_failed_gates():
    system=mean_system(3);before=system.initial.copy();result=resolve_thermal(system)
    u=result.temperature[0][1].copy()
    with pytest.raises(ThermalQualityError) as error:accept_resolved(system,result)
    assert not error.value.receipt['physical_heat_passed']
    assert not error.value.receipt['retained_wall_passed']
    np.testing.assert_array_equal(system.initial,before)
    np.testing.assert_array_equal(result.temperature[0][1],u)


def test_quality_pass_remains_explicitly_sampled_and_retained():
    system=mean_system(8);result=resolve_thermal(system)
    proof=accept_resolved(system,result)
    assert proof['passed'] and proof['sampled_radii']==129
    assert not proof['spatial_refinement_checked'] and not proof['general_strong_pde_qualified']
    assert not proof['station_usable'] and not proof['validated']


def test_solver_failure_leaves_copied_initial_state_unchanged(monkeypatch):
    from atomic_model_station_thermal_resolved import system as module
    system=mean_system();before=system.initial.copy()
    monkeypatch.setattr(module,'solve_ivp',lambda *args,**kwargs:SimpleNamespace(success=False))
    with pytest.raises(ValueError,match='integration failed'):resolve_thermal(system)
    np.testing.assert_array_equal(system.initial,before)


def test_absent_temperature_modes_are_included_in_l2_distance():
    m=resolved_velocity_mode(0,0,.628319,order=6,quadrature=128)
    s=resolved_scalar_mode(0,0,.628319,order=6,quadrature=128)
    v=resolved_velocity_mode(0,1,.628319,order=6,quadrature=128)
    t=resolved_scalar_mode(0,1,.628319,order=6,quadrature=128)
    a=ThermalResult([(m,np.zeros(m.dimension))],[(s,np.zeros(s.dimension))],[],[],{},dict(time=0.))
    b=ThermalResult(a.velocity+[(v,np.zeros(v.dimension))],a.temperature+[(t,t.fit(t.radius*(1-t.radius**2)))],[],[],{},dict(time=0.))
    proof=compare_thermal(a,b,s.radius,s.weights)
    assert proof['velocity_l2']==0
    assert proof['temperature_l2']==pytest.approx(np.sqrt(1/12),abs=1e-12)
    assert proof['temperature_common_l2']==0 and proof['absent_modes_included']


def test_audit_rejects_other_equation_sources():
    system=mean_system();result=resolve_thermal(system,end_time=.01)
    other=ResolvedThermalSystem(system.velocity,system.temperature,sources={(0,0):[.01]})
    with pytest.raises(ValueError,match='sources'):audit_resolved(other,result,radius=np.linspace(.01,1,31))


def test_expanded_scalar_fft_matches_pairs_at_full_121_mode_budget():
    from atomic_model_station_refined.products import signed_modes
    from atomic_model_station_thermal_resolved.products import scalar_products
    rng=np.random.default_rng(81);state=[];theta={};radial={}
    for m in range(6):
        for k in (range(6) if m==0 else range(-5,6)):
            v=resolved_velocity_mode(k,m,.628319,order=3,quadrature=128)
            q=.0001*(rng.normal(size=v.dimension)+1j*rng.normal(size=v.dimension))
            if k==m==0:q=np.zeros(v.dimension,complex)
            state.append((v,q))
            s=resolved_scalar_mode(k,m,.628319,order=3,quadrature=128)
            t=.001*(rng.normal(size=3)+1j*rng.normal(size=3))
            if k==m==0:t=t.real.astype(complex)
            key=(k,m);theta[key],radial[key],_=s.jets(t,s.radius)
            if key!=(0,0):theta[(-k,-m)],radial[(-k,-m)]=theta[key].conj(),radial[key].conj()
    r=state[0][0].radius;w=state[0][0].weights
    velocity,_=signed_modes(state,r,.628319)
    assert len(velocity)==121
    a=scalar_products(r,.628319,velocity,theta,radial)
    b=scalar_products(r,.628319,velocity,theta,radial,method='direct')
    for key in a:np.testing.assert_allclose(a[key],b[key],atol=1e-12)
    power=sum(np.real(np.vdot(theta[key],w*a[key])) for key in a)
    assert abs(power)<1e-12
