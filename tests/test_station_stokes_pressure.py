import json
import numpy as np
import pytest
from scipy.linalg import solve
from atomic_model_multiphysics.provenance import seal,unseal
from atomic_model_station_regular import regular_mode,StokesSession
from atomic_model_station_stokes_pressure import pressure_basis,weak_pressure_map,recover_pressure,PressureStokesSession,PressureField


@pytest.mark.parametrize('k,m',[(0,0),(2,0),(0,1),(2,1),(-2,2),(2,4)])
def test_manufactured_pressure_gradient_load_recovery_and_stationary_velocity(k,m):
    model=regular_mode(k,m,.628319,order=5);p,g=pressure_basis(model,model.radius)
    coefficients=np.arange(6)*.01+.02j
    if k==m==0:coefficients[0]=0
    force=model.load(np.einsum('cni,i->cn',g,coefficients))
    field,proof=recover_pressure(model,force)
    np.testing.assert_allclose(field.coefficients,coefficients,atol=1e-10)
    assert proof['weak_momentum_relative_residual']<1e-10
    session=PressureStokesSession(model,np.zeros(model.dimension));receipt,stepped=session.advance(.005,force)
    assert np.linalg.norm(session.q)<1e-12
    np.testing.assert_allclose(stepped.coefficients,coefficients,atol=1e-10)
    assert receipt['physical_pressure_reconstructed'] and not proof['strong_wall_momentum_qualified']


@pytest.mark.parametrize('k,m',[(0,0),(2,1),(-2,2),(2,4)])
def test_recovered_step_matches_independent_monolithic_velocity_pressure_solve(k,m):
    model=regular_mode(k,m,.628319,order=5);dt=.005;nu=.1
    rng=np.random.default_rng(73);q=rng.normal(size=model.dimension)*.001
    force=rng.normal(size=15)*.001+1j*rng.normal(size=15)*.001
    session=PressureStokesSession(model,q,viscosity=nu);receipt,field=session.advance(dt,force)
    # Independently assemble the physical gradient load by quadrature, not C*H.
    _,gradient=pressure_basis(model,model.radius)
    g=np.column_stack([model.load(gradient[:,:,j]) for j in range(model.order+1)])
    start=1 if k==m==0 else 0;gp=g[:,start:]
    a=model.mass_full+.5*dt*nu*model.stiffness_full
    system=np.block([[a,dt*gp],[dt*gp.conj().T,np.zeros((gp.shape[1],gp.shape[1]))]])
    rhs=np.r_[(model.mass_full-.5*dt*nu*model.stiffness_full)@model.coefficients(q)+dt*force,np.zeros(gp.shape[1])]
    direct=solve(system,rhs,assume_a='her')
    np.testing.assert_allclose(model.coefficients(session.q),direct[:15],atol=1e-10)
    np.testing.assert_allclose(field.coefficients[start:],direct[15:],atol=1e-9)
    assert receipt['pressure']['weak_momentum_relative_residual']<1e-9


def test_pressure_sign_gauge_and_boundary_normal_derivative_independently():
    model=regular_mode(0,0,.628319,order=5);r=model.radius
    # p=r^4-1/3 has zero radial-area mean; its physical gradient is4*r^3.
    force=np.zeros((3,len(r)),complex);force[0]=4*r**3
    field,proof=recover_pressure(model,model.load(force))
    p,g=field.evaluate(np.array([0.,.2,.7,1.]))
    np.testing.assert_allclose(p,np.array([0.,.2,.7,1.])**4-1/3,atol=1e-11)
    np.testing.assert_allclose(g[0],4*np.array([0.,.2,.7,1.])**3,atol=1e-10)
    assert proof['mean_gauge_absolute_residual']<1e-12
    assert proof['wall_gradient_real'][0]==pytest.approx(4.,abs=1e-10)


def test_scalar_pressure_axis_regularity_and_exact_weak_integration_by_parts():
    model=regular_mode(2,3,.628319);r=model.radius;p,g=pressure_basis(model,r)
    direct=np.column_stack([model.load(g[:,:,j]) for j in range(model.order+1)])
    np.testing.assert_allclose(direct,weak_pressure_map(model),atol=1e-12)
    coefficients=np.arange(7)*.01;field=PressureField(model,coefficients)
    point=np.array([0.,1e-6,2e-6]);value,gradient=field.evaluate(point)
    assert value[0]==0 and np.all(gradient[:,0]==0)
    np.testing.assert_allclose(value[1]/point[1]**3,value[2]/point[2]**3,rtol=1e-7,atol=1e-8)


def test_unsupported_reaction_is_refused_instead_of_labeled_pressure():
    model=regular_mode(2,1,.628319);reaction=model.mass_full@model.coefficients(np.ones(model.dimension))
    with pytest.raises(ValueError,match='represented'):recover_pressure(model,reaction)


def test_wrapper_restart_is_exact_and_base_v72_dynamics_unchanged():
    model=regular_mode(2,1,.628319);q=np.arange(model.dimension)*.001
    a=PressureStokesSession(model,q);base=StokesSession(model,q)
    for i in range(5):
        result,field=a.advance(.005);base.advance(.005)
        np.testing.assert_array_equal(a.q,base.q)
    b=PressureStokesSession.restore(json.loads(json.dumps(a.checkpoint(),sort_keys=True)))
    for i in range(5):
        ra,pa=a.advance(.005);rb,pb=b.advance(.005)
        assert ra==rb;np.testing.assert_array_equal(pa.coefficients,pb.coefficients)
    assert a.checkpoint()==b.checkpoint()


def test_recovery_failure_rolls_back_already_computed_candidate(monkeypatch):
    import atomic_model_station_stokes_pressure.session as module
    model=regular_mode(2,1,.628319);a=PressureStokesSession(model,np.ones(model.dimension)*.001);before=a.checkpoint()
    def fail(*args,**kwargs):raise ValueError('injected recovery failure')
    monkeypatch.setattr(module,'recover_step_pressure',fail)
    with pytest.raises(ValueError,match='injected'):a.advance(.005)
    assert a.checkpoint()==before


@pytest.mark.parametrize('kind',['shape','nan','real_radius','policy','missing'])
def test_invalid_pressure_inputs_and_checkpoint(kind):
    model=regular_mode(2,1,.628319)
    if kind in ('shape','nan'):
        reaction=np.zeros(3*model.order-1 if kind=='shape' else 3*model.order,complex)
        if kind=='nan':reaction[0]=np.nan
        with pytest.raises(ValueError):recover_pressure(model,reaction)
    elif kind=='real_radius':
        with pytest.raises(ValueError):pressure_basis(model,[-.1])
    else:
        a=PressureStokesSession(model,np.zeros(model.dimension));p=unseal(a.checkpoint())
        if kind=='policy':p['policy']['pressure']='0'*64
        else:del p['core']
        with pytest.raises(ValueError):PressureStokesSession.restore(seal(p))
