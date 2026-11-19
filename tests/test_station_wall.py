import numpy as np
import pytest
from atomic_model_station_regular import regular_mode
from atomic_model_station_stokes_pressure import PressureStokesSession,PressureField
from atomic_model_station_wall import velocity_jets,audit_momentum


def axial_force(r,m,nu):
    f=np.zeros((3,len(r)),complex);f[0]=(m+2)*r**(m+1);f[1]=1j*m*r**(m+1)
    f[2]=4*nu*(m+1)*r**m;return f


@pytest.mark.parametrize('m',[0,1,2,4])
def test_vector_laplacian_against_independent_cylindrical_finite_differences(m):
    model=regular_mode(2,m,.628319,order=5);q=np.arange(model.dimension)*.001
    r=np.linspace(.1,.9,11);h=2e-4;u,dr,lap=velocity_jets(model,q,r)
    first=(-model.evaluate(q,r+2*h)+8*model.evaluate(q,r+h)-8*model.evaluate(q,r-h)+model.evaluate(q,r-2*h))/(12*h)
    second=(-model.evaluate(q,r+2*h)+16*model.evaluate(q,r+h)-30*model.evaluate(q,r)+16*model.evaluate(q,r-h)-model.evaluate(q,r-2*h))/(12*h*h)
    direct=second+first/r-(model.alpha*model.k)**2*u
    direct[0]-=(m*m+1)*u[0]/r**2+2j*m*u[1]/r**2
    direct[1]-=(m*m+1)*u[1]/r**2-2j*m*u[0]/r**2
    direct[2]-=m*m*u[2]/r**2
    np.testing.assert_allclose(lap,direct,atol=2e-7)
    np.testing.assert_allclose(dr,first,atol=1e-8)
    np.testing.assert_allclose(u,model.evaluate(q,r),atol=1e-13)


@pytest.mark.parametrize('m',[0,1,2,4])
def test_stationary_axial_solution_pressure_and_wall_traction(m):
    model=regular_mode(0,m,.628319,order=6);nu=.1
    u=np.zeros((3,len(model.radius)),complex);u[2]=model.radius**m*(1-model.radius**2)
    q=model.fit(u);session=PressureStokesSession(model,q)
    result,p=session.advance(.005,model.load(axial_force(model.radius,m,nu)))
    r=np.linspace(0,1,51);value,derivative,lap=velocity_jets(model,session.q,r)
    np.testing.assert_allclose(lap[2],-4*(m+1)*r**m,atol=1e-8)
    a=audit_momentum(model,q,session.q,p,dt=.005,viscosity=nu,radius=r,force=axial_force(r,m,nu))
    assert a['sampled_momentum_passed'] and a['wall_momentum_passed']
    np.testing.assert_allclose(a['wall_traction']['real'],[-.5 if m==0 else -1.,0,-2*nu],atol=1e-8)
    assert max(abs(np.array(a['wall_traction']['imag'])))<1e-8


def test_small_weak_residual_does_not_imply_strong_wall_acceptance():
    model=regular_mode(2,1,.628319,order=4);q=np.arange(model.dimension)*.001
    session=PressureStokesSession(model,q);receipt,p=session.advance(.005)
    a=audit_momentum(model,q,session.q,p,dt=.005,viscosity=.1,radius=np.linspace(0,1,31),force=np.zeros((3,31)))
    assert receipt['pressure']['weak_momentum_relative_residual']<1e-9
    assert a['wall_momentum_residual']>1e-5 and not a['wall_momentum_passed']
    assert not a['station_usable'] and not a['nonlinear_advection_included']


def test_axis_limits_are_finite_for_all_spin_orders():
    for m in (0,1,2,4):
        model=regular_mode(2,m,.628319);q=np.arange(model.dimension)*.001
        for value in velocity_jets(model,q,[0.,1e-8,1.]):assert np.all(np.isfinite(value))


@pytest.mark.parametrize('kind',['force_shape','force_nan','missing_wall','radius_order','wrong_pressure','dt','viscosity','tolerance'])
def test_invalid_or_unidentified_strong_momentum_problem_is_rejected(kind):
    model=regular_mode(2,1,.628319);q=np.zeros(model.dimension);p=PressureField(model,np.zeros(model.order+1))
    r=np.linspace(0,1,11);f=np.zeros((3,len(r)));args=dict(dt=.005,viscosity=.1,radius=r,force=f)
    if kind=='force_shape':args['force']=np.zeros(3*model.order)
    if kind=='force_nan':f[0,0]=np.nan
    if kind=='missing_wall':args['radius']=r*.99
    if kind=='radius_order':args['radius']=r[::-1]
    if kind=='wrong_pressure':p=PressureField(regular_mode(3,1,.628319),np.zeros(model.order+1))
    if kind=='dt':args['dt']=0
    if kind=='viscosity':args['viscosity']=-1
    if kind=='tolerance':args['tolerance']=np.nan
    with pytest.raises(ValueError):audit_momentum(model,q,q,p,**args)


def test_checked_session_rejects_candidate_and_preserves_every_state_field():
    from atomic_model_station_wall import WallCheckedSession,WallMomentumError
    model=regular_mode(2,1,.628319,order=4);q=np.arange(model.dimension)*.001
    session=WallCheckedSession(model,q,radius=np.linspace(0,1,31));before=session.checkpoint()
    with pytest.raises(WallMomentumError) as error:session.advance(.005)
    assert not error.value.audit['wall_momentum_passed']
    assert session.checkpoint()==before


def test_checked_stationary_solution_has_exact_restart():
    from atomic_model_station_wall import WallCheckedSession
    model=regular_mode(0,2,.628319);u=np.zeros((3,len(model.radius)),complex);u[2]=model.radius**2*(1-model.radius**2)
    a=WallCheckedSession(model,model.fit(u),radius=np.linspace(0,1,51))
    force=lambda r,time:axial_force(r,2,.1)
    a.advance(.005,force);b=WallCheckedSession.restore(a.checkpoint())
    ra,pa=a.advance(.005,force);rb,pb=b.advance(.005,force)
    assert ra==rb;np.testing.assert_array_equal(pa.coefficients,pb.coefficients)
    assert a.checkpoint()==b.checkpoint()


def test_checked_force_must_be_explicit_and_finite_on_both_grids():
    from atomic_model_station_wall import WallCheckedSession
    model=regular_mode(0,0,.628319);a=WallCheckedSession(model,np.zeros(model.dimension),radius=np.linspace(0,1,31));before=a.checkpoint()
    with pytest.raises(ValueError):a.advance(.005,np.zeros(3*model.order))
    def bad(r,time):
        return np.full((3,len(r)),np.nan) if len(r)==31 else np.zeros((3,len(r)))
    with pytest.raises(ValueError):a.advance(.005,bad)
    assert a.checkpoint()==before


def test_wall_gate_policy_cannot_be_silently_changed_at_restart():
    from atomic_model_station_wall import WallCheckedSession
    from atomic_model_multiphysics.provenance import seal,unseal
    model=regular_mode(0,0,.628319);a=WallCheckedSession(model,np.zeros(model.dimension),radius=[0,1]);p=unseal(a.checkpoint())
    p['policy']['operators']='0'*64
    with pytest.raises(ValueError):WallCheckedSession.restore(seal(p))
