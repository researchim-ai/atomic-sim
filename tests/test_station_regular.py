import copy,json
import numpy as np
import pytest
from scipy.linalg import expm,solve
from scipy.special import jv,jn_zeros
from atomic_model_station_regular import regular_mode,StokesSession
from atomic_model_multiphysics.provenance import seal,unseal


@pytest.mark.parametrize('k,m',[(0,0),(2,0),(0,1),(3,1),(-2,2),(3,4),(1,6)])
def test_regular_space_has_no_slip_and_analytic_divergence(k,m):
    model=regular_mode(k,m,.628319,order=5)
    rng=np.random.default_rng(71);q=.01*(rng.normal(size=model.dimension)+1j*rng.normal(size=model.dimension))
    assert np.max(abs(model.evaluate(q,[1.])))<1e-12
    r=np.linspace(0,1,101)
    assert np.max(abs(model.analytic_divergence(q,r)))<1e-10
    np.testing.assert_allclose(model.mass,np.eye(model.dimension),atol=1e-11)
    assert min(np.linalg.eigvalsh(model.stiffness))>0


@pytest.mark.parametrize('m',[0,1,2,4])
def test_spin_components_obey_cartesian_axis_powers(m):
    model=regular_mode(2,m,.628319,order=4);q=np.linspace(.001,.01,model.dimension)
    r=np.array([0.,1e-6,2e-6]);u=model.evaluate(q,r)
    spins=model.spin_components(q,r)
    for component,power in zip(spins,[m+1,abs(m-1),m]):
        if power:assert abs(component[0])<1e-14
        ratio=component[1:]/r[1:]**power
        np.testing.assert_allclose(ratio[0],ratio[1],rtol=1e-5,atol=1e-7)


def test_continuity_against_independent_finite_difference_of_velocity():
    model=regular_mode(2,2,.628319,order=5);q=np.arange(model.dimension)*.001
    r=np.linspace(.05,.95,19);h=1e-4
    u=model.evaluate(q,r)
    derivative=(-model.evaluate(q,r+2*h)[0]+8*model.evaluate(q,r+h)[0]-8*model.evaluate(q,r-h)[0]+model.evaluate(q,r-2*h)[0])/(12*h)
    div=derivative+u[0]/r+1j*model.m*u[1]/r+1j*model.alpha*model.k*u[2]
    assert max(abs(div))<1e-8


def test_cylindrical_swirl_against_independent_bessel_diffusion():
    root=jn_zeros(1,1)[0];errors=[]
    for order in (3,5,7,9):
        model=regular_mode(0,0,.628319,order=order)
        field=np.zeros((3,len(model.radius)),complex);field[1]=jv(1,root*model.radius)
        q=model.fit(field);end=expm(-.02*solve(model.mass,model.stiffness,assume_a='pos'))@q
        errors.append(float(np.sqrt(np.sum(model.weights*abs(model.evaluate(end,model.radius)-field*np.exp(-.02*root**2))**2))))
    assert all(b<a/100 for a,b in zip(errors,errors[1:])) and errors[-1]<1e-10


def test_time_order_energy_decay_and_exact_native_restart():
    model=regular_mode(2,1,.628319,order=5);q=np.arange(model.dimension)*.001
    exact=expm(-.02*solve(model.mass,model.stiffness,assume_a='pos'))@q;errors=[]
    for dt in (.01,.005,.0025):
        session=StokesSession(model,q);last=.5*np.real(np.vdot(q,model.mass@q))
        for i in range(round(.2/dt)):
            receipt=session.advance(dt);assert receipt['kinetic_energy']<=last+1e-14;last=receipt['kinetic_energy']
        errors.append(np.linalg.norm(session.q-exact));assert abs(receipt['total_energy_residual'])<1e-12
    assert min(np.log2(np.array(errors[:-1])/errors[1:]))>1.99
    a=StokesSession(model,q)
    for i in range(10):a.advance(.005)
    b=StokesSession.restore(json.loads(json.dumps(a.checkpoint(),sort_keys=True)))
    for i in range(10):assert a.advance(.005)==b.advance(.005)
    assert a.checkpoint()==b.checkpoint()


def test_forced_step_has_joint_constraint_reaction_and_work_budget():
    model=regular_mode(3,2,.628319,order=5);session=StokesSession(model,np.zeros(model.dimension))
    rng=np.random.default_rng(4);force=rng.normal(size=3*model.order)+1j*rng.normal(size=3*model.order)
    receipt=session.advance(.01,force)
    assert receipt['work']>0 and receipt['dissipation']>0
    assert abs(receipt['total_energy_residual'])<1e-12
    assert receipt['constraint_reaction_relative_residual']<1e-10
    assert abs(receipt['constraint_reaction_work'])<1e-12
    assert receipt['wall_maximum']<1e-12 and receipt['analytic_divergence_maximum']<1e-10
    assert not receipt['physical_pressure_reconstructed'] and not receipt['station_usable']


@pytest.mark.parametrize('change',[dict(k=True),dict(k=17),dict(m=-1),dict(m=7),dict(alpha=0),dict(alpha=float('nan')),dict(order=1),dict(order=11),dict(quadrature=8)])
def test_invalid_regular_space_is_rejected(change):
    args=dict(k=2,m=1,alpha=.628319);args.update(change)
    with pytest.raises(ValueError):regular_mode(**args)


@pytest.mark.parametrize('dt',[0,-1,True,float('nan'),float('inf'),1e4])
def test_bad_step_is_atomic(dt):
    model=regular_mode(2,1,.628319);session=StokesSession(model,np.zeros(model.dimension));before=session.checkpoint()
    with pytest.raises(ValueError):session.advance(dt)
    assert session.checkpoint()==before


@pytest.mark.parametrize('kind',['short_force','nan_force','q','time','ledger','identity','integrator','shape'])
def test_invalid_force_and_tampered_checkpoints(kind):
    model=regular_mode(2,1,.628319);s=StokesSession(model,np.zeros(model.dimension));before=s.checkpoint()
    if kind in ('short_force','nan_force'):
        force=np.zeros(3*model.order-1 if kind=='short_force' else 3*model.order)
        if kind=='nan_force':force[0]=np.nan
        with pytest.raises(ValueError):s.advance(.01,force)
        assert s.checkpoint()==before;return
    p=unseal(before)
    if kind=='q':p['q_real'][0]=1
    if kind=='time':p['time']=-1
    if kind=='ledger':p['dissipation']=1
    if kind=='identity':p['fingerprint']='0'*64
    if kind=='integrator':p['integrator_sha256']='0'*64
    if kind=='shape':p['q_imag']=[]
    with pytest.raises(ValueError):StokesSession.restore(seal(p))


def test_stable_spin_evaluation_agrees_with_cartesian_components():
    model=regular_mode(2,2,.628319);q=np.arange(model.dimension)*.001;r=np.linspace(.01,1,30)
    u=model.evaluate(q,r);spins=model.spin_components(q,r)
    np.testing.assert_allclose(spins,np.array([u[0]+1j*u[1],u[0]-1j*u[1],u[2]]),atol=1e-12)


def test_missing_checkpoint_field_is_refused():
    model=regular_mode(2,1,.628319);s=StokesSession(model,np.zeros(model.dimension));p=unseal(s.checkpoint());del p['work']
    with pytest.raises(ValueError):StokesSession.restore(seal(p))


def test_unrepresentable_clock_increment_rolls_back_after_candidate_solve():
    model=regular_mode(2,1,.628319);s=StokesSession(model,np.zeros(model.dimension));p=unseal(s.checkpoint());p['time']=1e20
    s=StokesSession.restore(seal(p));before=s.checkpoint()
    with pytest.raises(ValueError,match='clock'):s.advance(.01)
    assert s.checkpoint()==before
