import json
import numpy as np
import pytest
from atomic_model_multiphysics.provenance import seal,unseal
from atomic_model_station_regular import regular_mode
from atomic_model_station_nonlinear import NonlinearSession,NonlinearConvergenceError
from scripts.qualify_station_advection import modes


def initial(scale=1.):return [(m,q*scale) for m,q in modes(1,3,32)]


def test_inviscid_energy_pressure_and_exact_restart():
    s=NonlinearSession(initial(),viscosity=0);energy=s.initial_energy
    for i in range(4):
        r,p=s.advance(.01)
        assert r['nonlinear_relative_residual']<=s.tolerance and r['nonlinear_pressure_recovered']
        assert abs(r['kinetic_energy']-energy)<1e-12
    restored=NonlinearSession.restore(json.loads(json.dumps(s.checkpoint(),sort_keys=True)))
    for i in range(3):
        a,pa=s.advance(.01);b,pb=restored.advance(.01)
        assert a==b
        for key in pa:np.testing.assert_array_equal(pa[key].coefficients,pb[key].coefficients)
    assert s.checkpoint()==restored.checkpoint()


def test_viscous_decay_and_method_agreement():
    a=NonlinearSession(initial(10),method='fft');b=NonlinearSession(initial(10),method='convolution');last=a.initial_energy
    for i in range(4):
        r,_=a.advance(.005);s,_=b.advance(.005)
        assert r['kinetic_energy']<last;last=r['kinetic_energy']
        assert r['viscous_dissipation']>0 and abs(r['total_energy_residual'])<1e-11
        for (ma,qa),(mb,qb) in zip(a.modes,b.modes):np.testing.assert_allclose(qa,qb,atol=1e-11)


def test_stationary_centrifugal_equilibrium_has_analytic_nonlinear_pressure():
    model=regular_mode(0,0,.628319,order=6);r=model.radius;u=np.zeros((3,len(r)),complex);u[1]=r*(1-r*r)
    q=model.fit(u);session=NonlinearSession([(model,q)],viscosity=0)
    receipt,fields=session.advance(.01)
    np.testing.assert_array_equal(session.modes[0][1],q)
    p,g=fields[(0,0)].evaluate(r)
    np.testing.assert_allclose(p,r*r/2-r**4/2+r**6/6-1/8,atol=1e-11)
    assert receipt['iterations']==1


def test_external_work_is_separate_from_advection_and_dissipation():
    state=initial();s=NonlinearSession([(m,np.zeros_like(q)) for m,q in state]);force={}
    for m,q in state:
        f=np.zeros((3,len(m.radius)),complex)
        if (m.k,m.m) in [(0,0),(1,1)]:f[2]=.1*m.radius**m.m*(1-m.radius**2)
        force[(m.k,m.m)]=m.load(f)
    r,_=s.advance(.01,force)
    assert r['external_work']>0 and r['viscous_dissipation']>0
    assert abs(r['step_energy_residual'])<1e-11 and abs(r['advection_power'])<1e-11


def test_iteration_budget_exhaustion_is_atomic_and_diagnostic():
    s=NonlinearSession(initial(10),max_iterations=1);before=s.checkpoint()
    with pytest.raises(NonlinearConvergenceError) as error:s.advance(.01)
    assert error.value.iterations==1 and error.value.residual>s.tolerance
    assert s.checkpoint()==before


def test_failed_pressure_recovery_does_not_commit_converged_velocity(monkeypatch):
    import atomic_model_station_nonlinear.session as module
    s=NonlinearSession(initial());before=s.checkpoint()
    def fail(*args,**kwargs):raise ValueError('injected pressure failure')
    monkeypatch.setattr(module,'recover_step_pressure',fail)
    with pytest.raises(ValueError,match='injected'):s.advance(.01)
    assert s.checkpoint()==before


@pytest.mark.parametrize('dt',[0,-1,True,float('nan'),float('inf'),1e4])
def test_bad_step_is_atomic(dt):
    s=NonlinearSession(initial());before=s.checkpoint()
    with pytest.raises(ValueError):s.advance(dt)
    assert s.checkpoint()==before


@pytest.mark.parametrize('kind',['missing','short','nan'])
def test_incomplete_or_nonfinite_force_is_rejected(kind):
    s=NonlinearSession(initial());before=s.checkpoint();force={(m.k,m.m):np.zeros(3*m.order,complex) for m,q in s.modes}
    key=next(iter(force))
    if kind=='missing':force.pop(key)
    if kind=='short':force[key]=force[key][:-1]
    if kind=='nan':force[key][0]=np.nan
    with pytest.raises(ValueError):s.advance(.01,force)
    assert s.checkpoint()==before


@pytest.mark.parametrize('change',[dict(viscosity=-1),dict(tolerance=0),dict(tolerance=1e-8),dict(max_iterations=True),dict(max_iterations=0),dict(method='alias')])
def test_invalid_policy_is_refused(change):
    with pytest.raises(ValueError):NonlinearSession(initial(),**change)


@pytest.mark.parametrize('kind',['policy','fingerprint','ledger','shape','schema'])
def test_corrupted_restart_is_refused(kind):
    s=NonlinearSession(initial());p=unseal(s.checkpoint())
    if kind=='policy':p['policy']['advection']='0'*64
    if kind=='fingerprint':p['modes'][0]['fingerprint']='0'*64
    if kind=='ledger':p['dissipation']=1
    if kind=='shape':p['modes'][0]['q_imag']=[]
    if kind=='schema':del p['work']
    with pytest.raises(ValueError):NonlinearSession.restore(seal(p))


def test_mixed_quadratures_duplicate_and_missing_mean_are_refused():
    state=initial()
    with pytest.raises(ValueError):NonlinearSession([state[0],state[0]])
    with pytest.raises(ValueError):NonlinearSession([(m,q) for m,q in state if (m.k,m.m)!=(0,0)])
    other=regular_mode(2,1,.628319,order=3,quadrature=64)
    with pytest.raises(ValueError):NonlinearSession(state+[(other,np.zeros(other.dimension))])


def test_unrepresentable_time_increment_does_not_mutate_state():
    s=NonlinearSession(initial());p=unseal(s.checkpoint());p['time']=1e20;s=NonlinearSession.restore(seal(p));before=s.checkpoint()
    with pytest.raises(ValueError,match='clock'):s.advance(.01)
    assert s.checkpoint()==before
