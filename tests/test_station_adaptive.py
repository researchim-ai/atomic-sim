import numpy as np
import pytest
from atomic_model_multiphysics.provenance import seal,unseal
from atomic_model_station_nonlinear import NonlinearSession
from atomic_model_station_adaptive import AdaptiveSession,AccuracyError
from scripts.qualify_station_advection import modes


def core(scale=10.):return NonlinearSession([(m,q*scale) for m,q in modes(1,3,32)],viscosity=.05)


def test_step_rejects_bad_estimates_and_commits_actual_two_half_steps():
    original=core();before=original.checkpoint();a=AdaptiveSession(original,max_step=.01)
    receipt,pressure=a.step(.01)
    assert receipt['rejected_trials']>0 and receipt['error_ratio']<=1
    assert not receipt['richardson_state_correction']
    manual=NonlinearSession.restore(before);manual.advance(receipt['actual_half_step']);manual.advance(receipt['actual_half_step'])
    assert unseal(a.checkpoint())['core']==manual.checkpoint()
    assert original.checkpoint()==before


def test_force_is_sampled_at_each_actual_midpoint():
    state=modes(0,3,32);base=NonlinearSession([(m,np.zeros_like(q)) for m,q in state],viscosity=0)
    a=AdaptiveSession(base);times=[]
    def force(time):
        times.append(time);return {(m.k,m.m):np.zeros(3*m.order,complex) for m,q in state}
    receipt,p=a.step(.0025,force)
    np.testing.assert_allclose(times,[.00125,.000625,.001875],atol=1e-16)
    assert receipt['pressure_time']==pytest.approx(.001875)
    assert a.time==pytest.approx(.0025)


def test_rejection_budget_and_minimum_step_are_atomic():
    for options in [dict(max_rejections=0),dict(min_step=.01)]:
        a=AdaptiveSession(core(),atol=1e-14,rtol=1e-12,max_step=.01,**options);before=a.checkpoint()
        with pytest.raises(AccuracyError) as error:a.step(.01)
        assert error.value.attempts>=1 and a.checkpoint()==before


def test_whole_interval_rolls_back_if_step_budget_exhausted():
    a=AdaptiveSession(core());before=a.checkpoint()
    with pytest.raises(AccuracyError,match='interval step budget'):a.advance_to(.1,max_steps=1)
    assert a.checkpoint()==before


def test_controller_restart_preserves_next_step_and_trajectory_exactly():
    a=AdaptiveSession(core());a.step(.01);b=AdaptiveSession.restore(a.checkpoint())
    for i in range(3):
        ra,pa=a.step(.01);rb,pb=b.step(.01)
        assert ra==rb
        for key in pa:np.testing.assert_array_equal(pa[key].coefficients,pb[key].coefficients)
    assert a.checkpoint()==b.checkpoint()


def test_interval_endpoint_and_energy_are_recorded_without_clock_reset():
    a=AdaptiveSession(core());r,p=a.advance_to(.005)
    assert abs(r['endpoint_difference'])<1e-15 and r['maximum_local_error_ratio']<=1
    assert abs(r['last_step']['core_last_step']['total_energy_residual'])<1e-10
    assert not r['global_error_bound_claimed']
    assert all(v['actual_half_step']==v['accepted_duration']/2 for v in r['steps'])


@pytest.mark.parametrize('change',[dict(atol=0),dict(atol=float('nan')),dict(rtol=0),dict(min_step=1e-12),dict(max_step=1e-8),dict(max_rejections=True),dict(max_rejections=-1)])
def test_invalid_controller_policy_is_rejected(change):
    with pytest.raises(ValueError):AdaptiveSession(core(),**change)


@pytest.mark.parametrize('kind',['code','next_step','counter','schema'])
def test_corrupt_controller_restart_is_refused(kind):
    a=AdaptiveSession(core());p=unseal(a.checkpoint())
    if kind=='code':p['controller_sha256']='0'*64
    if kind=='next_step':p['next_step']=1.
    if kind=='counter':p['accepted']=-1
    if kind=='schema':del p['core']
    with pytest.raises(ValueError):AdaptiveSession.restore(seal(p))


def test_invalid_force_does_not_commit_any_trial():
    a=AdaptiveSession(core());before=a.checkpoint()
    def force(time):return {key:np.full(3*m.order,np.nan) for m,q in a.modes for key in [(m.k,m.m)]}
    with pytest.raises(ValueError):a.step(.01,force)
    assert a.checkpoint()==before


def test_nonlinear_trial_failure_reduces_step_without_accepting_failed_state(monkeypatch):
    from atomic_model_station_nonlinear import NonlinearConvergenceError
    original=NonlinearSession.advance;calls=[0]
    def first_failure(self,*args,**kwargs):
        calls[0]+=1
        if calls[0]==1:raise NonlinearConvergenceError(40,1.)
        return original(self,*args,**kwargs)
    monkeypatch.setattr(NonlinearSession,'advance',first_failure)
    a=AdaptiveSession(core());row,p=a.step(.01)
    assert row['rejected_trials']>=1 and row['error_ratio']<=1


def test_bad_force_on_second_half_step_is_atomic():
    state=modes(0,3,32);a=AdaptiveSession(NonlinearSession([(m,np.zeros_like(q)) for m,q in state],viscosity=0));before=a.checkpoint()
    def force(time):
        return {(m.k,m.m):np.full(3*m.order,np.nan if abs(time-.001875)<1e-15 else 0.) for m,q in state}
    with pytest.raises(ValueError):a.step(.0025,force)
    assert a.checkpoint()==before


def test_unattainable_endpoint_does_not_reset_clock_or_accept_partial_interval():
    state=modes(0,3,32);a=AdaptiveSession(NonlinearSession([(m,np.zeros_like(q)) for m,q in state],viscosity=0),min_step=.001,max_step=.001);before=a.checkpoint()
    with pytest.raises(AccuracyError,match='endpoint remainder'):a.advance_to(.0025)
    assert a.checkpoint()==before
