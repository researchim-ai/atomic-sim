from dataclasses import replace
import json
import numpy as np
import pytest
from scipy.linalg import expm
from core.status import DomainError
from coupling.core_channels import ChannelConfig,CoreChannels,ChannelSession
from atomic_model_multiphysics.provenance import seal,unseal


def test_analytic_jacobian_matches_finite_differences_with_local_feedback():
    model=CoreChannels(ChannelConfig(reactivity=(1e-4,0.,-5e-5,8e-5)))
    y=model.initial();y[model.k:model.k+model.m]+=[.1,.2,.3]
    exact=model.jacobian(0.,y);numeric=np.zeros_like(exact)
    for j in range(model.size):
        h=1e-6*max(abs(y[j]),1.);up=y.copy();down=y.copy();up[j]+=h;down[j]-=h
        if j>=model.size-2:
            numeric[:,j]=0;continue
        numeric[:,j]=(model.rhs(0.,up)-model.rhs(0.,down))/(2*h)
    np.testing.assert_allclose(exact,numeric,rtol=2e-6,atol=2e-4)


def test_critical_uniform_neutrons_and_thermal_state_are_stationary():
    s=ChannelSession(ChannelConfig(reactivity=(0.,)*4));initial=s.state.copy()
    result=s.advance(.1,step_s=.01)
    np.testing.assert_allclose(s.state[:-2],initial[:-2],rtol=1e-12,atol=1e-11)
    assert result['audit']['energy_relative_defect']<1e-10
    assert result['audit']['precursor_inventory_relative_defect']<1e-10
    assert s.state[-2]==pytest.approx(1e5) and s.state[-1]==pytest.approx(1e5)


def test_uniform_precursors_follow_independent_point_matrix():
    model=CoreChannels(ChannelConfig(alpha_per_k=0.));y=model.initial()
    point=expm(model.kinetics.matrix(1e-4)*.1)@model.kinetics.initial_state()
    from scipy.integrate import solve_ivp
    ref=solve_ivp(model.rhs,(0,.1),y,method='Radau',jac=model.jacobian,rtol=1e-10,atol=1e-12)
    assert ref.success
    projected=model.core.project(ref.y[:model.k,-1])
    np.testing.assert_allclose(projected,point,rtol=2e-9,atol=1e-9)


def test_local_temperature_changes_local_feedback_and_conserves_power():
    model=CoreChannels();y=model.initial();y[model.k+1]+=10
    rho,powers,mapped,_=model.terms(y)
    assert np.ptp(rho)>0 and np.max(rho)<=1e-4
    assert mapped.sum()==pytest.approx(powers.sum(),rel=1e-14)
    assert model.audit(y)['pairing_relative_defect']<1e-14


def test_json_restart_and_following_interval_are_exact():
    s=ChannelSession();s.advance(.05,step_s=.01)
    saved=json.loads(json.dumps(s.checkpoint()))
    restored=ChannelSession.restore(saved)
    assert restored.checkpoint()==s.checkpoint()
    assert restored.advance(.05,step_s=.005)==s.advance(.05,step_s=.005)
    assert restored.checkpoint()==s.checkpoint()


def test_newton_failure_rolls_back_entire_interval(monkeypatch):
    s=ChannelSession();before=s.checkpoint()
    monkeypatch.setattr(np.linalg,'solve',lambda *args:(_ for _ in ()).throw(np.linalg.LinAlgError('deliberate refusal')))
    with pytest.raises(np.linalg.LinAlgError):s.advance(.1)
    assert s.checkpoint()==before


@pytest.mark.parametrize('field,value',[('state',[1]),('time_s',.3),('history',[{'duration_s':.1,'bad':True}]),('format','other')])
def test_resealed_corrupted_checkpoint_refused(field,value):
    s=ChannelSession();s.advance(.01);p=unseal(s.checkpoint());p[field]=value
    with pytest.raises(ValueError):ChannelSession.restore(seal(p))


@pytest.mark.parametrize('kwargs',[dict(neutron_cells=True),dict(channel_faces=[0,4]),dict(channel_faces=(0,3)),
    dict(reactivity=(.1,)*4),dict(alpha_per_k=1e-5),dict(power_w=10**1000),dict(sink_k=float('nan'))])
def test_invalid_problem_configuration_refused(kwargs):
    with pytest.raises((ValueError,DomainError)):replace(ChannelConfig(),**kwargs)


@pytest.mark.parametrize('dt,h',[(0,.01),(.1,0),(.1,float('nan')),(True,.01),(1,.00001)])
def test_work_bounds_and_typed_intervals_leave_state_unchanged(dt,h):
    s=ChannelSession();before=s.checkpoint()
    with pytest.raises(ValueError):s.advance(dt,step_s=h)
    assert before==s.checkpoint()


def test_late_rhs_failure_rolls_back_already_computed_trial_steps(monkeypatch):
    s=ChannelSession();before=s.checkpoint();rhs=CoreChannels.rhs;visited=[]
    def rejected(self,t,y):
        visited.append(t)
        if t>.035:raise ValueError('deliberate late domain refusal')
        return rhs(self,t,y)
    monkeypatch.setattr(CoreChannels,'rhs',rejected)
    with pytest.raises(ValueError,match='late domain'):s.advance(.1,step_s=.01)
    assert any(t<=.03 for t in visited) and any(t>.035 for t in visited)
    assert s.checkpoint()==before


@pytest.mark.parametrize('field',['reactivity','channel_faces'])
def test_resealed_nonvector_configuration_refused(field):
    p=unseal(ChannelSession().checkpoint());p['config'][field]=None
    with pytest.raises(ValueError,match='vectors'):ChannelSession.restore(seal(p))
