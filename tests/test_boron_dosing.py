from copy import deepcopy
from dataclasses import replace
import math

import pytest
from coupling.boron_dosing import plan_boron_dose,execute_boron_dose,_mass
from coupling.pwr_boron import BoronExchangeConfig,BoronExchangePWRSession
from coupling.refinement import coupled_pwr_mesh


@pytest.fixture(scope='module')
def seed():
    return BoronExchangePWRSession(**coupled_pwr_mesh(2),boron=BoronExchangeConfig(
        pressure_pa=1e6,regenerator_ua_w_k=500000.,require_liquid_ports=True),max_step_s=.1,rtol=2e-9)


def test_mass_matches_integrated_actuator_for_nonzero_initial_flow():
    from scipy.integrate import quad
    q0=.3;q=1.2;tau=.2;opening=.7;tail=1.
    close=q+(q0-q)*math.exp(-opening/tau)
    exact=quad(lambda t:q+(q0-q)*math.exp(-t/tau),0.,opening)[0]+quad(lambda t:close*math.exp(-t/tau),0.,tail)[0]
    assert _mass(opening,tail,q,q0,tau)==pytest.approx(exact,abs=1e-13)


@pytest.mark.parametrize('mass,flow',[(.01,1.),(.3,1.),(59.99865241060018,1.),(120.,1.)])
def test_planner_includes_finite_lag_and_eventual_tail_without_mutation(seed,mass,flow):
    s=deepcopy(seed);before=s.checkpoint();p=plan_boron_dose(s,mass_kg=mass,flow_kg_s=flow)
    assert s.checkpoint()==before
    assert _mass(p.opening_duration_s,p.closing_duration_s,flow,0.,.2)==pytest.approx(mass,abs=1e-10)
    assert p.eventual_transfer_mass_kg==mass+p.future_tail_mass_kg
    assert p.future_tail_mass_kg==.2*p.residual_flow_kg_s>0.
    assert all(0<x['duration_s']<=60. for x in p.steps())
    assert p.steps()[-1]['patch']=={'boron_flow_target_kg_s':0.}
    changed=p.steps();changed[0]['patch']['boron_flow_target_kg_s']=50.
    assert p.steps()[0]['patch']['boron_flow_target_kg_s']==flow


@pytest.mark.parametrize('kwargs',[dict(mass_kg=True,flow_kg_s=1.),dict(mass_kg=float('nan'),flow_kg_s=1.),
    dict(mass_kg=-1.,flow_kg_s=1.),dict(mass_kg=1.,flow_kg_s=0.),dict(mass_kg=1.,flow_kg_s=100.),
    dict(mass_kg=1.,flow_kg_s=1.,tail_time_constants=4.),dict(mass_kg=1.,flow_kg_s=1.,tail_time_constants=21.)])
def test_invalid_inputs_refused(seed,kwargs):
    with pytest.raises(ValueError):plan_boron_dose(seed,**kwargs)


def test_reserve_accounts_for_mass_after_finite_tail(seed):
    s=deepcopy(seed);s.boron_config=replace(s.boron_config,minimum_mass_kg=1000.)
    # Planning reads actual inventory, not configured initial mass.
    s.state[s.boron_index]=1000.3005
    s.state[s.boron_index+1]=1000.3005*s.boron_u
    with pytest.raises(ValueError,match='future tail.*reserve'):plan_boron_dose(s,mass_kg=.3,flow_kg_s=1.)


def test_receiver_pressure_domain_refused_before_simulation(seed):
    with pytest.raises(ValueError,match='receiver'):plan_boron_dose(seed,mass_kg=2500.,flow_kg_s=1.)


def test_time_and_replay_budgets_refused(seed,monkeypatch):
    s=deepcopy(seed);s.time=3599.5
    with pytest.raises(ValueError,match='time budget'):plan_boron_dose(s,mass_kg=.3,flow_kg_s=1.)
    s=deepcopy(seed);monkeypatch.setattr(s,'_maximum_history_steps',lambda:1)
    with pytest.raises(ValueError,match='history budget'):plan_boron_dose(s,mass_kg=.3,flow_kg_s=1.)


def test_whole_dose_executes_native_commands_and_replans_from_residual_flow(seed):
    s=deepcopy(seed);p=plan_boron_dose(s,mass_kg=.3,flow_kg_s=1.)
    r=execute_boron_dose(s,p)
    assert r['transferred_mass_kg']==pytest.approx(.3,abs=1e-7)
    assert r['received_mass_kg']==pytest.approx(.3,abs=1e-7)
    assert r['residual_flow_kg_s']>0 and s.controls['boron_flow_target_kg_s']==0.
    assert list(s.history)==p.steps()
    next_plan=plan_boron_dose(s,mass_kg=.3,flow_kg_s=1.)
    assert next_plan.opening_duration_s<p.opening_duration_s
    execute_boron_dose(s,next_plan)
    assert 5000.-s._tanks(s.state)['supply']['mass_kg']==pytest.approx(.6,abs=1e-7)
    with pytest.raises(ValueError,match='achievable'):plan_boron_dose(s,mass_kg=.001,flow_kg_s=1.)


def test_stale_and_tampered_plans_cannot_run(seed):
    s=deepcopy(seed);p=plan_boron_dose(s,mass_kg=.3,flow_kg_s=1.)
    before=s.checkpoint()
    with pytest.raises(ValueError,match='Altered'):execute_boron_dose(s,replace(p,opening_duration_s=1.))
    assert s.checkpoint()==before
    s.advance(.01)
    before=s.checkpoint()
    with pytest.raises(ValueError,match='Stale'):execute_boron_dose(s,p)
    assert s.checkpoint()==before


def test_late_failure_rolls_back_earlier_successful_interval(seed,monkeypatch):
    s=deepcopy(seed);p=plan_boron_dose(s,mass_kg=.3,flow_kg_s=1.);before=s.checkpoint()
    advance=BoronExchangePWRSession.advance;calls=[]
    def fail_closing(self,duration_s,patch=None):
        calls.append(self.time)
        if len(calls)==2:raise ValueError('Injected late failure')
        return advance(self,duration_s,patch)
    monkeypatch.setattr(BoronExchangePWRSession,'advance',fail_closing)
    with pytest.raises(ValueError,match='late failure'):execute_boron_dose(s,p)
    assert len(calls)==2 and calls[1]>0 and s.checkpoint()==before


def test_actual_phase_failure_leaves_entire_dose_uncommitted(seed):
    # Known unsupported strict-port recipe: fail after part of the open interval.
    s=BoronExchangePWRSession(**coupled_pwr_mesh(2),boron=BoronExchangeConfig(
        pressure_pa=1e6,regenerator_ua_w_k=200000.,require_liquid_ports=True),max_step_s=.05,rtol=2e-9)
    p=plan_boron_dose(s,mass_kg=20.,flow_kg_s=50.);before=s.checkpoint()
    with pytest.raises(ValueError,match='liquid'):execute_boron_dose(s,p)
    assert s.checkpoint()==before


def test_close_only_plan_includes_unavoidable_current_flow(seed):
    s=deepcopy(seed);s.advance(.1,{'boron_flow_target_kg_s':2.})
    q0=float(s.state[s.boron_index+6]);mass=q0*.2*(-math.expm1(-5.))
    p=plan_boron_dose(s,mass_kg=mass,flow_kg_s=.1)
    assert p.opening_duration_s==0. and len(p.steps())==1
    assert p.eventual_transfer_mass_kg==pytest.approx(q0*.2,abs=1e-12)
    r=execute_boron_dose(s,p)
    assert r['transferred_mass_kg']==pytest.approx(mass,abs=1e-7)


def test_active_initial_flow_above_new_target_is_included(seed):
    s=deepcopy(seed);s.advance(.1,{'boron_flow_target_kg_s':2.})
    q0=float(s.state[s.boron_index+6]);p=plan_boron_dose(s,mass_kg=.3,flow_kg_s=.1)
    assert q0>.1
    assert _mass(p.opening_duration_s,p.closing_duration_s,.1,q0,.2)==pytest.approx(.3,abs=1e-12)
    assert execute_boron_dose(s,p)['transferred_mass_kg']==pytest.approx(.3,abs=1e-7)


def test_postcondition_failure_discards_all_intervals(seed,monkeypatch):
    s=deepcopy(seed);p=plan_boron_dose(s,mass_kg=.3,flow_kg_s=1.);before=s.checkpoint()
    advance=BoronExchangePWRSession.advance
    def corrupt(self,duration_s,patch=None):
        row=advance(self,duration_s,patch)
        if patch and patch.get('boron_flow_target_kg_s')==0.:
            self.state[self.boron_index]+=.01
        return row
    monkeypatch.setattr(BoronExchangePWRSession,'advance',corrupt)
    with pytest.raises(ValueError,match='postcondition'):execute_boron_dose(s,p)
    assert s.checkpoint()==before


@pytest.mark.parametrize('owner',[None,{},object()])
def test_unrelated_owner_cannot_receive_dose(owner):
    with pytest.raises(ValueError,match='session'):plan_boron_dose(owner,mass_kg=.3,flow_kg_s=1.)


def test_power_and_mass_resolution_limits(seed):
    s=deepcopy(seed);s.controls['grid_available']=False
    with pytest.raises(ValueError,match='grid'):plan_boron_dose(s,mass_kg=.3,flow_kg_s=1.)
    with pytest.raises(ValueError,match='0.001'):plan_boron_dose(seed,mass_kg=1e-10,flow_kg_s=1.)


@pytest.mark.parametrize('change',[dict(opening_duration_s=float('inf')),dict(closing_duration_s=1e100),
    dict(opening_duration_s=-1.),dict(target_flow_kg_s=True),dict(origin_sha256='unbound')])
def test_handcrafted_plan_cannot_generate_unbounded_steps(seed,change):
    p=plan_boron_dose(seed,mass_kg=.3,flow_kg_s=1.)
    with pytest.raises(ValueError):replace(p,**change)
