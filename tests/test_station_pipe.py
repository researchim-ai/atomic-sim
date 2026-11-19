from copy import deepcopy
from dataclasses import replace
from importlib.resources import files
import json
from math import log
import numpy as np
import pytest
from core.status import DomainError,SolverError
from atomic_model_multiphysics.provenance import seal,unseal
from atomic_model_station_pipe import PipeCase,PipeSession,steady_state,exact_profiles,exponential_reference


def pulse(case):
    s=PipeSession(case);x=s.x;w=s.w
    # Define a discrete zero-area-mean initial perturbation, not a post-step projection.
    a=float(w@((1-x*x)*x*x)/(w@(1-x*x)));shape=(1-x*x)*(x*x-a)
    y=s.state;y[:case.cells]+=.02*shape;y[case.cells:]-=.05*shape
    return y.tolist()


@pytest.mark.parametrize('kwargs',[dict(reynolds=True),dict(reynolds=0),dict(reynolds=10001),dict(prandtl=float('nan')),
 dict(prandtl=.1),dict(buoyancy_c=-1),dict(buoyancy_c=11),dict(cells=True),dict(cells=8),dict(cells=257),dict(max_step=0),dict(max_step=float('inf'))])
def test_case_requires_a_bounded_research_domain(kwargs):
    with pytest.raises(ValueError):PipeCase(**kwargs)


@pytest.mark.parametrize('c',[0.,3.,5.,7.5,10.])
def test_radial_steady_solution_converges_to_published_polynomial_and_controllers(c):
    errors=[]
    for n in (32,64,128):
        s=PipeSession(PipeCase(cells=n,buoyancy_c=c));p=s.snapshot();ref=exact_profiles(s.x,c)
        error=max(np.max(np.abs(np.array(p[k])-np.array(ref[k]))) for k in ('velocity','temperature'))
        errors.append(error)
        assert p['area_mean_velocity']==pytest.approx(.5,abs=5e-12)
        assert p['area_mean_temperature']==pytest.approx(.5,abs=5e-12)
        assert abs(p['pressure_beta']+c/3)<.02 and abs(p['sink_ratio_nu']-1)<.001
        assert not p['station_usable'] and not p['dns_qualified']
    assert errors[0]>3.8*errors[1]>14.4*errors[2]
    assert errors[-1]<3e-4


@pytest.mark.parametrize('c',[0.,5.,10.])
def test_discrete_steady_state_stays_fixed_under_cn(c):
    s=PipeSession(PipeCase(buoyancy_c=c));before=s.state
    p=s.advance(300)
    assert np.max(np.abs(s.state-before))<2e-11 and p['checks']['relative_maximum']<1e-10


def test_means_are_volume_means_and_nu_is_not_the_uhf_nusselt():
    p=PipeSession(PipeCase(cells=128)).snapshot()
    assert p['area_mean_temperature']==pytest.approx(.5,abs=5e-12)
    assert p['mixing_cup_temperature']==pytest.approx(1/3,abs=5e-5)
    assert p['sink_ratio_nu']==pytest.approx(1,rel=1e-4) and 'not hD/k' in p['nu_definition']


def test_transient_convergence_to_independent_augmented_matrix_exponential():
    case=PipeCase(cells=32);initial=pulse(case)
    reference=exponential_reference(case,initial,200,buoyancy_c=5.)
    errors=[]
    for dt in (20.,10.,5.):
        s=PipeSession(replace(case,max_step=dt),initial_state=initial);r=s.advance(200,buoyancy_c=5.)
        errors.append(float(np.max(np.abs(s.state-reference))))
        assert r['checks']['relative_maximum']<1e-10
    assert errors[0]>3.5*errors[1]>12*errors[2]
    assert errors[-1]<2e-5


def test_adverse_pressure_work_is_allowed_and_both_energy_ledgers_close():
    s=PipeSession(PipeCase(buoyancy_c=5.));p=s.advance(100)
    assert p['pressure_coefficient']<0 and p['ledger']['pressure_work']<0
    assert p['ledger']['buoyancy_work']>0 and p['ledger']['viscous_loss']>0
    assert p['ledger']['wall_heat']==pytest.approx(p['ledger']['sink_heat'],abs=1e-11)
    assert abs(p['checks']['kinetic_residual'])<1e-11


def test_native_restart_keeps_profile_controllers_ledgers_and_next_parameter_change_exact():
    case=PipeCase(cells=32);s=PipeSession(case,initial_state=pulse(case));s.advance(100,buoyancy_c=3.)
    cp=json.loads(json.dumps(s.checkpoint()));other=PipeSession.restore(cp)
    assert other.checkpoint()==cp
    assert s.advance(100,buoyancy_c=7.5)==other.advance(100,buoyancy_c=7.5)
    assert s.checkpoint()==other.checkpoint()


def test_checkpoint_and_array_views_do_not_mutate_the_owner():
    s=PipeSession();before=s.checkpoint();saved=s.checkpoint();saved['payload']['initial_state'][0]=123.
    view=s.state;view[0]=123.;ledger=s.ledger;ledger['wall_heat']=123.
    assert s.checkpoint()==before


@pytest.mark.parametrize('which',['state','source','command','extra'])
def test_resealed_checkpoint_cannot_override_replayed_physics(which):
    s=PipeSession(PipeCase(cells=16));s.advance(10);p=unseal(s.checkpoint())
    if which=='state':p['snapshot']['temperature'][0]+=.01
    if which=='source':p['source']['files']['model.py']='0'*64
    if which=='command':p['history'][0]['duration']=-1
    if which=='extra':p['unrecognized']=True
    with pytest.raises(ValueError):PipeSession.restore(seal(p))


def test_rejected_late_substep_rolls_back_clock_profiles_ledgers_and_history(monkeypatch):
    s=PipeSession(PipeCase(cells=16,max_step=1));before=s.checkpoint();original=PipeSession._admit;calls=0
    def reject(self,y):
        nonlocal calls
        calls+=1
        if calls==2:raise DomainError('Deliberate rejected radial state')
        return original(self,y)
    monkeypatch.setattr(PipeSession,'_admit',reject)
    with pytest.raises(DomainError):s.advance(10,buoyancy_c=5.)
    assert calls==2 and s.checkpoint()==before


def test_nonfinite_linear_solver_failure_is_atomic(monkeypatch):
    s=PipeSession(PipeCase(cells=16));before=s.checkpoint()
    class Bad:
        def solve(self,rhs):return np.full(len(rhs),np.nan)
    monkeypatch.setattr(PipeSession,'_factor',lambda *a:Bad())
    with pytest.raises(SolverError):s.advance(10)
    assert s.checkpoint()==before


@pytest.mark.parametrize('kwargs',[dict(duration=True),dict(duration=float('nan')),dict(duration=-1),dict(duration=2001),dict(duration=10,buoyancy_c=20)])
def test_advance_input_errors_leave_state_unchanged(kwargs):
    s=PipeSession(PipeCase(cells=16));before=s.checkpoint()
    with pytest.raises(ValueError):s.advance(**kwargs)
    assert s.checkpoint()==before


def test_work_budget_is_rejected_before_a_large_step():
    s=PipeSession(PipeCase(cells=16,max_step=.01));before=s.checkpoint()
    with pytest.raises(ValueError,match='budget'):s.advance(1000)
    assert s.checkpoint()==before


def test_initial_mean_violation_is_not_hidden_by_projection():
    case=PipeCase(cells=16);y=steady_state(case);y[0]+=.1
    with pytest.raises(DomainError,match='mean'):PipeSession(case,initial_state=y)
    with pytest.raises(ValueError):steady_state(False)


def test_source_fixture_is_analytic_not_digitized_dns():
    f=json.loads(files('atomic_model_station_pipe').joinpath('data/jfm2021_base.json').read_text())
    assert f['doi']=='10.1017/jfm.2021.371' and f['sink_ratio_nu']==1.
    assert not f['dns_qualified'] and not f['station_usable'] and not f['validated']


@pytest.mark.parametrize('n',[32,64,128])
def test_relaxation_rates_match_independent_bessel_j2_roots(n):
    from atomic_model_station_pipe import relaxation_scales
    r=relaxation_scales(PipeCase(cells=n))
    assert r['relative_errors'][0]<3/n**2
    assert r['slowest_temperature_time']==pytest.approx(.7*r['slowest_velocity_time'],rel=2e-15)
    assert not r['dns_qualified']


def test_sector_spectrum_does_not_claim_three_dimensional_stability():
    from atomic_model_station_pipe import relaxation_scales
    r=relaxation_scales(PipeCase(cells=64));s=relaxation_scales(PipeCase(cells=128))
    assert r['relative_errors'][0]>3.8*s['relative_errors'][0]
    assert 'no DNS' in r['scope']


def test_total_substep_budget_is_enforced_across_commands():
    s=PipeSession(PipeCase(cells=16,max_step=.01));s.advance(40);s.advance(40)
    assert s.snapshot()['substeps']==8000
    before=s.checkpoint()
    with pytest.raises(ValueError,match='budget'):s.advance(10)
    assert s.checkpoint()==before


@pytest.mark.parametrize('payload',[None,[],{'cells':32}])
def test_malformed_checkpoint_config_is_a_controlled_input_error(payload):
    s=PipeSession(PipeCase(cells=16));p=unseal(s.checkpoint());p['case']=payload
    with pytest.raises(ValueError):PipeSession.restore(seal(p))


def test_legacy_sources_allow_only_evidenced_exact_cache_migration():
    from pathlib import Path
    from scripts.qualify_station_inventory import read
    from scripts.qualify_station_shutdown import identity as station_identity
    from scripts.qualify_station_mixed_convection import identity as mixed_identity
    root=Path(__file__).resolve().parents[1]
    from tests._station_history import assert_source_compatible
    assert_source_compatible(read(root/'docs/benchmarks/station_low_flow_v63.json')['source'],station_identity())
    assert read(root/'docs/benchmarks/station_mixed_convection_v64.json')['source']==mixed_identity()
