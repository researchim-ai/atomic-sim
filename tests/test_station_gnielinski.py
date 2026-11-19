from copy import deepcopy
import json
import math
import pytest
from core.status import DomainError,ModelLimitError
from physics.materials import iapws
from atomic_model_station.film import FilmConfig,film_transfer
from atomic_model_station.film_correlations import nusselt_gnielinski,MAX_RICHARDSON_D
from atomic_model_station_faults import FaultSession
from atomic_model_station_campaign import Episode
from atomic_model_station_campaign.engine import run_episode
from atomic_model_station_env import EnvConfig,StationEnv
from atomic_model_station_hot import HotConfig,HotSession


@pytest.mark.parametrize('re,denominator',[(1e4,5.64),(1e5,7.46),(1e6,9.28)])
def test_unit_prandtl_reduces_to_an_independent_darcy_reference(re,denominator):
    # At Pr=1 the entire denominator is1; for decade Re the ANL log10 law is exact arithmetic.
    expected=(re-1000)/(8*denominator**2)
    assert nusselt_gnielinski(re,1.).value==pytest.approx(expected,rel=2e-15)


@pytest.mark.parametrize('re,pr',[(2999.,1.),(1e6+1,1.),(1e4,.59),(1e4,201.),(float('nan'),1.),(1e4,float('inf')),(True,1.),(1e4,True)])
def test_no_clamping_of_unsupported_gnielinski_inputs(re,pr):
    with pytest.raises(DomainError):nusselt_gnielinski(re,pr)


def test_lower_re_heating_is_continuous_and_passes_a_declared_buoyancy_screen():
    s=iapws.properties_pt(10e6,500.);g=dict(diameter_m=.01,length_m=4.,flow_area_m2=.01,heated_area_m2=4.)
    flow=lambda re:re*s.viscosity_pa_s*g['flow_area_m2']/g['diameter_m']
    low=film_transfer(s,flow(5000),g,500.01,correlation='gnielinski')
    assert low['heat_w']>0 and low['richardson_d']<MAX_RICHARDSON_D
    assert low['model_id']=='gnielinski-bulk-smooth-tube' and not low['validated']
    with pytest.raises(DomainError):film_transfer(s,flow(5000),g,500.01)
    left=film_transfer(s,flow(9999.99),g,500.01,correlation='gnielinski')
    right=film_transfer(s,flow(10000.01),g,500.01,correlation='gnielinski')
    assert abs(right['heat_w']/left['heat_w']-1)<1e-5
    with pytest.raises(ModelLimitError,match='buoyancy') as info:film_transfer(s,flow(3100),g,510.,correlation='gnielinski')
    assert info.value.details()['criterion']=='film_richardson_d'
    assert info.value.observed>info.value.limit and info.value.limit==.01


@pytest.mark.parametrize('cls,args',[(FilmConfig,{'correlation':True}),(EnvConfig,{'film_model':'unknown'}),
    (HotConfig,{'film_model':'laminar'}),(Episode,{'name':'bad','ticks':1,'film_model':None})])
def test_configurations_reject_unknown_closure_choices(cls,args):
    with pytest.raises(ValueError):cls(**args)


@pytest.fixture(scope='module')
def solved():
    return FaultSession(equilibrium_start=True,film_model='gnielinski',tick_s=.01)


def test_alternative_changes_heat_law_without_resizing_geometry_or_operating_state_projection(solved):
    old=FaultSession(equilibrium_start=True,tick_s=.01)
    p=solved.control.plant.snapshot();q=old.control.plant.snapshot()
    assert p['film_geometry']==q['film_geometry']
    assert p['clad_temperature_k']>q['clad_temperature_k']
    assert p['film_config']['correlation']=='gnielinski' and q['film_config']['correlation']=='dittus_boelter'
    r=p['equilibrium_initialization']['report']
    assert r['constrained_rank']==49 and not r['parameters_changed'] and not r['operating_state_projected']
    from atomic_model_station.foundation import validate_registry
    assert validate_registry(solved.control.plant.parameter_registry())
    assert solved.control.plant.parameter_registry()['parameters']['film.gnielinski_selected']['value']==1


def test_native_scenario_replay_keeps_the_selected_closure(solved):
    s=deepcopy(solved);s.step(modes={k:'auto' for k in s.control.loops})
    saved=json.loads(json.dumps(s.checkpoint()));other=FaultSession.restore(saved)
    assert s.step()==other.step() and s.checkpoint()==other.checkpoint()


def test_campaign_and_measured_agent_reset_use_the_selected_closure():
    from atomic_model_multiphysics.provenance import unseal
    episode=Episode('film',2,tick_s=.01,equilibrium_start=True,film_model='gnielinski',checkpoint_ticks=(1,))
    r=unseal(run_episode(episode));assert r['accepted']
    assert r['episode']['film_model']=='gnielinski'
    env=StationEnv(EnvConfig(max_steps=2,tick_s=.01,equilibrium_start=True,film_model='gnielinski'))
    obs,info=env.reset();assert 'film' not in obs and info['steps']==0
    cfg=unseal(env.checkpoint())['config'];assert cfg['film_model']=='gnielinski'
    env.step();copy=StationEnv.restore(env.checkpoint());assert env.step()==copy.step()


def test_hot_lifecycle_carries_closure_without_changing_it_on_restore():
    from atomic_model_station_readiness import ReadinessConfig
    c=HotConfig(maximum_warmup_s=.03,hold_s=.01,tick_s=.01,equilibrium_start=True,film_model='gnielinski',
        readiness=ReadinessConfig(window_s=.02,maximum_gap_s=.01,relative_span=.1))
    s=HotSession(c);s.step();restored=HotSession.restore(s.checkpoint())
    assert restored.config.film_model=='gnielinski' and restored.checkpoint()==s.checkpoint()
