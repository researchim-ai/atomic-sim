import json
import math
import numpy as np
import pytest

from atomic_model_station import PWRStagedSession,PWRSteamSession,StageConfig,CoreConfig
from atomic_model_station.stages import staged_path
from atomic_model_station.foundation import validate_registry
from physics.materials import iapws
from atomic_model_multiphysics.provenance import seal,unseal


def point(config=None,**kwargs):
    return staged_path(6e6,iapws.saturation_properties(6e6)[1].h_j_kg,8000.,1000.,558.,
                       config or StageConfig(),**kwargs)


def test_ideal_zero_reheat_cycle_matches_independent_entropy_reference():
    cfg=StageConfig(hp_efficiency=1.,lp_efficiency=1.,reheat_ua_w_k=0.,
                    minimum_hp_exit_quality=.7,minimum_lp_exit_quality=.8)
    r=point(cfg);inlet=iapws.saturation_properties(6e6)[1]
    f,g=iapws.saturation_properties(cfg.intermediate_pressure_pa)
    xf=(inlet.s_j_kg_k-f.s_j_kg_k)/(g.s_j_kg_k-f.s_j_kg_k)
    hp_h=f.h_j_kg+xf*(g.h_j_kg-f.h_j_kg)
    a,b=iapws.saturation_properties(8000.)
    xlp=(g.s_j_kg_k-a.s_j_kg_k)/(b.s_j_kg_k-a.s_j_kg_k)
    lp_h=a.h_j_kg+xlp*(b.h_j_kg-a.h_j_kg)
    expected=1000*(inlet.h_j_kg-hp_h)+1000*xf*(g.h_j_kg-lp_h)
    assert r['total_work_w']==pytest.approx(expected,rel=1e-12)
    assert r['separator']['steam_flow_kg_s']==pytest.approx(1000*xf)
    assert r['reheat_w']==0.


def test_separator_conserves_mass_and_energy_and_drain_reaches_condenser():
    r=point(StageConfig(separator_efficiency=.95))
    sep=r['separator'];hp=r['hp'];lp=r['lp']
    assert sep['drain_flow_kg_s']+sep['steam_flow_kg_s']==1000.
    assert sep['steam_quality']>hp['outlet_quality']
    assert abs(sep['energy_residual_w'])<1e-5
    assert r['condenser_energy_w']==pytest.approx(sep['drain_flow_kg_s']*sep['drain_enthalpy_j_kg']+lp['mass_flow_kg_s']*lp['outlet_enthalpy_j_kg'])
    assert abs(r['energy_residual_w'])<1e-5


def test_reheater_is_bounded_by_donor_temperature_and_flow_enthalpy():
    r=point();heater=r['reheater']
    assert heater['inlet_temperature_k']<heater['outlet_temperature_k']<558.
    assert 0<heater['effectiveness']<1
    assert r['reheat_w']==pytest.approx(r['lp']['mass_flow_kg_s']*(heater['outlet_enthalpy_j_kg']-heater['inlet_enthalpy_j_kg']))
    assert r['lp_exit_quality']>=.88
    assert r['condenser_mixture']['quality']<r['lp_exit_quality']
    assert min(r['entropy_generation_w_k'].values())>=-.01


def test_more_available_area_increases_reheat_without_creating_external_heat():
    a=point(StageConfig(reheat_ua_w_k=2e6));b=point(StageConfig(reheat_ua_w_k=5e6))
    assert a['reheat_w']<b['reheat_w']
    assert a['lp_exit_quality']<b['lp_exit_quality']
    for r in [a,b]:assert r['total_work_w']+r['condenser_energy_w']==pytest.approx(1000*r['hp']['inlet_enthalpy_j_kg']+r['reheat_w'])


@pytest.mark.parametrize('options',[dict(reheater_available=False),dict(separator_available=False)])
def test_nominal_missing_condition_is_refused_by_moisture_limit(options):
    with pytest.raises(ValueError):point(**options)


def test_zero_flow_has_no_heat_or_work_even_when_equipment_unavailable():
    r=staged_path(6e6,2.78e6,8000.,0.,400.,StageConfig(),separator_available=False,reheater_available=False)
    assert not r['active'] and r['active_stage_count']==0
    assert r['total_work_w']==r['reheat_w']==r['condenser_energy_w']==0.
    assert r['lp_exit_quality'] is None


def test_finite_existing_wall_pays_for_reheat_in_directional_energy_rates():
    old=PWRSteamSession();s=PWRStagedSession()
    assert np.array_equal(s.state,old.state) and s._stored()==old._stored()
    a=old._rhs(0.,old.state);b=s._rhs(0.,s.state);path=s.staged_terms;n=s.base_size
    wall=(b[n-1]-a[n-1])*s.plant.sg_wall_capacity
    assert wall==pytest.approx(-path['reheat_w'],rel=1e-12)
    assert wall+(b[n-2]-a[n-2])+(b[n]-a[n])==pytest.approx(0.,abs=1e-5)
    assert validate_registry(s.parameter_registry())


def test_live_path_has_distinct_stream_and_heat_audit():
    s=PWRStagedSession(max_step_s=.2);r=s.advance(.2)
    audit=r['steam_port_audit']
    assert 'enthalpy_drop_work_w' not in audit
    assert audit['shaft_work_w']==r['powers']['steam_work_w']
    assert abs(audit['fluid_enthalpy_drop_w']+audit['reheater_heat_w']-audit['shaft_work_w'])<.001
    assert r['energy_relative']<1e-8 and r['mass_relative']<1e-10
    assert r['expansion']['lp_outlet_quality']==r['staged_turbine']['lp_exit_quality']


@pytest.mark.parametrize('patch',[dict(reheater_available=False),dict(separator_available=False),dict(reheater_available=1)])
def test_unavailable_or_invalid_equipment_rolls_back_entire_station(patch):
    s=PWRStagedSession();saved=s.checkpoint()
    with pytest.raises(ValueError):s.advance(.1,patch)
    assert s.checkpoint()==saved


def test_closed_station_can_disable_equipment_without_phantom_reheat():
    s=PWRStagedSession(max_step_s=.2);s.advance(1.,dict(scram=True,turbine_trip=True))
    r=s.advance(.1,dict(reheater_available=False,separator_available=False))
    assert not r['staged_turbine']['active']
    assert r['powers']['steam_work_w']==r['powers']['reheater_heat_w']==0.


def test_restart_during_motion_preserves_stage_paths_and_wall_energy():
    s=PWRStagedSession(max_step_s=.2);s.advance(.1,dict(scram=True,turbine_trip=True))
    saved=json.loads(json.dumps(s.checkpoint()));t=PWRStagedSession.restore(saved)
    assert t.advance(.5)==s.advance(.5)
    with pytest.raises((ValueError,KeyError)):PWRSteamSession.restore(saved)
    p=unseal(saved);p['stages_source']={}
    with pytest.raises(ValueError,match='source'):PWRStagedSession.restore(seal(p))


@pytest.mark.parametrize('values',[dict(stage_count=3),dict(stage_count=True),dict(hp_efficiency=1.1),dict(minimum_lp_exit_quality=float('nan'))])
def test_stage_configuration_domains(values):
    with pytest.raises(ValueError):StageConfig(**values)


def test_spatial_owner_is_compatible_with_two_stage_thermal_coupling():
    s=PWRStagedSession(core=CoreConfig(kind='diffusion',cells=2),max_step_s=.2)
    r=s.advance(.1)
    assert len(r['core']['owned_state'])==14 and r['energy_relative']<1e-8


def test_staged_extended_study_horizon_preserves_base_budget():
    s=PWRStagedSession();old=PWRSteamSession()
    assert old._maximum_time_s()==3600. and s._maximum_time_s()==7200.
    s.time=7200.;before=s.checkpoint()
    with pytest.raises(ValueError):s.advance(.1)
    assert s.checkpoint()==before
