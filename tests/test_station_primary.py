import json
import math
import numpy as np
import pytest
from atomic_model_station import PWRPrimarySession,PrimaryConfig,PWRCoreSession
from physics.hydraulics.network import G


def test_primary_log_coordinate_preserves_nominal_energy_and_has_single_owner():
    s=PWRPrimarySession(max_step_s=.2);initial=s.snapshot();r=s.advance(.2)
    assert r['primary']['speed_rad_s']==pytest.approx(initial['primary']['speed_rad_s'],rel=1e-6)
    assert r['pwr_state'][11]==s.plant.omega_nom*math.exp(r['primary']['log_speed'])
    assert r['energy_relative']<1e-8 and r['mass_relative']<1e-10
    assert any(m['name']=='primary_shaft_speed' for m in r['exact_mode_reduction'])
    assert len(r['core']['owned_state'])==7


def test_closed_steam_follows_exact_positive_exponential_without_reverse_work():
    s=PWRPrimarySession(max_step_s=.5);q=s.snapshot()['steam_flow_kg_s'];tau=s.plant.network.junctions['steam'].flow_time_constant_s
    r=s.advance(5.,{'scram':True,'turbine_trip':True,'primary_pump_on':False})
    assert r['steam_flow_kg_s']==pytest.approx(q*math.exp(-5/tau),rel=1e-13)
    assert r['powers']['steam_work_w']>=0
    assert any(m['name']=='closed_turbine_steam' for m in r['exact_mode_reduction'])


def test_bypass_request_moves_gradually_and_inhibits_motor_without_assigning_flow():
    s=PWRPrimarySession(max_step_s=.2);initial=s.snapshot()
    r=s.advance(.5,{'scram':True,'turbine_trip':True,'primary_bypass_open':True})
    p=r['primary'];assert p['bypass_fraction']==pytest.approx(.5/60)
    assert p['bypass_selected'] and not p['effective_motor_on']
    assert s.controls['primary_pump_on']
    assert p['pump_head_pa']==pytest.approx(s.plant.dp_pump_nom*(p['speed_rad_s']/s.plant.omega_nom)**2*(1-p['bypass_fraction']))
    assert p['mean_flow_kg_s']>0 and p['mean_flow_kg_s']!=initial['primary']['mean_flow_kg_s']


def test_frozen_zero_head_momentum_matches_independent_four_column_gravity():
    s=PWRPrimarySession();s.controls['primary_bypass_open']=True;y=s.state.copy();y[s.primary_index+1]=1.
    rate=s._rhs(0.,y);p=s.plant;d=s.primary_terms;rho={n:p.network.volumes[n].thermo.rho_kg_m3 for n in ['core','hot_leg','sg_primary','cold_leg']}
    expected=G*p.height_m*(-.25*rho['core']-.5*rho['hot_leg']+.25*rho['sg_primary']+.5*rho['cold_leg'])
    assert d['buoyancy_head_pa']==pytest.approx(expected,abs=1e-8) and d['pump_head_pa']==0.
    assert abs(d['momentum_residual_pa'])<1e-6
    assert p.inertia*p.omega*rate[11]==pytest.approx(-d['shaft_drag_energy_out_w'],rel=1e-12)
    assert d['frozen_zero_head_reference_kg_s']>0


def test_zero_height_has_no_buoyancy_even_with_positive_frozen_initial_flow():
    from atomic_model_station.primary import frozen_gravity_head
    s=PWRPrimarySession();rho={n:s.plant.network.volumes[n].thermo.rho_kg_m3 for n in ['core','hot_leg','sg_primary','cold_leg']}
    assert frozen_gravity_head(rho,0.)==0.
    assert s.snapshot()['primary']['mean_flow_kg_s']>0.


def test_json_restart_preserves_valve_motion_steam_mode_and_new_coordinate():
    s=PWRPrimarySession(max_step_s=.2);s.advance(.2,{'scram':True,'turbine_trip':True,'primary_bypass_open':True})
    saved=json.loads(json.dumps(s.checkpoint()));t=PWRPrimarySession.restore(saved)
    assert t.advance(.2)==s.advance(.2)
    with pytest.raises((ValueError,KeyError)):PWRCoreSession.restore(saved)


def test_powered_near_stall_start_is_refused_without_cold_start_model():
    s=PWRPrimarySession();y=s.state.copy();y[s.primary_index]=-20.;y=s._reconstruct_state(y)
    with pytest.raises(ValueError,match='near-stall'):s._rhs(0.,y)


@pytest.mark.parametrize('patch',[{'primary_bypass_open':1},{'unknown_primary_control':False}])
def test_invalid_primary_commands_roll_back_every_state(patch):
    s=PWRPrimarySession();before=s.checkpoint()
    with pytest.raises(ValueError):s.advance(.1,dict(patch,scram=True))
    assert s.checkpoint()==before


@pytest.mark.parametrize('values',[{'height_m':True},{'bypass_transition_time_s':0.},{'maximum_speed_fraction':float('nan')}])
def test_primary_configuration_domain(values):
    with pytest.raises(ValueError):PrimaryConfig(**values)


def test_reverse_spray_outside_explicit_roundoff_band_is_refused():
    s=PWRPrimarySession();y=s.state.copy();j=13+3*len(s.plant.network.volumes)+s.plant.network.junction_names().index('spray');y[j]=-1e-5
    with pytest.raises(ValueError,match='spray'):s._admit(y)
    assert y[j]==-1e-5


def test_registry_records_study_geometry_and_extended_rotor_band():
    from atomic_model_station.foundation import validate_registry
    s=PWRPrimarySession();assert validate_registry(s.parameter_registry())
    assert s.config.minimum_speed_fraction==.5 and s.primary_config.bypass_transition_time_s==60.
    assert s.parameter_registry()['parameters']['primary_study.height_m']['family']=='illustrative'
