import json
import math

import numpy as np
import pytest

from atomic_model_station import PWRSteamSession,SteamConfig,PWRFilmSession,CoreConfig
from atomic_model_station.steam import actual_expansion
from atomic_model_station.foundation import validate_registry
from physics.materials import iapws
from physics.power.steam_turbine_train import expansion
from atomic_model_multiphysics.provenance import seal,unseal


def test_actual_inlet_expansion_matches_independent_dry_saturated_limit():
    pin,pout,eta=6e6,8000.,.85
    inlet=iapws.saturation_properties(pin)[1]
    actual=actual_expansion(pin,inlet.h_j_kg,pout,eta)
    old=expansion(dict(inlet_pressure_pa=pin,outlet_pressure_pa=pout,isentropic_efficiency=eta))
    for key in old:assert actual[key]==pytest.approx(old[key],rel=1e-12,abs=1e-8)
    ideal=actual_expansion(pin,inlet.h_j_kg,pout,1.)
    assert ideal['outlet_entropy_j_kg_k']==pytest.approx(inlet.s_j_kg_k,abs=1e-7)


def test_actual_enthalpy_is_not_reset_to_saturated_vapor():
    pin,pout=6e6,8000.;liquid,vapor=iapws.saturation_properties(pin)
    h=.995*vapor.h_j_kg+.005*liquid.h_j_kg
    r=actual_expansion(pin,h,pout,.85)
    assert r['inlet_enthalpy_j_kg']==h
    assert r['inlet_quality']==pytest.approx(.995)
    assert r['work_j_kg']==pytest.approx(h-r['outlet_enthalpy_j_kg'])
    assert r['outlet_entropy_j_kg_k']>=r['inlet_entropy_j_kg_k']


@pytest.mark.parametrize('args',[(6e6,2.8e6,7e6,.85),(6e6,2.8e6,8000.,1.1),(6e6,1e6,8000.,.85)])
def test_unsupported_expansion_rejected(args):
    with pytest.raises(ValueError):actual_expansion(*args)


def test_header_adds_real_stocks_without_changing_existing_initial_inventories():
    s=PWRSteamSession();old=PWRFilmSession();r=s.snapshot();i=s.steam_index
    assert np.allclose(s.state[:i],old.state,rtol=1e-14,atol=1e-10)
    assert s._stored()-old._stored()==pytest.approx(s.state[i+1],rel=1e-12)
    assert r['steam_header']['quality']>=s.steam_config.minimum_quality
    assert validate_registry(s.parameter_registry())
    assert not s.controls['governor_enabled']


def test_admission_moves_in_finite_time_and_changes_mass_and_pressure():
    s=PWRSteamSession(max_step_s=.2);before=s.snapshot()
    moving=s.advance(.005,{'steam_admission':.98})
    assert moving['steam_header']['outlet_position']==pytest.approx(.99)
    assert moving['steam_header']['outlet_position']!=.98
    assert moving['effective_valve']==moving['steam_header']['outlet_position']
    assert moving['actuator_status']['steam_outlet_position']==moving['effective_valve']
    r=s.advance(.2);h=r['steam_header']
    assert h['outlet_position']==.98
    assert h['mass_kg']!=before['steam_header']['mass_kg']
    assert h['pressure_pa']!=before['steam_header']['pressure_pa']
    assert h['inlet_flow_kg_s']!=h['turbine_flow_kg_s']
    assert r['energy_relative']<1e-8 and r['mass_relative']<1e-10


def test_flow_coordinates_are_derived_not_second_owners():
    s=PWRSteamSession();y=s.state.copy();y[s.steam_index+2]=-123.
    reconstructed=s._reconstruct_state(y)
    assert reconstructed[s.steam_index+2]>0
    r=s.advance(.1)
    modes={m['name']:m['kind'] for m in r['exact_mode_reduction']}
    assert modes['header_inlet_flow']=='derived_equilibrium_flux'
    assert modes['header_outlet_flow']=='derived_equilibrium_flux'


def test_trip_closes_actual_valves_and_retains_finite_header_inventory():
    s=PWRSteamSession(max_step_s=.2)
    r=s.advance(1.,dict(scram=True,turbine_trip=True));h=r['steam_header']
    assert h['inlet_position']==h['outlet_position']==0.
    assert h['inlet_flow_kg_s']==h['turbine_flow_kg_s']==0.
    assert h['mass_kg']>1 and h['energy_j']>0
    assert any(e['kind']=='steam_valve_arrived' for e in r['events'])
    mass,energy=h['mass_kg'],h['energy_j']
    later=s.advance(.2)['steam_header']
    assert later['mass_kg']==pytest.approx(mass,abs=1e-9)
    assert later['energy_j']==pytest.approx(energy,abs=1e-4)


def test_header_heat_loss_goes_to_existing_cooling_stock_and_entropy_is_nonnegative():
    s=PWRSteamSession(steam=SteamConfig(heat_loss_ua_w_k=1000.),max_step_s=.2)
    r=s.advance(.2);h=r['steam_header']
    assert h['heat_to_cooling_w']>0 and r['cooling']['header_received_w']==h['heat_to_cooling_w']
    assert h['heat_entropy_w_k']>0
    assert h['inlet_valve_entropy_w_k']>=0 and h['mixing_entropy_w_k']>=-1e-5
    assert abs(h['routing_energy_residual_w'])<.001
    assert abs(r['condenser_balance']['energy_residual_w'])<.001
    assert r['steam_port_audit']['enthalpy_drop_work_w']==pytest.approx(r['powers']['steam_work_w'])
    assert r['energy_relative']<1e-8


def test_restart_during_valve_stroke_crosses_analytical_arrival_exactly():
    s=PWRSteamSession(max_step_s=.2);s.advance(.1,dict(scram=True,turbine_trip=True))
    saved=json.loads(json.dumps(s.checkpoint()));t=PWRSteamSession.restore(saved)
    assert t.advance(.6)==s.advance(.6)
    with pytest.raises((ValueError,KeyError)):PWRFilmSession.restore(saved)
    payload=unseal(saved);payload['steam_source']['material_sha256']='0'*64
    with pytest.raises(ValueError,match='source'):PWRSteamSession.restore(seal(payload))


@pytest.mark.parametrize('patch',[{'steam_admission':1.1},{'steam_inlet_isolated':1},{'governor_enabled':True}])
def test_invalid_controls_roll_back_full_header_state(patch):
    s=PWRSteamSession();saved=s.checkpoint()
    with pytest.raises(ValueError):s.advance(.1,dict(patch,scram=True))
    assert s.checkpoint()==saved


@pytest.mark.parametrize('values',[{'volume_m3':True},{'heat_loss_ua_w_k':-1.},{'minimum_quality':float('nan')}])
def test_header_configuration_domains(values):
    with pytest.raises(ValueError):SteamConfig(**values)


def test_low_quality_header_is_refused_without_projection():
    s=PWRSteamSession();y=s.state.copy();p=iapws.saturation_properties(6.3e6)
    x=.98;rho=1/((1-x)/p[0].rho_kg_m3+x/p[1].rho_kg_m3);u=(1-x)*p[0].u_j_kg+x*p[1].u_j_kg
    i=s.steam_index;y[i]=s.steam_config.volume_m3*rho;y[i+1]=y[i]*u
    with pytest.raises(ValueError,match='phase limit'):s._admit(s._reconstruct_state(y.copy()))
    assert y[i+1]==s.steam_config.volume_m3*rho*u


def test_spatial_core_can_drive_header_and_live_turbine_inlet():
    s=PWRSteamSession(core=CoreConfig(kind='diffusion',cells=2),max_step_s=.2)
    r=s.advance(.1)
    assert len(r['core']['owned_state'])==14 and r['energy_relative']<1e-8


def test_forged_flow_alias_is_refused_at_admission():
    s=PWRSteamSession();y=s.state.copy();y[s.steam_index+2]*=1.01
    with pytest.raises(ValueError,match='alias'):s._admit(y)
