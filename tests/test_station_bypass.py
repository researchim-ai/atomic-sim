import json
import numpy as np
import pytest

from atomic_model_station import PWRBypassSession,PWRStagedSession,BypassConfig,CoreConfig
from atomic_model_station.bypass import nozzle_flow,critical_nozzle
from atomic_model_station.foundation import validate_registry
from atomic_model_multiphysics.provenance import seal,unseal
from physics.materials import iapws


def initial():
    s=PWRStagedSession();p=s.header.thermo
    return float(p.p_pa),float(p.h_j_kg)


def test_nozzle_maximum_matches_independent_saturated_isentrope_grid():
    pressure,h=initial();up=iapws.properties_ph(pressure,h);values=[]
    for p in np.linspace(.2*pressure,.99*pressure,301):
        f,g=iapws.saturation_properties(p);x=(up.s_j_kg_k-f.s_j_kg_k)/(g.s_j_kg_k-f.s_j_kg_k)
        assert 0<=x<=1
        ht=(1-x)*f.h_j_kg+x*g.h_j_kg;rho=1/((1-x)/f.rho_kg_m3+x/g.rho_kg_m3)
        values.append(rho*np.sqrt(2*(h-ht)))
    r=critical_nozzle(pressure,h)
    assert max(values)<=r['mass_flux_kg_m2_s']*(1+1e-10)
    assert abs(max(values)/r['mass_flux_kg_m2_s']-1)<1e-4
    assert r['enthalpy_j_kg']+.5*r['velocity_m_s']**2==pytest.approx(h,rel=1e-12)


def test_choked_plateau_unchoked_branch_and_area_scaling():
    p,h=initial();a=nozzle_flow(p,h,8000.,1.,.01,.8);b=nozzle_flow(p,h,16000.,1.,.01,.8)
    assert a['regime']==b['regime']=='choked' and a['flow_kg_s']==b['flow_kg_s']
    c=nozzle_flow(p,h,.9*p,1.,.01,.8)
    assert c['regime']=='unchoked' and 0<c['flow_kg_s']<a['flow_kg_s']
    d=nozzle_flow(p,h,8000.,.5,.02,.8)
    assert d['flow_kg_s']==pytest.approx(a['flow_kg_s'])
    assert nozzle_flow(p,h,8000.,0.,.01,.8)['flow_kg_s']==0.
    assert nozzle_flow(p,h,p*1.01,1.,.01,.8)['regime']=='blocked'


def test_flux_continuity_at_critical_backpressure():
    p,h=initial();critical=critical_nozzle(p,h)['critical_pressure_pa']
    a=nozzle_flow(p,h,critical*(1-1e-6),1.,.01,.8)
    b=nozzle_flow(p,h,critical*(1+1e-6),1.,.01,.8)
    assert a['flow_kg_s']==pytest.approx(b['flow_kg_s'],rel=1e-6)


def test_closed_bypass_reduces_to_existing_model_without_added_energy_stock():
    old=PWRStagedSession();s=PWRBypassSession()
    assert np.array_equal(s.state[:-1],old.state) and s._stored()==old._stored()
    assert np.array_equal(s._rhs(0.,s.state)[:-1],old._rhs(0.,old.state))
    assert validate_registry(s.parameter_registry())


def test_manual_motion_transfers_header_mass_and_enthalpy_without_shaft_work():
    s=PWRBypassSession(max_step_s=.2)
    r=s.advance(.02,dict(steam_bypass=.2));b=r['header_bypass']
    assert b['position']==pytest.approx(.04) and b['target']==.2
    assert b['flow_kg_s']>0 and b['shaft_work_w']==0.
    assert b['recovered_enthalpy_j_kg']==b['stagnation_enthalpy_j_kg']
    assert abs(r['bypass_port_audit']['energy_residual_w'])<1e-6
    assert abs(r['condenser_balance']['energy_residual_w'])<.001
    assert r['energy_relative']<1e-8 and r['mass_relative']<1e-10
    assert b['entropy_generation_w_k']>0


def test_pressure_switch_never_projects_mass_or_energy():
    s=PWRBypassSession();y=s.state.copy();c=s.bypass_config;p=c.supply_close_pa
    props=iapws.properties_ph(p,s.header.thermo.h_j_kg);i=s.steam_index
    y[i]=s.steam_config.volume_m3*props.rho_kg_m3;y[i+1]=y[i]*props.u_j_kg
    before=y.copy();s._custom_root('bypass_supply_close',y,0.)
    assert np.array_equal(y,before) and not s.bypass_supply_permitted
    assert s.events[-1]['inventories_projected'] is False


def test_trip_releases_actual_finite_header_then_closes_on_supply_floor():
    s=PWRBypassSession(max_step_s=.5)
    early=s.advance(1.,dict(scram=True,turbine_trip=True))
    assert early['header_bypass']['flow_kg_s']>0 and early['steam_flow_kg_s']==0.
    late=s.advance(15.)
    assert late['header_bypass']['flow_kg_s']==0. and not late['header_bypass']['supply_permitted']
    assert late['steam_header']['mass_kg']<early['steam_header']['mass_kg']
    assert late['steam_header']['quality']>=.99
    assert any(e['kind']=='bypass_supply_close' for e in late['events'])
    assert late['energy_relative']<1e-8


def test_pressure_relief_latches_and_has_real_valve_motion():
    s=PWRBypassSession(bypass=BypassConfig(relief_open_pa=6.36e6,relief_close_pa=6.33e6),max_step_s=.2)
    r=s.advance(.1)
    assert r['header_bypass']['relief_latched'] and r['header_bypass']['position']==pytest.approx(.2)
    assert any(e['kind']=='bypass_relief_open' for e in r['events'])


def test_unavailable_bypass_closes_with_finite_motion_not_instant_state_reset():
    s=PWRBypassSession(max_step_s=.2);s.advance(.1,dict(steam_bypass=.2))
    r=s.advance(.02,dict(bypass_available=False));assert r['bypass_position']==pytest.approx(.16)
    r=s.advance(.1);assert r['bypass_position']==0. and r['header_bypass']['flow_kg_s']==0.


def test_restart_preserves_interlocks_and_valve_motion():
    s=PWRBypassSession(max_step_s=.2);s.advance(.1,dict(scram=True,turbine_trip=True))
    saved=json.loads(json.dumps(s.checkpoint()));t=PWRBypassSession.restore(saved)
    assert t.advance(.5)==s.advance(.5)
    with pytest.raises((ValueError,KeyError)):PWRStagedSession.restore(saved)
    p=unseal(saved);p['bypass_source']={}
    with pytest.raises(ValueError,match='source'):PWRBypassSession.restore(seal(p))


@pytest.mark.parametrize('patch',[dict(steam_bypass=1.1),dict(bypass_available=1)])
def test_invalid_commands_preserve_every_inventory_and_latch(patch):
    s=PWRBypassSession();before=s.checkpoint()
    with pytest.raises(ValueError):s.advance(.1,dict(patch,scram=True))
    assert s.checkpoint()==before


@pytest.mark.parametrize('args',[(4e6,2.8e6,8000.,1.,.01,.8),(6e6,2.8e6,8000.,1.1,.01,.8),(6e6,2.8e6,8000.,1.,.04,.8)])
def test_nozzle_outside_declared_domain_is_refused(args):
    with pytest.raises(ValueError):nozzle_flow(*args)


@pytest.mark.parametrize('values',[dict(area_m2=True),dict(discharge_coefficient=float('nan')),dict(supply_close_pa=6.3e6)])
def test_bypass_configuration_domain_and_hysteresis(values):
    with pytest.raises(ValueError):BypassConfig(**values)


def test_spatial_core_and_extra_actuator_coordinate_remain_compatible():
    s=PWRBypassSession(core=CoreConfig(kind='diffusion',cells=2),max_step_s=.2)
    r=s.advance(.02,dict(steam_bypass=.1))
    assert len(r['core']['owned_state'])==14 and r['header_bypass']['flow_kg_s']>0
    assert r['energy_relative']<1e-8
