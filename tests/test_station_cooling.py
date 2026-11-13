import json
import math

import numpy as np
import pytest

from atomic_model_station import CoolingConfig,PWRCooledSession,PWRTrainSession


def test_nominal_sizing_is_explicit_and_auxiliary_power_counted_once():
    s=PWRCooledSession();a=s.snapshot();b=PWRTrainSession().snapshot();c=a['cooling']
    assert not a['cooling_design']['measured'] and not a['cooling_design']['inventories_projected']
    assert a['powers']['own_needs_w']-b['powers']['own_needs_w']==pytest.approx(c['pump_electric_w'])
    assert b['powers']['net_export_w']-a['powers']['net_export_w']==pytest.approx(c['pump_electric_w'])
    assert c['external_removed_w']==pytest.approx(c['condenser_received_w']+c['pump_electric_w'])
    assert abs(c['cold_energy_rate_w'])<1e-4 and abs(c['hot_energy_rate_w'])<1e-4
    assert min(c['entropy_generation_w_k'].values())>=0
    assert a['model_id']!=b['model_id'] and a['finite_cooling_connected']


def test_pump_coastdown_matches_independent_closed_form_and_energy_transfer():
    s=PWRCooledSession(max_step_s=.5);q=s.snapshot()['cooling']['flow_kg_s'];duration=10.
    r=s.advance(duration,{'cooling_pump_on':False});c=r['cooling']
    exact=q/(1+s.cooling_resistance*q*duration/s.cooling_config.pipe_inertance_m_inv)
    assert c['flow_kg_s']==pytest.approx(exact,rel=2e-7)
    assert c['pump_electric_w']==0. and c['kinetic_rate_w']<0 and c['friction_heat_w']>0
    assert c['friction_heat_w']==pytest.approx(-c['kinetic_rate_w'],rel=1e-13)
    assert r['energy_relative']<1e-8 and r['mass_relative']<1e-10
    assert c['hot_temperature_k']>s.cooling_design['initial_hot_temperature_k']


def test_external_water_loss_heats_finite_inventory_and_changes_backpressure():
    a=PWRCooledSession(max_step_s=.5);b=PWRCooledSession(max_step_s=.5)
    disturbed=a.advance(10.,{'external_water_fraction':0.});nominal=b.advance(10.)
    c=disturbed['cooling']
    assert c['external_removed_w']==0. and c['external_flow_kg_s']==0.
    assert c['cold_temperature_k']>nominal['cooling']['cold_temperature_k']
    assert c['hot_temperature_k']>nominal['cooling']['hot_temperature_k']
    assert disturbed['condenser_pressure_pa']>nominal['condenser_pressure_pa']
    assert disturbed['powers']['gross_electrical_w']<nominal['powers']['gross_electrical_w']
    assert disturbed['energy_relative']<1e-8


def test_cooling_checkpoint_replays_controls_temperatures_kinetic_stock_and_ledgers():
    s=PWRCooledSession(max_step_s=.5);s.advance(.5,{'cooling_pump_fraction':.8,'external_water_fraction':.7})
    saved=json.loads(json.dumps(s.checkpoint()));restored=PWRCooledSession.restore(saved)
    assert restored.advance(.5)==s.advance(.5)
    with pytest.raises(ValueError):PWRTrainSession.restore(saved)


def test_external_water_boundary_has_finite_capacity_and_nonnegative_entropy():
    s=PWRCooledSession(max_step_s=.5);r=s.advance(2.,{'external_water_fraction':.5})
    c=r['cooling'];cp=s.cooling_config.heat_capacity_j_kg_k
    assert c['external_inlet_temperature_k']<c['external_outlet_temperature_k']<c['cold_temperature_k']
    assert c['external_removed_w']==pytest.approx(c['external_flow_kg_s']*cp*(c['external_outlet_temperature_k']-c['external_inlet_temperature_k']),rel=1e-12)
    assert min(c['entropy_generation_w_k'].values())>=0.
    assert abs(c['energy_residual_w'])<1e-4 and abs(c['pump_energy_residual_w'])<1e-5


def test_cooling_registry_declares_constant_properties_and_source_families():
    from atomic_model_station.foundation import validate_registry
    s=PWRCooledSession();r=s.parameter_registry();assert validate_registry(r)
    assert r['parameters']['cooling.density_kg_m3']['family']=='illustrative'
    assert r['parameters']['cooling.cold_mass_kg']['value']==1e6
    json.dumps(s.snapshot(),allow_nan=False)


@pytest.mark.parametrize('patch',[{'cooling_pump_on':1},{'cooling_pump_fraction':True},{'external_water_fraction':-1.},
                                 {'external_water_temperature_k':float('nan')},{'cooler_available':0},{'external_water_fraction':2.}])
def test_invalid_mixed_commands_do_not_latch_trip_or_change_cooling(patch):
    s=PWRCooledSession();before=s.checkpoint()
    with pytest.raises(ValueError):s.advance(.1,dict(patch,scram=True))
    assert s.checkpoint()==before


def test_physical_domain_failure_rolls_back_entire_requested_interval():
    s=PWRCooledSession(cooling=CoolingConfig(cold_mass_kg=1e4,hot_mass_kg=1e4),max_step_s=.2);before=s.checkpoint()
    from core.status import DomainError
    with pytest.raises((ValueError,DomainError)):s.advance(30.,{'external_water_fraction':0.,'scram':True,'turbine_trip':True})
    assert s.checkpoint()==before


def test_new_thermal_and_flow_columns_are_not_treated_as_passive_ledgers():
    s=PWRCooledSession();assert all(s._passive_column(i) for i in range(s.ledger_index,s.ledger_index+4))
    assert all(not s._passive_column(i) for i in range(s.cooling_index,s.cooling_index+3))
    y=s.state.copy();r0=s._rhs(0.,y);y[s.cooling_index+1]+=s.cooling_config.hot_mass_kg*s.cooling_config.heat_capacity_j_kg_k
    r1=s._rhs(0.,y)
    assert not np.isclose(r1[s.base_size-2],r0[s.base_size-2])
    assert r1[s.cooling_index+1]!=r0[s.cooling_index+1]
