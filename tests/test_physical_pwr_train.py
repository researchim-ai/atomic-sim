import json
import math
import numpy as np
import pytest
from atomic_model_station.pwr_train import PWRTrainSession,TrainConfig
from physics.power.steam_turbine_train import SteamTurbineTrain


def test_coupled_nominal_energy_mass_entropy_and_distinct_rotors():
    s=PWRTrainSession(max_step_s=.5);r=s.advance(.5)
    assert r['reactor_connected'] and r['steam_cycle_connected'] and not r['complete_package_b']
    assert r['energy_relative']<1e-8 and r['mass_relative']<1e-10 and r['boron_relative']<1e-10
    assert abs(r['instantaneous_energy_defect_w'])<.01
    assert r['powers']['gross_electrical_w']>r['powers']['net_export_w']>0.
    assert r['expansion']['outlet_entropy_j_kg_k']>=r['expansion']['inlet_entropy_j_kg_k']
    assert abs(r['fuel_to_clad_w']-r['clad_to_coolant_w'])/r['fuel_to_clad_w']<1e-7
    assert s.config.inertia_kg_m2!=s.plant.inertia and s.base_size==len(s.plant.pack())


def test_load_causally_changes_steam_flow_through_speed_governor():
    a=PWRTrainSession(max_step_s=.3);b=PWRTrainSession(max_step_s=.3)
    x=a.advance(2.,{'load_fraction':.9});y=b.advance(2.,{'load_fraction':.9,'governor_enabled':False})
    assert x['effective_valve']<y['effective_valve'] and x['steam_flow_kg_s']<y['steam_flow_kg_s']
    assert x['turbine_speed_rad_s']<y['turbine_speed_rad_s'] and x['steam_generator_pressure_pa']>y['steam_generator_pressure_pa']
    assert x['clad_temperature_k']>300 and x['energy_relative']<1e-6


def test_frozen_port_mechanical_powers_match_existing_turbine_kernel():
    s=PWRTrainSession();p=s.plant;c=s.config;r=s.snapshot()
    config=dict(inlet_pressure_pa=p.network.volumes['sg_secondary'].pressure_pa,outlet_pressure_pa=p._condenser_pressure_pa,
        isentropic_efficiency=p.eta_turbine,inertia_kg_m2=c.inertia_kg_m2,bearing_drag_nm_s_rad=s.bearing_drag,
        generator_efficiency=c.generator_efficiency,min_omega_rad_s=.9*c.nominal_speed_rad_s,max_omega_rad_s=1.1*c.nominal_speed_rad_s,energy_tolerance_j=1.)
    controls=dict(mass_flow_kg_s=r['steam_flow_kg_s'],generator_conductance_nm_s_rad=s.generator_drag,generator_connected=True)
    kernel=SteamTurbineTrain(config,controls,c.nominal_speed_rad_s)
    work,gen,bearing,_=kernel._powers(r['turbine_energy_j'])
    np.testing.assert_allclose([work,gen,bearing],[r['powers']['steam_work_w'],r['powers']['generator_mechanical_w'],r['powers']['turbine_bearing_w']],rtol=1e-13)


def test_json_restart_replays_actual_thermodynamic_ports_and_clad_state():
    s=PWRTrainSession(max_step_s=.5);s.advance(.2,{'load_fraction':.95});saved=json.loads(json.dumps(s.checkpoint()))
    restored=PWRTrainSession.restore(saved)
    assert restored.advance(.2)==s.advance(.2)


def test_limit_rejection_rolls_back_whole_physical_interval():
    s=PWRTrainSession(TrainConfig(inertia_kg_m2=1e5));before=s.checkpoint()
    with pytest.raises(ValueError):s.advance(1.,{'generator_connected':False,'governor_enabled':False})
    assert s.checkpoint()==before


@pytest.mark.parametrize('patch',[{'load_fraction':True},{'generator_connected':1},{'valve_request':1.5},{'unknown':0.}])
def test_invalid_commands_are_atomic(patch):
    s=PWRTrainSession();before=s.checkpoint()
    with pytest.raises(ValueError):s.advance(.1,patch)
    assert s.checkpoint()==before


def test_unsupported_clad_surface_temperature_rejected_at_initialization():
    with pytest.raises(ValueError):PWRTrainSession(TrainConfig(clad_resistance_fraction=.2))


def test_train_registry_and_snapshot_are_json_admitted_without_source_promotion():
    from atomic_model_station.foundation import validate_registry
    s=PWRTrainSession();assert validate_registry(s.parameter_registry())
    assert all(v['family']=='illustrative' for v in s.parameter_registry()['parameters'].values())
    assert json.loads(json.dumps(s.snapshot()))['complete_package_b'] is False


def test_live_steam_connection_contracts_close_mass_and_shaft_enthalpy_drop():
    s=PWRTrainSession();r=s.advance(.1,{'load_fraction':.95});a=r['steam_port_audit']
    assert a['inlet']['residual_mass_kg_s']==0. and a['outlet']['residual_mass_kg_s']==0.
    assert math.isclose(a['enthalpy_drop_work_w'],r['powers']['steam_work_w'],rel_tol=1e-12)


def test_exact_known_modes_and_domain_aware_jacobian_preserve_stocks():
    s=PWRTrainSession(max_step_s=1.);names=s.plant.network.volume_names()
    index=13+3*names.index('sg_secondary')+2;before=float(s.state[index])
    r=s.advance(10.)
    assert r['pwr_state'][index]==before
    assert {v['name'] for v in r['exact_mode_reduction']}=={'secondary_boron','rod_position','spray'}
    assert r['mass_relative']<1e-10 and r['boron_relative']<1e-10
