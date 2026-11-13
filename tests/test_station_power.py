import json
import pytest
from atomic_model_station import PWRPowerSession,BackupConfig,PWRFeedSession


def test_grid_loss_overrides_drive_requests_and_latches_declared_trip():
    s=PWRPowerSession(max_step_s=.2)
    r=s.advance(.2,{'grid_available':False,'feed_pump_on':True,'condensate_pump_on':True,'cooling_pump_on':True,'primary_pump_on':True})
    e=r['electrical']
    assert not any(e['effective_drives'].values()) and not e['main_bus_available']
    assert r['controls']['feed_pump_on'] and r['controls']['primary_pump_on']
    assert e['emergency_latched'] and r['actuator_status']['scram_latched'] and r['actuator_status']['turbine_trip_latched']
    assert r['powers']['gross_electrical_w']==r['powers']['net_export_w']==r['powers']['grid_import_w']==0.
    assert e['dhr_active'] and e['dhr_from_backup'] and r['powers']['own_needs_w']==1e5
    assert r['energy_relative']<1e-8 and r['mass_relative']<1e-10
    kinds=[event['kind'] for event in r['events']]
    assert kinds.index('grid_lost')<kinds.index('power_loss_trip')<kinds.index('main_bus_deenergized')


def test_exact_energy_use_and_exhaustion_switch_with_json_restart():
    s=PWRPowerSession(backup=BackupConfig(initial_energy_j=1e5),max_step_s=.2)
    a=s.advance(.5,{'grid_available':False});assert a['electrical']['backup_energy_j']==pytest.approx(5e4,abs=1e-6)
    saved=json.loads(json.dumps(s.checkpoint()));t=PWRPowerSession.restore(saved)
    r=s.advance(.75);assert t.advance(.75)==r
    assert r['electrical']['backup_energy_j']==0. and r['electrical']['backup_exhausted']
    assert not r['electrical']['dhr_active'] and r['backup_used_j']==pytest.approx(1e5,abs=1e-6)
    assert r['dhr_transferred_j']==pytest.approx(1e8,rel=1e-8)
    event=[event for event in r['events'] if event['kind']=='backup_exhausted']
    assert len(event)==1 and event[0]['time_s']==pytest.approx(1.,abs=1e-9)
    assert abs(event[0]['root_residual_j'])<=1e-3
    assert len(s.history)==2


def test_grid_restoration_does_not_recharge_or_reset_and_avoids_zero_drive_chatter():
    s=PWRPowerSession(backup=BackupConfig(initial_energy_j=1e5),max_step_s=.2)
    s.advance(1.25,{'grid_available':False})
    r=s.advance(.5,{'grid_available':True})
    assert r['electrical']['main_bus_available'] and r['electrical']['backup_energy_j']==0.
    assert r['electrical']['dhr_active'] and not r['electrical']['dhr_from_backup']
    assert r['actuator_status']['scram_latched'] and r['actuator_status']['turbine_trip_latched']
    assert r['powers']['grid_import_w']>0. and r['powers']['gross_electrical_w']==0.
    assert r['feed_train']['check_opening_band_pa']==1.
    assert len(r['events'])<20


def test_actual_grid_exchange_is_separate_from_net_generation_and_backup_use():
    s=PWRPowerSession(max_step_s=.2);r=s.advance(.5,{'grid_available':False})
    assert r['net_exported_j']==0. and r['net_generated_j']<0.
    assert r['net_generated_j']+r['backup_used_j']==pytest.approx(r['net_exported_j'],abs=1e-6)
    assert r['gross_generated_j']-r['own_needs_j']==pytest.approx(r['net_generated_j'],abs=1e-6)
    assert r['electrical']['energy_transfer_residual_w']==0.


def test_backup_disabled_has_no_free_dhr_heat_and_causal_primary_temperature_difference():
    a=PWRPowerSession(max_step_s=.2);b=PWRPowerSession(max_step_s=.2)
    x=a.advance(.5,{'grid_available':False});y=b.advance(.5,{'grid_available':False,'backup_enabled':False})
    assert x['powers']['dhr_heat_w']>0 and y['powers']['dhr_heat_w']==0.
    assert y['electrical']['backup_energy_j']==5e6 and y['backup_used_j']==0.
    assert x['electrical']['dhr_donor_temperature_k']<y['electrical']['dhr_donor_temperature_k']
    assert x['electrical']['dhr_receiving_temperature_k']>y['electrical']['dhr_receiving_temperature_k']
    assert x['electrical']['dhr_entropy_generation_w_k']>0.


def test_grid_supplies_requested_dhr_without_discharging_backup():
    s=PWRPowerSession(max_step_s=.2);r=s.advance(.2,{'dhr_requested':True})
    assert r['electrical']['dhr_active'] and not r['electrical']['dhr_from_backup']
    assert r['electrical']['backup_energy_j']==5e6 and r['backup_used_j']==0.
    assert r['dhr_transferred_j']>0. and r['powers']['dhr_electric_w']==1e5
    assert r['energy_relative']<1e-8


@pytest.mark.parametrize('patch',[{'grid_available':1},{'backup_enabled':0},{'primary_pump_on':'yes'},{'dhr_requested':None}])
def test_invalid_power_commands_do_not_trip_or_mutate_any_reserve(patch):
    s=PWRPowerSession();before=s.checkpoint()
    with pytest.raises(ValueError):s.advance(.1,patch)
    assert s.checkpoint()==before


def test_failed_physical_call_rolls_back_power_trip_and_backup_use():
    from atomic_model_station import TrainConfig
    s=PWRPowerSession(TrainConfig(inertia_kg_m2=1e5));before=s.checkpoint()
    with pytest.raises(ValueError):s.advance(1.,{'grid_available':False})
    assert s.checkpoint()==before and not s.emergency_latched


def test_power_registry_and_backend_identity_are_source_bound():
    from atomic_model_station.foundation import validate_registry
    s=PWRPowerSession();assert validate_registry(s.parameter_registry())
    assert s.parameter_registry()['parameters']['backup.initial_energy_j']['family']=='illustrative'
    assert len(s.snapshot()['feed_inventory'])==7
    with pytest.raises((ValueError,KeyError)):PWRFeedSession.restore(s.checkpoint())


def test_electric_integral_tolerance_scales_with_units_and_rejects_real_imbalance():
    from scripts.qualify_station_power import grid_energy_admitted
    assert grid_energy_admitted(dict(net_exported_j=1e12+.02,net_generated_j=1e12,backup_used_j=0.))
    assert not grid_energy_admitted(dict(net_exported_j=1e12+100.,net_generated_j=1e12,backup_used_j=0.))
    assert not grid_energy_admitted(dict(net_exported_j=.02,net_generated_j=0.,backup_used_j=0.))
