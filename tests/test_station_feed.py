import json
import pytest
from atomic_model_station import PWRFeedSession,FeedConfig,PWRCooledSession


def test_finite_feed_nominal_energy_and_gas_compression_accounts():
    s=PWRFeedSession(max_step_s=.5);r=s.advance(1.)
    from atomic_model_station.foundation import validate_registry
    assert validate_registry(s.parameter_registry())
    assert r['finite_feed_connected'] and len(r['feed_inventory'])==7
    assert r['energy_relative']<1e-8 and r['mass_relative']<1e-10
    assert abs(r['feed_train']['internal_energy_residual_w'])<1e-5
    assert abs(r['condenser_balance']['energy_residual_w'])<1e-6
    assert abs(r['instantaneous_energy_defect_w'])<.01
    assert r['powers']['feed_pump_electric_w']>r['powers']['feed_pump_extra_loss_w']>0
    assert r['feed_train']['tanks']['feed']['gas_energy_j']>0


def test_feed_pump_trip_closes_check_and_restoration_reopens_with_restart():
    s=PWRFeedSession(max_step_s=.2);r=s.advance(.2,{'feed_pump_on':False})
    assert r['feed_train']['feed_flow_kg_s']==0. and not r['feed_train']['feed_check_open']
    assert r['powers']['feed_pump_electric_w']==0.
    assert r['feed_train']['condensate_flow_kg_s']>0.
    assert any(e['kind']=='feed_check_closed' for e in r['events'])
    saved=json.loads(json.dumps(s.checkpoint()));t=PWRFeedSession.restore(saved)
    r=s.advance(.5,{'feed_pump_on':True});assert t.advance(.5,{'feed_pump_on':True})==r
    assert r['feed_train']['feed_check_open'] and r['feed_train']['feed_flow_kg_s']>0
    assert any(e['kind']=='feed_check_opened' for e in r['events'])
    assert len(s.history)==2


def test_condensate_pump_trip_uses_finite_feed_reserve_and_blocks_reverse_flow():
    s=PWRFeedSession(max_step_s=.2);initial=s.snapshot();r=s.advance(2.,{'condensate_pump_on':False})
    assert r['feed_train']['condensate_flow_kg_s']==0. and not r['feed_train']['condensate_check_open']
    assert r['feed_train']['feed_flow_kg_s']>0.
    assert r['feed_train']['tanks']['feed']['mass_kg']<initial['feed_train']['tanks']['feed']['mass_kg']
    assert r['mass_relative']<1e-10 and r['energy_relative']<1e-8


def test_makeup_and_drain_transfer_mass_and_enthalpy_between_finite_tanks():
    s=PWRFeedSession(max_step_s=.5);a=s.snapshot();b=s.advance(2.,{'makeup_valve':1.,'drain_valve':.4})
    assert b['feed_train']['tanks']['makeup']['mass_kg']<a['feed_train']['tanks']['makeup']['mass_kg']
    assert b['feed_train']['tanks']['drain']['mass_kg']>a['feed_train']['tanks']['drain']['mass_kg']
    assert b['mass_relative']<1e-10 and b['energy_relative']<1e-8
    for tank in b['feed_train']['tanks'].values():assert tank['level_m']>0


def test_saturated_head_command_cannot_assign_requested_feed_flow():
    s=PWRFeedSession(max_step_s=.2);r=s.advance(.1,{'feed_pump_fraction':.1})
    assert r['feed_train']['feed_controller']['saturated']
    assert r['feed_train']['feed_controller']['requested_flow_kg_s']>1000
    assert r['feed_train']['feed_flow_kg_s']==0


@pytest.mark.parametrize('patch',[{'feed_pump_on':1},{'feed_valve_fraction':0.},{'makeup_valve':True},{'condensate_pump_fraction':float('nan')}])
def test_invalid_feed_commands_are_atomic(patch):
    s=PWRFeedSession();before=s.checkpoint()
    with pytest.raises(ValueError):s.advance(.1,dict(patch,scram=True))
    assert s.checkpoint()==before


def test_insufficient_suction_head_rejected_without_cavitation_surrogate():
    with pytest.raises(ValueError,match='Suction'):PWRFeedSession(feed=FeedConfig(suction_static_head_m=1.))


def test_depleted_reserve_rejects_requested_interval_and_rolls_back_all_transfers():
    s=PWRFeedSession(feed=FeedConfig(makeup_mass_kg=21.),max_step_s=.2);before=s.checkpoint()
    with pytest.raises(ValueError):s.advance(1.,{'makeup_valve':1.})
    assert s.checkpoint()==before


def test_checkpoint_cannot_be_loaded_into_cooling_only_backend():
    with pytest.raises((ValueError,KeyError)):PWRCooledSession.restore(PWRFeedSession().checkpoint())
