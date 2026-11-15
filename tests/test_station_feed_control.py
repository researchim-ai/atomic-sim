import json
import math
from copy import deepcopy

import numpy as np
import pytest
from atomic_model_station import PWRBypassSession
from atomic_model_station.levels import liquid_level,collapsed_level
from atomic_model_station_control import feed_control_session,ControlSession,PI,PIConfig
from physics.materials import iapws


def test_level_geometry_has_independent_volume_fraction_reference():
    s=PWRBypassSession();r=s.snapshot();v=s.plant.network.volumes['sg_secondary']
    f,g=iapws.saturation_properties(v.pressure_pa)
    liquid_volume=v.volume_m3*((v.mass_kg/v.volume_m3)-g.rho_kg_m3)/(f.rho_kg_m3-g.rho_kg_m3)
    assert r['level_readouts']['sg_collapsed_level_m']==pytest.approx(liquid_volume/s.feed_config.sg_area_m2,rel=1e-12)
    liquid_mass=s.plant.hotwell_mass_kg*(1-s.plant.condenser.state().quality)
    assert r['level_readouts']['hotwell_level_m']==pytest.approx(liquid_mass/s.plant.true_hotwell['rho']/s.feed_config.hotwell_area_m2)
    assert liquid_mass<s.plant.hotwell_mass_kg
    assert r['level_readouts']['feed_tank_level_m']==r['feed_train']['tanks']['feed']['level_m']
    assert r['level_readouts']['sg_outflow_kg_s']==pytest.approx(r['sg_steam_flow_kg_s']+r['dump_flow_kg_s'])
    assert liquid_level(1000.,1000.,2.)==.5
    assert collapsed_level(1000.,.5,1000.,2.,4.)==.25


@pytest.mark.parametrize('args',[(1.,-1.,1000.,1.,1.),(1.,1.1,1000.,1.,1.),(10000.,0.,1000.,1.,1.)])
def test_geometry_refuses_invalid_phase_or_impossible_liquid_volume(args):
    with pytest.raises(ValueError):collapsed_level(*args)


def test_external_feed_request_changes_head_not_inventories_or_instantaneous_flow():
    s=PWRBypassSession();before=s.state.copy();q=s.snapshot()['feed_train']['feed_flow_kg_s']
    s._set_controls(dict(feed_flow_control=True,feed_flow_fraction=.55))
    rate=s._rhs(0.,s.state)
    assert np.array_equal(s.state,before)
    assert s.plant.network.junctions['feed'].mass_flow_kg_s==q
    assert s.plant.feed_controller['requested_flow_kg_s']==pytest.approx(.55*s.feed_flow_scale_kg_s)
    assert 0<s.plant.feed_controller['delivered_head_pa']<=s.plant.maximum_feed_head_pa
    assert rate[s.plant.feed_index]>0


@pytest.mark.parametrize('fraction',[.3,.5,.7])
def test_split_range_is_exclusive_and_conserves_finite_tank_mass(fraction):
    s=PWRBypassSession();base=s._rhs(0.,s.state).copy()
    s._set_controls(dict(feed_inventory_control=True,feed_inventory_fraction=fraction))
    changed=s._rhs(0.,s.state);t=s.feed_terms;i=s.feed_index
    assert t['makeup_opening']*t['drain_opening']==0.
    assert t['makeup_flow_kg_s']*t['drain_flow_kg_s']==0.
    assert sum(changed[j]-base[j] for j in (i,i+2,i+4))==pytest.approx(0.,abs=1e-10)
    assert changed[i]-base[i]==pytest.approx(t['makeup_flow_kg_s']-t['drain_flow_kg_s'],abs=1e-10)


@pytest.mark.parametrize('patch',[dict(feed_flow_control=1),dict(condensate_flow_fraction=1.1),dict(feed_inventory_control=1),dict(feed_inventory_fraction=True)])
def test_bad_feed_command_rolls_back_whole_plant_and_trip(patch):
    s=PWRBypassSession();before=s.checkpoint()
    with pytest.raises(ValueError):s.advance(.1,dict(patch,scram=True))
    assert s.checkpoint()==before


def test_feedforward_is_causal_and_part_of_bumpless_tracking():
    p=PI(PIConfig('x','feed_flow_fraction',1.,1.,bias=.5,kp=.2,ki=0.))
    first=p.propose(1.,1.,.5,1.,entering=True,feedforward=.1)
    assert first['command']==pytest.approx(.5)
    p.commit(first,.5,1.,automatic=True)
    second=p.propose(1.,1.,.5,1.,feedforward=.2)
    assert second['command']==pytest.approx(.6)
    p.commit(second,.6,1.,automatic=False)
    assert p.propose(1.,1.,.6,1.,entering=True,feedforward=.2)['command']==pytest.approx(.6)


def test_six_loops_enable_real_inner_flow_servo_without_public_flow_truth_leak():
    s=feed_control_session();r=s.step(modes={k:'auto' for k in s.loops})
    assert len(s.loops)==6 and not r['receipt']['trip_latched']
    assert s.plant.controls['feed_flow_control'] and s.plant.controls['condensate_flow_control'] and s.plant.controls['feed_inventory_control']
    for name in ('sg_level','feed_level','hotwell_level'):assert r['observation']['measurements'][name]['unit']=='m'
    assert 'feed_flow_fraction' not in r['receipt']['actuator_feedback']
    assert 'condensate_flow_fraction' not in r['receipt']['actuator_feedback']
    assert s.plant.snapshot()['energy_relative']<1e-8


def test_external_flow_manual_auto_transfer_and_sorted_file_restart():
    s=feed_control_session();s.step(modes={k:'auto' for k in s.loops})
    s.step(modes={'feed_flow_fraction':'manual'})
    assert s.plant.controls['feed_flow_control']
    before=s.instruments.operator_frame()['measurements']['feed_flow']['value']/s.plant.feed_flow_scale_kg_s
    r=s.step(modes={'feed_flow_fraction':'auto'})
    assert r['receipt']['decisions']['feed_flow_fraction']['selected']==pytest.approx(before)
    saved=json.loads(json.dumps(s.checkpoint(),sort_keys=True));restored=ControlSession.restore(saved)
    assert restored.step()==s.step()


def test_missing_level_measurement_trips_active_feed_controller_without_truth_fallback():
    from atomic_model_station_signals import StationSignals,Sensor
    from atomic_model_station_control import ProtectionConfig
    base=feed_control_session()
    specs=[Sensor(**dict(s.__dict__,delay_ticks=2)) if s.name=='sg_level' else s for s in base.instruments.sensors]
    plant=PWRBypassSession();bank=StationSignals(specs)
    owner=ControlSession(plant,instruments=bank,loops=[p.config for p in base.loops.values()],protection=ProtectionConfig())
    r=owner.step(modes={k:'auto' for k in owner.loops})
    assert r['receipt']['trip_latched'] and 'measurement_unavailable:sg_level' in r['receipt']['trip_reasons']
    assert not owner.plant.controls['feed_flow_control'] and not owner.plant.controls['condensate_flow_control']
    assert not owner.plant.controls['feed_inventory_control']


def test_hotwell_geometry_rejects_insufficient_capacity_at_initialization():
    from atomic_model_station import FeedConfig
    with pytest.raises(ValueError,match='geometric capacity'):
        PWRBypassSession(feed=FeedConfig(hotwell_area_m2=1.,hotwell_sump_height_m=.1))


def test_level_arithmetic_refuses_nonfinite_result_and_oversized_integer():
    for args in [(10**400,1000.,1.),(1e300,1e-300,1.)]:
        with pytest.raises(ValueError):liquid_level(*args)


@pytest.mark.parametrize('area',[50.,100.,200.])
def test_hotwell_area_is_shared_by_level_and_suction_head(area):
    from atomic_model_station import FeedConfig
    s=PWRBypassSession(feed=FeedConfig(hotwell_area_m2=area))
    liquid=s.plant.hotwell_mass_kg*(1-s.plant.condenser.state().quality)
    expected=s.feed_config.suction_static_head_m+(liquid-.95*s.plant.condenser_mass_ref_kg)/(s.feed_rho*area)
    assert s.snapshot()['condensate_suction_m']==pytest.approx(expected)
    assert s.snapshot()['level_readouts']['hotwell_level_m']==pytest.approx(liquid/s.plant.true_hotwell['rho']/area)


def test_derated_feed_pump_cannot_deliver_requested_flow_above_available_head():
    s=PWRBypassSession(max_step_s=.1)
    initial=s.snapshot()['feed_train']['feed_flow_kg_s']
    final=s.advance(.05,dict(feed_flow_control=True,feed_flow_fraction=1.,feed_pump_fraction=.72))
    head=final['feed_train']['feed_controller']
    available=s.plant.maximum_feed_head_pa*.72**2
    assert head['saturated']
    assert head['requested_head_pa']>available
    assert head['delivered_head_pa']==available
    assert final['feed_train']['feed_flow_kg_s']<initial<head['requested_flow_kg_s']
    assert final['mass_relative']<1e-10
    assert final['energy_relative']<1e-6
