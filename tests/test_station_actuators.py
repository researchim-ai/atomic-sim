import json
import math

import pytest

from atomic_model_station import PWRTrainSession
from atomic_model_station.actuators import RodDrive


def test_exact_drive_has_both_stroke_stops_and_no_overshoot():
    drive=RodDrive()
    for start,target in ((.5,0.),(.5,1.),(.5,.51),(.5,.49)):
        at,motion=drive.trajectory(start,target,False,2.)
        assert at(2.)==start
        assert at(motion['arrival_s'])==target==at(motion['arrival_s']+100.)
        assert at((2.+motion['arrival_s'])/2)==pytest.approx((start+target)/2)
    at,motion=drive.trajectory(.5,0.,True,0.)
    assert at(2.5)==1. and motion['target']==1.


@pytest.mark.parametrize('speed',[0.,True,float('nan'),-.01,1.])
def test_invalid_rod_drive_parameters_refused(speed):
    with pytest.raises(ValueError):RodDrive(operating_speed_s=speed)


def test_requested_position_is_not_assigned_to_actual_position():
    s=PWRTrainSession(max_step_s=.2);initial=s.snapshot();pos=initial['pwr_state'][12]
    r=s.advance(.5,{'rod_target':pos+.02})
    assert r['actuator_status']['rod_requested']==pos+.02
    assert r['pwr_state'][12]==pos+.005
    assert r['powers']['nuclear_heat_w']<initial['powers']['nuclear_heat_w']
    r=s.advance(2.)
    assert r['pwr_state'][12]==pos+.02
    arrivals=[event for event in r['events'] if event['kind']=='rod_arrived']
    assert len(arrivals)==1 and arrivals[0]['time_s']==pytest.approx(2.)


def test_trip_priority_and_latches_survive_request_clearing_and_restart():
    s=PWRTrainSession(max_step_s=.2)
    r=s.advance(.2,{'rod_target':0.,'scram':True,'turbine_trip':True,'generator_connected':True,'valve_request':1.})
    assert r['actuator_status']['scram_latched'] and r['actuator_status']['turbine_trip_latched']
    assert r['actuator_status']['rod_requested']==0. and r['pwr_state'][12]>.25
    assert not r['actuator_status']['effective_generator_connected']
    assert r['powers']['gross_electrical_w']==0. and r['effective_valve']==0.
    saved=json.loads(json.dumps(s.checkpoint()));restored=PWRTrainSession.restore(saved)
    patch={'scram':False,'turbine_trip':False,'rod_target':0.,'generator_connected':True}
    r=s.advance(.2,patch)
    assert restored.advance(.2,patch)==r
    assert r['actuator_status']['scram_latched'] and r['actuator_status']['turbine_trip_latched']
    assert r['effective_valve']==0. and r['powers']['gross_electrical_w']==0.
    assert any(e['kind']=='command_inhibited' for e in r['events'])
    assert not r['actuator_status']['reset_available']


def test_scram_full_stroke_opens_real_bypass_and_rotor_runs_down():
    s=PWRTrainSession(max_step_s=.2);p=s.plant
    secondary_index=13+3*p.network.volume_names().index('sg_secondary')
    initial=s.snapshot();mass=initial['pwr_state'][secondary_index]+initial['hotwell_mass_kg']
    early=s.advance(4.,{'scram':True,'turbine_trip':True})
    late=s.advance(2.)
    assert early['pwr_state'][12]==late['pwr_state'][12]==1.
    assert late['dump_flow_kg_s']>early['dump_flow_kg_s']>0.
    assert late['steam_flow_kg_s']<early['steam_flow_kg_s']
    assert late['turbine_speed_rad_s']<early['turbine_speed_rad_s']
    assert late['powers']['steam_work_w']<late['powers']['turbine_bearing_w']
    assert late['powers']['nuclear_heat_w']<.15*initial['powers']['nuclear_heat_w']
    assert late['pwr_state'][secondary_index]+late['hotwell_mass_kg']==pytest.approx(mass,rel=1e-12)
    b=late['condenser_balance'];assert abs(b['energy_residual_w'])<1e-6 and b['mass_residual_kg_s']==0.
    assert late['energy_relative']<1e-8 and late['mass_relative']<1e-10
    assert late['minimum_sampled_dump_flow_kg_s']>=-late['dump_signed_roundoff_band_kg_s']


@pytest.mark.parametrize('patch',[{'rod_target':True},{'rod_target':1.01},{'scram':1},{'turbine_trip':'yes'},{'boron_ppm':900.}])
def test_bad_or_unsupported_commands_leave_no_accepted_command_or_trip(patch):
    s=PWRTrainSession();before=s.checkpoint()
    with pytest.raises(ValueError):s.advance(.1,patch)
    assert s.checkpoint()==before


def test_boron_has_no_unaccounted_inventory_setter():
    s=PWRTrainSession();r=s.snapshot()
    b=r['actuator_status']['boron']
    assert not b['available'] and not b['direct_inventory_assignment_allowed'] and b['command'] is None
    assert 'finite' in b['reason']


def test_out_of_band_bypass_reverse_flow_refused_without_projection():
    s=PWRTrainSession();y=s.state.copy()
    j=13+3*len(s.plant.network.volumes)+s.plant.network.junction_names().index('dump')
    y[j]=-1e-5
    with pytest.raises(ValueError,match='Reverse steam bypass'):s._admit(y)
    assert y[j]==-1e-5


def test_pending_latches_and_journal_rollback_when_physical_call_is_rejected():
    from atomic_model_station import TrainConfig
    s=PWRTrainSession(TrainConfig(inertia_kg_m2=1e5));before=s.checkpoint()
    # Turbine trip is not requested: manual reactor trip closes steam but leaves
    # the disconnected rotor exposed to the finite initial steam-flow tail.
    with pytest.raises(ValueError):s.advance(1.,{'scram':True,'generator_connected':False})
    assert s.checkpoint()==before and not s.scram_latched and not s.events


def test_qualification_evidence_serializes_numpy_predicates_and_refuses_nan(tmp_path):
    import numpy as np
    from atomic_model_multiphysics.provenance import unseal
    from scripts.qualify_station_actuators import save
    path=tmp_path/'evidence.json'
    save(path,{'trials':[{'coastdown':np.bool_(True),'power':np.float64(12.),'steps':np.int64(3)}]})
    result=unseal(json.loads(path.read_text()))
    assert result=={'trials':[{'coastdown':True,'power':12.,'steps':3}]}
    assert type(result['trials'][0]['coastdown']) is bool
    with pytest.raises(ValueError):save(tmp_path/'bad.json',{'power':np.float64(float('nan'))})
