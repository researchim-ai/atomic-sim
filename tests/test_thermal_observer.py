from copy import deepcopy
import math
import numpy as np
import pytest
from atomic_model_thermal_observer import ThermalMonitor,MonitorConfig
from atomic_model_thermal_observer.monitor import predict
from atomic_model_hem_heated.model import canonical
from atomic_model_multiphysics.provenance import seal,unseal


def obs(wall=0.,power=0.,time=0.):return np.array([wall,0.,.3,-10.,power,time])


def test_exact_affine_wall_update_matches_exponential():
    assert predict(440.,430.,430.,0.,.001,100.,1000.,0.)==pytest.approx(430+10*math.exp(-.01))


def test_observer_uses_applied_power_and_accepts_exact_nominal_model():
    cfg=MonitorConfig(capacity_min=150.,capacity_max=150.,conductance_min=8000.,conductance_max=8000.,ambient_ua_min=20.,ambient_ua_max=20.,noise_margin_k=0.,model_margin_k=1e-10)
    m=ThermalMonitor(cfg);m.observe(obs())
    wall=predict(440.,440.,430.,50000.,cfg.dt_s,150.,8000.,20.)
    diagnostic=m.observe(obs(wall-440.,.5,.1))
    assert not diagnostic['alarm'] and diagnostic['inconsistency_gap_k']==0.
    wrong=ThermalMonitor(cfg);wrong.observe(obs())
    assert wrong.observe(obs(wall-440.,0.,.1))['alarm']


def test_bias_jump_is_inconsistency_without_unique_fault_diagnosis():
    m=ThermalMonitor();m.observe(obs());r=m.observe(obs(-.15,time=.1))
    assert r['alarm'] and r['first_alarm']['index']==1
    assert r['reasons']==['measurement_model_inconsistency']
    assert 'no unique fault diagnosis' in r['interpretation']


def test_repeat_monitor_and_alarm_are_latched():
    m=ThermalMonitor();m.observe(obs());m.observe(obs());r=m.observe(obs())
    assert 'repeated_reading' in r['reasons']
    assert m.observe(obs(.001,time=.1))['alarm']


def test_replay_and_snapshots_are_detached_from_live_monitor():
    m=ThermalMonitor();m.observe(obs());r=m.observe(obs(-.15,time=.1));saved=m.checkpoint()
    other=ThermalMonitor.restore(saved);assert canonical(saved)==canonical(other.checkpoint())
    r['first_alarm']['reasons'].append('forged');saved['payload']['last']['alarm']=False
    assert m.checkpoint()['payload']['last']['alarm']
    bad=unseal(other.checkpoint());bad['history'][1][0]=.5
    with pytest.raises(ValueError):ThermalMonitor.restore(seal(bad))

@pytest.mark.parametrize('key,value',[('dt_s',0),('noise_margin_k',float('nan')),('capacity_min',True),('capacity_max',99),('conductance_min',0),('repeated_samples',True)])
def test_invalid_monitor_config_refused(key,value):
    with pytest.raises(ValueError):MonitorConfig(**{key:value})

@pytest.mark.parametrize('damage',['nan','power','temperature','shape'])
def test_invalid_sample_is_atomic(damage):
    m=ThermalMonitor();m.observe(obs());before=m.checkpoint();sample=obs()
    if damage=='nan':sample[0]=float('nan')
    elif damage=='power':sample[4]=1.01
    elif damage=='temperature':sample[0]=1e200
    else:sample=sample[:5]
    with pytest.raises(ValueError):m.observe(sample)
    assert canonical(m.checkpoint())==canonical(before)


def test_model_mismatch_also_causes_residual_not_a_sensor_verdict():
    m=ThermalMonitor(MonitorConfig(capacity_min=100.,capacity_max=100.,conductance_min=1000.,conductance_max=1000.,ambient_ua_min=0.,ambient_ua_max=0.,noise_margin_k=0.,model_margin_k=.001))
    m.observe(obs());actual=predict(440.,440.,430.,100000.,.00025,500.,1000.,0.)
    r=m.observe(obs(actual-440.,1.,.1))
    assert r['alarm'] and r['reasons']==['measurement_model_inconsistency']
