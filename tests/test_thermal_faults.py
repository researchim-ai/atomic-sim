from copy import deepcopy
import numpy as np
import pytest
from atomic_model_thermal_agent import ThermalConfig
from scripts import benchmark_thermal_faults as faults


def test_bias_changes_only_measurement_and_preserves_common_noise_and_physics():
    cfg=ThermalConfig(steps=4,action_dt_s=1e-5)
    a=faults.SensorEnv(cfg,'healthy');b=faults.SensorEnv(cfg,'bias_low')
    oa,_=a.reset(seed=7);ob,_=b.reset(seed=7);np.testing.assert_array_equal(oa,ob)
    for i in range(3):
        ra=a.step([0.]);rb=b.step([0.]);oa,ob=ra[0],rb[0]
        np.testing.assert_array_equal(oa[1:],ob[1:])
        assert ob[0]-oa[0]==pytest.approx(-.15 if i+1>=faults.ONSET else 0.)
        assert ra[4]['wall_temperature_k']==rb[4]['wall_temperature_k']


def test_stuck_reading_freezes_and_reset_restarts_fault_state():
    env=faults.SensorEnv(ThermalConfig(steps=4,action_dt_s=1e-5),'stuck')
    env.reset(seed=3);values=[env.step([0.])[0][0] for _ in range(4)]
    assert values[1]==values[2]==values[3]
    env.reset(seed=3);assert env._frozen is None
    assert [env.step([0.])[0][0] for _ in range(4)]==values


def actor():return faults.Actor('guarded',{'learned':.6,'pi':100000.},np.zeros(5),.00025,100000.)


def test_monitor_uses_observations_and_latches_after_three_equal_samples():
    a=actor();obs=np.array([0.,0.,.3,-10.,0.,0.])
    assert a.act(obs)==0.;assert a.act(obs)==0.;assert not a.alarm
    assert a.act(obs)==.04 and a.alarm
    obs[0]=1.;assert a.act(obs)==.04


def test_moving_biased_signal_is_not_falsely_claimed_detectable():
    a=actor()
    for value in [-.15,-.14,-.13]:a.act(np.array([value,0.,.3,-10.,0.,0.]))
    assert not a.alarm


def test_nonfinite_policy_input_rejected():
    with pytest.raises(ValueError):actor().act(np.array([float('nan')]*6))


def rows():
    return [dict(fault=f,policy=p,case=c,seed=s,iae_k_s=1.,energy_j=1.,completed=True,alarm_index=None,false_alarm=False)
      for f in faults.FAULTS for p in faults.POLICIES for c in range(4) for s in faults.SEEDS]


def test_paired_reference_is_same_policy_case_and_seed():
    records=rows();r=next(r for r in records if r['fault']=='bias_low' and r['policy']=='learned');r['iae_k_s']=2.
    result=faults.paired(records);row=next(r for r in result if r['fault']=='bias_low' and r['policy']=='learned')
    assert row['mean_delta_iae_vs_healthy_k_s']==.125 and row['tracking_degraded']==1

@pytest.mark.parametrize('damage',['missing','duplicate','nan'])
def test_bad_pairing_evidence_refused(damage):
    records=rows()
    if damage=='missing':records.pop()
    elif damage=='duplicate':records.append(deepcopy(records[0]))
    else:records[0]['energy_j']=float('nan')
    with pytest.raises(ValueError):faults.paired(records)

@pytest.mark.parametrize('kind',['unknown',''])
def test_unsupported_fault_refused(kind):
    with pytest.raises(ValueError):faults.SensorEnv(ThermalConfig(),kind)
