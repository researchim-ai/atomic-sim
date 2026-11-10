from dataclasses import replace
from copy import deepcopy
import numpy as np
import pytest
from atomic_model_thermal_agent import ThermalEnv,ThermalConfig
from atomic_model_hem_heated.model import canonical
from atomic_model_multiphysics.provenance import unseal,seal
from scripts.train_thermal_agent import cases,features

@pytest.mark.parametrize('key,value',[('capacity_j_k',99),('noise_std_k',float('nan')),('delay_steps',True),('steps',129),('max_step_s',1e-4)])
def test_invalid_configuration(key,value):
    with pytest.raises(ValueError):ThermalConfig(**{key:value})

@pytest.mark.parametrize('action',[[True],[float('nan')],[-.1],[1.1],[0.,1.]])
def test_invalid_action_is_atomic(action):
    env=ThermalEnv(ThermalConfig(steps=2));env.reset(seed=3);before=env.checkpoint()
    with pytest.raises(ValueError):env.step(action)
    assert canonical(before)==canonical(env.checkpoint())


def test_seed_changes_measurement_not_initial_physics():
    a=ThermalEnv();b=ThermalEnv();oa,_=a.reset(seed=1);ob,_=b.reset(seed=2)
    assert not np.array_equal(oa,ob)
    assert canonical(a.model.checkpoint())==canonical(b.model.checkpoint())


def test_delay_and_slew_are_physical_actuator_behavior():
    env=ThermalEnv(ThermalConfig(steps=3,delay_steps=1,slew_w_per_s=1e7));env.reset(seed=1)
    _,_,_,_,a=env.step([1.]);_,_,_,_,b=env.step([0.])
    assert a['requested_power_w']==100000 and a['applied_power_w']==0
    assert b['delayed_power_w']==100000 and b['applied_power_w']==2500.
    assert abs(b['global_ledger']['energy_residual_j'])<1e-5


def test_replay_preserves_noise_queue_and_continuation():
    env=ThermalEnv(ThermalConfig(steps=3));env.reset(seed=27);env.step([.4])
    saved=env.checkpoint();other=ThermalEnv.restore(saved)
    assert canonical(saved)==canonical(other.checkpoint())
    a=env.step([.1]);b=other.step([.1])
    np.testing.assert_array_equal(a[0],b[0]);assert a[1:]==b[1:]
    bad=unseal(deepcopy(saved));bad['queue'][0]=0.
    with pytest.raises(ValueError):ThermalEnv.restore(seal(bad))


def test_horizon_and_target_change_are_explicit():
    env=ThermalEnv(ThermalConfig(steps=2,noise_std_k=0));obs,_=env.reset(seed=0)
    after,_,terminated,truncated,info=env.step([0.])
    assert info['target_k']==440.4 and after[2]==pytest.approx(.2)
    assert not terminated and not truncated
    assert env.step([0.])[3]
    with pytest.raises(ValueError):env.step([0.])


def test_train_test_parameters_are_disjoint_and_features_exclude_hidden_parameters():
    assert {c.capacity_j_k for c in cases('train')}.isdisjoint(c.capacity_j_k for c in cases('test'))
    assert features(np.array([.1,.2,.3,-10.,.5,0.])).shape==(5,)
