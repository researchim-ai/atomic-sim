import math
import pytest
import numpy as np
from atomic_model_thermal_agent import ThermalConfig
from scripts import optimize_thermal_control as energy


def test_score_uses_dimensionless_tracking_and_energy():
    row=dict(completed=True,iae_k_s=.001,energy_j=10.,config=dict(steps=8,action_dt_s=.00025,target_rise_k=.5,max_power_w=100000.))
    assert energy.score(row)==pytest.approx(1.+.25*.05)
    row['completed']=False;assert energy.score(row) is None

@pytest.mark.parametrize('metric,value',[('iae_k_s',float('nan')),('energy_j',-1.),('energy_j',True)])
def test_nonfinite_or_negative_metrics_refused(metric,value):
    row=dict(completed=True,iae_k_s=.001,energy_j=1.,config={});row[metric]=value
    with pytest.raises(ValueError):energy.score(row)


def test_train_selection_excludes_incomplete_and_uses_stable_tie():
    rows=[dict(parameter=.6,all_completed=True,mean_score=.1),dict(parameter=.3,all_completed=True,mean_score=.1),dict(parameter=.15,all_completed=False,mean_score=None)]
    assert energy.select(rows)==energy.select(list(reversed(rows)))==.3

@pytest.mark.parametrize('rows',[[],[dict(parameter=.3,all_completed=False,mean_score=None)],[dict(parameter=.3,all_completed=True,mean_score=math.nan)]])
def test_no_silent_selection_when_training_invalid(rows):
    with pytest.raises(ValueError):energy.select(rows)


def test_tail_keeps_original_target_switch_and_drains_delayed_energy():
    cfg=ThermalConfig(steps=2,delay_steps=2,action_dt_s=1e-5,max_power_w=20000.,slew_w_per_s=1e10)
    env=energy.TailEnv(cfg)
    assert env._target(0)==pytest.approx(440.4) and env._target(1)==pytest.approx(440.2)
    record=energy.rollout(cfg,'hold',1.,np.zeros(5),17)
    assert record['completed'] and record['actuator_drained']
    assert record['active_energy_j']==0. and record['tail_energy_j']>0
    assert record['energy_j']==record['tail_energy_j']
    assert all(s['requested_power_w']==0. for s in record['trace'] if s['tail'])


def test_seed_pairs_do_not_overlap_and_search_budgets_equal():
    assert set(energy.TRAIN_SEEDS).isdisjoint(energy.TEST_SEEDS)
    assert len(energy.SCALES)==len(energy.PI_GAINS)==4
