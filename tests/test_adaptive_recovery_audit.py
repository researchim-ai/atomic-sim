from copy import deepcopy
import pytest
from atomic_model_thermal_agent import ThermalConfig
from scripts.train_adaptive_thermal_recovery import rollout
from scripts.audit_adaptive_thermal_recovery import replay_episode

TRAINING = dict(selected={'learned': .6}, imitation={'weights': [.1, .2, 0., 0., 0.]})


@pytest.fixture(scope='module')
def episode():
    return rollout(ThermalConfig(steps=4, action_dt_s=1e-5, delay_steps=0, slew_w_per_s=1e10),
                   ('bias_low', -.1, 2), 'model_mid', TRAINING, 810, 'train')


def test_decisions_replay_without_solver(episode):
    assert replay_episode(episode, TRAINING)


@pytest.mark.parametrize('damage', ['state', 'action', 'monitor', 'length', 'power'])
def test_audit_refuses_resealed_decision_corruption(episode, damage):
    row = deepcopy(episode)
    if damage == 'state': row['trace'][2]['recovery']['estimated_wall_k'] += 1.
    elif damage == 'action': row['trace'][2]['action'] += .01
    elif damage == 'monitor': row['trace'][2]['diagnostic']['index'] = 99
    elif damage == 'length': row['trace'].insert(1, deepcopy(row['trace'][0]))
    else: row['trace'][1]['policy_observation'][4] += .01
    with pytest.raises(ValueError): replay_episode(row, TRAINING)
