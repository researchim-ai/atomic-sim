from copy import deepcopy
import numpy as np
import pytest
from atomic_model_thermal_agent import ThermalConfig
from scripts import benchmark_thermal_drift as drift
from scripts.audit_adaptive_thermal_recovery import replay_episode


def test_initial_bias_is_present_at_reset_and_drift_starts_at_zero():
    o = np.zeros(6)
    assert drift.sensor_observation(o, drift.SCENARIOS[1], 0, 16)[0] == -.15
    assert drift.sensor_observation(o, drift.SCENARIOS[3], 0, 16)[0] == 0.
    assert drift.sensor_observation(o, drift.SCENARIOS[3], 15, 16)[0] == -.24
    assert drift.sensor_observation(o, drift.SCENARIOS[3], 20, 16)[0] == -.24
    np.testing.assert_array_equal(o, np.zeros(6))


def test_fault_changes_only_wall_and_preserves_noise_pairing():
    cfg = ThermalConfig(steps=4, action_dt_s=1e-5)
    healthy = drift.DriftEnv(cfg, drift.SCENARIOS[0]); biased = drift.DriftEnv(cfg, drift.SCENARIOS[1])
    a, _ = healthy.reset(seed=7); b, _ = biased.reset(seed=7)
    np.testing.assert_array_equal(a[1:], b[1:]); assert b[0] == a[0] - .15
    for _ in range(3):
        a = healthy.step([.1]); b = biased.step([.1])
        np.testing.assert_array_equal(a[0][1:], b[0][1:]); assert a[4] == b[4]
    c, _ = biased.reset(seed=7); assert c[0] == healthy.reset(seed=7)[0][0] - .15


@pytest.mark.parametrize('kind', ['scenario', 'index', 'steps', 'nan', 'shape'])
def test_invalid_transform_refused(kind):
    o = [0.] * 6; s = drift.SCENARIOS[0]; index = 0; steps = 16
    if kind == 'scenario': s = ('unknown', 0., 0)
    elif kind == 'index': index = True
    elif kind == 'steps': steps = 1
    elif kind == 'nan': o[0] = float('nan')
    else: o.pop()
    with pytest.raises(ValueError): drift.sensor_observation(o, s, index, steps)


def rows():
    return [dict(policy=p, case=c, scenario=list(s), seed=seed, split='test', completed=True,
                 iae_k_s=.0001, energy_j=1., score=.5, alarm_index=None)
            for p in drift.POLICIES for c in range(4) for s in drift.SCENARIOS for seed in drift.SEEDS]


def test_summary_is_order_independent_and_healthy_paired():
    data = rows()
    for row in data:
        if row['scenario'][0] == 'drift_low': row['iae_k_s'] = .001
    result = drift.summarize(data)
    assert result == drift.summarize(list(reversed(data)))
    assert all(r['tracking_degraded'] == 8 for r in result if r['scenario'] == 'drift_low')


@pytest.mark.parametrize('kind', ['missing', 'duplicate', 'nan', 'leak', 'score'])
def test_bad_stress_matrix_refused(kind):
    data = rows()
    if kind == 'missing': data.pop()
    elif kind == 'duplicate': data.append(deepcopy(data[0]))
    elif kind == 'nan': data[0]['energy_j'] = float('nan')
    elif kind == 'leak': data[0]['split'] = 'train'
    else: data[0]['score'] = float('nan')
    with pytest.raises(ValueError): drift.summarize(data)


def test_one_initial_bias_episode_replays_causal_decisions():
    training = dict(selected={'learned': .6}, imitation={'weights': [.1, .2, 0., 0., 0.]})
    row = drift.rollout(ThermalConfig(steps=4, action_dt_s=1e-5, delay_steps=0, slew_w_per_s=1e10),
                        drift.SCENARIOS[1], 'selected', 'model_light', training, 1010)
    assert row['completed'] and replay_episode(row, training)
