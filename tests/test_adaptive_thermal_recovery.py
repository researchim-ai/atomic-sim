from copy import deepcopy
import pytest
from atomic_model_multiphysics.provenance import seal, unseal
from atomic_model_thermal_recovery import RecoveryConfig, ThermalRecovery


def obs(wall=0., power=.2):
    return [wall, 0., .3, -10., power, .2]


def test_normal_passes_through_and_alarm_transition_is_limited():
    c = ThermalRecovery()
    assert c.act(obs(), False, .8)['action'] == .8
    r = c.act(obs(-.3), True, 0.)
    assert abs(r['action'] - .8) <= .1 + 1e-15
    assert c.act(obs(), False, 1.)['alarm']


def test_alarm_freezes_wall_assimilation_but_retains_other_measurements():
    a = ThermalRecovery(); b = ThermalRecovery()
    a.act(obs(), False, .2); b.act(obs(), False, .2)
    for _ in range(4):
        assert a.act(obs(-.4), True, .1) == b.act(obs(.4), True, .9)
    changed = obs(); changed[2] = .6
    assert a.act(changed, True, .1) != b.act(obs(), True, .1)


def test_exact_checkpoint_continuation_and_detachment():
    c = ThermalRecovery(); c.act(obs(), False, .3); c.act(obs(-.2), True, .1)
    saved = c.checkpoint(); restored = ThermalRecovery.restore(saved)
    saved['payload']['history'][0]['observation'][0] = 100.
    assert restored.act(obs(), False, .5) == c.act(obs(), False, .5)
    assert restored.checkpoint() == c.checkpoint()


@pytest.mark.parametrize('bad', ['nan', 'bool', 'shape', 'power', 'alarm', 'normal'])
def test_bad_input_does_not_mutate_state(bad):
    c = ThermalRecovery(); before = c.checkpoint(); o = obs(); alarm = False; normal = .1
    if bad == 'nan': o[0] = float('nan')
    elif bad == 'bool': o = [True] * 6
    elif bad == 'shape': o.pop()
    elif bad == 'power': o[4] = 2.
    elif bad == 'alarm': alarm = 1
    else: normal = True
    with pytest.raises(ValueError): c.act(o, alarm, normal)
    assert c.checkpoint() == before


@pytest.mark.parametrize('key,value', [('dt_s', 0.), ('gain_w_k', float('nan')), ('max_command_step', True)])
def test_invalid_configuration(key, value):
    with pytest.raises(ValueError): RecoveryConfig(**{key: value})


def test_resealed_checkpoint_tampering_refused():
    c = ThermalRecovery(); c.act(obs(), False, .2)
    payload = unseal(deepcopy(c.checkpoint())); payload['estimate'] += 1.
    with pytest.raises(ValueError): ThermalRecovery.restore(seal(payload))


def test_sample_budget_and_initial_alarm():
    c = ThermalRecovery()
    assert abs(c.act(obs(), True, .8)['action'] - .2) <= .1 + 1e-15
    for _ in range(255): c.act(obs(), True, .8)
    before = c.checkpoint()
    with pytest.raises(ValueError): c.act(obs(), True, .8)
    assert c.checkpoint() == before


def train_rows():
    from scripts import train_adaptive_thermal_recovery as experiment
    return [dict(candidate=k, case=c, scenario=list(s), seed=810, split='train', completed=True, score=.5)
            for k in experiment.CANDIDATES for c in range(2) for s in experiment.TRAIN]


def test_adaptive_selection_is_order_independent_and_train_only():
    from scripts import train_adaptive_thermal_recovery as experiment
    rows = train_rows()
    for row in rows:
        if row['candidate'] == 'model_mid': row['score'] = .3
    result = experiment.select(rows)
    assert result['selected'] == 'model_mid' and result['test_used'] is False
    assert experiment.select(list(reversed(rows))) == result


@pytest.mark.parametrize('damage', ['missing', 'duplicate', 'leak', 'nan', 'failed'])
def test_adaptive_selection_refuses_invalid_matrix(damage):
    from scripts import train_adaptive_thermal_recovery as experiment
    rows = train_rows()
    if damage == 'missing': rows.pop()
    elif damage == 'duplicate': rows.append(deepcopy(rows[0]))
    elif damage == 'leak': rows[0]['split'] = 'test'
    elif damage == 'nan': rows[0]['score'] = float('nan')
    else:
        for row in rows: row['completed'] = False
    with pytest.raises(ValueError): experiment.select(rows)
