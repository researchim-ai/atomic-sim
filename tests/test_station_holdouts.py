from dataclasses import replace
import pytest
from atomic_model_station_env import EnvConfig
from atomic_model_station_faults import Fault
from atomic_model_station_evaluation import EvaluationCase, hold_policy
from atomic_model_station_holdouts import audit_splits, evaluate_holdouts


def cohort():
    cases = [EvaluationCase(str(i), EnvConfig(max_steps=5, tick_s=.01,
        automatic=True, inventory_deadband_m=i*.0004,
        faults=(Fault('fault', 'bias', 'pressure', i+1, value=(i+1)*150000.),)),
        'single_station', 'label'+str(i), 'arbitrary', 'arbitrary') for i in range(3)]
    return cases, {str(i): split for i, split in enumerate(('train','dev','test'))}


def test_audit_derives_factors_and_copies_inputs():
    cases, assignment = cohort()
    a = audit_splits(cases, assignment, axes=['inventory_deadband_m', 'fault_magnitude', 'fault_onset_s'])
    assert a == audit_splits(list(reversed(cases)), dict(reversed(list(assignment.items()))),
                             axes=['fault_onset_s', 'fault_magnitude', 'inventory_deadband_m'])
    assert a['physical_factor_disjoint'] and not a['unseen_objects_validated']
    assert not a['operating_history_holdout'] and not a['statistical_independence_proven']
    assignment['0'] = 'test'
    assert a['cases'][0]['split'] == 'train'


@pytest.mark.parametrize('axes', [[], ['object_family'], ['seed'], ['rtol'], ['max_steps'],
                                 ['parameter_family'], ['fault_onset_s']*2, [None]])
def test_unphysical_or_invalid_axes_refused(axes):
    with pytest.raises(ValueError, match='supported physical axes'):
        audit_splits(*cohort(), axes=axes)


@pytest.mark.parametrize('axis', ['inventory_deadband_m', 'fault_magnitude', 'fault_onset_s'])
def test_new_labels_do_not_hide_actual_factor_leakage(axis):
    cases, assignment = cohort()
    c0, c2 = cases[0].config, cases[2].config
    if axis == 'inventory_deadband_m':
        cfg = replace(c2, inventory_deadband_m=c0.inventory_deadband_m)
    else:
        f0, f2 = c0.faults[0], c2.faults[0]
        cfg = replace(c2, faults=(replace(f2, name='disguised',
            **({'value': f0.value} if axis == 'fault_magnitude' else {'start_tick': f0.start_tick})),))
    cases[2] = replace(cases[2], config=cfg)
    with pytest.raises(ValueError, match='Physical cross-split leakage on '+axis):
        audit_splits(cases, assignment, axes=[axis])


def test_shared_mechanism_rejected_even_if_amplitude_and_labels_differ():
    with pytest.raises(ValueError, match='fault_mechanism'):
        audit_splits(*cohort(), axes=['fault_mechanism'])


def test_mixed_fault_sets_cannot_hide_one_shared_factor():
    cases, assignment = cohort()
    cfg = replace(cases[2].config, faults=(*cases[2].config.faults,
        Fault('second', 'bias', 'pressure_reference', 1, value=200000.)))
    cases[2] = replace(cases[2], config=cfg)
    with pytest.raises(ValueError, match='fault_onset_s'):
        audit_splits(cases, assignment, axes=['fault_onset_s'])


def test_no_fault_and_unused_magnitude_cannot_establish_fault_holdout():
    cases, assignment = cohort()
    cases[0] = replace(cases[0], config=replace(cases[0].config, faults=()))
    with pytest.raises(ValueError, match='exercised fault'):
        audit_splits(cases, assignment, axes=['fault_onset_s'])
    cases, assignment = cohort()
    cases[0] = replace(cases[0], config=replace(cases[0].config,
        faults=(Fault('unused', 'failure', 'pressure', 1),)))
    with pytest.raises(ValueError, match='active magnitude'):
        audit_splits(cases, assignment, axes=['fault_magnitude'])


def test_future_fault_rejected_before_policy_runs():
    cases, assignment = cohort()
    cases[0] = replace(cases[0], config=replace(cases[0].config,
        faults=(Fault('future', 'bias', 'pressure', 5, value=10.),)))
    with pytest.raises(ValueError, match='Unexercised'):
        evaluate_holdouts(cases, assignment, lambda: pytest.fail('must not run'), axes=['fault_onset_s'])


def test_numeric_fault_times_use_seconds_not_tick_labels():
    cases, assignment = cohort()
    cases[2] = replace(cases[2], config=replace(cases[2].config, tick_s=.02,
        faults=(replace(cases[2].config.faults[0], start_tick=1),)))
    with pytest.raises(ValueError, match='fault_onset_s'):
        audit_splits(cases, assignment, axes=['fault_onset_s'])


def test_real_station_results_keep_failed_episode_in_denominator():
    cases, assignment = cohort()
    seen=[]
    def factory():
        def policy(obs, schema):
            assert 'config' not in obs and 'faults' not in schema
            assert 'label2' not in str((obs,schema))
            seen.append(obs['time_s'])
            return {}
        return policy
    result = evaluate_holdouts(cases, assignment, factory, axes=['fault_magnitude'])
    assert result['summary']['attempted'] == result['summary']['full_horizon_completed'] == 1
    assert seen == pytest.approx([0,.01,.02,.03,.04])
    bad = evaluate_holdouts(cases, assignment,
        lambda: lambda *_: {'operator': {'load_fraction':99}}, axes=['fault_magnitude'])
    assert bad['summary']['attempted'] == bad['summary']['execution_failures'] == 1
    assert bad['summary']['full_horizon_completed'] == 0
    assert bad['evaluation']['episodes'][0]['failure']['code'] == 'invalid_action'


def test_early_termination_not_counted_as_full_horizon(monkeypatch):
    from atomic_model_station_holdouts import model
    monkeypatch.setattr(model, 'evaluate', lambda *a, **kw: dict(episodes=[dict(
        case='2', status='completed', simulated_s=.01, wall_s=.1)]))
    r = evaluate_holdouts(*cohort(), lambda:hold_policy, axes=['fault_magnitude'])
    assert r['summary']['early_terminations'] == 1
    assert r['summary']['full_horizon_completed'] == 0


def test_shared_arbitrary_labels_do_not_override_actual_factors():
    cases, assignment = cohort()
    cases = [replace(c, parameter_family='same', object_family='same',
                     fault_family='same', time_family='same') for c in cases]
    assert audit_splits(cases, assignment, axes=['fault_magnitude'])['physical_factor_disjoint']


def test_manual_deadband_is_not_an_effective_parameter_holdout():
    cases, assignment = cohort()
    cases[0] = replace(cases[0], config=replace(cases[0].config, automatic=False))
    with pytest.raises(ValueError, match='automatic control'):
        audit_splits(cases, assignment, axes=['inventory_deadband_m'])


def test_numeric_representation_cannot_hide_deadband_overlap():
    cases, assignment = cohort()
    cases[2] = replace(cases[2], config=replace(cases[2].config, inventory_deadband_m=0))
    with pytest.raises(ValueError, match='inventory_deadband_m'):
        audit_splits(cases, assignment, axes=['inventory_deadband_m'])


@pytest.mark.parametrize('mutation', ['seed','rtol','names'])
def test_solver_noise_or_renaming_cannot_create_physical_scenario(mutation):
    cases, assignment = cohort()
    cfg = cases[0].config
    if mutation == 'names':
        cfg = replace(cfg, faults=(replace(cfg.faults[0], name='renamed'),))
    else:
        cfg = replace(cfg, **({'seed':42} if mutation == 'seed' else {'rtol':2e-8}))
    cases[2] = replace(cases[2], config=cfg)
    with pytest.raises(ValueError, match='scenario'):
        audit_splits(cases, assignment, axes=['fault_magnitude'])


@pytest.mark.parametrize('cases,assignment', [(None,{}), ([],{}), ([{}],{}), (*cohort()[:1],{})])
def test_invalid_collection_or_assignment_refused(cases, assignment):
    with pytest.raises(ValueError):
        audit_splits(cases, assignment, axes=['fault_magnitude'])


def test_signed_zero_does_not_create_new_deadband():
    cases, assignment = cohort()
    cases[2] = replace(cases[2], config=replace(cases[2].config, inventory_deadband_m=-0.0))
    with pytest.raises(ValueError, match='inventory_deadband_m'):
        audit_splits(cases, assignment, axes=['inventory_deadband_m'])


@pytest.mark.parametrize('fault', [Fault('zero','bias','pressure',1,value=-0.),
    Fault('zero','drift','pressure',1,value=0),
    Fault('zero','cooling_loss','external_water',1,value=1)])
def test_no_effect_fault_cannot_create_held_out_conditions(fault):
    cases, assignment = cohort()
    cases[0] = replace(cases[0], config=replace(cases[0].config, faults=(fault,)))
    with pytest.raises(ValueError, match='No-effect fault'):
        audit_splits(cases, assignment, axes=['fault_onset_s'])
