import pytest
from scripts import station_agent_suite as suite


def test_four_objectives_have_separate_metrics_and_explicit_baselines():
    recipe=suite.recipe()
    assert set(recipe['tasks'])=={'tracking','electrical_budget','diagnosis','recovery'}
    for name,task in recipe['tasks'].items():
        assert task['metrics'] and task['reference_policy'] and task['baseline']
    assert recipe['tasks']['electrical_budget']['minimum_net_export_j']==2e7
    assert 'separate objectives' in recipe['score_aggregation']
    assert not recipe['industrial_utility_validated'] and not recipe['full_operational_matrix']
    assert suite.recipe()==recipe


def test_simple_baselines_need_no_hidden_station_access():
    assert suite.hold_electrical({}, {}, {})=={}
    assert suite.dhr_electrical({}, {}, {})=={'operator':{'dhr_requested':True}}
    assert suite.diagnostic_silent({}, {}, {})=={'action':{},'alarm':False}
    assert suite.invalid_measured({}, {}, {})['alarm'] is False


def test_fresh_and_non_symlink_output_required(tmp_path):
    with pytest.raises(ValueError,match='Fresh'):suite.run(tmp_path)
    link=tmp_path/'link';link.symlink_to(tmp_path,target_is_directory=True)
    with pytest.raises(ValueError,match='Fresh'):suite.run(link/'child')
