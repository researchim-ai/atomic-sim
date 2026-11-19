import json
import pytest
from atomic_model_station_examples import NAMES,run_example
from atomic_model_station_examples import model


@pytest.mark.parametrize('name',NAMES)
def test_examples_reach_declared_horizon_and_preserve_scope(tmp_path,name):
    out=tmp_path/name;r=run_example(name,out)
    assert r['passed'] and r['example_completed'] and all(r['checks'].values())
    assert not r['industrial_qualification'] and not r['physical_scenario_qualified']
    assert not r['task_success_evaluated']
    trajectory=json.loads((out/'trajectory.json').read_text())['payload']
    assert len(trajectory['transitions'])==r['config']['max_steps']
    assert trajectory['observation_visibility']=='measured_only'
    assert trajectory['checkpoint_visibility']=='privileged_research_data'
    assert (out/'checkpoint.json').is_file()
    if name=='fault':assert r['checks']['requested_on_effective_off'] and r['checks']['fault_epoch']
    if name=='load':assert r['checks']['load_commands_selected']
    if name=='replay':assert r['checks']['replay_checkpoint_exact'] and r['checks']['replay_transitions_exact']


def test_agent_receives_only_public_frame_and_schema(tmp_path,monkeypatch):
    received=[];real=model.tracking_policy
    def spy(observation,schema,objective):
        assert 'measurements' in observation and 'state' not in observation and 'checkpoint' not in observation
        assert 'operator' in schema and 'checkpoint' not in schema
        assert set(objective)=={'target_w','diagnostic_threshold_pa'}
        received.append(observation['time_s'])
        return real(observation,schema,objective)
    monkeypatch.setattr(model,'tracking_policy',spy)
    run_example('agent',tmp_path/'agent')
    assert received==[0.,.02,.04]


def test_missing_initial_measurement_is_refused_instead_of_using_latent_truth(tmp_path,monkeypatch):
    monkeypatch.setattr(model,'valid_reading',lambda *a:None)
    with pytest.raises(ValueError,match='evidence retained'):run_example('agent',tmp_path/'agent')
    report=json.loads((tmp_path/'agent/report.json').read_text())['payload']
    assert not report['passed'] and report['simulated_s']==0.


def test_unknown_name_and_existing_output_are_rejected(tmp_path):
    with pytest.raises(ValueError):run_example('unknown',tmp_path/'new')
    assert not (tmp_path/'new').exists()
    (tmp_path/'keep').write_text('keep')
    with pytest.raises(ValueError):run_example('first',tmp_path)
    assert (tmp_path/'keep').read_text()=='keep'


def test_source_change_is_reported_as_failure(tmp_path,monkeypatch):
    versions=iter([1,2]);monkeypatch.setattr(model,'identity',lambda:next(versions))
    with pytest.raises(ValueError,match='evidence retained'):run_example('first',tmp_path/'first')
    report=json.loads((tmp_path/'first/report.json').read_text())['payload']
    assert not report['passed'] and not report['checks']['source_unchanged']
