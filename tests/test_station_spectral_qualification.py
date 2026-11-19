from pathlib import Path
import json
from importlib.resources import files
import pytest
from atomic_model_multiphysics.provenance import seal
from atomic_model_training.spec import atomic_json
from atomic_model_station_spectral.data import load_fixture
from scripts import qualify_station_spectral as study
from scripts.qualify_station_inventory import read


@pytest.fixture(scope='module')
def evidence(tmp_path_factory):
    root=tmp_path_factory.mktemp('spectral')
    assert study.run(root/'source')['passed']
    assert study.run(root/'repeat',root/'source')['passed']
    return root/'source'


def test_real_spatial_execution_is_separate_from_dns_acceptance(evidence):
    p=read(evidence/'report.json')
    assert p['passed'] and p['external_cases']==5 and p['real_spatial_modes_present']
    assert p['native_restart_prefix_exact'] and p['native_restart_fields_exact'] and p['maximum_restart_rms']==0
    assert not p['numerical_resolution_qualified'] and not p['thermal_statistics_qualified'] and not p['turbulent_heat_transfer_qualified']
    assert not p['integration_gates']['time_half']['passed']
    assert not p['station_usable'] and not p['validated'] and not p['r1_accepted'] and p['closed_tasks']==[]
    assert p['duration']==.2 and p['time_unit']=='R/(2Ub)' and len(list(evidence.glob('*.json')))==5


def test_short_warmup_never_gets_a_fake_statistical_confidence_interval(evidence):
    for p in read(evidence/'statistics.json').values():
        assert not p['qualified']
        for s in p['observables'].values():
            assert 'warmup_not_completed' in s['reasons'] and s['confidence_interval_95'] is None and s['effective_samples'] is None


def test_independent_native_and_trace_audits_agree_without_hiding_divergence(evidence):
    for p in read(evidence/'audits.json').values():
        assert p['independent_native_agreement'] and p['maximum_native_error']<1e-9 and p['maximum_trace_error']<1e-9
        assert p['maximum_divergence_l2']>1e-7
        assert all(v['spatial_modes_present'] and v['walls_passed'] for v in p['checks'])
        assert all(v['means_passed'] for v in p['checks'][1:])
        assert p['checks'][0]['means_passed'] == (p['checks'][0]['time'] > 0)


def test_changed_gate_cannot_reuse_reference(evidence,tmp_path):
    p=read(evidence/'protocol.json');p['criteria']['divergence_l2']=1
    ref=tmp_path/'wrong';ref.mkdir();atomic_json(ref/'protocol.json',seal(p))
    with pytest.raises(ValueError,match='identity'):study.run(tmp_path/'out',ref)
    assert not (tmp_path/'out').exists()


def test_wrong_independent_native_result_fails_software_verification(tmp_path,monkeypatch):
    data=load_fixture();data['records']['whole']['native_audits'][0]['divergence_l2']+=.1
    monkeypatch.setattr(study,'load_fixture',lambda:data)
    with pytest.raises(ValueError,match='qualification failed'):study.run(tmp_path/'refused')
    assert not read(tmp_path/'refused/report.json')['passed']


def test_wrong_legacy_trace_units_are_detected(tmp_path,monkeypatch):
    data=load_fixture();data['records']['whole']['trace']['kinetic'][0]*=2
    monkeypatch.setattr(study,'load_fixture',lambda:data)
    with pytest.raises(ValueError,match='qualification failed'):study.run(tmp_path/'refused')
    assert not read(tmp_path/'refused/report.json')['passed']


def test_corrupted_compact_fixture_is_rejected(tmp_path,monkeypatch):
    import atomic_model_station_spectral.data as module
    raw=json.loads(files('atomic_model_station_spectral').joinpath('data/spatial_probe.json').read_text())
    raw['payload']['records']['whole']['spec']['k']=1
    target=tmp_path/'data';target.mkdir();(target/'spatial_probe.json').write_text(json.dumps(raw))
    monkeypatch.setattr(module,'files',lambda _:tmp_path)
    with pytest.raises(ValueError):module.load_fixture()
