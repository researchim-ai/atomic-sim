import json
from importlib.resources import files
import pytest
from atomic_model_multiphysics.provenance import seal
from atomic_model_training.spec import atomic_json
from atomic_model_station_quadrature.data import load_fixture
from scripts import qualify_station_quadrature as study
from scripts.qualify_station_inventory import read


@pytest.fixture(scope='module')
def evidence(tmp_path_factory):
    root=tmp_path_factory.mktemp('corrected-quadrature')
    assert study.run(root/'source')['passed']
    assert study.run(root/'repeat',root/'source')['passed']
    return root/'source'


def test_full_corrected_fine_matrix_is_accepted_with_original_refusal_preserved(evidence):
    p=read(evidence/'report.json');rows=read(evidence/'comparison.json')
    assert p['passed'] and p['corrected_external_continuum_comparison_passed'] and p['external_cases']==14
    assert all(v['passed'] for v in rows.values() if v['cells']==64)
    assert p['maximum_corrected_fine_profile_absolute']<2e-5 and p['maximum_original_fine_profile_absolute']>2e-5
    assert not p['original_external_continuum_gate_closed'] and not p['original_v67_fixture_updated']
    assert not p['station_usable'] and not p['turbulent'] and not p['validated'] and not p['r1_accepted'] and not p['npp_11_03_accepted']
    assert p['closed_tasks']==[] and len(list(evidence.glob('*.json')))==5


def test_moment_radial_and_temporal_errors_are_checked_separately(evidence):
    p=read(evidence/'moments.json');r=read(evidence/'refinement.json')
    assert set(p)=={'32','64'} and all(v['passed'] and v['rule']['minimum_weight']>0 for v in p.values())
    assert max(v['independent_qr_absolute'] for v in p.values())<5e-14
    assert all(v['passed'] for v in r['radial'].values())
    assert max(r['temporal_absolute'].values())<2e-7


def test_criteria_cannot_be_changed_to_reuse_an_old_protocol(evidence,tmp_path):
    p=read(evidence/'protocol.json');p['criteria']['external_profile_absolute']=1
    ref=tmp_path/'wrong';ref.mkdir();atomic_json(ref/'protocol.json',seal(p))
    with pytest.raises(ValueError,match='identity'):study.run(tmp_path/'out',ref)
    assert not (tmp_path/'out').exists()


def test_altered_fine_profile_fails_the_scientific_gate(tmp_path,monkeypatch):
    data=load_fixture();row=next(v for v in data['records'].values() if v['cells']==64)
    row['final']['temperature'][0]+=.1
    monkeypatch.setattr(study,'load_fixture',lambda:data)
    with pytest.raises(ValueError,match='qualification failed'):study.run(tmp_path/'refused')
    assert not read(tmp_path/'refused/report.json')['passed']


def test_altered_external_weights_are_not_silently_recomputed(tmp_path,monkeypatch):
    data=load_fixture();data['rules']['32']['corrected_weights'][0]+=.001
    monkeypatch.setattr(study,'load_fixture',lambda:data)
    with pytest.raises(ValueError,match='rule differs'):study.run(tmp_path/'refused')
    assert not (tmp_path/'refused/report.json').exists()


def test_broken_fixture_seal_is_rejected(tmp_path,monkeypatch):
    import atomic_model_station_quadrature.data as module
    raw=json.loads(files('atomic_model_station_quadrature').joinpath('data/corrected_reference.json').read_text())
    raw['payload']['subspace']['turbulent']=True
    target=tmp_path/'data';target.mkdir();(target/'corrected_reference.json').write_text(json.dumps(raw))
    monkeypatch.setattr(module,'files',lambda _:tmp_path)
    with pytest.raises(ValueError):module.load_fixture()
