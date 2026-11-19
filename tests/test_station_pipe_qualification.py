from pathlib import Path
import pytest
from atomic_model_multiphysics.provenance import seal
from atomic_model_training.spec import atomic_json
from scripts import qualify_station_pipe as study
from scripts.qualify_station_inventory import read


@pytest.fixture(scope='module')
def evidence(tmp_path_factory):
    root=tmp_path_factory.mktemp('pipe-qualification')
    assert study.run(root/'source')['passed']
    assert study.run(root/'repeat',root/'source')['passed']
    return root/'source'


def test_research_qualification_has_exact_native_replay_without_dns_or_station_acceptance(evidence):
    r=read(evidence/'report.json')
    assert r['passed'] and r['native_prefix_exact'] and r['native_suffix_and_final_exact']
    assert not r['dns_qualified'] and not r['npp_11_03_accepted'] and not r['validated']
    assert r['duration']==800 and r['time_unit']=='R/(2Ub)' and r['closed_tasks']==[]
    assert len(list(evidence.glob('*.json')))==8


def test_every_saved_transition_has_closed_heat_and_kinetic_ledgers(evidence):
    path=read(evidence/'trajectory.json');assert len(path['records'])==32
    assert path['maximum_absolute']<5e-5
    for record in path['records']:
        p=record['snapshot']
        assert p['checks']['relative_maximum']<1e-10
        assert abs(p['area_mean_velocity']-.5)<5e-11 and abs(p['area_mean_temperature']-.5)<5e-11


def test_temporal_degenerate_prandtl_case_is_compared_independently(evidence):
    temporal=read(evidence/'temporal.json')
    assert set(temporal)=={'0.7','1.0'}
    assert all(r['passed'] and min(r['observed_orders'])>1.8 for r in temporal.values())
    assert read(evidence/'report.json')['first_mode_relative']<2e-4


def test_changed_reference_criteria_cannot_reuse_a_study_protocol(evidence,tmp_path):
    p=read(evidence/'protocol.json');p['criteria']['whole_path_absolute']=1.
    ref=tmp_path/'wrong';ref.mkdir();atomic_json(ref/'protocol.json',seal(p))
    with pytest.raises(ValueError,match='identity'):study.run(tmp_path/'out',ref)
    assert not (tmp_path/'out').exists()


def test_an_inaccurate_independent_reference_fails_the_gate(tmp_path,monkeypatch):
    original=study.exponential_reference
    def wrong(*args,**kwargs):
        y=original(*args,**kwargs);y[0]+=.01;return y
    monkeypatch.setattr(study,'exponential_reference',wrong)
    with pytest.raises(ValueError,match='qualification failed'):study.run(tmp_path/'refused')
    assert not read(tmp_path/'refused/report.json')['passed']
