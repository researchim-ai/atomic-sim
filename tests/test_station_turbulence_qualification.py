from pathlib import Path
import pytest
from atomic_model_multiphysics.provenance import seal
from atomic_model_training.spec import atomic_json
from scripts import qualify_station_turbulence as study
from scripts.qualify_station_inventory import read


@pytest.fixture(scope='module')
def evidence(tmp_path_factory):
    root=tmp_path_factory.mktemp('turbulence')
    assert study.run(root/'source')['passed']
    assert study.run(root/'repeat',root/'source')['passed']
    return root/'source'


def test_low_reynolds_dns_refusal_is_preserved(evidence):
    p=read(evidence/'report.json');f=read(evidence/'friction.json')
    assert p['passed'] and not p['dns_momentum_comparison_passed']
    assert p['dns_refused_cases']==[180] and p['dns_accepted_cases']==[360,550,1000]
    assert .064<f['180']['relative_error']<.065
    assert max(f[k]['relative_error'] for k in ('360','550','1000'))<.005
    assert not p['model_is_dns'] and not p['thermal_qualified'] and not p['buoyancy_qualified']
    assert not p['station_usable'] and not p['validated'] and not p['npp_11_03_accepted'] and p['closed_tasks']==[]
    assert len(list(evidence.glob('*.json')))==5


def test_spatial_and_profile_gates_are_separate(evidence):
    s=read(evidence/'spatial.json');d=read(evidence/'dns.json');p=read(evidence/'protocol.json')
    assert all(v['passed'] for v in s.values())
    assert all(min(v['orders'])>1.8 for v in s.values())
    assert p['constants']==dict(kappa=.42,Aplus=27) and p['criteria']['friction_relative']==.05
    assert not d['180']['passed'] and d['180']['profile_relative_l2']<.05


def test_changed_criteria_cannot_reuse_reference(evidence,tmp_path):
    p=read(evidence/'protocol.json');p['criteria']['friction_relative']=1
    ref=tmp_path/'wrong';ref.mkdir();atomic_json(ref/'protocol.json',seal(p))
    with pytest.raises(ValueError,match='identity'):study.run(tmp_path/'out',ref)
    assert not (tmp_path/'out').exists()


def test_wrong_quadrature_reference_fails_numerical_gate(tmp_path,monkeypatch):
    original=study.velocity
    def wrong(*args):return original(*args)+1.
    monkeypatch.setattr(study,'velocity',wrong)
    with pytest.raises(ValueError,match='qualification failed'):study.run(tmp_path/'refused')
    assert not read(tmp_path/'refused/report.json')['passed']
