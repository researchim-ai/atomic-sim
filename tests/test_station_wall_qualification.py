import pytest
from scripts import qualify_station_wall as study
from scripts.qualify_station_inventory import read


@pytest.fixture(scope='module')
def source(tmp_path_factory):
    p=tmp_path_factory.mktemp('wall')/'source';study.run(p);return p


def test_selected_wall_benchmarks_and_negative_results_are_separate(source):
    r=read(source/'report.json');ref=read(source/'refinement.json')
    assert r['passed'] and r['selected_linear_wall_benchmarks_qualified'] and r['stationary_cases']==12
    assert r['finest_swirl_wall_residual']<1e-8 and r['maximum_traction_error']<1e-8
    assert r['coarse_candidate_refused_atomically'] and r['finest_guarded_restart_exact']
    assert not r['general_wall_momentum_qualified'] and not r['station_usable'] and r['closed_tasks']==[]
    assert ref['weak_strong_counterexample']['strong_wall_rejected']
    assert ref['weak_strong_counterexample']['weak_momentum_residual']<1e-9
    assert [v['wall_passed'] for v in ref['swirl']]==[False,False,False,True]


def test_all_four_scientific_json_repeat_exactly(source,tmp_path):
    assert study.run(tmp_path/'repeat',source)['passed']
    assert {p.name for p in source.glob('*.json')}=={'protocol.json','cases.json','refinement.json','report.json'}


def test_false_pass_flag_cannot_hide_large_numeric_residual(tmp_path,monkeypatch):
    original=study.audit_momentum
    def wrong(*args,**kwargs):
        result=original(*args,**kwargs);result['normalized_wall_residual']=1.;result['wall_momentum_passed']=True
        return result
    monkeypatch.setattr(study,'audit_momentum',wrong)
    with pytest.raises(ValueError,match='qualification failed'):study.run(tmp_path/'rejected')
    assert not read(tmp_path/'rejected/report.json')['passed']
