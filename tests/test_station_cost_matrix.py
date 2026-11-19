import json
import pytest
from scripts import profile_station_matrix as matrix


@pytest.mark.parametrize('path,name,expected',[
    ('/a/eos.py','pt','eos'),('/a/scipy/radau.py','solve_collocation_system','nonlinear_solver'),
    ('/a/scipy/radau.py','_validate_jac','jacobian'),('/a/bypass.py','event','events'),
    ('/a/json/encoder.py','iterencode','serialization'),('/a/linalg.py','lu_factor','linear_algebra'),
])
def test_categories(path,name,expected):assert matrix.category(path,name)==expected


def test_exclusive_times_do_not_double_count_nested_functions():
    stats={('/a/eos.py',1,'pt'):(1,2,1.,10.,{}),('/b/radau.py',2,'solve_collocation_system'):(1,1,2.,20.,{})}
    result=matrix.summarize(stats)
    assert result['total_own_s']==3.
    assert sum(g['own_s'] for g in result['groups'].values())==3.
    assert result['groups']['eos']['calls']==2
    assert all('/' not in r['file'] for r in result['functions'])


def sample():
    return dict(transitions=[1],checkpoint=dict(sha256='same'),serialization_bytes=5,
        restore_semantics='replay',phases={k:dict(wall_s=.1,profile=None) for k in ('initialize','advance','checkpoint','serialize','restore')})


def test_failed_attempt_is_retained_and_later_case_executes(tmp_path,monkeypatch):
    calls=[]
    def execute(actions,profiled):
        calls.append(1)
        if len(calls)==1:raise ValueError('solver refused')
        return sample()
    monkeypatch.setattr(matrix,'execute',execute)
    with pytest.raises(ValueError,match='every attempt'):matrix.run(tmp_path/'out',repeats=1)
    report=json.loads((tmp_path/'out/report.json').read_text())['payload']
    assert report['attempts']==2 and report['accepted']==1 and report['failed']==1
    assert len(list((tmp_path/'out').glob('*.json')))==3


def test_profile_changed_result_is_refused(tmp_path,monkeypatch):
    def execute(actions,profiled):
        result=sample()
        if profiled:result['transitions']=[2]
        return result
    monkeypatch.setattr(matrix,'execute',execute)
    with pytest.raises(ValueError,match='matrix failed'):matrix.run(tmp_path/'out',repeats=1)
    report=json.loads((tmp_path/'out/report.json').read_text())['payload']
    assert report['accepted']==0 and report['failed']==2


@pytest.mark.parametrize('repeats',[True,0,6])
def test_invalid_budget_creates_no_output(tmp_path,repeats):
    with pytest.raises(ValueError):matrix.run(tmp_path/'out',repeats=repeats)
    assert not (tmp_path/'out').exists()


def test_existing_output_is_never_overwritten(tmp_path):
    (tmp_path/'keep').write_text('keep')
    with pytest.raises(ValueError):matrix.run(tmp_path,repeats=1)
    assert (tmp_path/'keep').read_text()=='keep'


def test_source_change_invalidates_report(tmp_path,monkeypatch):
    versions=iter([1,2]);monkeypatch.setattr(matrix,'signature',lambda:next(versions))
    monkeypatch.setattr(matrix,'execute',lambda *a:sample())
    with pytest.raises(ValueError,match='Source changed'):matrix.run(tmp_path/'out',repeats=1)
    assert not (tmp_path/'out/report.json').exists()
