from pathlib import Path
import json
import pytest
from atomic_model_multiphysics.provenance import seal
from atomic_model_training.spec import atomic_json
from scripts import requalify_station_nominal as driver


def source_tree(tmp_path,monkeypatch):
    root=tmp_path/'source';math=root/'mathematics';math.mkdir(parents=True)
    monkeypatch.setattr(driver.study,'identity',lambda:dict(epoch='current'))
    monkeypatch.setattr(driver.study,'rows',lambda p:[None]*721)
    atomic_json(math/'protocol.json',seal(dict(epoch='current',duration_s=3600.,tick_s=5.)))
    atomic_json(math/'report.json',seal(dict(passed=True,source_unchanged=True,duration_s=3600.)))
    for role in ('coarse','fine'):
        (math/role).mkdir()
        atomic_json(math/role/'report.json',seal(dict(passed=True,completed=True,source_unchanged=True,
            duration_s=3600.,window_count=601,all_600s_windows_ready=True)))
    return root


@pytest.mark.parametrize('change',[
    {'epoch':'old'}, {'duration_s':60.}, {'tick_s':1.},
])
def test_short_or_stale_protocol_refused_before_build(tmp_path,monkeypatch,change):
    root=source_tree(tmp_path,monkeypatch);p=root/'mathematics/protocol.json'
    data=json.loads(p.read_text())['payload'];data.update(change);atomic_json(p,seal(data))
    monkeypatch.setattr(driver,'launch',lambda *a:pytest.fail('must not build'))
    with pytest.raises(ValueError):driver.run(tmp_path/'out',root)
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('change',[
    dict(window_count=600),dict(completed=False),dict(passed=False),
    dict(all_600s_windows_ready=False),dict(source_unchanged=False),dict(duration_s=3595.),
])
def test_partial_role_cannot_be_accepted_as_full_hour(tmp_path,monkeypatch,change):
    root=source_tree(tmp_path,monkeypatch);p=root/'mathematics/fine/report.json'
    data=json.loads(p.read_text())['payload'];data.update(change);atomic_json(p,seal(data))
    with pytest.raises(ValueError,match='Complete nominal role'):driver.validate_source(root)


def test_missing_interior_trajectory_is_refused(tmp_path,monkeypatch):
    root=source_tree(tmp_path,monkeypatch)
    monkeypatch.setattr(driver.study,'rows',lambda p:[None]*720)
    with pytest.raises(ValueError,match='trajectory'):driver.validate_source(root)


def test_math_mutation_changes_hash_but_log_timing_does_not(tmp_path):
    atomic_json(tmp_path/'proof.json',seal({'value':1}))
    log=tmp_path/'worker.log';log.write_text('elapsed 1')
    before=driver.tree_hashes(tmp_path);log.write_text('elapsed 2')
    assert before==driver.tree_hashes(tmp_path)
    atomic_json(tmp_path/'proof.json',seal({'value':2}))
    assert before!=driver.tree_hashes(tmp_path)


def test_evidence_symlink_is_refused(tmp_path):
    (tmp_path/'target.json').write_text('{}')
    (tmp_path/'link.json').symlink_to(tmp_path/'target.json')
    with pytest.raises(ValueError,match='symlinks'):driver.tree_hashes(tmp_path)


def test_build_refusal_is_persisted_without_accepting_source_only(tmp_path,monkeypatch):
    root=source_tree(tmp_path,monkeypatch)
    def refused(*args,**kwargs):raise RuntimeError('build refused')
    monkeypatch.setattr(driver,'launch',refused)
    with pytest.raises(ValueError,match='evidence retained'):driver.run(tmp_path/'out',root)
    report=json.loads((tmp_path/'out/report.json').read_text())['payload']
    assert not report['passed'] and report['failure']['message']=='build refused'
    assert report['full_r1_accepted'] is False and report['closed_tasks']==[]


def test_existing_output_is_never_replaced(tmp_path):
    marker=tmp_path/'keep';marker.write_text('keep')
    with pytest.raises(ValueError):driver.run(tmp_path)
    assert marker.read_text()=='keep'


def test_running_source_needs_complete_coarse_and_never_grants_final_acceptance(tmp_path,monkeypatch):
    root=source_tree(tmp_path,monkeypatch)
    (root/'mathematics/report.json').unlink()
    (root/'mathematics/fine/report.json').unlink()
    assert driver.validate_source(root,allow_running=True) is None
    with pytest.raises(FileNotFoundError):driver.validate_source(root)
    p=root/'mathematics/coarse/report.json';data=json.loads(p.read_text())['payload']
    data['window_count']=600;atomic_json(p,seal(data))
    with pytest.raises(ValueError,match='Complete nominal role'):driver.validate_source(root,allow_running=True)


def test_running_mode_pins_coarse_but_full_comparison_still_detects_fine_changes(tmp_path,monkeypatch):
    root=source_tree(tmp_path,monkeypatch)
    pinned=driver.pinned_hashes(root,True);full=driver.tree_hashes(root/'mathematics')
    atomic_json(root/'mathematics/fine/new.json',seal(dict(record='new')))
    assert pinned==driver.pinned_hashes(root,True)
    assert full!=driver.tree_hashes(root/'mathematics')
    atomic_json(root/'mathematics/coarse/new.json',seal(dict(record='tampered')))
    assert pinned!=driver.pinned_hashes(root,True)
