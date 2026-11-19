"""Safe bounded recovery of sealed per-case scientific work."""
import copy
import pytest
from atomic_model_multiphysics.provenance import seal,sha
from atomic_model_training.spec import atomic_json
from scripts import resume_station_thermal_radial_paths as recovery
from scripts.qualify_station_inventory import read


def failed_row(case):
    checks={name:False for name in ('complete','same_initial_fields','weak_pressure','budgets','sampled_wall',*recovery.study.VARIANTS[1:])}
    return dict(case=case.specification(),accepted=False,checks=checks,comparisons={},
        stages={name:dict(completed=False,records=[],reason='deliberate refusal') for name in recovery.study.VARIANTS})


def prepare(tmp_path):
    out=tmp_path/'study';out.mkdir()
    atomic_json(out/'protocol.json',seal(dict(source=recovery.study.identity(),
        cases=[c.specification() for c in recovery.study.thermal_catalog()],
        times=list(recovery.study.TIMES),variants=list(recovery.study.VARIANTS))))
    return out


def test_bounded_resume_keeps_old_cases_and_never_accepts_partial_work(tmp_path,monkeypatch):
    out=prepare(tmp_path);cases=recovery.study.thermal_catalog()
    atomic_json(out/(cases[0].name+'.json'),seal(failed_row(cases[0])))
    original=sha(out/(cases[0].name+'.json'));calls=[]
    def calculate(case):
        calls.append(case.name);return failed_row(case)
    monkeypatch.setattr(recovery.study,'qualify_case',calculate)
    partial=recovery.resume(out,max_cases=1)
    assert not partial['completed'] and calls==[cases[1].name]
    assert not (out/'report.json').exists()
    assert sha(out/(cases[0].name+'.json'))==original
    final=recovery.resume(out,max_cases=5)
    assert final['completed'] and calls==[c.name for c in cases[1:]]
    report=read(out/'report.json')
    assert not report['passed'] and len(report['refused_cases'])==5
    assert sha(out/(cases[0].name+'.json'))==original


def test_source_mismatch_refuses_before_numerical_work(tmp_path,monkeypatch):
    out=prepare(tmp_path)
    monkeypatch.setattr(recovery.study,'identity',lambda:{'changed':True})
    with pytest.raises(ValueError,match='Protocol/source mismatch'):
        recovery.resume(out)
    assert not (out/'report.json').exists()


def test_corrupt_saved_case_is_not_silently_recomputed(tmp_path):
    out=prepare(tmp_path);case=recovery.study.thermal_catalog()[0]
    (out/(case.name+'.json')).write_text('{"payload":{},"sha256":"invalid"}')
    with pytest.raises(ValueError,match='Invalid content envelope'):
        recovery.resume(out)


def test_unexpected_file_blocks_recovery(tmp_path):
    out=prepare(tmp_path);(out/'unexpected.json').write_text('{}')
    with pytest.raises(ValueError,match='Unexpected study artifact'):
        recovery.resume(out)


def test_inconsistent_acceptance_is_rejected():
    case=recovery.study.thermal_catalog()[0];row=failed_row(case);row['accepted']=True
    with pytest.raises(ValueError,match='Inconsistent case acceptance'):
        recovery.validate_case(row,case)


def test_missing_completed_epochs_are_rejected():
    case=recovery.study.thermal_catalog()[0];row=failed_row(case)
    row['stages']['main']['completed']=True
    with pytest.raises(ValueError,match='Complete ordered epochs'):
        recovery.validate_case(row,case)


def test_symlink_directory_refused(tmp_path):
    out=prepare(tmp_path);link=tmp_path/'alias';link.symlink_to(out,target_is_directory=True)
    with pytest.raises(ValueError,match='ordinary study directory'):
        recovery.resume(link)


@pytest.mark.parametrize('budget',[0,6,True])
def test_invalid_case_budget_refused(tmp_path,budget):
    with pytest.raises(ValueError,match='Case work budget'):
        recovery.resume(tmp_path,max_cases=budget)
