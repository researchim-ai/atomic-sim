"""Refusals and completeness of independent radial-axis qualification."""
import pytest

from atomic_model_station_thermal_cases import thermal_catalog
from scripts import qualify_station_thermal_radial_catalog as driver
from scripts.qualify_station_inventory import read


@pytest.mark.parametrize('value', [float('nan'),float('inf'),-1.,1e-7])
def test_invalid_or_threshold_norm_is_not_accepted(value):
    assert not driver.below(dict(velocity_l2=value,temperature_l2=0.),1e-7)


def test_missing_numerical_paths_cannot_pass(monkeypatch):
    def refused(*args,**kwargs):
        raise ValueError('deliberate numerical refusal')
    monkeypatch.setattr(driver,'build_thermal_case',refused)
    row=driver.qualify_case(thermal_catalog()[0])
    assert not row['accepted'] and not any(row['checks'].values())
    assert len(row['stages'])==4 and not row['comparisons']
    assert all(s['reason']=='deliberate numerical refusal' for s in row['stages'].values())


def test_failed_case_is_preserved_in_complete_report(tmp_path,monkeypatch):
    def row(case):
        return dict(case=case.specification(),accepted=case.name!='negative_b',
                    stages={'main':dict(completed=case.name!='negative_b')})
    monkeypatch.setattr(driver,'qualify_case',row)
    report=driver.run(tmp_path/'study')
    assert not report['passed'] and report['refused_cases']==['negative_b']
    assert report['completed_paths']==4 and len(report['case_evidence'])==5
    assert read(tmp_path/'study/report.json')==report
    assert not report['intermediate_radial_epochs_qualified']
    assert not report['shell4_radial_refinement_qualified']
    assert not report['station_usable'] and report['closed_tasks']==[]


def test_source_change_invalidates_final_receipt(tmp_path,monkeypatch):
    versions=iter([{'v':1},{'v':2}])
    monkeypatch.setattr(driver,'identity',lambda:next(versions))
    monkeypatch.setattr(driver,'qualify_case',lambda c:dict(case=c.specification(),accepted=True))
    with pytest.raises(ValueError,match='source changed'):
        driver.run(tmp_path/'study')
    assert not (tmp_path/'study/report.json').exists()


def test_requires_new_output_directory(tmp_path):
    with pytest.raises(ValueError,match='Fresh ordinary'):
        driver.run(tmp_path)
