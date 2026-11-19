"""Failure accounting in the extended finite-catalog scientific driver."""
import pytest

from atomic_model_station_thermal_cases import thermal_catalog
from scripts import qualify_station_thermal_case_paths as driver
from scripts.qualify_station_inventory import read


def test_incomplete_paths_are_retained_and_never_accepted(monkeypatch):
    class Refusal:
        def __init__(self, *args, **kwargs):
            pass

        def advance_to(self, epoch):
            raise ValueError('deliberate numerical refusal')

    monkeypatch.setattr(driver, 'build_thermal_case', lambda *a, **kw: None)
    monkeypatch.setattr(driver, 'ThermalTrajectory', Refusal)
    row = driver.qualify_case(thermal_catalog()[0])
    assert not row['accepted']
    assert not any(row['checks'].values())
    assert row['comparisons'] == {}
    assert set(row['stages']) == {'main', 'time', 'fourier'}
    assert all(s['error'] == 'deliberate numerical refusal' for s in row['stages'].values())


def test_matrix_keeps_failed_cases_and_publishes_report(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, 'qualify_case', lambda case: dict(
        case=case.specification(), accepted=case.name != 'negative_b'))
    report = driver.run(tmp_path / 'study')
    assert not report['passed']
    assert report['refused_cases'] == ['negative_b']
    assert len(report['accepted_cases']) == 4
    assert len(report['case_evidence']) == 5
    assert read(tmp_path / 'study' / 'report.json') == report
    assert not report['radial_convergence_at_extended_horizon']
    assert not report['station_usable'] and report['closed_tasks'] == []


def test_source_mutation_refuses_final_receipt(tmp_path, monkeypatch):
    identities = iter([{'version': 1}, {'version': 2}])
    monkeypatch.setattr(driver, 'identity', lambda: next(identities))
    monkeypatch.setattr(driver, 'qualify_case', lambda case: dict(case=case.specification(), accepted=True))
    with pytest.raises(ValueError, match='source changed'):
        driver.run(tmp_path / 'study')
    assert not (tmp_path / 'study' / 'report.json').exists()


def test_output_directory_must_be_fresh(tmp_path):
    with pytest.raises(ValueError, match='Fresh ordinary'):
        driver.run(tmp_path)
