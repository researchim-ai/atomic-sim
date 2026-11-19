import pytest
from scripts.qualify_station_inventory import read
from scripts.qualify_station_thermal_cases import run


@pytest.fixture(scope='module')
def evidence(tmp_path_factory):
    output=tmp_path_factory.mktemp('thermal-cases')/'study'
    report=run(output)
    return report,read(output/'matrix.json'),read(output/'counterexample.json')


def test_five_physical_cases_and_twenty_five_controlled_paths_pass(evidence):
    report,matrix,negative=evidence
    assert report['passed'] and report['qualified_cases']==5 and report['completed_numerical_paths']==25
    assert all(row['accepted'] and all(row['checks'].values()) for row in matrix['cases'])
    assert report['maximum_main_heat_residual']<1e-10
    assert report['maximum_main_wall_residual']<1e-8
    assert report['maximum_temporal_l2']<1e-8 and report['maximum_fourier_l2']<1e-7
    assert report['maximum_radial_l2']<1e-7 and report['maximum_temperature_refinement_l2']<1e-7
    assert not negative['quality']['passed']


def test_exact_tested_points_and_negative_reference_are_preserved(evidence):
    report,matrix,negative=evidence
    assert len(report['tested_case_specifications'])==5
    assert not matrix['input_box_qualified'] and not matrix['interpolation_qualified']
    assert not report['unseen_case_generalization'] and not report['statistical_convergence_qualified']
    assert not report['station_usable'] and not report['validated'] and report['closed_tasks']==[]
    flat=next(row for row in matrix['cases'] if row['case']['name']=='flat_passive')
    assert flat['stages']['main']['initial_wall']['compatible']
    assert negative['solver']['maximum_physical_heat_budget_residual']>1e-10
    assert len(matrix['cases'][0]['comparisons'])==4
