import pytest
from scripts.qualify_station_inventory import read
from scripts.qualify_station_thermal import run


@pytest.fixture(scope='module')
def evidence(tmp_path_factory):
    path=tmp_path_factory.mktemp('coupled-thermal')/'study'
    report=run(path)
    return report,read(path/'physics.json'),read(path/'dynamics.json')


def test_analytic_heat_and_coupled_time_reference_pass(evidence):
    report,physics,dynamics=evidence
    assert report['passed'] and all(report['checks'].values())
    assert report['selected_coupled_ode_qualified'] and report['selected_analytic_heat_qualified']
    assert max(report['independent_difference'].values())<1e-8
    assert physics['analytic']['equal_diffusivities']['temperature_error']<1e-9
    assert dynamics['nonlinear']['buoyancy_feedback']['velocity_l2']>1e-7
    assert dynamics['nonlinear']['buoyancy_feedback']['temperature_l2']>1e-10


def test_coarse_heat_and_general_spatial_failure_are_preserved(evidence):
    report,physics,dynamics=evidence
    coarse=physics['analytic']['coarse']['solver']
    assert coarse['maximum_variance_budget_residual']<1e-9<coarse['maximum_physical_heat_budget_residual']
    assert not report['general_heated_spatial_gate_passed']
    assert not dynamics['nonlinear']['general_spatial_gate_passed']
    assert not report['station_usable'] and not report['validated'] and report['closed_tasks']==[]
    assert not report['general_heated_strong_pde_qualified']
    assert not report['physical_heat_gate_enforced_by_engine']
