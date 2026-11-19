import pytest
from scripts.qualify_station_inventory import read
from scripts.qualify_station_thermal_resolved import run


@pytest.fixture(scope='module')
def evidence(tmp_path_factory):
    path=tmp_path_factory.mktemp('resolved-thermal')/'study'
    report=run(path)
    return report,read(path/'cases.json')['cases'],read(path/'comparison.json')


def test_selected_heated_spatial_endpoint_and_full_heat_prefix_pass(evidence):
    report,cases,comparison=evidence
    assert report['passed'] and report['selected_heated_endpoint_and_heat_qualified']
    assert all(comparison['checks'].values())
    assert report['maximum_physical_heat_budget_residual']<1e-10
    assert max(report['maximum_wall_momentum_residual'],report['maximum_wall_thermal_residual'])<1e-8
    for name in ('final','temperature48','tight'):assert cases[name]['gate']['passed']
    assert report['fourier_velocity_l2']<1e-7 and report['fourier_temperature_l2']<1e-7


def test_coarse_failures_and_initial_incompatibility_remain_visible(evidence):
    report,cases,comparison=evidence
    assert not cases['legacy_v80']['gate']['passed']
    assert not cases['radial']['gate']['passed']
    assert cases['radial']['solver']['maximum_physical_heat_budget_residual']>1e-10
    assert not cases['final']['initial_wall']['compatible']
    assert cases['final']['initial_wall']['maximum_temperature_rate_mismatch']>.01
    assert report['heat_prefix_starts_at_zero'] and not report['initial_field_modified']
    assert not report['all_time_wall_qualified'] and not report['general_strong_pde_qualified']
    assert not report['validated'] and not report['station_usable'] and report['closed_tasks']==[]
