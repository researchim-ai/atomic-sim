from scripts.qualify_station_inventory import read
import pytest
from scripts.qualify_station_refined import run

@pytest.fixture(scope='module')
def evidence(tmp_path_factory):
    path=tmp_path_factory.mktemp('refined')/'study'
    report=run(path)
    return report,read(path/'comparison.json'),read(path/'cases.json')

def test_selected_refinement_passes_without_erasing_coarse_failures(evidence):
    report,comparison,cases=evidence
    assert report['passed'] and all(comparison['checks'].values())
    assert comparison['comparisons']['fourier_previous']['velocity_l2']>1e-7
    assert cases['cases']['k2p20']['momentum']['maximum_wall_residual']>1e-8
    assert report['wall_residual']<1e-8

def test_analytic_pressure_and_scope_are_explicit(evidence):
    report,_,cases=evidence
    assert cases['analytic_vortex']['pressure_error']<1e-9
    for name in ('general_strong_pde_qualified','thermal_model_supported','experimental_validation','native_backend_changed','station_usable','validated'):assert report[name] is False
    assert report['closed_tasks']==[]
