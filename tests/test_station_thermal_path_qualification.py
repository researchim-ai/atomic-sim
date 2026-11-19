import pytest
from scripts.qualify_station_inventory import read
from scripts.qualify_station_thermal_path import run
from atomic_model_station_thermal_path import ThermalTrajectory


@pytest.fixture(scope='module')
def evidence(tmp_path_factory):
    output=tmp_path_factory.mktemp('thermal-path')/'study'
    report=run(output)
    return report,read(output/'paths.json'),read(output/'continuation.json')


def test_longer_sampled_path_and_cumulative_budgets_are_qualified(evidence):
    report,paths,continuation=evidence
    assert report['passed'] and report['selected_longer_thermal_path_qualified']
    assert all(report['checks'].values()) and report['relative_horizon']==10
    assert report['global_heat_bound']<1e-10
    assert report['maximum_sampled_fourier_velocity_l2']<1e-7
    assert report['maximum_sampled_temporal_velocity_l2']<1e-8
    assert len(paths['main'])==5 and report['accepted_main_segments']==5
    for row in paths['main']:
        status=row['status']
        assert status['origin_time']==0 and status['conservative_composition_bound']
        assert not status['continuous_error_bound']


def test_boundary_restart_and_scope_are_preserved(evidence):
    report,paths,continuation=evidence
    path=ThermalTrajectory.restore(continuation['final_checkpoint'])
    assert path.time==.5 and path.status()['accepted_segments']==5
    assert continuation['boundary_restart_exact'] and continuation['partial_failure_atomic']
    assert continuation['replay_checkpoint_sha256']==continuation['final_checkpoint']['sha256']
    assert not continuation['bdf_internal_history_serialized']
    assert not continuation['uninterrupted_bitwise_equivalence_claimed']
    assert paths['raw_endpoints']['coarse_radial']['maximum_physical_heat_budget_residual']>1e-10
    assert not report['statistical_convergence_qualified'] and not report['physical_duration_qualified']
    assert not report['station_usable'] and not report['validated'] and report['closed_tasks']==[]
