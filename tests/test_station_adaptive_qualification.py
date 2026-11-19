import pytest
from atomic_model_station_adaptive import AdaptiveSession
from scripts import qualify_station_adaptive as study
from scripts.qualify_station_inventory import read


@pytest.fixture(scope='module')
def source(tmp_path_factory):
    p=tmp_path_factory.mktemp('adaptive-time')/'source';study.run(p);return p


def test_selected_temporal_target_improves_fixed_step_against_independent_reference(source):
    r=read(source/'report.json');accuracy=read(source/'accuracy.json')
    assert r['passed'] and r['selected_temporal_target_qualified']
    assert r['adaptive_velocity_error']<1e-7<r['fixed_velocity_error'] and r['accuracy_gain']>5
    assert r['adaptive_reference_agreement']<1e-10 and r['maximum_local_error_ratio']<=1
    assert r['controller_restart_exact'] and r['rejection_atomic']
    assert all(accuracy['checks'].values())


def test_actual_half_step_trace_and_unclosed_spatial_scope_are_preserved(source):
    r=read(source/'report.json');proof=read(source/'controller.json');interval=proof['interval']
    endpoint=AdaptiveSession.restore(proof['endpoint_checkpoint'])
    assert abs(endpoint.time-.1)<1e-15
    assert interval['accepted_steps']==len(interval['steps']) and interval['rejected_trials']>0
    assert all(row['error_ratio']<=1 for row in interval['steps'])
    assert not r['global_error_bound_claimed'] and not r['spatial_convergence_qualified'] and not r['general_strong_wall_qualified']
    assert not r['station_usable'] and r['closed_tasks']==[]
    assert proof['wall_audit']['nonlinear_advection_included']
