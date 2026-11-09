"""External benchmark admission; synthetic harness controls are not peer results."""
from copy import deepcopy
import math
import sys
from types import SimpleNamespace
import numpy as np
import pytest
from atomic_model_spatial import Model,eigenvalue
from atomic_model_multiphysics.provenance import unseal
from atomic_model_training.spec import read_json
from scripts import benchmark_external as bench


def test_shared_state_list_has_explicit_phase_pairs_and_supported_coordinates():
    points=bench.points()
    assert len(points)==36 and len({p['id'] for p in points})==36
    assert sum(p['kind']=='pt' for p in points)==16
    assert sum(p['kind']=='px' for p in points)==20
    for p in points:
        values=bench.atomic_water(p)
        assert all(math.isfinite(v) for v in values.values())


@pytest.mark.parametrize('ratio',bench.RATIOS)
@pytest.mark.parametrize('cells',[1,8,32])
def test_atomic_neutron_common_problem_matches_exact_no_leakage(ratio,cells):
    config=bench.neutron_config(ratio,cells)
    result=eigenvalue(Model(config))
    assert abs(result['k']-ratio)<1e-8
    assert result['relative_residual']<1e-10
    assert config['diffusion_m'][0][0]==pytest.approx(1/(6*.09))
    assert config['nu_fission_per_m'][0][0]/config['absorption_per_m'][0][0]==pytest.approx(ratio,rel=1e-15)


@pytest.mark.parametrize('values',[(1.,1.,-1.,1.),(float('nan'),1.,.001,1.),(1.,float('inf'),.001,1.),(1.,1.,.001,0.)])
def test_invalid_monte_carlo_numbers_refused(values):
    with pytest.raises(ValueError):bench.admit_neutron(*values)


def test_accuracy_and_uncertainty_requirements_cannot_substitute_each_other():
    assert all(bench.admit_neutron(1.,1.001,.001,1.).values())
    assert not bench.admit_neutron(1.,1.01,.001,1.)['monte_carlo_analytic']
    assert not bench.admit_neutron(1.,1.,.01,1.)['precision']
    assert not bench.admit_neutron(1.001,1.,.001,1.)['atomic_analytic']
    assert all(bench.admit_neutron(1.,1.,0.,1.).values())


@pytest.mark.parametrize('damage',['missing','nan','infinite'])
def test_eos_vectors_cannot_hide_missing_or_nonfinite_data(damage):
    ref=bench.atomic_water(bench.points()[0]);measured=deepcopy(ref)
    if damage=='missing':measured.pop('density_kg_m3')
    else:measured['density_kg_m3']=float('nan') if damage=='nan' else float('inf')
    with pytest.raises(ValueError):bench.relative_errors(measured,ref)


def test_relative_errors_keep_absolute_floor_near_zero():
    ref={k:0. for k in bench.EOS_BUDGETS}
    measured=dict(ref,entropy_j_kg_k=1e-5)
    assert bench.relative_errors(measured,ref)['entropy_j_kg_k']==1e-5


def test_modelica_export_is_explicitly_unexecuted(tmp_path):
    result=bench.export_modelica(tmp_path,bench.points())
    assert result['status']=='prepared_not_run'
    text=(tmp_path/'AtomicSimWaterBenchmark.mo').read_text()
    assert 'parameter Integer n = 36;' in text and 'W.setState_pT' in text and 'W.setState_ph' in text
    assert (tmp_path/'run.mos').exists()


@pytest.mark.parametrize('failure',['missing','failure'])
def test_external_failure_is_retained_and_does_not_prevent_other_peer(tmp_path,monkeypatch,failure):
    monkeypatch.setitem(sys.modules,'threadpoolctl',SimpleNamespace(threadpool_info=lambda:[]))
    def refused(*args):
        if failure=='missing':raise ImportError('optional peer absent')
        raise ValueError('peer refused physical state')
    monkeypatch.setattr(bench,'coolprop_profile',refused)
    monkeypatch.setattr(bench,'openmc_profile',lambda *args:dict(status='completed',passed=True,scope='synthetic harness fixture'))
    result=bench.run(tmp_path/'report')
    assert not result['passed'] and result['profiles']['openmc']['passed']
    assert result['profiles']['coolprop']['status']==('unavailable' if failure=='missing' else 'failed')
    assert not result['whole_plant_compared'] and not result['superiority_established']
    assert unseal(read_json(tmp_path/'report/report.json'))['completed']


def test_parallel_blas_run_refused_before_output(tmp_path,monkeypatch):
    monkeypatch.setitem(sys.modules,'threadpoolctl',SimpleNamespace(threadpool_info=lambda:[dict(num_threads=8)]))
    with pytest.raises(ValueError,match='single-thread BLAS'):bench.run(tmp_path/'refused')
    assert not (tmp_path/'refused').exists()


def test_external_high_pressure_vapor_regression():
    # Frozen independent CoolProp8.0.0 IF97 values at identical PT inputs.
    # Native receipt: external-components-v1/water-points.json; IAPWS table11,row21.
    state=bench.atomic_water(dict(kind='pt',pressure_pa=15.5e6,temperature_k=622.9415516035506))
    assert state['density_kg_m3']==pytest.approx(94.3474611419409,rel=2e-11)
    assert state['internal_energy_j_kg']==pytest.approx(2491370.410590831,rel=2e-11)
