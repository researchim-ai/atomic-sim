"""Forward chart differences versus frozen actual external derivative calls."""
import json
from pathlib import Path
import pytest
from atomic_model_multiphysics.provenance import unseal
from scripts import audit_msl_saturation as audit
from scripts.verify_phase_event import reference_event

DATA=unseal(json.loads(Path(__file__).with_name('fixtures').joinpath('msl_saturation_tangent_v1.json').read_text()))
ROWS=[(v['variant'],r) for v in DATA['variants'] for r in v['rows']]


@pytest.mark.parametrize('variant,row',ROWS)
def test_actual_peer_capacity_distinguishes_proxy_and_forward_path(variant,row):
    fd=audit.forward_derivative(row['t'],row['rho'])
    assert fd['cv_refinement']<=audit.BUDGETS['cv_fd_j_kg_k']
    if variant=='forward':
        assert abs(row['peer']['cv']-fd['cv'])<=audit.BUDGETS['cv_fd_j_kg_k']
        assert abs(row['peer']['dpdt']-fd['dpdt'])<=audit.BUDGETS['dpdt_fd_pa_k']
    else:
        assert abs(audit.proxy_capacity(row['t'],row['rho'])-row['peer']['cv'])<=audit.BUDGETS['cv_fd_j_kg_k']


@pytest.mark.parametrize('case',['liquid_exit','vapor_exit'])
def test_independent_proxy_integrators_reproduce_measured_prediction(case):
    actual=audit.proxy_event(case)
    expected=next(r for r in DATA['variants'][0]['proxy_predictions'] if r['case']==case)
    assert actual['integrators_agree']
    assert actual['ode'][0]['time_s']==pytest.approx(expected['ode'][0]['time_s'],abs=2e-10)
    if case=='vapor_exit':
        delta=actual['ode'][0]['time_s']-reference_event(case)['quadrature_time_s']
        assert 1.35e-7<delta<1.36e-7


@pytest.mark.parametrize('t,rho',[(float('nan'),1.),(650.,1.),(450.,0.),(450.,float('nan'))])
def test_derivative_trials_outside_declared_domain_refused(t,rho):
    with pytest.raises(ValueError):audit.forward_derivative(t,rho)


def test_single_phase_approach_cannot_be_silently_replaced_by_mixture_proxy():
    with pytest.raises(ValueError):audit.proxy_event('vapor_entry')


def test_unknown_parent_cannot_be_patched(tmp_path):
    parent=tmp_path/'library';parent.mkdir()
    with pytest.raises((ValueError,OSError)):audit.corrected_copy(parent,tmp_path/'patched')
    assert not (tmp_path/'patched').exists()
