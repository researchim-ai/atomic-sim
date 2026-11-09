"""Classifier diagnostics use identical coordinates, never loosen dynamic gates."""
import pytest
from scripts import benchmark_phase_boundary as bench


def test_declared_matrix_resolves_both_sides_of_each_boundary():
    rows=bench.states()
    assert len(rows)==88
    assert {r['side'] for r in rows}=={'liquid','vapor'}
    assert len({r['pressure_pa'] for r in rows})==4
    assert all('phase_dT' in bench.render([r]) for r in rows)


@pytest.mark.parametrize('rho,t',[(0.,450.),(float('nan'),450.),(1.,299.),(1.,623.15)])
def test_invalid_static_coordinates_refused(rho,t):
    with pytest.raises(ValueError):bench.reference(rho,t)


def test_phase_and_region_endpoint_conventions_are_distinct():
    ref=bench.reference(5.,450.)
    vapor=ref['vapor_density']
    at=bench.reference(vapor,450.)
    assert at['region']==4 and at['phase']==1
    assert bench.reference(vapor*(1+1e-6),450.)['phase']==2
    assert bench.reference(vapor*(1-1e-6),450.)['region']==2


def test_disagreement_is_reported_without_being_promoted():
    row=dict(density=5.,temperature=450.)
    ref=bench.reference(5.,450.)
    values=dict(liquid1=ref['liquid_density'],vapor1=ref['vapor_density'],phase1=1.,region1=2.)
    record=bench.evaluate([row],values)[0]
    assert not record['phase_matches'] and not record['region_matches']
    assert record['outside_measured_boundary_shift']
    assert 'passed' not in record


@pytest.mark.parametrize('key,value',[('phase1',3.),('region1',3.),('liquid1',float('nan')),('vapor1',-1.)])
def test_external_nonfinite_or_unsupported_classification_refused(key,value):
    row=dict(density=5.,temperature=450.)
    values=dict(liquid1=900.,vapor1=4.,phase1=2.,region1=4.)
    values[key]=value
    with pytest.raises(ValueError):bench.evaluate([row],values)


def test_missing_runtime_leaves_no_fake_report(tmp_path):
    with pytest.raises(ValueError):bench.run(tmp_path/'out',omc=tmp_path/'missing',library=tmp_path)
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('tolerance',[0.,float('nan'),1e-3,1e-15])
def test_refinement_precision_rejected_before_campaign_or_output(tmp_path,tolerance):
    from scripts.refine_modelica_wall import run
    with pytest.raises(ValueError):run(tmp_path/'missing',tmp_path/'out',tolerance=tolerance)
    assert not (tmp_path/'out').exists()
