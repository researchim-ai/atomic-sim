import numpy as np
import pytest

from atomic_model_station_thermal_cases import ThermalCase,thermal_catalog,build_thermal_case,evaluate_thermal_case,evaluate_thermal_matrix
from atomic_model_station_thermal_resolved import initial_wall_compatibility,compare_thermal,resolved_velocity_mode,resolved_scalar_mode,ResolvedThermalSystem
from atomic_model_station_thermal import ThermalResult
from atomic_model_station_thermal_cases.evaluate import CRITERIA


def endpoint(system):
    return ThermalResult(system.velocity,system.temperature,[],[],{},dict(time=0.))


def test_catalog_is_finite_distinct_and_roundtrips_exactly():
    rows=thermal_catalog()
    assert len(rows)==5 and len({row.name for row in rows})==5
    assert {np.sign(row.buoyancy) for row in rows}=={-1.,0.,1.}
    assert any(row.source_mean<0 for row in rows)
    for row in rows:assert ThermalCase.from_specification(row.specification())==row


@pytest.mark.parametrize('change',[dict(name='../bad'),dict(name=''),dict(seed=True),dict(seed=-1),dict(radial_power=0),dict(viscosity=0),dict(diffusivity=float('nan')),dict(phase=True),dict(source_mean=.1),dict(alpha=0)])
def test_case_input_envelope_is_explicit(change):
    with pytest.raises(ValueError):ThermalCase(**change)


def test_case_specification_rejects_missing_or_unknown_fields():
    spec=ThermalCase().specification()
    with pytest.raises(ValueError):ThermalCase.from_specification({'name':'partial'})
    with pytest.raises(ValueError):ThermalCase.from_specification(dict(spec,unknown=1))
    with pytest.raises(AttributeError):ThermalCase().seed=100


def test_baseline_matches_prior_physical_seed_within_roundoff():
    from scripts.qualify_station_thermal_path import long_system
    current=build_thermal_case(ThermalCase());prior=long_system()
    proof=compare_thermal(endpoint(current),endpoint(prior),current.radius,current.weights)
    assert max(proof['velocity_l2'],proof['temperature_l2'])<1e-10


def test_factory_reproducibility_and_independence_of_case_order():
    a,b=thermal_catalog()[:2]
    first=build_thermal_case(a)
    other=build_thermal_case(b)
    again=build_thermal_case(a)
    np.testing.assert_array_equal(first.initial,again.initial)
    assert not np.array_equal(first.initial,other.initial)
    assert first.equation_fingerprint!=other.equation_fingerprint
    assert not np.shares_memory(first.initial,again.initial)


def test_extra_fourier_modes_start_at_zero_and_keep_same_physical_fields():
    case=ThermalCase()
    a=build_thermal_case(case,band=4);b=build_thermal_case(case,band=5)
    proof=compare_thermal(endpoint(a),endpoint(b),a.radius,a.weights)
    assert max(proof['velocity_l2'],proof['temperature_l2'])<1e-10
    for (m,q),(s,t) in zip(b.velocity,b.temperature):
        if max(abs(m.k),m.m)==5:
            assert not np.any(q) and not np.any(t)


def test_flat_thermal_profile_is_initially_wall_compatible():
    flat=build_thermal_case(thermal_catalog()[-1])
    baseline=build_thermal_case(ThermalCase())
    assert initial_wall_compatibility(flat)['compatible']
    assert not initial_wall_compatibility(baseline)['compatible']


def test_signed_mean_fields_and_source_have_prescribed_physical_profiles():
    case=thermal_catalog()[3]
    system=build_thermal_case(case)
    v,q=system.velocity[system.mean];s,t=system.temperature[system.mean]
    r=np.linspace(0,1,101)
    np.testing.assert_allclose(s.evaluate(t,r),case.temperature_mean*(1-r*r),atol=1e-12)
    np.testing.assert_allclose(v.evaluate(q,r)[2],case.mean_axial*(1-r*r)**2,atol=1e-12)
    np.testing.assert_allclose(system.source((0,0),r),case.source_mean,atol=1e-15)
    assert case.temperature_mean<0 and case.source_mean<0


@pytest.mark.parametrize('change',[dict(band=0),dict(band=6),dict(velocity_order=2),dict(temperature_order=49),dict(quadrature=64)])
def test_case_space_limits_reject_unresolved_or_unsupported_inputs(change):
    with pytest.raises(ValueError):build_thermal_case(ThermalCase(),**change)


def test_matrix_rejects_empty_duplicate_and_invalid_case_lists_before_running():
    case=ThermalCase()
    for rows in ([],[case,case],[{}]):
        with pytest.raises(ValueError):evaluate_thermal_matrix(rows)
    with pytest.raises(ValueError):evaluate_thermal_matrix([case],horizon=0)


def test_criteria_are_immutable():
    with pytest.raises(TypeError):CRITERIA['wall']=1.


def test_numerical_stage_failure_is_recorded_and_never_accepted(monkeypatch):
    from atomic_model_station_thermal_cases import evaluate as module
    calls=[]
    def fail(*args,**kwargs):calls.append(kwargs);raise ValueError('injected numerical refusal')
    monkeypatch.setattr(module,'resolve_thermal',fail)
    report=evaluate_thermal_case(ThermalCase())
    assert len(calls)==5 and not report['accepted'] and not report['checks']['complete']
    assert all(not stage['completed'] and stage['error_type']=='ValueError' for stage in report['stages'].values())
    assert not report['station_usable'] and not report['validated']


def test_matrix_keeps_rejected_rows_and_does_not_infer_parameter_box(monkeypatch):
    from atomic_model_station_thermal_cases import evaluate as module
    def fake(case,**kwargs):return dict(accepted=case.name=='baseline',case=case.specification())
    monkeypatch.setattr(module,'evaluate_thermal_case',fake)
    report=evaluate_thermal_matrix(thermal_catalog()[:2])
    assert report['accepted_count']==1 and report['total']==2 and not report['accepted']
    assert len(report['cases'])==2
    assert not report['input_box_qualified'] and not report['interpolation_qualified']
    assert report['tested_case_specifications']==[row.specification() for row in thermal_catalog()[:2]]


def test_case_observer_receives_detached_results(monkeypatch):
    from atomic_model_station_thermal_cases import evaluate as module
    def fake(case,**kwargs):return dict(accepted=True,case=case.specification())
    monkeypatch.setattr(module,'evaluate_thermal_case',fake)
    calls=[]
    def observer(row):
        calls.append(row['case']['name']);row['accepted']=False;row['case']['name']='mutated'
    report=evaluate_thermal_matrix([ThermalCase()],on_case=observer)
    assert calls==['baseline'] and report['accepted']
    assert report['cases'][0]['case']['name']=='baseline'
    with pytest.raises(ValueError):evaluate_thermal_matrix([ThermalCase()],on_case='invalid')
