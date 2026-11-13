import math
import pytest
from scripts import benchmark_beavrs_temperature as b


def test_pinned_itc_units_and_temperature_context():
    data=b.measurements()
    assert [r['measured_pcm_per_f'] for r in data['rows']]==[-1.75,-2.75,-8.01]
    assert [r['measured_pcm_per_k'] for r in data['rows']]==[-3.15,-4.95,-14.418]
    assert data['context_inlet_temperature_f']==560.
    assert math.isclose(data['context_temperature_k'],566.4833333333333)
    assert not data['temperature_context_is_exact_itc_test_condition']
    with pytest.raises(ValueError): b.measurements(b'altered')


def test_actual_reactivity_derivative_matches_analytic_components():
    point=b.probe(b.measurements()['context_temperature_k'],15.7e6,.05)
    assert abs(point['numerical_error_pcm_per_k'])<b.NUMERICAL_TOLERANCE_PCM_K
    assert point['fuel_pcm_per_k']<0 and point['moderator_pcm_per_k']<0
    assert point['analytic_pcm_per_k']==point['fuel_pcm_per_k']+point['moderator_pcm_per_k']


def test_step_refinement_and_rod_independence_are_measured_not_fitted():
    t=b.measurements()['context_temperature_k']
    coarse=b.probe(t,15.7e6,.2);fine=b.probe(t,15.7e6,.05)
    assert abs(fine['numerical_error_pcm_per_k'])<abs(coarse['numerical_error_pcm_per_k'])
    inserted=b.probe(t,15.7e6,.05,1.)
    assert math.isclose(fine['numerical_pcm_per_k'],inserted['numerical_pcm_per_k'],abs_tol=1e-8)


@pytest.mark.parametrize('args',[(float('nan'),15.7e6,.1),(566.,float('inf'),.1),(566.,15.7e6,0.),
                                (566.,15.7e6,True),(566.,15.7e6,.1,2.),(500.,15.7e6,.1),(566.,1e6,.1)])
def test_bad_probe_refused(args):
    with pytest.raises(ValueError): b.probe(*args)


@pytest.mark.parametrize('damage',['none','missing','duplicate','nan','fine_error','divergence'])
def test_numerical_admission_requires_complete_convergent_matrix(damage):
    points=[dict(pressure_pa=p,step_k=h,rod_fraction=r,numerical_error_pcm_per_k=h*h*.001)
            for p in b.PRESSURES_PA for h in b.STEPS_K for r in b.RODS]
    if damage=='missing':points.pop()
    elif damage=='duplicate':points.append(dict(points[0]))
    elif damage=='nan':points[0]['numerical_error_pcm_per_k']=float('nan')
    elif damage=='fine_error':points[6]['numerical_error_pcm_per_k']=.01
    elif damage=='divergence':points[0]['numerical_error_pcm_per_k']=0.
    if damage in ('missing','duplicate','nan'):
        with pytest.raises(ValueError):b.qualify(points)
    else:assert b.qualify(points)['passed'] is (damage=='none')
