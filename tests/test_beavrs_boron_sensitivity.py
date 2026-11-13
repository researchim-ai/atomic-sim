from copy import deepcopy
import math
import pytest
from scripts import analyze_beavrs_boron as analysis


def data(): return analysis.benchmark.measurements()


def test_analytic_derivatives_match_finite_difference_with_common_reference():
    d = data(); rows = d['rows'][:2]; ref = d['reference_boron_ppm']
    fit = analysis.calibrate(ref, rows, ('D', 'CD')); h = 1e-3
    coefficient = lambda reference, rs: analysis.calibrate(reference, rs, ('D', 'CD'))['coefficient_pcm_per_ppm']
    fd = (coefficient(ref+h, rows)-coefficient(ref-h, rows))/(2*h)
    assert math.isclose(fd, fit['coefficient_per_reference_ppm'], rel_tol=1e-7)
    for derivative in fit['local_derivatives']:
        for key, field in [('critical_boron_ppm','coefficient_per_boron_ppm'), ('measured_rod_worth_pcm','coefficient_per_worth_pcm')]:
            plus = deepcopy(rows); minus = deepcopy(rows)
            next(r for r in plus if r['id']==derivative['id'])[key] += h
            next(r for r in minus if r['id']==derivative['id'])[key] -= h
            fd = (coefficient(ref,plus)-coefficient(ref,minus))/(2*h)
            assert math.isclose(fd, derivative[field], rel_tol=1e-7)


def test_prediction_reference_derivative_includes_shared_reference_effect():
    d = data(); ref = d['reference_boron_ppm']; result = analysis.diagnose(d)
    view = next(v for v in result['views'] if v['name']=='D_CD')
    derivative = next(v for v in view['prediction_sensitivity'] if v['id']=='ALL')['prediction_per_reference_ppm']
    def pred(reference):
        c = analysis.calibrate(reference,d['rows'][:2],('D','CD'))['coefficient_pcm_per_ppm']
        return analysis.benchmark.predict(reference,6042.,c)
    h = 1e-3
    assert math.isclose((pred(ref+h)-pred(ref-h))/(2*h), derivative, rel_tol=1e-7)


def test_no_view_selected_and_secants_not_used_in_predictions():
    d = data(); first = analysis.diagnose(d)
    assert first['selected_view'] is None and not first['test_used_for_selection'] and not first['uncertainty_propagated']
    changed = deepcopy(d); changed['rows'][2]['critical_boron_ppm'] += 20.
    second = analysis.diagnose(changed)
    for a,b in zip(first['views'],second['views'],strict=True):
        assert a['calibration']==b['calibration']
        assert [r['predicted_boron_ppm'] for r in a['evaluation']['rows']]==[r['predicted_boron_ppm'] for r in b['evaluation']['rows']]
    assert first['descriptive_secants'] != second['descriptive_secants']


def test_all_current_evaluation_predictions_are_extrapolation():
    result = analysis.diagnose(data())
    assert all(p['measured_concentration_outside_calibration_range'] and p['predicted_concentration_outside_calibration_range'] for v in result['views'] for p in v['prediction_sensitivity'])


def test_calibration_and_diagnostics_are_exactly_order_independent():
    d = data(); expected = analysis.diagnose(d); d['rows'].reverse()
    actual = analysis.diagnose(d)
    assert expected==actual


@pytest.mark.parametrize('damage', ['test_leak','duplicate','unknown','degenerate','nan'])
def test_bad_subset_refused(damage):
    d = data(); rows = deepcopy(d['rows'][:2]); ids = ('D','CD')
    if damage=='test_leak': rows[0]=d['rows'][2]
    elif damage=='duplicate': rows[1]=deepcopy(rows[0])
    elif damage=='unknown': ids=('ABCD','ALL')
    elif damage=='degenerate':
        for r in rows:r['critical_boron_ppm']=975.-1e-8
    else: rows[0]['critical_boron_ppm']=float('nan')
    with pytest.raises(ValueError):analysis.calibrate(975.,rows,ids)


@pytest.mark.parametrize('damage',['missing','duplicate','split','nan'])
def test_bad_diagnostic_matrix_refused(damage):
    d = data()
    if damage=='missing': d['rows'].pop()
    elif damage=='duplicate': d['rows'][3]=deepcopy(d['rows'][2])
    elif damage=='split': d['rows'][2]['split']='calibration'
    else: d['rows'][2]['critical_boron_ppm']=float('nan')
    with pytest.raises(ValueError):analysis.diagnose(d)
