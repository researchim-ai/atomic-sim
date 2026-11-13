from copy import deepcopy
import math
import pytest
from scripts import benchmark_beavrs_boron as b


def test_pinned_source_extracts_actual_table_and_sequential_sums():
    data = b.measurements()
    assert data['reference_boron_ppm'] == 975.
    assert [r['critical_boron_ppm'] for r in data['rows']] == [902., 810., 686., 508.]
    assert [r['measured_rod_worth_pcm'] for r in data['rows']] == [788., 1991., 3710., 6042.]
    assert data['uncertainty_boron_ppm'] is None
    with pytest.raises(ValueError): b.measurements(b'tampered')


def test_train_only_fit_exact_order_independent():
    data = b.measurements(); rows = data['rows'][:2]
    result = b.fit(975., rows)
    assert result == b.fit(975., list(reversed(rows)))
    assert result['test_used'] is False
    assert result['coefficient_pcm_per_ppm'] == (73 * 788 + 165 * 1991) / (73**2 + 165**2)


@pytest.mark.parametrize('damage', ['missing', 'duplicate', 'leak', 'negative', 'nan'])
def test_bad_calibration_refused(damage):
    rows = deepcopy(b.measurements()['rows'][:2])
    if damage == 'missing': rows.pop()
    elif damage == 'duplicate': rows[1] = deepcopy(rows[0])
    elif damage == 'leak': rows[0] = deepcopy(b.measurements()['rows'][2])
    elif damage == 'negative': rows[0]['measured_rod_worth_pcm'] = -1.
    else: rows[0]['critical_boron_ppm'] = float('nan')
    with pytest.raises(ValueError): b.fit(975., rows)


@pytest.mark.parametrize('args', [(975., 1., 0.), (975., -1., 8.), (True, 1., 8.), (975., float('nan'), 8.), (1., 100., 1.)])
def test_bad_prediction_refused(args):
    with pytest.raises(ValueError): b.predict(*args)


def test_actual_plant_coefficient_and_calibrated_probe():
    assert math.isclose(b.probe(), 8., rel_tol=1e-10)
    assert math.isclose(b.probe(11.), 11., rel_tol=1e-10)


def test_held_out_values_do_not_affect_fit_or_predictions():
    data = b.measurements(); train = data['rows'][:2]
    c = b.fit(975., train)['coefficient_pcm_per_ppm']; first = b.evaluate(data, c)
    changed = deepcopy(data); changed['rows'][2]['critical_boron_ppm'] += 100.
    assert b.fit(975., changed['rows'][:2]) == b.fit(975., train)
    second = b.evaluate(changed, c)
    assert [r['predicted_boron_ppm'] for r in first['rows']] == [r['predicted_boron_ppm'] for r in second['rows']]
    assert first['evaluation_mae_ppm'] != second['evaluation_mae_ppm']
