"""Calibration sensitivity and extrapolation audit of conditional BEAVRS boron."""
import argparse
from copy import deepcopy
import math
from pathlib import Path

from atomic_model_multiphysics.provenance import seal, sha
from atomic_model_training.spec import atomic_json
from scripts import benchmark_beavrs_boron as benchmark

CALIBRATION_VIEWS = (('D_only', ('D',)), ('CD_only', ('CD',)), ('D_CD', ('D', 'CD')))


def calibrate(reference, rows, ids):
    """Diagnostic train subsets only; never select a subset by evaluation score."""
    if tuple(ids) not in [v[1] for v in CALIBRATION_VIEWS]:
        raise ValueError('Declared calibration subset required')
    if len(rows) != len(ids) or {r['id'] for r in rows} != set(ids) or any(r['split'] != 'calibration' for r in rows):
        raise ValueError('Calibration subset provenance')
    if type(reference) not in (int, float) or not math.isfinite(reference) or not 0 < reference <= 5000:
        raise ValueError('Finite reference domain required')
    pairs = []
    for row in sorted(rows, key=lambda r: r['id']):
        concentration = row['critical_boron_ppm']; worth = row['measured_rod_worth_pcm']
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in (concentration, worth)) or not 0 <= concentration < reference or not 0 < worth <= 1e5:
            raise ValueError('Finite compensation data required')
        pairs.append((row, reference - concentration, worth))
    denominator = math.fsum(x*x for _, x, _ in pairs)
    if denominator < 1e-12: raise ValueError('Degenerate calibration leverage')
    coefficient = math.fsum(x*w for _, x, w in pairs) / denominator
    derivatives = [dict(id=row['id'], coefficient_per_boron_ppm=(2*coefficient*x-w)/denominator,
                        coefficient_per_worth_pcm=x/denominator) for row, x, w in pairs]
    return dict(coefficient_pcm_per_ppm=coefficient, ids=list(ids), test_used=False,
                coefficient_per_reference_ppm=math.fsum((w-2*coefficient*x)/denominator for _, x, w in pairs),
                local_derivatives=derivatives)


def diagnose(data):
    # Validate the frozen four-state dataset layout before any descriptive use.
    rows = data['rows']
    if (len(rows) != 4 or {r['id'] for r in rows} != set((*benchmark.CALIBRATION, *benchmark.EVALUATION))
            or any(r['split'] != ('calibration' if r['id'] in benchmark.CALIBRATION else 'evaluation') for r in rows)):
        raise ValueError('Declared BEAVRS matrix required')
    rows = sorted(rows, key=lambda r: r['id'])
    data = dict(data, rows=rows)
    reference = data['reference_boron_ppm']
    for row in rows:
        benchmark.predict(reference, row['measured_rod_worth_pcm'], 100.)
        b = row['critical_boron_ppm']
        if type(b) not in (int, float) or not math.isfinite(b) or not 0 <= b < reference:
            raise ValueError('Finite measured concentration required')
    views = []
    for name, ids in CALIBRATION_VIEWS:
        train = [r for r in rows if r['id'] in ids]
        calibration = calibrate(reference, train, ids)
        coefficient = calibration['coefficient_pcm_per_ppm']
        evaluation = benchmark.evaluate(data, coefficient)
        train_low = min(r['critical_boron_ppm'] for r in train)
        train_high = max(r['critical_boron_ppm'] for r in train)
        predictions = []
        for row in sorted(rows, key=lambda r: r['id']):
            if row['split'] != 'evaluation': continue
            worth = row['measured_rod_worth_pcm']; factor = worth / coefficient**2
            predictions.append(dict(id=row['id'],
                measured_concentration_outside_calibration_range=not train_low <= row['critical_boron_ppm'] <= train_high,
                predicted_concentration_outside_calibration_range=not train_low <= benchmark.predict(reference, worth, coefficient) <= train_high,
                prediction_per_reference_ppm=1.+factor*calibration['coefficient_per_reference_ppm'],
                prediction_per_input_worth_pcm=-1./coefficient,
                calibration_input_derivatives=[dict(id=d['id'],
                    prediction_per_boron_ppm=factor*d['coefficient_per_boron_ppm'],
                    prediction_per_worth_pcm=factor*d['coefficient_per_worth_pcm']) for d in calibration['local_derivatives']]))
        views.append(dict(name=name, calibration=calibration, evaluation=evaluation,
                          calibration_boron_range_ppm=[train_low, train_high], prediction_sensitivity=predictions))
    effective = [dict(id=r['id'], split=r['split'],
        effective_secant_pcm_per_ppm=r['measured_rod_worth_pcm']/(reference-r['critical_boron_ppm']))
        for r in sorted(rows, key=lambda r: r['id'])]
    return dict(views=views, descriptive_secants=effective, selected_view=None,
                test_used_for_selection=False, uncertainty_propagated=False,
                measured_state_secants_used_for_prediction=False)


def run(output):
    output = Path(output).absolute()
    if any(p.is_symlink() for p in (output, *output.parents)): raise ValueError('Symlink output')
    data = benchmark.measurements(); source = benchmark.identity()
    drivers = {str(Path(p).absolute()): sha(p) for p in (__file__, benchmark.__file__)}
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output/'protocol.json', seal(dict(source=source, drivers=drivers, source_data=data['source'],
        calibration_views=CALIBRATION_VIEWS, evaluation_ids=benchmark.EVALUATION,
        scope='Diagnostic subsets and analytic local derivatives; no winner selection or new validation',
        historical_evaluation_now_development_data=True, measurement_uncertainty_available=False)))
    analysis = diagnose(deepcopy(data))
    unchanged = source == benchmark.identity() and all(sha(p)==h for p,h in drivers.items())
    report = dict(completed=True, passed=unchanged, source_unchanged=unchanged, analysis=analysis,
                  validated=False, plant_referenced=False, whole_plant_compared=False,
                  limitations=['Four conditional states, common reference and correlated cumulative worth inputs',
                               'Evaluation concentrations outside fitted-state concentration ranges',
                               'Local derivatives are not measurement uncertainties or confidence intervals',
                               'Secants combine boron compensation and configuration assumptions; not unique physical coefficients'])
    atomic_json(output/'report.json', seal(report)); return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); run(args.output)
