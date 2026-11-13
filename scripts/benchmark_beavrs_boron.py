"""Conditional BEAVRS Cycle-1 boron compensation; not full-core validation."""
import argparse
from importlib.resources import files
import hashlib
import json
import math
from pathlib import Path

from atomic_model_multiphysics.provenance import seal, sha
from atomic_model_training.spec import atomic_json
from atomic_model_plant_extension.pwr import SourceBoundPWR
from atomic_model_plant_extension.session import identity

SOURCE_SHA = '13eea6a808136dfee603d607cb1ac97ededbdfb31a7bb6a8728a937aa183e801'
COMMIT = '1ccae04dc3830a5977f08ed7f5f7e1e51ded0d22'
CALIBRATION = ('D', 'CD')
EVALUATION = ('ABCD', 'ALL')
STATES = {'ARO': '\\ac{ARO}', 'D': 'D in', 'CD': 'C, D in',
          'ABCD': 'A, B, C, D in', 'ALL': 'A, B, C, D, SE, SD, SC in'}
BANKS = ('D in', 'C with D in', 'B with D, C in', 'A with D, C, B in',
         'SE with D, C, B, A in', 'SD with D, C, B, A, SE in', 'SC with D, C, B, A, SE, SD in')


def measurements(raw=None):
    if raw is None:
        raw = files('atomic_model_plant_extension').joinpath('data/beavrs/expdata.tex').read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA:
        raise ValueError('BEAVRS source hash mismatch')
    table = raw.decode().split('\\label{tbl:meas_c1phys}}', 1)[1].split('\\end{table}', 1)[0]
    critical = {}; increments = {}
    for line in table.splitlines():
        columns = [v.strip() for v in line.split('&')]
        if len(columns) != 6: continue
        if columns[0] in STATES.values(): critical[columns[0]] = float(columns[1])
        if columns[2] in BANKS: increments[columns[2]] = float(columns[3])
    if set(critical) != set(STATES.values()) or set(increments) != set(BANKS):
        raise ValueError('Incomplete source table')
    cumulative = [math.fsum(increments[k] for k in BANKS[:n]) for n in (1, 2, 4, 7)]
    rows = [dict(id=k, critical_boron_ppm=critical[STATES[k]], measured_rod_worth_pcm=w,
                 split='calibration' if k in CALIBRATION else 'evaluation')
            for k, w in zip((*CALIBRATION, *EVALUATION), cumulative, strict=True)]
    return dict(reference_boron_ppm=critical[STATES['ARO']], rows=rows,
                uncertainty_boron_ppm=None, uncertainty_rod_worth_pcm=None,
                source=dict(commit=COMMIT, sha256=SOURCE_SHA, table='tbl:meas_c1phys',
                            url=f'https://github.com/mit-crpg/BEAVRS/blob/{COMMIT}/docs/specifications/expdata/expdata.tex'))


def fit(reference_boron_ppm, rows):
    if (len(rows) != 2 or {r['id'] for r in rows} != set(CALIBRATION)
            or any(r['split'] != 'calibration' for r in rows)):
        raise ValueError('Only declared calibration states allowed')
    if type(reference_boron_ppm) not in (int, float) or not math.isfinite(reference_boron_ppm):
        raise ValueError('Finite reference concentration required')
    rows = sorted(rows, key=lambda r: r['id'])
    differences = []
    for r in rows:
        b = r['critical_boron_ppm']; w = r['measured_rod_worth_pcm']
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in (b, w)) or not 0 <= b < reference_boron_ppm or w <= 0:
            raise ValueError('Positive finite boron compensation data required')
        differences.append((reference_boron_ppm - b, w))
    coefficient = math.fsum(d * w for d, w in differences) / math.fsum(d * d for d, w in differences)
    return dict(coefficient_pcm_per_ppm=coefficient, calibration_ids=list(CALIBRATION), test_used=False,
                criterion='Unweighted least squares through origin: measured worth = b * boron removal')


def predict(reference_boron_ppm, worth_pcm, coefficient_pcm_per_ppm):
    values = (reference_boron_ppm, worth_pcm, coefficient_pcm_per_ppm)
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in values) or reference_boron_ppm < 0 or worth_pcm < 0 or coefficient_pcm_per_ppm <= 0:
        raise ValueError('Finite conditional prediction inputs required')
    concentration = reference_boron_ppm - worth_pcm / coefficient_pcm_per_ppm
    if concentration < 0: raise ValueError('Negative critical boron prediction outside model domain')
    return concentration


def probe(coefficient=None):
    """Probe the actual plant reactivity method at fixed synthetic thermal state."""
    plant = SourceBoundPWR()
    if coefficient is not None:
        if type(coefficient) not in (int, float) or not math.isfinite(coefficient) or coefficient <= 0:
            raise ValueError('Positive finite coefficient required')
        plant.boron_coef = -coefficient * 1e-5
    plant.set_boron_ppm(1000.); before = plant.reactivity()
    plant.set_boron_ppm(1001.); after = plant.reactivity()
    return -(after - before) * 1e5


def evaluate(data, coefficient):
    rows = []
    for row in data['rows']:
        predicted = predict(data['reference_boron_ppm'], row['measured_rod_worth_pcm'], coefficient)
        rows.append(dict(**row, predicted_boron_ppm=predicted,
            residual_boron_ppm=predicted - row['critical_boron_ppm'],
            residual_reactivity_pcm=coefficient * (data['reference_boron_ppm'] - row['critical_boron_ppm']) - row['measured_rod_worth_pcm']))
    held = [r for r in rows if r['split'] == 'evaluation']
    return dict(coefficient_pcm_per_ppm=coefficient, rows=rows,
                evaluation_mae_ppm=math.fsum(abs(r['residual_boron_ppm']) for r in held) / len(held),
                evaluation_max_abs_error_ppm=max(abs(r['residual_boron_ppm']) for r in held))


def run(output):
    output = Path(output).absolute()
    if any(p.is_symlink() for p in (output, *output.parents)): raise ValueError('Symlink output')
    data = measurements(); source = identity(); driver = sha(__file__)
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / 'protocol.json', seal(dict(source=source, driver_sha256=driver, source_data=data['source'],
        calibration_ids=CALIBRATION, evaluation_ids=EVALUATION,
        model='Linear conditional boron compensation with measured sequential rod worth inputs',
        reference='ARO concentration975ppm is an imposed measured boundary, not a prediction',
        uncertainty_available=False, blind_test=False, validated=False, plant_referenced=False)))
    calibration = fit(data['reference_boron_ppm'], [r for r in data['rows'] if r['split'] == 'calibration'])
    atomic_json(output / 'calibration.json', seal(calibration))
    baseline = probe(); fitted = calibration['coefficient_pcm_per_ppm']; actual = probe(fitted)
    unchanged = source == identity() and driver == sha(__file__)
    report = dict(completed=True, passed=unchanged and math.isclose(actual, fitted, rel_tol=1e-10), source_unchanged=unchanged,
        calibration=calibration, baseline=evaluate(data, baseline), calibrated=evaluate(data, fitted),
        actual_plant_fitted_coefficient_pcm_per_ppm=actual, uncertainty_available=False,
        calibrated_plant_used_in_dynamics=False, validated=False, plant_referenced=False,
        limitations=['Measured rod worths are inputs, not predictions', 'No full-core geometry or spatial neutronics',
                     'Two held-out critical concentrations from the same campaign; correlations unknown',
                     'No uncertainty-based acceptance; no synthetic uncertainty assigned',
                     'No full-plant HZP state or dynamic trajectory simulated'])
    atomic_json(output / 'report.json', seal(report)); return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); run(args.output)
