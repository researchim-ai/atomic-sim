"""Conditional isothermal feedback audit against BEAVRS Cycle-1 measurements."""
import argparse
import hashlib
from importlib.resources import files
import math
from pathlib import Path
import re

from atomic_model_multiphysics.provenance import seal, sha
from atomic_model_training.spec import atomic_json
from atomic_model_plant_extension import water
from plants.vver import doppler_slope_per_k
from scripts import benchmark_beavrs_boron as boron

STEPS_K = (.2, .1, .05)
PRESSURES_PA = (15e6, 15.7e6, 16e6)
RODS = (0., .5, 1.)
LABELS = {'\\ac{ARO}': 'ARO', 'D in': 'D', 'C, D in': 'CD'}
NUMERICAL_TOLERANCE_PCM_K = 1e-5


def measurements(raw=None):
    if raw is None: raw = files('atomic_model_plant_extension').joinpath('data/beavrs/expdata.tex').read_bytes()
    if hashlib.sha256(raw).hexdigest() != boron.SOURCE_SHA: raise ValueError('BEAVRS source hash mismatch')
    text = raw.decode(); table = text.split('\\label{tbl:meas_c1phys}}', 1)[1].split('\\end{table}', 1)[0]
    rows = []
    for line in table.splitlines():
        columns = [v.strip() for v in line.split('&')]
        if len(columns) != 6 or columns[4] not in LABELS: continue
        value = float(columns[5].split('\\\\', 1)[0].strip())
        rows.append(dict(id=LABELS[columns[4]], measured_pcm_per_f=value, measured_pcm_per_k=value*1.8))
    if len(rows) != 3 or {r['id'] for r in rows} != set(LABELS.values()): raise ValueError('Incomplete temperature table')
    context = text.split('\\label{tbl:hzp_c1}', 1)[1].split('\\end{table}', 1)[0]
    match = re.search(r'Inlet Coolant Temperature\s*&\s*([0-9.]+)', context)
    if match is None: raise ValueError('Missing HZP inlet context')
    temperature_f = float(match[1]); temperature_k = (temperature_f-32.)/1.8+273.15
    return dict(rows=rows, context_inlet_temperature_f=temperature_f, context_temperature_k=temperature_k,
                temperature_context_is_exact_itc_test_condition=False,
                uncertainty_pcm_per_k=None, source=boron.measurements(raw)['source'])


def probe(temperature_k, pressure_pa, step_k, rod=0.):
    for value in (temperature_k, pressure_pa, step_k, rod):
        if type(value) not in (float, int) or not math.isfinite(value): raise ValueError('Finite feedback inputs required')
    if not 550 <= temperature_k <= 580 or not 14e6 <= pressure_pa <= 16.5e6 or not .001 <= step_k <= 1. or not 0 <= rod <= 1:
        raise ValueError('Conditional feedback domain')
    plant = boron.SourceBoundPWR(); plant.rod = rod
    def response(t):
        # A prescribed thermodynamic path, not a hydraulic equilibrium or trajectory.
        for name in ('cold_leg', 'core'):
            plant.network.volumes[name].set_liquid(pressure_pa, t, 975.)
        plant.fuel_temperature = t
        return plant.reactivity()
    low = response(temperature_k-step_k); high = response(temperature_k+step_k)
    numerical = (high-low)/(2*step_k)*1e5
    state = water.pt(pressure_pa, temperature_k)
    if state.region != 1: raise ValueError('Liquid-only feedback path')
    dv_dt = float(water.single_jacobian(pressure_pa, temperature_k, 1)[0,1])
    moderator = plant.density_coef * (-state.density_kg_m3**2 * dv_dt) * 1e5
    fuel = doppler_slope_per_k(temperature_k, plant.doppler_coef) * 1e5
    return dict(temperature_k=temperature_k, pressure_pa=pressure_pa, step_k=step_k, rod_fraction=rod,
        numerical_pcm_per_k=numerical, analytic_pcm_per_k=moderator+fuel,
        moderator_pcm_per_k=moderator, fuel_pcm_per_k=fuel,
        numerical_error_pcm_per_k=numerical-(moderator+fuel))


def qualify(probes):
    expected = {(p,h,r) for p in PRESSURES_PA for h in STEPS_K for r in RODS}
    lookup = {}
    for point in probes:
        key = (point['pressure_pa'],point['step_k'],point['rod_fraction'])
        if key in lookup or key not in expected or not math.isfinite(point['numerical_error_pcm_per_k']):
            raise ValueError('Derivative matrix coverage/finiteness')
        lookup[key] = point
    if set(lookup) != expected: raise ValueError('Incomplete derivative matrix')
    groups = []
    for p in PRESSURES_PA:
        for r in RODS:
            errors = [abs(lookup[(p,h,r)]['numerical_error_pcm_per_k']) for h in STEPS_K]
            groups.append(dict(pressure_pa=p,rod_fraction=r,absolute_errors_pcm_per_k=errors,
                passed=errors[-1]<NUMERICAL_TOLERANCE_PCM_K and all(b<=a for a,b in zip(errors,errors[1:]))))
    return dict(passed=all(g['passed'] for g in groups),groups=groups,
                criterion='Finest step within fixed tolerance and nonincreasing error under step halving')


def run(output):
    output = Path(output).absolute()
    if any(p.is_symlink() for p in (output,*output.parents)): raise ValueError('Symlink output')
    data = measurements(); source = boron.identity()
    drivers = {str(Path(p).absolute()):sha(p) for p in (__file__,boron.__file__)}
    output.mkdir(parents=True,exist_ok=False)
    atomic_json(output/'protocol.json',seal(dict(source=source,drivers=drivers,source_data=data['source'],
        context_temperature_k=data['context_temperature_k'],pressures_pa=PRESSURES_PA,steps_k=STEPS_K,rod_proxies=RODS,
        pressure_is_model_assumption=True,rod_proxies_are_not_beavrs_bank_mapping=True,
        exact_measurement_conditions_reconstructed=False,numerical_tolerance_pcm_per_k=NUMERICAL_TOLERANCE_PCM_K,
        numerical_criterion='Finest step within tolerance and nonincreasing analytic-error under step halving',
        scope='Prescribed equal fuel/coolant temperature at fixed pressure and boron; no dynamics or fitting')))
    probes = [probe(data['context_temperature_k'],p,h,r) for p in PRESSURES_PA for h in STEPS_K for r in RODS]
    representative = next(x for x in probes if x['pressure_pa']==15.7e6 and x['step_k']==.05 and x['rod_fraction']==0.)
    prediction = representative['numerical_pcm_per_k']
    comparisons = [dict(**row,conditional_prediction_pcm_per_k=prediction,
                        residual_pcm_per_k=prediction-row['measured_pcm_per_k']) for row in data['rows']]
    spreads = [max(x['numerical_pcm_per_k'] for x in probes if x['pressure_pa']==p and x['step_k']==h)
               -min(x['numerical_pcm_per_k'] for x in probes if x['pressure_pa']==p and x['step_k']==h)
               for p in PRESSURES_PA for h in STEPS_K]
    unchanged = source==boron.identity() and all(sha(p)==v for p,v in drivers.items())
    qualification = qualify(probes)
    report = dict(completed=True,passed=unchanged and qualification['passed'],numerical_qualification=qualification,
        source_unchanged=unchanged,measurements=data,probes=probes,comparisons=comparisons,
        maximum_rod_proxy_spread_pcm_per_k=max(spreads),bank_dependent_temperature_feedback_implemented=False,
        fitted=False,validated=False,plant_referenced=False,whole_plant_compared=False,
        limitations=['HZP inlet context is not confirmed exact temperature for each ITC measurement',
                     'Pressure sweep is a model assumption, not measured uncertainty',
                     'No individual-bank state mapping or spatial neutronic feedback',
                     'Numerical differentiation acceptance is not experimental agreement'])
    atomic_json(output/'report.json',seal(report)); return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.output)
