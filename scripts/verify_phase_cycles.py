"""Uniform isochoric HEM heat cycles with exact stocks and forward-only IF97 roots."""

import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, root

from atomic_model_hem_qualified import eos
from atomic_model_hem_qualified.cases import baseline
from atomic_model_hem_qualified.model import QualifiedHEMNetwork, signature
from atomic_model_hem_qualified.qualification import CRITERIA
from atomic_model_multiphysics.provenance import seal, sha
from atomic_model_training.spec import atomic_json
from physics.materials import iapws

CASES = {'liquid_entry': (0., 50.), 'liquid_exit': (0., -50.),
         'vapor_entry': (1., 50.), 'vapor_exit': (1., -50.)}
MESHES = (2, 4)
CAPS = (2e-5, 1e-5)
PRESSURES = (.5e6, 1e6, 8e6, 15.5e6)
DURATIONS = (.0005, .0005, .0005, .0005)
BUDGETS = dict(pressure_pa=.1, temperature_k=1e-5, specific_energy_j_kg=2e-6,
               mass_kg=1e-9, total_energy_j=1e-5, velocity_m_s=1e-6,
               quality=1e-8, density_relative=2e-10)


def quality_error(cell, reference):
    """Allow a one-sided chart only at an exact saturation endpoint."""
    measured, expected = cell['quality'], reference.quality
    if expected is None:
        if measured is not None:
            raise ValueError('Single-phase reference has a mixture observation')
        return 0.
    if measured is None:
        phase = 'liquid' if expected == 0. else 'vapor' if expected == 1. else None
        if phase is None or cell['phase'] != phase:
            raise ValueError('Interior mixture cannot be replaced by a single-phase chart')
        return 0.
    if type(measured) not in (int,float) or not np.isfinite(measured) or not 0 <= measured <= 1:
        raise ValueError('Invalid observed quality')
    return abs(measured-expected)


def forward_reference(endpoint, energy_offset, *, pressure_pa=1e6):
    """Solve forward charts at fixed density; do not call production rho_u."""
    if endpoint not in (0., 1.) or not np.isfinite(energy_offset) or abs(energy_offset) > 100:
        raise ValueError('Declared local saturation neighbourhood only')
    pressure_pa = eos.number(pressure_pa, "cycle pressure")
    saturated = eos.px(pressure_pa, endpoint)
    volume = 1 / saturated.density_kg_m3
    target = saturated.internal_energy_j_kg + energy_offset
    if abs(energy_offset) < 1e-8:
        return saturated
    if energy_offset < 0:
        def trial(p):
            f, g = eos.saturation(p)
            x = (volume - 1 / f.density_kg_m3) / (1 / g.density_kg_m3 - 1 / f.density_kg_m3)
            return x, (1 - x) * f.internal_energy_j_kg + x * g.internal_energy_j_kg - target
        pressure = brentq(lambda p: trial(p)[1], max(eos.P_MIN, .99 * pressure_pa), min(eos.P_MAX, 1.01 * pressure_pa), xtol=1e-7)
        quality, residual = trial(pressure)
        if not 0 <= quality <= 1 or abs(residual) > 2e-6:
            raise ValueError('Forward mixture root is outside physical chart')
        return eos.px(pressure, quality)
    raw = iapws._region1_raw if endpoint == 0 else iapws._region2_raw
    def residual(logarithms):
        p, t = np.exp(logarithms)
        values = raw(float(t), float(p * 1e-6))
        return [(values['v'] - volume) / volume, (values['u'] * 1000 - target) / 1e6]
    solved = root(residual, np.log([pressure_pa, saturated.temperature_k]), tol=1e-10)
    p, t = np.exp(solved.x)
    if not solved.success or max(abs(np.array(residual(solved.x)))) > 2e-12:
        raise ValueError('Independent forward single-phase root did not converge')
    result = eos.pt(float(p), float(t))
    if abs(result.internal_energy_j_kg - target) > 2e-6 or abs(result.density_kg_m3 * volume - 1) > 2e-12:
        raise ValueError('Forward stock residual exceeds budget')
    return result


def request_for(case, cells=2, cap=2e-5, *, pressure_pa=1e6):
    if case not in CASES or cells not in MESHES or cap not in CAPS:
        raise ValueError('Declared cycle/mesh/cap only')
    endpoint, offset = CASES[case]
    # Admit both extrema before creating any production trajectory.
    extrema = {value: forward_reference(endpoint, value, pressure_pa=pressure_pa) for value in (-abs(offset), abs(offset))}
    state = extrema[offset]
    saturated = eos.px(pressure_pa, endpoint)
    request = baseline(endpoint, velocity=0., cells=cells, pressure=pressure_pa)
    spec = request['model']
    spec['max_step_s'] = cap
    point = dict(kind='rho_u', density_kg_m3=saturated.density_kg_m3,
                 internal_energy_j_kg=saturated.internal_energy_j_kg + offset,
                 pressure_seed_pa=state.pressure_pa, temperature_seed_k=state.temperature_k, velocity_m_s=0.)
    for node in spec['nodes']:
        node.update(initial=deepcopy(point), volume_m3=.001)
    for edge in spec['edges']:
        edge['initial'] = [deepcopy(point) for _ in range(cells)]
    for wall in spec['walls']:
        wall.update(conductance_w_k=0., initial_temperature_k=state.temperature_k,
                    ambient_temperature_k=state.temperature_k)
    return request


def offset_at(start, time):
    if not 0 <= time <= .002:
        raise ValueError('Cycle time outside0..0.002s')
    fraction = time / .001 if time <= .001 else (.002 - time) / .001
    return start * (1 - 2 * fraction)


def compare_snapshot(snapshot, initial, reference, specific_increment):
    if len(snapshot['cells']) != len(initial['cells']):
        raise ValueError('Cycle cell count changed')
    errors = {key: 0. for key in BUDGETS}
    for cell, old in zip(snapshot['cells'], initial['cells'], strict=True):
        if cell['id'] != old['id']:
            raise ValueError('Cycle cell order changed')
        measured = dict(
            pressure_pa=abs(cell['pressure_pa'] - reference.pressure_pa),
            temperature_k=abs(cell['temperature_k'] - reference.temperature_k),
            specific_energy_j_kg=abs(cell['internal_energy_j_kg'] - reference.internal_energy_j_kg),
            mass_kg=abs(cell['mass_kg'] - old['mass_kg']),
            total_energy_j=abs(cell['total_energy_j'] - old['total_energy_j'] - old['mass_kg'] * specific_increment),
            velocity_m_s=abs(cell['velocity_m_s']),
            quality=quality_error(cell,reference),
            density_relative=abs(cell['density_kg_m3'] / reference.density_kg_m3 - 1),
        )
        if any(not np.isfinite(v) for v in measured.values()):
            raise ValueError('Nonfinite cycle measurement')
        errors = {key:max(errors[key],measured[key]) for key in errors}
    return errors


def execute(case, cells, cap, *, pressure_pa=1e6):
    request = request_for(case,cells,cap,pressure_pa=pressure_pa)
    model = QualifiedHEMNetwork(request)
    initial = model.snapshot()
    observations = [initial]
    endpoint, offset = CASES[case]
    masses = {c['id']:c['mass_kg'] for c in initial['cells']}
    calls, saved = [], None
    for index, duration in enumerate(DURATIONS):
        rate = -2 * offset / .001 * (1 if index < 2 else -1)
        controls = dict(fluid={name:mass*rate for name,mass in masses.items()}, walls={w['id']:0. for w in initial['walls']})
        command = dict(controls=controls, expected_revision=index, controller_state={'cycle_tick':index})
        observations.append(model.step(duration_s=duration, **command))
        calls.append(dict(duration_s=duration,command=command))
        if index == 1:
            saved = model.checkpoint()
    maxima = {key:0. for key in BUDGETS}
    references = []
    for snapshot in observations:
        target = offset_at(offset,snapshot['time_s'])
        reference = forward_reference(endpoint,target,pressure_pa=pressure_pa)
        references.append(reference.as_dict())
        error = compare_snapshot(snapshot,initial,reference,target-offset)
        maxima = {key:max(maxima[key],error[key]) for key in maxima}
    single = 'liquid' if endpoint == 0 else 'vapor'
    expected = ['two_phase',single] if offset < 0 else [single,'two_phase']
    checks = dict(exact_stock_and_forward_eos=all(v <= BUDGETS[k] for k,v in maxima.items()),
                  phase_crossing=all(c['phase'] == expected[0] for c in observations[0]['cells']) and all(c['phase'] == expected[1] for c in observations[2]['cells']),
                  phase_return=all(c['phase'] == expected[0] for c in observations[-1]['cells']),
                  mass_balance=all(abs(s['global_ledger']['mass_residual_kg']) <= CRITERIA['mass_balance_kg'] for s in observations),
                  energy_balance=all(abs(s['global_ledger']['energy_residual_j']) <= CRITERIA['energy_balance_j'] for s in observations),
                  entropy_residual=all(abs(s['global_ledger']['numerical_entropy_residual_j_k']) <= abs(CRITERIA['numerical_entropy_lower_j_k']) for s in observations),
                  positive_mass=all(c['mass_kg'] > 0 for s in observations for c in s['cells']),
                  exact_external_heat=all(abs(s['global_ledger']['external_fluid_heat_j'] - sum(masses.values()) * (offset_at(offset,s['time_s'])-offset)) <= CRITERIA['energy_balance_j'] for s in observations),
                  zero_net_heat=abs(observations[-1]['global_ledger']['external_fluid_heat_j']) <= CRITERIA['energy_balance_j'])
    if cells == MESHES[-1] and cap == CAPS[-1]:
        restored = QualifiedHEMNetwork(request)
        restored.restore(saved)
        for call in calls[2:]:
            resumed = restored.step(duration_s=call['duration_s'], **call['command'])
        checks['restart_exact'] = resumed == observations[-1] and restored.checkpoint() == model.checkpoint()
    return dict(request=request, program=calls, observations=observations, references=references,
                checkpoint=saved, record=dict(case=case,pressure_pa=pressure_pa,cells=cells,cap_s=cap,checks={k:bool(v) for k,v in checks.items()},
                                              passed=bool(all(checks.values())), maximum_errors=maxima))


def verify(output, *, pressures=(1e6,)):
    if not pressures or any(type(p) not in (int,float) or p not in PRESSURES for p in pressures) or len(set(pressures)) != len(pressures):
        raise ValueError("Choose distinct declared pressures")
    # Preflight the whole campaign before publishing an output directory.
    for pressure in pressures:
        for endpoint, offset in CASES.values():
            for value in (-abs(offset), abs(offset)):
                forward_reference(endpoint,value,pressure_pa=pressure)
    output = Path(output).absolute()
    if any(p.is_symlink() for p in (output,*output.parents)):
        raise ValueError('Output cannot be a symlink')
    output.mkdir(parents=True,exist_ok=False)
    source,driver = signature(),sha(__file__)
    declared = dict(source=source,driver_sha256=driver,cases=CASES,meshes=MESHES,caps_s=CAPS,
                    pressures_pa=pressures,durations_s=DURATIONS,budgets=BUDGETS,balance_criteria=CRITERIA,
                    reference='Exact fixed-density energy path, independent forward-chart roots, shared IF97',
                    validated=False,plant_referenced=False)
    atomic_json(output/'protocol.json',seal(declared))
    records=[]
    for pressure in pressures:
        for case in CASES:
            for cells in MESHES:
                for cap in CAPS:
                    result=execute(case,cells,cap,pressure_pa=pressure)
                    atomic_json(output/f'p-{pressure:g}-{case}-mesh-{cells}-cap-{cap:g}.json',seal(result))
                    records.append(result['record'])
                    print(pressure,case,cells,cap,result['record']['passed'],flush=True)
    checks=dict(native_runs=all(r['passed'] for r in records),source_unchanged=source==signature() and driver==sha(__file__))
    report=dict(schema_version=1,passed=bool(all(checks.values())),checks=checks,runs=records,
                scope='Uniform fixed-density heat cycles at selected declared pressures; no wall boiling/CHF validation or spatial convergence order',
                validated=False,plant_referenced=False)
    atomic_json(output/'report.json',seal(report))
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--pressure-matrix',action='store_true',help='Run all four declared saturation pressures, including15.5MPa')
    args=parser.parse_args(argv)
    result=verify(args.output,pressures=PRESSURES if args.pressure_matrix else (1e6,))
    print(json.dumps(dict(passed=result['passed'])))
    return 0 if result['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
