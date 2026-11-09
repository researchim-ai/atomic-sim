"""Finite-receiver flashing in branched HEM graphs; bounded numerical evidence."""

import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from atomic_model_hem_qualified.model import QualifiedHEMNetwork, signature
from atomic_model_hem_qualified.qualification import CRITERIA, errors
from atomic_model_hem_qualified.reference import solve
from atomic_model_multiphysics.provenance import seal, sha, unseal
from atomic_model_training.spec import atomic_json, read_json
from scripts.verify_phase_release import CALLS, inventories, release_request

VARIANTS = ('parallel', 'split_equal', 'split_unequal', 'reversed', 'permuted')
MESHES = (4, 8)
CAPS = (1e-5, 5e-6)
FIELDS = {'pressure_pa': .1, 'temperature_k': 1e-5, 'velocity_m_s': 1e-6,
          'mass_kg': 1e-9, 'total_energy_j': 1e-5, 'momentum_kg_m_s': 1e-9,
          'quality': 1e-8}
TIME_FIELDS = {k: FIELDS[k] for k in ('pressure_pa', 'temperature_k', 'velocity_m_s', 'quality')}


def comparison_passed(diff, *, temporal=False):
    budgets = TIME_FIELDS if temporal else FIELDS
    if set(diff) != set(FIELDS) or any(type(v) not in (int, float) or not np.isfinite(v) or v < 0 for v in diff.values()):
        raise ValueError('Incomplete/nonfinite comparison')
    return bool(all(diff[k] <= budget for k, budget in budgets.items()))


def request_for(variant, cells=4, cap=1e-5):
    if variant not in VARIANTS or cells not in MESHES or cap not in CAPS:
        raise ValueError('Declared topology/mesh/cap only')
    request = release_request(cells, cap)
    spec = request['model']
    if variant.startswith('split_'):
        receiver = spec['nodes'].pop()
        fraction = .5 if variant == 'split_equal' else 1 / 3
        for index, share in enumerate((fraction, 1 - fraction)):
            node = deepcopy(receiver)
            node.update(id=f'b{index}', volume_m3=receiver['volume_m3'] * share)
            spec['nodes'].append(node)
            spec['edges'][index]['to_node'] = node['id']
    elif variant == 'reversed':
        for edge in spec['edges']:
            edge['from_node'], edge['to_node'] = edge['to_node'], edge['from_node']
            edge['initial'].reverse()
            edge['heat_power_w'].reverse()
            for point in edge['initial']:
                point['velocity_m_s'] *= -1
        for wall in spec['walls']:
            edge, cell = wall['cell'].split('/')
            wall['cell'] = f'{edge}/{cells - 1 - int(cell)}'
    elif variant == 'permuted':
        spec['nodes'].reverse()
        spec['edges'].reverse()
    return request


def canonical(snapshot, variant, cells):
    """Map orientation and equal split to the same physical cells/stocks."""
    result = {}
    for original in snapshot['cells']:
        cell = deepcopy(original)
        name = cell['id']
        if variant == 'reversed' and '/' in name:
            edge, index = name.split('/')
            name = f'{edge}/{cells - 1 - int(index)}'
            cell['velocity_m_s'] *= -1
            cell['momentum_kg_m_s'] *= -1
        result[name] = cell
    if variant == 'split_equal':
        a, b = result.pop('b0'), result.pop('b1')
        # Extensive stocks add; intensive equality is checked independently.
        combined = deepcopy(a)
        for key in ('mass_kg', 'total_energy_j', 'momentum_kg_m_s', 'volume_m3'):
            combined[key] = a[key] + b[key]
        result['b'] = combined
    return result


def differences(a, b):
    if set(a) != set(b):
        raise ValueError('Nonidentical physical cell sets')
    output = {}
    for field in FIELDS:
        values = []
        for name in a:
            x, y = a[name][field], b[name][field]
            if type(x) not in (int, float) or type(y) not in (int, float) or not np.isfinite([x, y]).all():
                raise ValueError('Missing/nonfinite topology observable')
            values.append(abs(x - y))
        output[field] = max(values)
    return output


def physical_checks(observations, variant):
    first, last = observations[0], observations[-1]
    a = {c['id']: c for c in first['cells']}
    b = {c['id']: c for c in last['cells']}
    receivers = ('b0', 'b1') if variant.startswith('split_') else ('b',)
    checks = dict(
        donor_loses_mass=b['a']['mass_kg'] < a['a']['mass_kg'],
        donor_flashes=a['a']['quality'] == 0 and b['a']['quality'] > 0 and b['a']['phase'] == 'two_phase',
        donor_depressurizes=b['a']['pressure_pa'] < a['a']['pressure_pa'],
        receivers_gain_mass=all(b[n]['mass_kg'] > a[n]['mass_kg'] for n in receivers),
        receivers_pressurize=all(b[n]['pressure_pa'] > a[n]['pressure_pa'] for n in receivers),
        positive_mass=all(c['mass_kg'] > 0 for s in observations for c in s['cells']),
    )
    for key, budget in (('mass_residual_kg', CRITERIA['mass_balance_kg']),
                        ('energy_residual_j', CRITERIA['energy_balance_j'])):
        checks[key] = all(abs(s['global_ledger'][key]) <= budget for s in observations)
    checks['entropy_lower'] = all(s['global_ledger']['numerical_entropy_residual_j_k'] >= CRITERIA['numerical_entropy_lower_j_k'] for s in observations)
    checks['closed_adiabatic'] = all(s['global_ledger'][key] == 0 for s in observations
                                   for key in ('external_fluid_heat_j', 'external_wall_heat_j', 'external_entropy_j_k'))
    if variant == 'split_equal':
        checks['equal_receiver_intensives'] = all(
            abs({c['id']: c for c in s['cells']}['b0'][key] - {c['id']: c for c in s['cells']}['b1'][key]) <= FIELDS[key]
            for s in observations for key in ('pressure_pa', 'temperature_k', 'quality'))
    if variant == 'split_unequal':
        checks['smaller_receiver_pressurizes_faster'] = b['b0']['pressure_pa'] > b['b1']['pressure_pa']
    return {k: bool(v) for k, v in checks.items()}


def verify(output, *, reuse_from=None):
    output = Path(output).absolute()
    if any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError('Output cannot be a symlink')
    source, driver = signature(), sha(__file__)
    helper = sha(Path(__file__).with_name('verify_phase_release.py'))
    declared = dict(source=source, driver_sha256=driver, release_helper_sha256=helper,
                    variants=VARIANTS, meshes=MESHES, caps_s=CAPS, program=CALLS,
                    criteria=CRITERIA, equivalence_budgets=FIELDS,
                    temporal_intensive_budgets=TIME_FIELDS,
                    temporal_stock_differences='Recorded solution sensitivity; not a conservation residual or admitted stock-error bound',
                    requests={v: {str(n): request_for(v, n) for n in MESHES} for v in VARIANTS},
                    scope='Closed 0.001s finite graph; equilibrium no-slip IF97; no nozzle/choking or plant validation',
                    validated=False, plant_referenced=False)
    reuse = Path(reuse_from).absolute() if reuse_from is not None else None
    reuse_hashes = {}
    if reuse is not None:
        previous = unseal(read_json(reuse / 'protocol.json'))
        # Checker criteria may change, but physical source, requests, program,
        # closure helper and discretizations must match before any reuse.
        for key in ('source', 'release_helper_sha256', 'variants', 'meshes', 'caps_s', 'program', 'criteria', 'requests'):
            if json.dumps(previous[key], sort_keys=True) != json.dumps(declared[key], sort_keys=True):
                raise ValueError('Reuse source/physical protocol differs: ' + key)
        names = ['protocol.json', *(f'references-{v}.json' for v in ('split_equal', 'split_unequal')),
                 *(f'{v}-mesh-{n}-cap-{cap:g}.json' for n in MESHES for cap in CAPS for v in VARIANTS)]
        reuse_hashes = {name: sha(reuse / name) for name in names}
        declared['reused_evidence'] = dict(directory=str(reuse), files=reuse_hashes,
                                         previous_checker_sha256=previous['driver_sha256'],
                                         previous_matrix_acceptance_inherited=False)
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / 'protocol.json', seal(declared))
    histories, runs, initial_stocks = {}, [], []
    references = {}
    for variant in ('split_equal', 'split_unequal'):
        request = request_for(variant)
        references[variant] = (unseal(read_json(reuse / f'references-{variant}.json')) if reuse else
                               {method: solve(request, CALLS, method, max_step=2e-5) for method in ('DOP853', 'Radau')})
        atomic_json(output / f'references-{variant}.json', seal(references[variant]))
    for cells in MESHES:
        for cap in CAPS:
            for variant in VARIANTS:
                request = request_for(variant, cells, cap)
                model = QualifiedHEMNetwork(request)
                initial_stocks.append(inventories(model))
                previous_run = unseal(read_json(reuse / f'{variant}-mesh-{cells}-cap-{cap:g}.json')) if reuse else None
                if previous_run is not None and previous_run['request'] != request:
                    raise ValueError('Reused native request differs')
                observations = previous_run['observations'] if previous_run is not None else [model.snapshot()]
                if previous_run is not None and (len(observations) != len(CALLS) + 1 or observations[0] != model.snapshot()
                    or [s['time_s'] for s in observations] != [0.0, .0002, .0005, .001]):
                    raise ValueError('Reused native observation grid/initial state differs')
                checkpoint = None
                if previous_run is None:
                    for index, call in enumerate(CALLS):
                        observations.append(model.advance(call['duration_s']))
                        if index == 1:
                            checkpoint = model.checkpoint()
                checks = physical_checks(observations, variant)
                if cells == 4 and cap == CAPS[-1]:
                    if previous_run is not None:
                        for call in CALLS[:2]:
                            model.advance(call['duration_s'])
                        checkpoint = model.checkpoint()
                        expected = model.advance(CALLS[-1]['duration_s'])
                        if expected != observations[-1]:
                            raise ValueError('Reused trajectory final state differs from replay')
                    restored = QualifiedHEMNetwork(request)
                    restored.restore(checkpoint)
                    resumed = restored.advance(CALLS[-1]['duration_s'])
                    checks['restart_exact'] = resumed == observations[-1] and restored.checkpoint() == model.checkpoint()
                    atomic_json(output / f'checkpoint-{variant}.json', checkpoint)
                reference_errors = {}
                if cells == 4 and variant in references:
                    for method, reference in references[variant].items():
                        reference_errors[method] = {k: max(errors(a, b)[k] for a, b in zip(observations, reference['observations'], strict=True))
                                                    for k in ('pressure_error_pa', 'temperature_error_k', 'velocity_error_m_s')}
                    checks['independent_reference'] = all(v <= CRITERIA[k] for e in reference_errors.values() for k, v in e.items())
                record = dict(variant=variant, cells=cells, cap_s=cap, passed=all(checks.values()), checks=checks, reference_errors=reference_errors)
                histories[(variant, cells, cap)] = observations
                runs.append(record)
                atomic_json(output / f'{variant}-mesh-{cells}-cap-{cap:g}.json', seal(dict(request=request, observations=observations, record=record)))
                print(variant, cells, cap, record['passed'], flush=True)
    comparisons = []
    for cells in MESHES:
        for cap in CAPS:
            baseline = histories[('parallel', cells, cap)]
            for variant in ('split_equal', 'reversed', 'permuted'):
                other = histories[(variant, cells, cap)]
                diff = {k: max(differences(canonical(a, 'parallel', cells), canonical(b, variant, cells))[k]
                               for a, b in zip(baseline, other, strict=True)) for k in FIELDS}
                comparisons.append(dict(kind='topology_equivalence', variant=variant, cells=cells, cap_s=cap,
                                        differences=diff, passed=comparison_passed(diff)))
        for variant in VARIANTS:
            coarse, fine = (histories[(variant, cells, cap)] for cap in CAPS)
            diff = {k: max(differences(canonical(a, variant, cells), canonical(b, variant, cells))[k]
                           for a, b in zip(coarse, fine, strict=True)) for k in FIELDS}
            comparisons.append(dict(kind='time_sensitivity', variant=variant, cells=cells,
                                    differences=diff, passed=comparison_passed(diff, temporal=True)))
    agreement = {variant: {key: max(errors(a, b)[key] for a, b in zip(ref['DOP853']['observations'], ref['Radau']['observations'], strict=True))
                           for key in ('pressure_error_pa', 'temperature_error_k', 'velocity_error_m_s')}
                 for variant, ref in references.items()}
    checks = dict(native_runs=all(r['passed'] for r in runs), equivalent_graphs_and_time=all(r['passed'] for r in comparisons),
                  independent_integrators=all(v <= CRITERIA[k] for d in agreement.values() for k, v in d.items()),
                  equal_initial_stocks=all(np.allclose(list(s.values()), list(initial_stocks[0].values()), rtol=2e-12, atol=0) for s in initial_stocks),
                  source_unchanged=source == signature() and driver == sha(__file__) and helper == sha(Path(__file__).with_name('verify_phase_release.py')))
    checks['reused_files_unchanged'] = all(sha(reuse / name) == value for name, value in reuse_hashes.items())
    report = dict(schema_version=1, passed=bool(all(checks.values())), checks={k: bool(v) for k,v in checks.items()},
                  runs=runs, comparisons=comparisons, reference_agreement=agreement,
                  scope=declared['scope'], spatial_convergence_claimed=False, validated=False, plant_referenced=False)
    atomic_json(output / 'report.json', seal(report))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reuse-from', type=Path, help='Readmit frozen trajectories with matching physical protocol; retain original failed report')
    args = parser.parse_args(argv)
    result = verify(args.output, reuse_from=args.reuse_from)
    print(json.dumps(dict(passed=result['passed'])))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
