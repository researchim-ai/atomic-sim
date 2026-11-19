"""Sampled long-path qualification of the finite thermal catalog (not a parameter box)."""
import argparse
import json
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss

from atomic_model_multiphysics.provenance import seal, sha
from atomic_model_training.spec import atomic_json
from atomic_model_station_thermal_cases import thermal_catalog, build_thermal_case
from atomic_model_station_thermal_cases.provenance import case_identity
from atomic_model_station_thermal_path import ThermalTrajectory, TrajectoryAccuracyError
from atomic_model_station_thermal_resolved import compare_thermal

TIMES = (.1, .2, .3, .4, .5)


def identity():
    return dict(engine=case_identity(), driver=sha(__file__))


def qualify_case(case):
    """Retain failed stages; missing comparisons can never count as acceptance."""
    x, w = leggauss(160)
    radius, weights = np.sqrt((x + 1) / 2), w / 4
    stages, endpoints = {}, {}
    for name, band, tight in [('main', 4, False), ('time', 4, True), ('fourier', 5, False)]:
        path = ThermalTrajectory(build_thermal_case(case, band=band), max_segment=.1,
                                 rtol=2e-12 if tight else 1e-10,
                                 atol=2e-15 if tight else 1e-13)
        records, fields, prefix = [], {}, None
        try:
            for epoch in TIMES:
                status = path.advance_to(epoch)
                fields[epoch] = path.endpoint()
                records.append(dict(status=status, audit=path.audit()))
                if epoch == .2:
                    prefix = json.loads(json.dumps(path.checkpoint()))
            replay_exact = None
            if name == 'main':
                replay = ThermalTrajectory.restore(prefix)
                for epoch in TIMES[2:]:
                    replay.advance_to(epoch)
                replay_exact = replay.checkpoint() == path.checkpoint()
            stages[name] = dict(completed=True, records=records, replay_exact=replay_exact,
                                final_checkpoint_sha256=path.checkpoint()['sha256'])
            endpoints[name] = fields
        except (ValueError, TrajectoryAccuracyError) as error:
            stages[name] = dict(completed=False, records=records, error=str(error),
                                error_type=type(error).__name__,
                                refusal=getattr(error, 'receipt', None))
        print(case.name, name, 'completed', stages[name]['completed'], flush=True)
    comparisons = {}
    for name in ('time', 'fourier'):
        if 'main' in endpoints and name in endpoints:
            comparisons[name] = {str(t): compare_thermal(endpoints['main'][t], endpoints[name][t], radius, weights)
                                 for t in TIMES}
    checks = dict(all_stages_completed=all(s['completed'] for s in stages.values()),
                  boundary_restart_exact=stages['main'].get('replay_exact') is True)
    for name, tolerance in [('time', 1e-8), ('fourier', 1e-7)]:
        checks[name] = name in comparisons and all(
            max(row['velocity_l2'], row['temperature_l2']) < tolerance
            for row in comparisons[name].values())
    records = [r for s in stages.values() for r in s['records']]
    checks['cumulative_budgets'] = bool(records) and all(
        max(r['status']['sampled_global_defect_upper_bounds'][:2]) < 1e-9
        and r['status']['sampled_global_defect_upper_bounds'][2] < 1e-10 for r in records)
    checks['sampled_wall'] = bool(records) and all(
        max(r['audit']['maximum_wall_momentum_residual'], r['audit']['maximum_wall_thermal_residual']) < 1e-8
        for r in records)
    return dict(case=case.specification(), accepted=all(checks.values()), checks=checks,
                stages=stages, comparisons=comparisons)


def run(output):
    out = Path(output).absolute()
    if out.exists() or any(p.is_symlink() for p in (out, *out.parents)):
        raise ValueError('Fresh ordinary output required')
    source = identity()
    out.mkdir(parents=True)
    cases = thermal_catalog()
    atomic_json(out / 'protocol.json', seal(dict(source=source, times=list(TIMES),
        cases=[c.specification() for c in cases],
        criteria=dict(temporal_l2=1e-8, fourier_l2=1e-7, heat=1e-10, energy=1e-9, wall=1e-8),
        scope='Nondimensional single-phase finite catalog; sampled epochs and cumulative budget bounds',
        exclusions=['Continuous-time wall guarantee', 'Radial convergence at extended horizon',
                    'Experimental validation', 'Parameter-box qualification', 'Plant applicability'])) )
    rows = []
    for case in cases:
        row = qualify_case(case)
        rows.append(row)
        atomic_json(out / (case.name + '.json'), seal(row))
        if source != identity():
            raise ValueError('Scientific source changed during qualification')
    report = dict(passed=all(r['accepted'] for r in rows), source=source,
                  accepted_cases=[r['case']['name'] for r in rows if r['accepted']],
                  refused_cases=[r['case']['name'] for r in rows if not r['accepted']],
                  total=len(rows), horizon=.5, sample_times=list(TIMES),
                  radial_convergence_at_extended_horizon=False,
                  parameter_box_qualified=False, validated=False, station_usable=False,
                  closed_tasks=[], case_evidence={c.name: sha(out / (c.name + '.json')) for c in cases})
    atomic_json(out / 'report.json', seal(report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    report = run(parser.parse_args().output)
    print(json.dumps(report, ensure_ascii=False))
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
