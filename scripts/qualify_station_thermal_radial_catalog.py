"""Independent radial axes at the extended finite-catalog endpoint."""
import argparse
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss

from atomic_model_multiphysics.provenance import seal, sha
from atomic_model_training.spec import atomic_json
from atomic_model_station_thermal import ThermalResult
from atomic_model_station_thermal_cases import thermal_catalog, build_thermal_case
from atomic_model_station_thermal_cases.provenance import case_identity
from atomic_model_station_thermal_resolved import (
    resolve_thermal, accept_resolved, ThermalQualityError, compare_thermal,
)

# Fourier shell4 remains24/24. Only inner radial orders change.
VARIANTS = (('main',32,40), ('velocity28',28,40),
            ('temperature32',32,32), ('temperature48',32,48))
HORIZON = .5


def identity():
    return dict(engine=case_identity(), driver=sha(__file__))


def below(row, tolerance):
    values = [row['velocity_l2'], row['temperature_l2']]
    return bool(np.all(np.isfinite(values)) and min(values) >= 0 and max(values) < tolerance)


def qualify_case(case):
    stages, results, systems = {}, {}, {}
    for name, pv, pt in VARIANTS:
        try:
            system = build_thermal_case(case, velocity_order=pv, temperature_order=pt)
            result = resolve_thermal(system, end_time=HORIZON)
            try:
                quality = accept_resolved(system, result)
            except ThermalQualityError as error:
                quality = error.receipt
            stages[name] = dict(completed=True, solver=result.report, quality=quality)
            results[name], systems[name] = result, system
        except ValueError as error:
            stages[name] = dict(completed=False, reason=str(error), error_type=type(error).__name__)
        print(case.name, name, 'completed', stages[name]['completed'], flush=True)
    x, w = leggauss(160)
    radius, weights = np.sqrt((x+1)/2), w/4
    comparisons, initials = {}, {}
    def initial(system):
        return ThermalResult(system.velocity, system.temperature, [], [], {}, dict(time=0.))
    if 'main' in results:
        for name in results:
            if name == 'main':
                continue
            comparisons[name] = compare_thermal(results['main'], results[name], radius, weights)
            initials[name] = compare_thermal(initial(systems['main']), initial(systems[name]), radius, weights)
    complete = len(results) == len(VARIANTS)
    checks = dict(complete=complete,
        same_initial_fields=complete and all(below(row,1e-10) for row in initials.values()),
        target_quality=complete and all(stages[name]['quality']['passed'] for name in ('main','temperature48')),
        energy=complete and all(max(r.report['maximum_kinetic_budget_residual'],
            r.report['maximum_variance_budget_residual']) < 1e-9 for r in results.values()),
        weak_pressure=complete and all(r.report['maximum_weak_pressure_residual'] < 1e-9 for r in results.values()),
        advective_power=complete and all(max(abs(r.report['endpoint_advection_power']),
            abs(r.report['endpoint_advection_variance'])) < 1e-11 for r in results.values()))
    for name in ('velocity28','temperature32','temperature48'):
        checks[name] = name in comparisons and below(comparisons[name],1e-7)
    return dict(case=case.specification(), accepted=all(checks.values()), checks=checks,
                stages=stages, comparisons=comparisons, initial_comparisons=initials)


def run(output):
    out = Path(output).absolute()
    if out.exists() or any(p.is_symlink() for p in (out,*out.parents)):
        raise ValueError('Fresh ordinary output required')
    source = identity()
    out.mkdir(parents=True)
    cases = thermal_catalog()
    atomic_json(out/'protocol.json',seal(dict(source=source,horizon=HORIZON,
        cases=[c.specification() for c in cases],variants=[list(v) for v in VARIANTS],
        band=4,shell4_orders=[24,24],quadrature=128,comparison_quadrature=160,
        rtol=1e-10,atol=1e-13,
        criteria=dict(initial_l2=1e-10,radial_l2=1e-7,heat=1e-10,wall=1e-8,
                      energy=1e-9,weak_pressure=1e-9,advective_power=1e-11),
        method='Uninterrupted BDF to identical endpoint;velocity and temperature axes varied separately',
        coarse_policy='Retain failed coarse heat/wall receipts;coarse fields are comparison references,not accepted targets',
        exclusions=['Intermediate-epoch radial convergence','Refinement of Fourier shell4 radial orders',
                    'Continuous parameter box','Experimental validation','Station acceptance'])))
    rows = []
    for case in cases:
        row = qualify_case(case)
        atomic_json(out/(case.name+'.json'),seal(row))
        rows.append(row)
        if source != identity():
            raise ValueError('Scientific source changed during radial qualification')
    report = dict(passed=all(row['accepted'] for row in rows),source=source,horizon=HORIZON,
        accepted_cases=[r['case']['name'] for r in rows if r['accepted']],
        refused_cases=[r['case']['name'] for r in rows if not r['accepted']],
        completed_paths=sum(s['completed'] for r in rows for s in r['stages'].values()),
        endpoint_inner_radial_comparison_passed=all(r['accepted'] for r in rows),
        intermediate_radial_epochs_qualified=False,shell4_radial_refinement_qualified=False,
        station_usable=False,validated=False,closed_tasks=[],
        case_evidence={c.name:sha(out/(c.name+'.json')) for c in cases})
    atomic_json(out/'report.json',seal(report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    report = run(parser.parse_args().output)
    print('passed',report['passed'],'accepted',report['accepted_cases'],'refused',report['refused_cases'])
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
