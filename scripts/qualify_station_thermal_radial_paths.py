"""Sampled radial path comparisons with independently refined outer shell axes."""
import argparse
from functools import lru_cache
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss

from atomic_model_multiphysics.provenance import seal, sha
from atomic_model_training.spec import atomic_json
from atomic_model_station_thermal_cases import thermal_catalog, build_thermal_case
from atomic_model_station_thermal_cases.provenance import case_identity
from atomic_model_station_thermal_path import ThermalTrajectory
from atomic_model_station_thermal_resolved import (
    resolved_velocity_mode, resolved_scalar_mode, ResolvedThermalSystem, compare_thermal,
)

TIMES = (.1,.2,.3,.4,.5)
VARIANTS = ('main','inner_velocity28','inner_temperature48','shell_velocity26','shell_temperature32')


def identity():
    return dict(engine=case_identity(),driver=sha(__file__))


@lru_cache(maxsize=64)
def shell_mode(kind,k,m,alpha):
    factory=resolved_velocity_mode if kind=='velocity' else resolved_scalar_mode
    return factory(k,m,alpha,order=26 if kind=='velocity' else 32,quadrature=128)


def build_variant(case,name):
    if name not in VARIANTS:
        raise ValueError('Declared radial path variant required')
    base=build_thermal_case(case,velocity_order=28 if name=='inner_velocity28' else 32,
                            temperature_order=48 if name=='inner_temperature48' else 40)
    if not name.startswith('shell_'):
        return base
    velocity,temperature=[],[]
    for v,q in base.velocity:
        if name=='shell_velocity26' and max(abs(v.k),v.m)==4:
            new=shell_mode('velocity',v.k,v.m,float(v.alpha))
            velocity.append((new,new.fit(v.evaluate(q,new.radius))))
        else:
            velocity.append((v,q))
    for t,q in base.temperature:
        if name=='shell_temperature32' and max(abs(t.k),t.m)==4:
            new=shell_mode('temperature',t.k,t.m,float(t.alpha))
            temperature.append((new,new.fit(t.evaluate(q,new.radius))))
        else:
            temperature.append((t,q))
    return ResolvedThermalSystem(velocity,temperature,viscosity=float(base.viscosity),
        diffusivity=float(base.diffusivity),buoyancy=float(base.buoyancy),sources=base.sources)


def below(row,tolerance):
    a=np.array([row['velocity_l2'],row['temperature_l2']])
    return bool(np.all(np.isfinite(a)) and np.all(a>=0) and np.max(a)<tolerance)


def qualify_case(case):
    x,w=leggauss(160);radius=np.sqrt((x+1)/2);weights=w/4
    stages,fields,initials={},{},{}
    for name in VARIANTS:
        records=[]
        try:
            path=ThermalTrajectory(build_variant(case,name),max_segment=.1)
            initials[name]=path.endpoint()
            endpoints={}
            for epoch in TIMES:
                status=path.advance_to(epoch)
                endpoint=path.endpoint()
                endpoints[epoch]=endpoint
                records.append(dict(status=status,audit=path.audit(),
                    weak_pressure=endpoint.report['maximum_weak_pressure_residual']))
            stages[name]=dict(completed=True,records=records)
            fields[name]=endpoints
        except ValueError as error:
            stages[name]=dict(completed=False,records=records,error_type=type(error).__name__,
                              reason=str(error),refusal=getattr(error,'receipt',None))
        print(case.name,name,'completed',stages[name]['completed'],flush=True)
    comparisons,initial_comparisons={},{}
    for name in VARIANTS[1:]:
        if 'main' in fields and name in fields:
            comparisons[name]={str(t):compare_thermal(fields['main'][t],fields[name][t],radius,weights) for t in TIMES}
        if 'main' in initials and name in initials:
            initial_comparisons[name]=compare_thermal(initials['main'],initials[name],radius,weights)
    complete=all(s['completed'] for s in stages.values())
    records=[r for s in stages.values() for r in s['records']]
    checks=dict(complete=complete,
        same_initial_fields=len(initial_comparisons)==4 and all(below(r,1e-10) for r in initial_comparisons.values()),
        weak_pressure=complete and all(r['weak_pressure']<1e-9 for r in records),
        budgets=complete and all(max(r['status']['sampled_global_defect_upper_bounds'][:2])<1e-9
            and r['status']['sampled_global_defect_upper_bounds'][2]<1e-10 for r in records),
        sampled_wall=complete and all(max(r['audit']['maximum_wall_momentum_residual'],
            r['audit']['maximum_wall_thermal_residual'])<1e-8 for r in records))
    for name in VARIANTS[1:]:
        checks[name]=name in comparisons and all(below(r,1e-7) for r in comparisons[name].values())
    return dict(case=case.specification(),accepted=all(checks.values()),checks=checks,
                stages=stages,comparisons=comparisons,initial_comparisons=initial_comparisons)


def run(output):
    out=Path(output).absolute()
    if out.exists() or any(p.is_symlink() for p in (out,*out.parents)):
        raise ValueError('Fresh ordinary output required')
    source=identity();out.mkdir(parents=True);cases=thermal_catalog()
    atomic_json(out/'protocol.json',seal(dict(source=source,times=list(TIMES),variants=list(VARIANTS),
        cases=[c.specification() for c in cases],band=4,quadrature=128,comparison_quadrature=160,
        criteria=dict(initial_l2=1e-10,radial_l2=1e-7,heat=1e-10,energy=1e-9,wall=1e-8,weak_pressure=1e-9),
        main_inner_orders=[32,40],main_shell4_orders=[24,24],
        shell_axes=dict(velocity=[24,26],temperature=[24,32]),
        method='Identical BDF segment schedule and strict cumulative quality gates;one radial axis changed per variant',
        exclusions=['Continuous-time PDE error bound','Parameter-box qualification','Statistical stationarity',
                    'Experimental validation','Station applicability'])) )
    rows=[]
    for case in cases:
        row=qualify_case(case);rows.append(row)
        atomic_json(out/(case.name+'.json'),seal(row))
        if source!=identity():
            raise ValueError('Scientific source changed during sampled radial qualification')
    passed=all(r['accepted'] for r in rows)
    report=dict(passed=passed,source=source,accepted_cases=[r['case']['name'] for r in rows if r['accepted']],
        refused_cases=[r['case']['name'] for r in rows if not r['accepted']],
        completed_paths=sum(s['completed'] for r in rows for s in r['stages'].values()),
        completed_epochs=sum(len(s['records']) for r in rows for s in r['stages'].values()),
        sampled_inner_and_shell_radial_comparisons_passed=passed,sample_times=list(TIMES),
        continuous_radial_error_bound=False,parameter_box_qualified=False,station_usable=False,
        validated=False,closed_tasks=[],case_evidence={c.name:sha(out/(c.name+'.json')) for c in cases})
    atomic_json(out/'report.json',seal(report))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    report=run(parser.parse_args().output)
    print('passed',report['passed'],'accepted',report['accepted_cases'],'refused',report['refused_cases'])
    if not report['passed']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
