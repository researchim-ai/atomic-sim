"""Controlled numerical axes for finite thermal case acceptance."""
import copy
from types import MappingProxyType

import numpy as np
from numpy.polynomial.legendre import leggauss

from atomic_model_multiphysics.provenance import seal
from atomic_model_station_regular.model import _finite_real
from atomic_model_station_thermal import ThermalResult
from atomic_model_station_thermal_resolved import (
    resolve_thermal,accept_resolved,ThermalQualityError,compare_thermal,initial_wall_compatibility,
)
from .catalog import ThermalCase
from .factory import build_thermal_case
from .provenance import case_identity


CRITERIA = MappingProxyType(dict(spatial_l2=1e-7,temporal_l2=1e-8,heat=1e-10,wall=1e-8,
                energy=1e-9,weak_pressure=1e-9,advection_power=1e-11,initial_l2=1e-10))
VARIANTS = (
    ('main',4,32,40,1e-10,1e-13),
    ('time',4,32,40,2e-12,2e-15),
    ('fourier',5,32,40,1e-10,1e-13),
    ('radial',4,28,32,1e-10,1e-13),
    ('temperature48',4,32,48,1e-10,1e-13),
)


def evaluate_thermal_case(case,*,horizon=.1):
    if type(case) is not ThermalCase:raise ValueError('ThermalCase required')
    _finite_real(horizon,'case horizon',.05,.5)
    source=case_identity();results={};systems={};stages={}
    for name,band,pv,pt,rtol,atol in VARIANTS:
        try:
            system=build_thermal_case(case,band=band,velocity_order=pv,temperature_order=pt)
            result=resolve_thermal(system,end_time=horizon,rtol=rtol,atol=atol)
            try:quality=accept_resolved(system,result)
            except ThermalQualityError as error:quality=error.receipt
            results[name]=result;systems[name]=system
            stages[name]=dict(completed=True,solver=result.report,quality=quality,
                initial_wall=initial_wall_compatibility(system))
        except ValueError as error:
            stages[name]=dict(completed=False,error_type=type(error).__name__,reason=str(error))
    x,w=leggauss(160);radius=np.sqrt((x+1)/2);weights=w/4
    comparisons={};initial={}
    if 'main' in results:
        for name in results:
            if name=='main':continue
            comparisons[name]=compare_thermal(results['main'],results[name],radius,weights)
            def endpoint(system):return ThermalResult(system.velocity,system.temperature,[],[],{},dict(time=0.))
            initial[name]=compare_thermal(endpoint(systems['main']),endpoint(systems[name]),radius,weights)
    def below(row,tolerance):return max(row['velocity_l2'],row['temperature_l2'])<tolerance
    complete=len(results)==len(VARIANTS)
    checks=dict(complete=complete,
        same_initial_field=complete and all(below(row,CRITERIA['initial_l2']) for row in initial.values()),
        main_quality=complete and all(stages[name]['quality']['passed'] for name in ('main','time','fourier','temperature48')),
        time=complete and below(comparisons['time'],CRITERIA['temporal_l2']),
        fourier=complete and below(comparisons['fourier'],CRITERIA['spatial_l2']),
        radial=complete and below(comparisons['radial'],CRITERIA['spatial_l2']),
        temperature=complete and below(comparisons['temperature48'],CRITERIA['spatial_l2']),
        energy=complete and all(max(result.report['maximum_kinetic_budget_residual'],result.report['maximum_variance_budget_residual'])<CRITERIA['energy'] for result in results.values()),
        weak_pressure=complete and all(result.report['maximum_weak_pressure_residual']<CRITERIA['weak_pressure'] for result in results.values()),
        power=complete and all(max(abs(result.report['endpoint_advection_power']),abs(result.report['endpoint_advection_variance']))<CRITERIA['advection_power'] for result in results.values()))
    if source!=case_identity():raise ValueError('Thermal case source changed during evaluation')
    return dict(accepted=bool(all(checks.values())),case=case.specification(),case_sha256=seal(case.specification())['sha256'],
        horizon=float(horizon),checks=checks,stages=stages,comparisons=comparisons,initial_comparisons=initial,
        criteria=dict(CRITERIA),source=source,
        interpolation_qualified=False,continuous_time_wall_qualified=False,statistical_convergence_qualified=False,
        station_usable=False,validated=False)


def evaluate_thermal_matrix(cases,*,horizon=.1,on_case=None):
    if (type(cases) not in (list,tuple) or not 1<=len(cases)<=16
            or any(type(case) is not ThermalCase for case in cases)):
        raise ValueError('Nonempty finite ThermalCase matrix1..16 required')
    if len({case.name for case in cases})!=len(cases):raise ValueError('Unique thermal case names required')
    _finite_real(horizon,'matrix horizon',.05,.5)
    if on_case is not None and not callable(on_case):raise ValueError('Callable case observer required')
    source=case_identity();rows=[]
    for case in cases:
        row=evaluate_thermal_case(case,horizon=horizon);rows.append(row)
        if on_case is not None:on_case(copy.deepcopy(row))
    if source!=case_identity():raise ValueError('Thermal matrix source changed during evaluation')
    count=sum(row['accepted'] for row in rows)
    return dict(accepted=count==len(cases),accepted_count=count,total=len(cases),cases=rows,
        tested_case_specifications=[case.specification() for case in cases],horizon=float(horizon),
        criteria=dict(CRITERIA),source=source,
        qualified_scope='Exactly the accepted case specifications and declared endpoint,not their Cartesian parameter box',
        input_box_qualified=False,interpolation_qualified=False,statistical_convergence_qualified=False,
        station_usable=False,validated=False)
