"""Selected heated 3D endpoint refinement and full-prefix physical heat budget."""
import argparse
import platform
from pathlib import Path

import numpy as np
import scipy
from numpy.polynomial.legendre import leggauss

from atomic_model_multiphysics.provenance import sha, seal
from atomic_model_training.spec import atomic_json
from atomic_model_station_thermal import ThermalResult, evolve_thermal
from atomic_model_station_thermal_resolved import (
    resolved_velocity_mode, resolved_scalar_mode, ResolvedThermalSystem, resolve_thermal,
    audit_resolved, accept_resolved, ThermalQualityError, initial_wall_compatibility, compare_thermal,
)
from scripts.qualify_station_advection import modes
from scripts.qualify_station_thermal import nonlinear_system


def identity():
    import atomic_model_station_thermal_resolved as package
    from scripts.qualify_station_thermal import identity as base_identity
    return dict(files={p.name:sha(p) for p in sorted(Path(package.__file__).parent.glob('*.py'))},
                driver=sha(__file__),base=base_identity(),python=platform.python_version(),
                numpy=np.__version__,scipy=scipy.__version__)


def resolved_system(band=4,velocity_order=32,temperature_order=40,outer_order=24):
    if type(band) is not int or not 1<=band<=4:raise ValueError('Qualification bands1..4 required')
    seed={(m.k,m.m):(m,q*10) for m,q in modes(1,3,96)}
    velocity=[];temperature=[]
    for m in range(band+1):
        for k in (range(band+1) if m==0 else range(-band,band+1)):
            key=(k,m);shell=max(abs(k),m)
            v=resolved_velocity_mode(k,m,.628319,order=outer_order if shell==4 else velocity_order,quadrature=128)
            s=resolved_scalar_mode(k,m,.628319,order=outer_order if shell==4 else temperature_order,quadrature=128)
            q=v.fit(seed[key][0].evaluate(seed[key][1],v.radius)) if key in seed else np.zeros(v.dimension,complex)
            amplitude=(.1 if key==(0,0) else .02*(1+.3j)) if key in seed else 0.
            velocity.append((v,q))
            temperature.append((s,s.fit(amplitude*s.radius**s.m*(1-s.radius**2))))
    return ResolvedThermalSystem(velocity,temperature)


def initial_endpoint(system):
    return ThermalResult(system.velocity,system.temperature,[],[],{},dict(time=0.))


def run(output,reference=None):
    out=Path(output).absolute();source=identity()
    if out.exists() or any(p.is_symlink() for p in (out,*out.parents)):raise ValueError('Fresh ordinary output required')
    out.mkdir(parents=True)
    criteria=dict(spatial_l2=1e-7,temporal_l2=1e-8,physical_heat=1e-10,wall=1e-8,
                  energy=1e-9,weak_pressure=1e-9,initial_l2=1e-10)
    atomic_json(out/'protocol.json',seal(dict(source=source,criteria=criteria,time=.05,viscosity=.05,diffusivity=.03,buoyancy=.2,
        initial_field='Unchanged v80 polynomial field; same velocity seed and thermal amplitudes; absent initial modes zero',
        quadrature=128,comparison_quadrature=160,
        graded_basis='Inner bands through3: velocity32/temperature40; shell4: both24. Separate shell20 and temperature48 comparisons.',
        solver='Real-split BDF, linear viscous/diffusive/buoyant sparse quasi-Newton block; actual nonlinear FFT RHS',
        strong_check='Direct pair convolutions, physical pressure gradient, analytic Laplacians;129radii through wall at t=0.05',
        heat_check='Physical heat and independently differentiated wall flux integrated from t=0, including initial incompatible layer',
        exclusions=['Strong wall acceptance at t=0 or all intermediate times','General continuum error bound','Experimental validation','Station acceptance','Bit-exact BDF restart'])))
    x,w=leggauss(160);radius=np.sqrt((x+1)/2);weights=w/4
    seed=nonlinear_system(10)
    legacy=evolve_thermal(seed,end_time=.05)
    legacy_adapter=ResolvedThermalSystem(seed.velocity,seed.temperature)
    baseline_gate=None
    try:accept_resolved(legacy_adapter,legacy)
    except ThermalQualityError as error:baseline_gate=error.receipt
    if baseline_gate is None:raise ValueError('Historical coarse failure unexpectedly disappeared')
    cases={'legacy_v80':dict(solver=legacy.report,gate=baseline_gate)}
    results={};systems={}
    specs=[('radial',4,28,32,24),('final',4,32,40,24),('temperature48',4,32,48,24),
           ('outer20',4,32,40,20),('fourier3',3,32,40,24),('tight',4,32,40,24)]
    for name,band,pv,pt,outer in specs:
        system=resolved_system(band,pv,pt,outer)
        initial=compare_thermal(initial_endpoint(seed),initial_endpoint(system),radius,weights)
        result=resolve_thermal(system,end_time=.05,rtol=2e-12 if name=='tight' else 1e-10,atol=2e-15 if name=='tight' else 1e-13)
        try:gate=accept_resolved(system,result)
        except ThermalQualityError as error:gate=error.receipt
        cases[name]=dict(solver=result.report,initial=initial,gate=gate,initial_wall=initial_wall_compatibility(system))
        results[name]=result;systems[name]=system
        print(name,'heat',result.report['maximum_physical_heat_budget_residual'],
              'walls',gate['audit']['maximum_wall_momentum_residual'],gate['audit']['maximum_wall_thermal_residual'],'gate',gate['passed'],flush=True)
    comparison={name:compare_thermal(results[name],results['final'],radius,weights) for name in ('radial','temperature48','outer20','fourier3','tight')}
    def fields_below(row,tolerance):return max(row['velocity_l2'],row['temperature_l2'])<tolerance
    checks=dict(initial=all(fields_below(cases[name]['initial'],criteria['initial_l2']) for name in results),
        radial=fields_below(comparison['radial'],criteria['spatial_l2']),
        thermal_radial=fields_below(comparison['temperature48'],criteria['spatial_l2']),
        outer_radial=fields_below(comparison['outer20'],criteria['spatial_l2']),
        fourier=fields_below(comparison['fourier3'],criteria['spatial_l2']),
        temporal=fields_below(comparison['tight'],criteria['temporal_l2']),
        final_quality=all(cases[name]['gate']['passed'] for name in ('final','temperature48','tight')),
        coarse_refusal=not cases['legacy_v80']['gate']['passed'] and not cases['radial']['gate']['passed'],
        initial_incompatibility_retained=all(not cases[name]['initial_wall']['compatible'] and not cases[name]['initial_wall']['state_modified'] for name in results),
        energy=all(max(case['solver']['maximum_kinetic_budget_residual'],case['solver']['maximum_variance_budget_residual'])<criteria['energy'] for case in cases.values()),
        pressure=all(case['solver']['maximum_weak_pressure_residual']<criteria['weak_pressure'] for case in cases.values()))
    atomic_json(out/'cases.json',seal(dict(cases=cases)))
    atomic_json(out/'comparison.json',seal(dict(comparison=comparison,checks=checks)))
    if source!=identity():raise ValueError('Resolved thermal qualification source changed')
    gate=cases['final']['gate'];audit=gate['audit']
    report=dict(passed=bool(all(checks.values())),source=source,checks=checks,
        selected_heated_endpoint_and_heat_qualified=bool(all(checks.values())),
        maximum_physical_heat_budget_residual=gate['maximum_physical_heat_budget_residual'],
        maximum_wall_momentum_residual=audit['maximum_wall_momentum_residual'],
        maximum_wall_thermal_residual=audit['maximum_wall_thermal_residual'],
        radial_velocity_l2=comparison['radial']['velocity_l2'],radial_temperature_l2=comparison['radial']['temperature_l2'],
        fourier_velocity_l2=comparison['fourier3']['velocity_l2'],fourier_temperature_l2=comparison['fourier3']['temperature_l2'],
        temporal_velocity_l2=comparison['tight']['velocity_l2'],temporal_temperature_l2=comparison['tight']['temperature_l2'],
        initial_field_modified=False,heat_prefix_starts_at_zero=True,selected_wall_gate_explicit=True,
        all_time_wall_qualified=False,general_strong_pde_qualified=False,experimental_validation=False,
        station_usable=False,validated=False,native_backend_changed=False,closed_tasks=[])
    atomic_json(out/'report.json',seal(report))
    if not report['passed']:raise ValueError('Selected resolved thermal qualification failed; evidence preserved')
    if reference and {p.name:sha(p) for p in Path(reference).glob('*.json')}!={p.name:sha(p) for p in out.glob('*.json')}:raise ValueError('Installed resolved thermal scientific JSON differ')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference',type=Path)
    args=parser.parse_args();print(run(args.output,args.reference)['passed'])

if __name__=='__main__':main()
