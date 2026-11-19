"""Longer nondimensional heated trajectories, cumulative budgets and boundary restart."""
import argparse
import json
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss

from atomic_model_multiphysics.provenance import sha,seal
from atomic_model_training.spec import atomic_json
from atomic_model_station_thermal_resolved import (
    ResolvedThermalSystem,resolved_velocity_mode,resolved_scalar_mode,resolve_thermal,
    compare_thermal,audit_resolved,
)
from atomic_model_station_thermal_path import ThermalTrajectory,TrajectoryAccuracyError
from atomic_model_station_thermal_path.provenance import runtime_identity
from scripts.qualify_station_thermal_resolved import resolved_system


TIMES=(.1,.2,.3,.4,.5)


def identity():
    from scripts.qualify_station_thermal_resolved import identity as base_identity
    return dict(runtime=runtime_identity(),driver=sha(__file__),base=base_identity())


def long_system(band=4,velocity_order=32,temperature_order=40):
    base=resolved_system(4,velocity_order,temperature_order,24)
    if band==4:return base
    if band!=5:raise ValueError('Long qualification bands4/5 required')
    velocity=list(base.velocity);temperature=list(base.temperature)
    existing=set(base.keys)
    for m in range(6):
        for k in (range(6) if m==0 else range(-5,6)):
            if (k,m) in existing:continue
            v=resolved_velocity_mode(k,m,.628319,order=16,quadrature=128)
            t=resolved_scalar_mode(k,m,.628319,order=16,quadrature=128)
            velocity.append((v,np.zeros(v.dimension,complex)))
            temperature.append((t,np.zeros(t.dimension,complex)))
    return ResolvedThermalSystem(velocity,temperature)


def run_path(system,*,tight=False):
    path=ThermalTrajectory(system,max_segment=.1,rtol=2e-12 if tight else 1e-10,atol=2e-15 if tight else 1e-13)
    fields={};records=[];prefix=None
    for time in TIMES:
        status=path.advance_to(time)
        fields[time]=path.endpoint()
        audit=path.audit()
        records.append(dict(status=status,audit=audit))
        if time==.2:prefix=json.loads(json.dumps(path.checkpoint()))
        print('path',len(system.keys),'tight',tight,'time',time,'bounds',status['sampled_global_defect_upper_bounds'],flush=True)
    return path,fields,records,prefix


def run(output,reference=None):
    out=Path(output).absolute();source=identity()
    if out.exists() or any(p.is_symlink() for p in (out,*out.parents)):raise ValueError('Fresh ordinary output required')
    out.mkdir(parents=True)
    criteria=dict(spatial_l2=1e-7,temporal_l2=1e-8,heat=1e-10,energy=1e-9,wall=1e-8,weak_pressure=1e-9)
    atomic_json(out/'protocol.json',seal(dict(source=source,criteria=criteria,times=list(TIMES),
        horizon=.5,horizon_relative_to_v81=10.,units='Nondimensional,not physical seconds or station operating duration',
        physical_seed='Unmodified v80/v81 polynomial seed; source zero; shell5 initial fields zero',
        quadrature=128,comparison_quadrature=160,
        basis='Inner through3: velocity32/temperature40;shell4:24/24;shell5:16/16',
        budgets='Global endpoint defect and conservative triangle composition of local sampled maxima,with roundoff guard;no reset at continuation',
        restart='Exact physical state,history and quadratures at declared segment boundary;fresh BDF history on every segment',
        comparisons='Temporal and Fourier at5 declared physical epochs;radial and scalar basis at final epoch;segmented vs uninterrupted BDF',
        exclusions=['Physical duration conversion','Full developed turbulent statistics','Continuous-time or all-state wall guarantee','Bitwise equivalence with uninterrupted BDF','Station acceptance','Experimental validation'])))
    x,w=leggauss(160);r=np.sqrt((x+1)/2);w=w/4
    system=long_system()
    main,fields,records,prefix=run_path(system)
    replay=ThermalTrajectory.restore(prefix)
    replay_exact=True
    for time in (.3,.4,.5):replay.advance_to(time)
    replay_exact=main.checkpoint()==replay.checkpoint()
    uninterrupted=resolve_thermal(system,end_time=.5)
    continuous_difference=compare_thermal(fields[.5],uninterrupted,r,w)
    tight,tight_fields,tight_records,_=run_path(system,tight=True)
    fourier,fourier_fields,fourier_records,_=run_path(long_system(5))
    temporal={str(t):compare_thermal(fields[t],tight_fields[t],r,w) for t in TIMES}
    spectral={str(t):compare_thermal(fields[t],fourier_fields[t],r,w) for t in TIMES}
    coarse_system=long_system(4,28,32)
    coarse=resolve_thermal(coarse_system,end_time=.5)
    scalar_system=long_system(4,32,48)
    scalar=resolve_thermal(scalar_system,end_time=.5)
    radial=compare_thermal(fields[.5],coarse,r,w)
    thermal_radial=compare_thermal(fields[.5],scalar,r,w)
    # Actual partial progress on the candidate must not change the public session.
    from scipy.special import jv,jn_zeros
    v=resolved_velocity_mode(0,0,.628319,order=8,quadrature=128)
    s=resolved_scalar_mode(0,0,.628319,order=8,quadrature=128)
    small=ResolvedThermalSystem([(v,np.zeros(v.dimension))],[(s,s.fit(.1*jv(0,jn_zeros(0,1)[0]*s.radius)))])
    refusal=ThermalTrajectory(small,max_segment=.05);before=refusal.checkpoint();atomic=False
    try:refusal.advance_to(.2,max_segments=1)
    except TrajectoryAccuracyError:atomic=refusal.checkpoint()==before
    all_records=records+tight_records+fourier_records
    path_bounds=[row['status']['sampled_global_defect_upper_bounds'] for row in all_records]
    def norm(row):return max(row['velocity_l2'],row['temperature_l2'])
    checks=dict(boundary_restart_exact=replay_exact,partial_failure_atomic=atomic,
        continuous=norm(continuous_difference)<criteria['temporal_l2'],
        temporal=all(norm(row)<criteria['temporal_l2'] for row in temporal.values()),
        fourier=all(norm(row)<criteria['spatial_l2'] for row in spectral.values()),
        radial=norm(radial)<criteria['spatial_l2'],thermal_radial=norm(thermal_radial)<criteria['spatial_l2'],
        heat=all(row[2]<criteria['heat'] for row in path_bounds),
        energy=all(max(row[:2])<criteria['energy'] for row in path_bounds),
        wall=all(max(row['audit']['maximum_wall_momentum_residual'],row['audit']['maximum_wall_thermal_residual'])<criteria['wall'] for row in all_records),
        weak_pressure=all(result.report['maximum_weak_pressure_residual']<criteria['weak_pressure'] for result in list(fields.values())+list(tight_fields.values())+list(fourier_fields.values())),
        coarse_heat_refused=coarse.report['maximum_physical_heat_budget_residual']>criteria['heat'])
    atomic_json(out/'paths.json',seal(dict(main=records,tight=tight_records,fourier=fourier_records,
        temporal_comparisons=temporal,fourier_comparisons=spectral,uninterrupted_difference=continuous_difference,
        radial_comparison=radial,thermal_radial_comparison=thermal_radial,
        raw_endpoints=dict(uninterrupted=uninterrupted.report,coarse_radial=coarse.report,temperature48=scalar.report))))
    final_checkpoint=main.checkpoint()
    atomic_json(out/'continuation.json',seal(dict(prefix_checkpoint=prefix,final_checkpoint=final_checkpoint,
        prefix_time=.2,final_time=.5,boundary_restart_exact=replay_exact,
        replay_checkpoint_sha256=replay.checkpoint()['sha256'],partial_failure_atomic=atomic,
        bdf_internal_history_serialized=False,uninterrupted_bitwise_equivalence_claimed=False)))
    if source!=identity():raise ValueError('Thermal trajectory qualification source changed')
    report=dict(passed=bool(all(checks.values())),source=source,checks=checks,
        selected_longer_thermal_path_qualified=bool(all(checks.values())),
        horizon=.5,relative_horizon=10.,sample_times=list(TIMES),accepted_main_segments=main.status()['accepted_segments'],
        global_heat_bound=main.status()['sampled_global_defect_upper_bounds'][2],
        global_kinetic_bound=main.status()['sampled_global_defect_upper_bounds'][0],
        maximum_sampled_fourier_velocity_l2=max(row['velocity_l2'] for row in spectral.values()),
        maximum_sampled_fourier_temperature_l2=max(row['temperature_l2'] for row in spectral.values()),
        maximum_sampled_temporal_velocity_l2=max(row['velocity_l2'] for row in temporal.values()),
        maximum_sampled_temporal_temperature_l2=max(row['temperature_l2'] for row in temporal.values()),
        uninterrupted_difference=continuous_difference,boundary_restart_exact=replay_exact,partial_failure_atomic=atomic,
        continuous_error_bound=False,statistical_convergence_qualified=False,physical_duration_qualified=False,
        general_strong_pde_qualified=False,station_usable=False,validated=False,native_backend_changed=False,closed_tasks=[])
    atomic_json(out/'report.json',seal(report))
    if not report['passed']:raise ValueError('Longer thermal path qualification failed;evidence preserved')
    if reference and {p.name:sha(p) for p in Path(reference).glob('*.json')}!={p.name:sha(p) for p in out.glob('*.json')}:raise ValueError('Installed thermal path scientific JSON differ')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--reference',type=Path)
    args=p.parse_args();print(run(args.output,args.reference)['passed'])

if __name__=='__main__':main()
