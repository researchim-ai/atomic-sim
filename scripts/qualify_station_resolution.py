"""Conditional Fourier/basis refinement and retained nonlinear wall diagnostics."""
import argparse,platform
from pathlib import Path
import numpy as np
import scipy
from numpy.polynomial.legendre import leggauss
from atomic_model_station_regular import regular_mode
from atomic_model_station_nonlinear import NonlinearSession
from atomic_model_station_resolution import compare_states,audit_nonlinear_momentum
from atomic_model_multiphysics.provenance import sha,seal
from atomic_model_training.spec import atomic_json
from scripts.qualify_station_advection import modes


def identity():
    import atomic_model_station_resolution as package
    from scripts.qualify_station_nonlinear import identity as base_identity
    root=Path(package.__file__).parent
    return dict(files={p.name:sha(p) for p in sorted(root.glob('*.py'))},driver=sha(__file__),base=base_identity(),python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__)


def initial_state(extent,order,seed):
    state=[]
    for m in range(extent+1):
        for k in (range(extent+1) if m==0 else range(-extent,extent+1)):
            model=regular_mode(k,m,.628319,order=order,quadrature=64);key=(k,m)
            q=model.fit(seed[key][0].evaluate(seed[key][1],model.radius)) if key in seed else np.zeros(model.dimension,complex)
            state.append((model,q))
    return state


def run(output,reference=None):
    out=Path(output).absolute();source=identity()
    if out.exists() or any(p.is_symlink() for p in (out,*out.parents)):raise ValueError('Fresh ordinary output required')
    out.mkdir(parents=True);criteria=dict(initial_velocity_rms=1e-10,fourier_rms=1e-7,basis_rms=1e-7,temporal_rms=1e-7,strong_wall=1e-8,minimum_fourier_ratio=10.)
    atomic_json(out/'protocol.json',seal(dict(source=source,criteria=criteria,
        initial_condition='Fixed physical polynomial field from extent1/order3 seeds; exact nested-space representation, new modes zero',
        comparison='Full signed velocity norm on independent96-point Gauss quadrature, including missing modes',
        fourier_family='Order3,time0.05,dt0.005,extent1..4',basis_family='Extent2,time0.1,dt0.0025,order3/5/7/9',
        temporal_family='Extent2/order9,time0.1,dt0.0025/0.00125',
        exclusions=['Families have different fixed backgrounds; no combined Navier-Stokes convergence claim','Unretained pressure not reconstructed'])))
    seed_list=[(m,q*10) for m,q in modes(1,3,64)];seed={(m.k,m.m):(m,q) for m,q in seed_list}
    x,w=leggauss(96);r=np.sqrt((x+1)/2);w=w/4;sample=np.linspace(.005,1,101)
    specs=[(f'fourier{k}',k,3,.005,10) for k in (1,2,3,4)]+[(f'basis{p}',2,p,.0025,40) for p in (3,5,7,9)]+[('time-half',2,9,.00125,80)]
    sessions={};cases={};software_passed=True
    for name,extent,order,dt,steps in specs:
        state=initial_state(extent,order,seed)
        start=compare_states(seed_list,state,r,w,first_time=0.,second_time=0.)
        session=NonlinearSession(state,viscosity=.05)
        for i in range(steps):
            before=session.modes;row,pressure=session.advance(dt)
        audit=audit_nonlinear_momentum(before,session.modes,pressure,dt=dt,viscosity=.05,radius=sample,tolerance=criteria['strong_wall'])
        software_passed=bool(software_passed and start['total_rms']<criteria['initial_velocity_rms'] and row['nonlinear_relative_residual']<=1e-12)
        cases[name]=dict(extent=extent,order=order,dt=dt,steps=steps,time=session.time,pressure_midpoint_time=row['midpoint_time'],initial_comparison=start,
            strong_audit=audit,final_step=row,final_checkpoint=session.checkpoint())
        sessions[name]=session
        print('resolution case',name,'wall',audit['maximum_retained_wall_residual'],flush=True)
    atomic_json(out/'cases.json',seal(cases))
    def difference(a,b):
        left=sessions[a];right=sessions[b]
        return compare_states(left.modes,right.modes,r,w,first_time=left.time,second_time=right.time)
    angular=[difference(f'fourier{k}',f'fourier{k+1}') for k in (1,2,3)]
    radial=[difference(f'basis{a}',f'basis{b}') for a,b in [(3,5),(5,7),(7,9)]]
    temporal=difference('basis9','time-half')
    ratios=[a['total_rms']/b['total_rms'] for a,b in zip(angular,angular[1:])]
    angular_ok=bool(angular[-1]['total_rms']<criteria['fourier_rms'] and min(ratios)>criteria['minimum_fourier_ratio'])
    basis_ok=bool(radial[-1]['total_rms']<criteria['basis_rms'])
    temporal_ok=bool(temporal['total_rms']<criteria['temporal_rms'])
    wall_ok=bool(cases['time-half']['strong_audit']['normalized_retained_wall_residual']<criteria['strong_wall'])
    comparison=dict(fourier=angular,basis=radial,time=temporal,fourier_ratios=ratios,
        fourier_conditional_gate_passed=angular_ok,basis_gate_passed=basis_ok,temporal_gate_passed=temporal_ok,finest_wall_gate_passed=wall_ok)
    atomic_json(out/'comparison.json',seal(comparison))
    if source!=identity():raise ValueError('Resolution audit source changed')
    report=dict(passed=software_passed,source=source,cases=len(cases),maximum_initial_velocity_error=max(v['initial_comparison']['total_rms'] for v in cases.values()),
        finest_fourier_difference=angular[-1]['total_rms'],finest_basis_difference=radial[-1]['total_rms'],finest_temporal_difference=temporal['total_rms'],
        finest_wall_residual=cases['time-half']['strong_audit']['maximum_retained_wall_residual'],
        fourier_conditional_gate_passed=angular_ok,basis_gate_passed=basis_ok,temporal_gate_passed=temporal_ok,finest_wall_gate_passed=wall_ok,
        all_omitted_modes_included=True,full_spatial_convergence_qualified=False,general_nonlinear_wall_qualified=False,
        thermal_model_supported=False,native_backend_changed=False,station_usable=False,validated=False,closed_tasks=[],
        scope='Reproducible conditional refinement measurements; passed validates execution and diagnostics, not physical gates')
    atomic_json(out/'report.json',seal(report))
    if not software_passed:raise ValueError('Resolution execution/initial-state qualification failed; evidence preserved')
    if reference and {p.name:sha(p) for p in Path(reference).glob('*.json')}!={p.name:sha(p) for p in out.glob('*.json')}:raise ValueError('Installed resolution scientific JSON differ')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--reference',type=Path)
    a=p.parse_args();print(run(a.output,a.reference)['passed'])

if __name__=='__main__':main()
