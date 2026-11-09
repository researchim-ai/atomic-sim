"""Reproducible external component comparisons; never a whole-simulator ranking."""

import argparse
import json
import math
import os
import platform
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

import numpy as np

from atomic_model_hem_qualified import eos
from atomic_model_hem_qualified.model import signature
from atomic_model_multiphysics.provenance import seal, sha
from atomic_model_spatial import Model, eigenvalue
from atomic_model_training.spec import atomic_json
from scripts.verify_phase_cycles import CASES, PRESSURES, execute as execute_cycle

PROVIDERS = ('coolprop', 'openmc')
SEEDS = (17, 29, 43)
RATIOS = (.8, 1., 1.2)
EOS_BUDGETS = dict(density_kg_m3=2e-9, internal_energy_j_kg=2e-8,
                   enthalpy_j_kg=2e-8, entropy_j_kg_k=2e-8, temperature_k=2e-9)
# Relative budgets, scaled by max(abs(reference), 1 SI unit).
SOURCE_URLS = dict(coolprop='https://coolprop.org/fluid_properties/IF97.html',
                   openmc='https://docs.openmc.org/en/stable/pythonapi/generated/openmc.XSdata.html',
                   openmodelica='https://openmodelica.org/download/download-linux/')


def points():
    result = []
    for p in PRESSURES:
        ts = eos.px(p, 0.).temperature_k
        for t in (350., ts-5., ts+5., 800.):
            result.append(dict(id=f'pt-{p:g}-{t:.12g}',kind='pt',pressure_pa=p,temperature_k=t))
        for x in (0., 1e-6, .05, .5, 1.):
            result.append(dict(id=f'px-{p:g}-{x:g}',kind='px',pressure_pa=p,quality=x))
    return result


def atomic_water(point):
    state = eos.pt(point['pressure_pa'],point['temperature_k']) if point['kind']=='pt' else eos.px(point['pressure_pa'],point['quality'])
    return {key:float(getattr(state,key)) for key in EOS_BUDGETS}


def relative_errors(measured, reference):
    if set(measured) != set(EOS_BUDGETS) or set(reference) != set(EOS_BUDGETS):
        raise ValueError('Incomplete property vector')
    if any(type(v) not in (int,float) or not math.isfinite(v) for v in [*measured.values(),*reference.values()]):
        raise ValueError('Nonfinite property vector')
    return {k:abs(measured[k]-reference[k])/max(abs(reference[k]),1.) for k in EOS_BUDGETS}


def timing(function, inputs, *, repeats=7, loops=20):
    for point in inputs: function(point)
    samples=[]
    for _ in range(repeats):
        start=time.perf_counter_ns()
        for _ in range(loops):
            for point in inputs: function(point)
        samples.append((time.perf_counter_ns()-start)/(loops*len(inputs)))
    return dict(samples_ns_per_state=samples,median_ns_per_state=statistics.median(samples),
                minimum_ns_per_state=min(samples),repeats=repeats,loops=loops,
                scope='Warm default caches; full five-property vector; Python API overhead included; not a plant simulation')


def export_modelica(output, cases):
    """Export the exact same static state list; runtime absence is not a pass."""
    rows=[]
    for point in cases:
        if point['kind']=='pt':
            expression=f"W.setState_pT({point['pressure_pa']:.17g}, {point['temperature_k']:.17g})"
        else:
            p=point['pressure_pa'];x=point['quality']
            expression=f'W.setState_ph({p:.17g}, W.bubbleEnthalpy(W.setSat_p({p:.17g}))+{x:.17g}*(W.dewEnthalpy(W.setSat_p({p:.17g}))-W.bubbleEnthalpy(W.setSat_p({p:.17g}))))'
        rows.append(expression)
    source='''model AtomicSimWaterBenchmark
  package W = Modelica.Media.Water.StandardWater;
  parameter Integer n = %d;
  W.ThermodynamicState states[n];
  output Real density[n], energy[n], enthalpy[n], entropy[n], temperature[n];
equation
  states = {%s};
  for i in 1:n loop
    density[i] = W.density(states[i]);
    energy[i] = W.specificInternalEnergy(states[i]);
    enthalpy[i] = W.specificEnthalpy(states[i]);
    entropy[i] = W.specificEntropy(states[i]);
    temperature[i] = W.temperature(states[i]);
  end for;
end AtomicSimWaterBenchmark;
'''%(len(rows),',\n    '.join(rows))
    (output/'AtomicSimWaterBenchmark.mo').write_text(source)
    (output/'run.mos').write_text('loadModel(Modelica);\nloadFile("AtomicSimWaterBenchmark.mo");\nsimulate(AtomicSimWaterBenchmark, stopTime=0.001, numberOfIntervals=1, outputFormat="csv");\ngetErrorString();\n')
    return dict(status='prepared_not_run',compiler_present=shutil.which('omc') is not None,
                reason='Modelica export has not been compiled or admitted; no comparison result',
                model_sha256=sha(output/'AtomicSimWaterBenchmark.mo'),source=SOURCE_URLS['openmodelica'])


def coolprop_profile(output, cases):
    import CoolProp
    import CoolProp.CoolProp as CP
    states={backend:CP.AbstractState(backend,'Water') for backend in ('IF97','HEOS')}
    def evaluate(point, backend='IF97'):
        state=states[backend]
        if point['kind']=='pt': state.update(CP.PT_INPUTS,point['pressure_pa'],point['temperature_k'])
        else: state.update(CP.PQ_INPUTS,point['pressure_pa'],point['quality'])
        return dict(density_kg_m3=float(state.rhomass()),internal_energy_j_kg=float(state.umass()),
                    enthalpy_j_kg=float(state.hmass()),entropy_j_kg_k=float(state.smass()),temperature_k=float(state.T()))
    rows=[]
    for point in cases:
        ours=atomic_water(point);reference=evaluate(point);heos=evaluate(point,'HEOS')
        differences=relative_errors(ours,reference)
        rows.append(dict(point=point,atomic=ours,coolprop_if97=reference,coolprop_heos=heos,
                         relative_errors= differences,heos_differences_diagnostic=relative_errors(ours,heos),
                         passed=all(differences[k]<=EOS_BUDGETS[k] for k in EOS_BUDGETS)))
    atomic_json(output/'water-points.json',seal(rows))
    trajectories=[]
    # Compare external forward properties on every actual thermal-cycle state.
    # CoolProp is not a transient network solver: no dynamic-solver claim.
    for p in PRESSURES:
        for case in CASES:
            cycle=execute_cycle(case,2,2e-5,pressure_pa=p)
            observed=[]
            for snapshot in cycle['observations']:
                for cell in snapshot['cells']:
                    point=dict(kind='px',pressure_pa=cell['pressure_pa'],quality=cell['quality']) if cell['quality'] is not None else dict(kind='pt',pressure_pa=cell['pressure_pa'],temperature_k=cell['temperature_k'])
                    measured={k:cell[k] for k in EOS_BUDGETS};reference=evaluate(point)
                    delta=relative_errors(measured,reference)
                    observed.append(dict(time_s=snapshot['time_s'],cell=cell['id'],point=point,
                                         relative_errors=delta,passed=all(delta[k]<=EOS_BUDGETS[k] for k in EOS_BUDGETS)))
            trajectories.append(dict(case=case,pressure_pa=p,native_passed=cycle['record']['passed'],observations=observed,
                                     passed=cycle['record']['passed'] and all(v['passed'] for v in observed)))
            atomic_json(output/f'cycle-{p:g}-{case}.json',seal(cycle))
    atomic_json(output/'water-cycle-checks.json',seal(trajectories))
    durations={}
    # Alternate order between repeated measurements to reduce order bias.
    functions={'atomic':atomic_water,'coolprop_if97':evaluate}
    samples={name:[] for name in functions}
    for repetition in range(7):
        order=list(functions) if repetition%2==0 else list(reversed(functions))
        for name in order:
            samples[name].append(timing(functions[name],cases,repeats=1)['median_ns_per_state'])
    for name, values in samples.items():
        durations[name]=dict(samples_ns_per_state=values,median_ns_per_state=statistics.median(values),
                             scope='36-state warm-cache mixed PT/PQ batch; 20 loops; five properties; API overhead included')
    return dict(status='completed',passed=all(r['passed'] for r in rows) and all(r['passed'] for r in trajectories),
                version=CoolProp.__version__,git_revision=CP.get_global_param_string('gitrevision'),
                module_sha256=sha(CP.__file__),static_states=len(rows),native_cycles=len(trajectories),
                dynamic_forward_checks=sum(len(r['observations']) for r in trajectories),
                maximum_relative_errors={k:max(r['relative_errors'][k] for r in rows) for k in EOS_BUDGETS},
                maximum_heos_difference_diagnostic={k:max(r['heos_differences_diagnostic'][k] for r in rows) for k in EOS_BUDGETS},
                timings=durations,whole_plant_comparison=False,
                scope='Independent IF97 implementation at identical PT/PQ coordinates; HEOS(IAPWS95) difference diagnostic, not experiment; cycle checks are properties along Atomic trajectories')


def neutron_config(ratio, cells=8):
    if ratio not in RATIOS or type(cells) is not int or not 1<=cells<=512:
        raise ValueError('Declared homogeneous benchmark only')
    absorption=.09
    return dict(faces_m=np.linspace(0.,50.,cells+1).tolist(),diffusion_m=[[1/(6*absorption)]*cells],
                absorption_per_m=[[absorption]*cells],nu_fission_per_m=[[absorption*ratio]*cells],
                scatter_per_m=[[[0.]*cells]],speed_m_s=[[1e7]*cells],chi_prompt=[[1.]*cells],
                chi_delayed=[[1.]*cells],beta=[[0.]*cells],decay_per_s=[.08],
                boundaries=['reflecting','reflecting'],
                provenance='Synthetic one-group homogeneous reflected cube; self-scatter=absorption, D=1/(3 total); no evaluated nuclear data')


def admit_neutron(k_atomic,k_mc,sigma,analytic):
    if any(type(v) not in (int,float) or not math.isfinite(v) for v in (k_atomic,k_mc,sigma,analytic)) or sigma<0 or analytic<=0:
        raise ValueError('Invalid neutron estimate')
    return dict(atomic_analytic=abs(k_atomic-analytic)<=1e-8,
                monte_carlo_analytic=abs(k_mc-analytic)<=max(5*sigma,5e-4),
                precision=sigma<=.003)


def openmc_case(output,ratio,seed):
    import openmc
    output.mkdir(exist_ok=False)
    openmc.reset_auto_ids()
    xs=openmc.XSdata('homogeneous',openmc.mgxs.EnergyGroups([0.,20e6]))
    xs.order=0
    a=.09/100 # 1/m to1/cm, both input and transversely uniform geometry converted.
    xs.set_total(np.array([2*a]));xs.set_absorption(np.array([a]))
    xs.set_scatter_matrix(np.array([[[a]]]))
    xs.set_fission(np.array([a*ratio/2.5]));xs.set_nu_fission(np.array([a*ratio]));xs.set_chi(np.array([1.]))
    library=openmc.MGXSLibrary(xs.energy_groups);library.add_xsdata(xs)
    library.export_to_hdf5(output/'mgxs.h5')
    material=openmc.Material();material.set_density('macro',1.);material.add_macroscopic('homogeneous')
    materials=openmc.Materials([material]);materials.cross_sections=str(output/'mgxs.h5')
    planes=[openmc.XPlane(x0=-2500,boundary_type='reflective'),openmc.XPlane(x0=2500,boundary_type='reflective'),
            openmc.YPlane(y0=-2500,boundary_type='reflective'),openmc.YPlane(y0=2500,boundary_type='reflective'),
            openmc.ZPlane(z0=-2500,boundary_type='reflective'),openmc.ZPlane(z0=2500,boundary_type='reflective')]
    cell=openmc.Cell(fill=material,region=+planes[0]&-planes[1]&+planes[2]&-planes[3]&+planes[4]&-planes[5])
    settings=openmc.Settings();settings.energy_mode='multi-group';settings.batches=50;settings.inactive=10
    settings.particles=5000;settings.seed=seed;settings.max_order=0
    settings.source=openmc.IndependentSource(space=openmc.stats.Box((-2499,)*3,(2499,)*3))
    settings.output={'summary':True,'tallies':False};settings.statepoint={'batches':[50]}
    model=openmc.Model(openmc.Geometry([cell]),materials,settings)
    model.export_to_xml(directory=output)
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    start=time.perf_counter()
    with (output/'openmc.log').open('x') as log:
        process=subprocess.run([shutil.which('openmc'),'-s','1'],cwd=output,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=120)
    if process.returncode: raise ValueError(f'OpenMC exited{process.returncode}; see {output}/openmc.log')
    wall=time.perf_counter()-start
    with openmc.StatePoint(output/'statepoint.50.h5') as statepoint:
        selected=[r for r in statepoint.global_tallies if r['name']==b'k-collision']
        if len(selected)!=1: raise ValueError('Missing fixed collision estimator')
        k,sigma=float(selected[0]['mean']),float(selected[0]['std_dev'])
    return dict(k=k,standard_error=sigma,estimator='collision',wall_s=wall,seed=seed,
                files={p.name:sha(p) for p in output.iterdir() if p.is_file()})


def openmc_profile(output):
    import openmc
    executable=shutil.which('openmc')
    if executable is None: raise FileNotFoundError('OpenMC executable absent')
    rows=[]
    for ratio in RATIOS:
        config=neutron_config(ratio)
        atomic_json(output/f'neutron-input-{ratio:g}.json',seal(config))
        measurements=timing(lambda _:eigenvalue(Model(config)),[None],repeats=7,loops=20)
        result=eigenvalue(Model(config));ka=float(result['k'])
        for seed in SEEDS:
            run=openmc_case(output/f'openmc-{ratio:g}-{seed}',ratio,seed)
            checks=admit_neutron(ka,run['k'],run['standard_error'],ratio)
            rows.append(dict(ratio=ratio,atomic_k=ka,atomic_relative_residual=float(result['relative_residual']),
                             openmc=run,checks=checks,passed=all(checks.values())))
            print('OpenMC',ratio,seed,rows[-1]['passed'],flush=True)
        atomic_json(output/f'atomic-neutron-timing-{ratio:g}.json',seal(measurements))
    atomic_json(output/'neutron-comparisons.json',seal(rows))
    return dict(status='completed',passed=all(r['passed'] for r in rows),runs=rows,
                version=openmc.__version__,executable_sha256=sha(executable),
                scope='Synthetic one-group homogeneous reflected cube, exact k=nuSigmaF/SigmaA; diffusion and transport agree here by no leakage; not evaluated-data or plant validation',
                timing_comparability='No speed ranking: deterministic eigenvalue versus finite-uncertainty Monte Carlo; setup retained separately')


def source_manifest():
    from atomic_model_spatial.workflow import identity
    return dict(qualified=signature(),spatial=identity(),driver_sha256=sha(__file__),
                cycle_driver_sha256=sha(Path(__file__).with_name('verify_phase_cycles.py')))


def run(output, *, providers=PROVIDERS):
    if not providers or len(set(providers))!=len(providers) or any(p not in PROVIDERS for p in providers):
        raise ValueError('Choose distinct supported providers')
    from threadpoolctl import threadpool_info
    pools=threadpool_info()
    if any(pool['num_threads']!=1 for pool in pools):
        raise ValueError('Benchmark requires single-thread BLAS: set OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1 before Python starts')
    output=Path(output).absolute()
    if any(p.is_symlink() for p in (output,*output.parents)): raise ValueError('Symlink output')
    output.mkdir(parents=True,exist_ok=False)
    source=source_manifest();cases=points()
    protocol=dict(schema_version=1,profile='external-component-benchmark-v1',source=source,providers=list(providers),
                  eos_points=cases,eos_relative_budgets=EOS_BUDGETS,neutron_ratios=RATIOS,seeds=SEEDS,
                  neutron=dict(particles=5000,batches=50,inactive=10,estimator='collision',absolute_k_floor=5e-4,sigma_multiplier=5,maximum_sigma=.003),
                  timing=dict(threads=1,warm_cache=True,repeats=7,loops=20),
                  runtime=dict(python=sys.version,platform=platform.platform(),machine=platform.machine(),
                               numpy=np.__version__,threadpools=pools,
                               cpu_model=next((l.split(':',1)[1].strip() for l in (Path('/proc/cpuinfo').read_text().splitlines() if Path('/proc/cpuinfo').is_file() else []) if l.startswith('model name')),platform.processor() or 'unknown')),
                  sources=SOURCE_URLS,validated=False,plant_referenced=False)
    atomic_json(output/'protocol.json',seal(protocol))
    peers=export_modelica(output,cases)
    profiles={}
    for provider in providers:
        try:
            print(provider,'started',flush=True)
            profiles[provider]=coolprop_profile(output,cases) if provider=='coolprop' else openmc_profile(output)
        except Exception as exc:
            profiles[provider]=dict(status='unavailable' if isinstance(exc,(ImportError,FileNotFoundError)) else 'failed',passed=False,
                                    error_type=type(exc).__name__,error=str(exc))
        atomic_json(output/f'{provider}-report.json',seal(profiles[provider]))
        print(provider,profiles[provider]['status'],profiles[provider]['passed'],flush=True)
    unchanged=source_manifest()==source
    report=dict(schema_version=1,completed=True,passed=unchanged and all(p['passed'] for p in profiles.values()),
                source_unchanged=unchanged,profiles=profiles,openmodelica=peers,
                all_default_providers_run=set(providers)==set(PROVIDERS),
                whole_plant_compared=False,superiority_established=False,
                scope='Component/code-to-code verification only; independent coefficients implementation is not experimental evidence',
                remaining=['Transient HEM network peer','Coupled whole-plant peer','Experimental accuracy','Cross-platform and memory profiling'],
                validated=False,plant_referenced=False)
    atomic_json(output/'report.json',seal(report))
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--providers',nargs='+',choices=PROVIDERS,default=list(PROVIDERS))
    args=parser.parse_args(argv)
    report=run(args.output,providers=args.providers)
    print(json.dumps(dict(passed=report['passed'],whole_plant_compared=False,superiority_established=False)))
    return 0 if report['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
