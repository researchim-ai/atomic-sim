"""Selected unforced nonlinear spatial, temporal and wall qualification."""
import argparse,platform
from pathlib import Path
import numpy as np
import scipy
from scipy.special import jv,jn_zeros
from scipy.integrate import quad
from numpy.polynomial.legendre import leggauss
from atomic_model_station_refined import refined_mode,evolve_refined,endpoint_momentum,compare_refined
from atomic_model_multiphysics.provenance import sha,seal
from atomic_model_training.spec import atomic_json
from scripts.qualify_station_advection import modes


def identity():
    import atomic_model_station_refined as package
    from scripts.qualify_station_adaptive import identity as base_identity
    return dict(files={p.name:sha(p) for p in sorted(Path(package.__file__).parent.glob('*.py'))},driver=sha(__file__),base=base_identity(),python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__)


def initial_state(extent,order,outer=16):
    seed={(m.k,m.m):(m,q*10) for m,q in modes(1,3,64)};state=[]
    for m in range(extent+1):
        for k in (range(extent+1) if m==0 else range(-extent,extent+1)):
            model=refined_mode(k,m,.628319,order=outer if max(abs(k),m)==5 else order)
            q=model.fit(seed[(k,m)][0].evaluate(seed[(k,m)][1],model.radius)) if (k,m) in seed else np.zeros(model.dimension,complex)
            state.append((model,q))
    return state


def run(output,reference=None):
    out=Path(output).absolute();source=identity()
    if out.exists() or any(p.is_symlink() for p in (out,*out.parents)):raise ValueError('Fresh ordinary output required')
    out.mkdir(parents=True)
    criteria=dict(velocity_l2=1e-7,strong_wall=1e-8,energy=1e-9,weak_pressure=1e-9,initial_l2=1e-10,analytic_vortex=1e-9)
    atomic_json(out/'protocol.json',seal(dict(source=source,criteria=criteria,time=.1,viscosity=.05,quadrature=96,
        seed='v77 physical polynomial seed, band1 order3, amplitude multiplied by10',
        solver='Real-split BDF, exact viscous sparse quasi-Newton block; nonlinear Jacobian omitted',
        norm='sqrt(sum over signed modes integral |delta u|^2 r dr); absent modes included',
        outer_basis='Band5 uses order16; separate order12 comparison. Conditioning guard unchanged.',
        exclusions=['Thermal physics','Experimental validation','General strong PDE convergence','Bit-exact BDF restart','Plant acceptance'])))
    x,w=leggauss(128);radius=np.sqrt((x+1)/2);weights=w/4
    seed=[(m,q*10) for m,q in modes(1,3,64)];results={};cases={}
    specs=[('k2p12',2,12,16),('k2p16',2,16,16),('k2p20',2,20,16),('k2p24',2,24,16),('k3p24',3,24,16),('k4p24',4,24,16),('k5p24',5,24,16),('k5p20',5,20,16),('k5outer12',5,24,12),('k5tight',5,24,16)]
    for name,k,p,o in specs:
        state=initial_state(k,p,o)
        initial=compare_refined(seed,state,radius,weights,first_time=0,second_time=0)['velocity_l2']
        result=evolve_refined(state,rtol=2e-12 if name=='k5tight' else 1e-10,atol=2e-15 if name=='k5tight' else 1e-13)
        strong=endpoint_momentum(result,radius=np.linspace(.005,1,101));results[name]=result
        cases[name]=dict(initial_l2=initial,solver=result.report,momentum=strong)
        print(name,'wall',strong['maximum_wall_residual'],flush=True)
    comparisons={}
    for label,a,b in [('radial','k5p20','k5p24'),('outer_radial','k5outer12','k5p24'),('fourier_previous','k3p24','k4p24'),('fourier','k4p24','k5p24'),('temporal','k5p24','k5tight')]:
        comparisons[label]=compare_refined(results[a].modes,results[b].modes,radius,weights,first_time=.1,second_time=.1)
    model=refined_mode(0,0,.628319,order=12);root=jn_zeros(1,1)[0]
    velocity=np.zeros((3,len(model.radius)),complex);velocity[1]=jv(1,root*model.radius)
    vortex=evolve_refined([(model,model.fit(velocity))]);points=np.linspace(0,1,101);amplitude=np.exp(-.05*root**2*.1)
    def g(r):return jv(1,root*r)**2/r if r else 0.
    gauge=quad(lambda r:g(r)*(1-r*r),0,1,epsabs=1e-13)[0]
    exact=amplitude**2*np.array([quad(g,0,r,epsabs=1e-13)[0]-gauge for r in points])
    pressure,_=vortex.pressure[(0,0)].evaluate(points)
    analytic=dict(velocity_error=float(np.max(abs(model.evaluate(vortex.modes[0][1],points)[1]-amplitude*jv(1,root*points)))),pressure_error=float(np.max(abs(pressure-exact))),wall=endpoint_momentum(vortex,radius=points[1:])['maximum_wall_residual'])
    checks={name:comparisons[name]['velocity_l2']<criteria['velocity_l2'] for name in ('radial','outer_radial','fourier','temporal')}
    checks.update(wall=all(cases[n]['momentum']['maximum_wall_residual']<criteria['strong_wall'] for n in ('k5p24','k5tight')),
        initial=all(c['initial_l2']<criteria['initial_l2'] for c in cases.values()),
        energy=all(c['solver']['maximum_energy_budget_residual']<criteria['energy'] for c in cases.values()),
        pressure=all(c['solver']['maximum_weak_pressure_residual']<criteria['weak_pressure'] for c in cases.values()),
        analytic=analytic['velocity_error']<criteria['analytic_vortex'] and analytic['pressure_error']<criteria['analytic_vortex'] and analytic['wall']<criteria['strong_wall'],
        coarse_wall_refused=all(cases[n]['momentum']['maximum_wall_residual']>criteria['strong_wall'] for n in ('k2p12','k2p16','k2p20')),
        previous_fourier_refused=comparisons['fourier_previous']['velocity_l2']>criteria['velocity_l2'])
    atomic_json(out/'cases.json',seal(dict(cases=cases,analytic_vortex=analytic)))
    atomic_json(out/'comparison.json',seal(dict(comparisons=comparisons,checks=checks)))
    if source!=identity():raise ValueError('Refined qualification source changed')
    report=dict(passed=bool(all(checks.values())),source=source,checks=checks,selected_spatial_wall_benchmarks_qualified=bool(all(checks.values())),
        wall_residual=cases['k5p24']['momentum']['maximum_wall_residual'],radial_velocity_l2=comparisons['radial']['velocity_l2'],fourier_velocity_l2=comparisons['fourier']['velocity_l2'],temporal_velocity_l2=comparisons['temporal']['velocity_l2'],
        general_strong_pde_qualified=False,thermal_model_supported=False,experimental_validation=False,native_backend_changed=False,station_usable=False,validated=False,closed_tasks=[])
    atomic_json(out/'report.json',seal(report))
    if not report['passed']:raise ValueError('Refined qualification failed; evidence preserved')
    if reference and {p.name:sha(p) for p in Path(reference).glob('*.json')}!={p.name:sha(p) for p in out.glob('*.json')}:raise ValueError('Installed refined scientific JSON differ')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--reference',type=Path)
    args=parser.parse_args();print(run(args.output,args.reference)['passed'])

if __name__=='__main__':main()
