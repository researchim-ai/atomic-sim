"""Qualify regular constrained single-mode Stokes dynamics, not turbulent DNS."""
import argparse,platform
from pathlib import Path
import numpy as np
import scipy
from scipy.linalg import expm,solve
from scipy.special import jv,jn_zeros
from atomic_model_station_regular import regular_mode,StokesSession
from atomic_model_multiphysics.provenance import sha,seal
from atomic_model_training.spec import atomic_json


def identity():
    import atomic_model_station_regular as package
    root=Path(package.__file__).parent
    return dict(files={p.name:sha(p) for p in sorted(root.glob('*.py'))},driver=sha(__file__),
        python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__)


def run(output,reference=None):
    out=Path(output).absolute();source=identity()
    if out.exists() or any(p.is_symlink() for p in (out,*out.parents)):raise ValueError('Fresh ordinary output required')
    out.mkdir(parents=True)
    criteria=dict(wall=1e-10,analytic_divergence=1e-8,energy=1e-9,reaction=1e-8,reaction_work=1e-10,
        minimum_time_order=1.99,maximum_bessel_error=1e-10)
    atomic_json(out/'protocol.json',seal(dict(source=source,criteria=criteria,
        model='Single complex Fourier mode, linear constant-viscosity Stokes, no-slip regular polynomial Galerkin space',
        pressure='Dual constraint reaction; physical collocated pressure not reconstructed',
        boundaries='No-slip at r=1 built into basis; Cartesian spin powers enforced analytically at r=0',
        native_backend_changed=False)))
    rows={};passed=True
    for m in (0,1,2,4):
        k=0 if m==0 else 2
        for order in (4,6,8):
            model=regular_mode(k,m,.628319,order=order);q=np.arange(model.dimension)*.0001
            force=np.arange(3*order)*.00001*(1+.2j);session=StokesSession(model,q)
            maxima={name:0. for name in ('wall_maximum','analytic_divergence_maximum','total_energy_residual','constraint_reaction_relative_residual','constraint_reaction_work')}
            restart=None;samples=[];exact_restart=True
            for i in range(80):
                result=session.advance(.0025,force)
                if restart is not None:
                    other=restart.advance(.0025,force)
                    exact_restart=bool(exact_restart and other==result and np.array_equal(session.q,restart.q))
                for name in maxima:maxima[name]=max(maxima[name],abs(result[name]))
                if i==39:restart=StokesSession.restore(session.checkpoint())
                if i in (0,39,79):samples.append(result)
            checks=dict(wall=maxima['wall_maximum']<criteria['wall'],analytic_divergence=maxima['analytic_divergence_maximum']<criteria['analytic_divergence'],
                energy=maxima['total_energy_residual']<criteria['energy'],reaction=maxima['constraint_reaction_relative_residual']<criteria['reaction'],
                reaction_work=maxima['constraint_reaction_work']<criteria['reaction_work'],restart=exact_restart)
            passed=passed and all(checks.values())
            rows[f'm{m}-order{order}']=dict(spec=model.spec,dimension=model.dimension,fingerprint=model.fingerprint,
                maxima=maxima,checks=checks,samples=samples,final_checkpoint=session.checkpoint())
    atomic_json(out/'dynamics.json',seal(rows))
    model=regular_mode(2,1,.628319,order=5);q=np.arange(model.dimension)*.001
    exact=expm(-.02*solve(model.mass,model.stiffness,assume_a='pos'))@q;temporal=[]
    for dt in (.01,.005,.0025):
        session=StokesSession(model,q)
        for i in range(round(.2/dt)):session.advance(dt)
        temporal.append(dict(dt=dt,error=float(np.linalg.norm(session.q-exact))))
    orders=np.log2(np.array([v['error'] for v in temporal[:-1]])/[v['error'] for v in temporal[1:]])
    root=float(jn_zeros(1,1)[0]);spatial=[]
    for order in (3,5,7,9):
        model=regular_mode(0,0,.628319,order=order);field=np.zeros((3,len(model.radius)),complex);field[1]=jv(1,root*model.radius)
        q=model.fit(field);end=expm(-.02*solve(model.mass,model.stiffness,assume_a='pos'))@q
        error=float(np.sqrt(np.sum(model.weights*abs(model.evaluate(end,model.radius)-field*np.exp(-.02*root**2))**2)))
        spatial.append(dict(order=order,error=error))
    refinement=dict(temporal=temporal,temporal_orders=orders.tolist(),minimum_time_order=float(min(orders)),bessel_root=root,spatial=spatial,
        spatial_reference='u_theta=J1(j_1,1*r)*exp(-nu*j_1,1^2*t); independent cylindrical diffusion solution',
        time_reference='Independent matrix exponential of the same semidiscrete system')
    atomic_json(out/'refinement.json',seal(refinement))
    passed=bool(passed and min(orders)>criteria['minimum_time_order'] and spatial[-1]['error']<criteria['maximum_bessel_error'] and all(b['error']<a['error']/100 for a,b in zip(spatial,spatial[1:])))
    if source!=identity():raise ValueError('Regular Stokes source changed')
    report=dict(passed=passed,source=source,cases=len(rows),linear_single_mode_stokes_qualified=passed,
        maximum_wall=max(v['maxima']['wall_maximum'] for v in rows.values()),maximum_analytic_divergence=max(v['maxima']['analytic_divergence_maximum'] for v in rows.values()),
        maximum_energy_residual=max(v['maxima']['total_energy_residual'] for v in rows.values()),
        maximum_reaction_residual=max(v['maxima']['constraint_reaction_relative_residual'] for v in rows.values()),
        maximum_reaction_work=max(v['maxima']['constraint_reaction_work'] for v in rows.values()),
        all_native_restarts_exact=all(v['checks']['restart'] for v in rows.values()),
        minimum_time_order=float(min(orders)),finest_bessel_error=spatial[-1]['error'],
        physical_pressure_reconstructed=False,nonlinear_advection_supported=False,thermal_model_supported=False,
        native_openpipeflow_backend_changed=False,station_usable=False,validated=False,dynamic_station_continuity_qualified=False,closed_tasks=[])
    atomic_json(out/'report.json',seal(report))
    if not passed:raise ValueError('Regular Stokes qualification failed; evidence retained')
    if reference:
        expected={p.name:sha(p) for p in Path(reference).glob('*.json')}
        if expected!={p.name:sha(p) for p in out.glob('*.json')}:raise ValueError('Regular Stokes installed scientific JSON differ')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--reference',type=Path)
    a=p.parse_args();print(run(a.output,a.reference)['passed'])

if __name__=='__main__':main()
