"""Strong wall momentum, traction and refinement for selected linear benchmarks."""
import argparse,platform
from pathlib import Path
import numpy as np
import scipy
from scipy.special import jv,jn_zeros
from atomic_model_station_regular import regular_mode
from atomic_model_station_stokes_pressure import PressureStokesSession
from atomic_model_station_wall import audit_momentum,WallCheckedSession,WallMomentumError
from atomic_model_multiphysics.provenance import sha,seal
from atomic_model_training.spec import atomic_json


def identity():
    import atomic_model_station_wall as package
    from scripts.qualify_station_stokes_pressure import identity as base_identity
    root=Path(package.__file__).parent
    return dict(files={p.name:sha(p) for p in sorted(root.glob('*.py'))},driver=sha(__file__),base=base_identity(),
        python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__)


def force_field(r,m,nu):
    f=np.zeros((3,len(r)),complex)
    f[0]=(m+2)*r**(m+1);f[1]=1j*m*r**(m+1);f[2]=4*nu*(m+1)*r**m
    return f


def run(output,reference=None):
    out=Path(output).absolute();source=identity()
    if out.exists() or any(p.is_symlink() for p in (out,*out.parents)):raise ValueError('Fresh ordinary output required')
    out.mkdir(parents=True);criteria=dict(normalized_momentum=1e-8,traction_error=1e-8,finest_swirl_residual=1e-8,minimum_refinement_ratio=100.)
    atomic_json(out/'protocol.json',seal(dict(source=source,criteria=criteria,
        scope='Selected strong linear Stokes wall benchmarks, explicit physical force and dimensionless density1 traction',
        stationary_solution='uz=r^m*(1-r^2),p=r^(m+2),k0; subtract mean1/2 for m0',
        dynamic_reference='Axisymmetric J1 viscous swirl with no forcing; boundary Laplacian vanishes in continuum',
        general_wall_acceptance=False,native_backend_changed=False)))
    radius=np.linspace(0,1,101);cases={};passed=True
    for m in (0,1,2,4):
        for order in (4,6,8):
            model=regular_mode(0,m,.628319,order=order);u=np.zeros((3,len(model.radius)),complex)
            u[2]=model.radius**m*(1-model.radius**2);q=model.fit(u);session=WallCheckedSession(model,q,radius=radius)
            step,p=session.advance(.005,lambda r,time:force_field(r,m,.1))
            result=audit_momentum(model,q,session.q,p,dt=.005,viscosity=.1,radius=radius,force=force_field(radius,m,.1))
            target=np.array([-.5 if m==0 else -1.,0.,-.2]);stress=np.array(result['wall_traction']['real'])+1j*np.array(result['wall_traction']['imag'])
            error=float(np.max(abs(stress-target)))
            checks=dict(momentum=result['normalized_momentum_residual']<criteria['normalized_momentum'],
                wall=result['normalized_wall_residual']<criteria['normalized_momentum'],traction=error<criteria['traction_error'])
            passed=bool(passed and all(checks.values()))
            cases[f'm{m}-order{order}']=dict(spec=model.spec,checks=checks,strong_audit=result,traction_error=error,pressure=step['pressure'])
    atomic_json(out/'cases.json',seal(cases))
    root=float(jn_zeros(1,1)[0]);swirl=[]
    for order in (3,5,7,9):
        model=regular_mode(0,0,.628319,order=order);u=np.zeros((3,len(model.radius)),complex);u[1]=jv(1,root*model.radius)
        session=PressureStokesSession(model,model.fit(u))
        for i in range(20):
            before=session.q;receipt,p=session.advance(.005)
        a=audit_momentum(model,before,session.q,p,dt=.005,viscosity=.1,radius=radius,force=np.zeros((3,len(radius))))
        swirl.append(dict(order=order,audit=a,wall_passed=bool(a['normalized_wall_residual']<criteria['finest_swirl_residual'])))
    ratios=[a['audit']['wall_momentum_residual']/b['audit']['wall_momentum_residual'] for a,b in zip(swirl,swirl[1:])]
    model=regular_mode(2,1,.628319,order=4);q=np.arange(model.dimension)*.001;session=PressureStokesSession(model,q)
    receipt,p=session.advance(.005)
    negative=audit_momentum(model,q,session.q,p,dt=.005,viscosity=.1,radius=radius,force=np.zeros((3,len(radius))))
    counterexample=dict(weak_momentum_residual=receipt['pressure']['weak_momentum_relative_residual'],strong_audit=negative,
        strong_wall_rejected=bool(negative['normalized_wall_residual']>criteria['normalized_momentum']))
    guarded=WallCheckedSession(model,q,radius=radius);initial=guarded.checkpoint();coarse_refused=False
    try:guarded.advance(.005)
    except WallMomentumError:coarse_refused=guarded.checkpoint()==initial
    fine=regular_mode(0,0,.628319,order=9);u=np.zeros((3,len(fine.radius)),complex);u[1]=jv(1,root*fine.radius)
    guarded=WallCheckedSession(fine,fine.fit(u),radius=radius);replay=None;guard_restart_exact=True
    for i in range(20):
        row,field=guarded.advance(.005)
        if replay is not None:
            other,op=replay.advance(.005)
            guard_restart_exact=bool(guard_restart_exact and row==other and np.array_equal(field.coefficients,op.coefficients))
        if i==9:replay=WallCheckedSession.restore(guarded.checkpoint())
    passed=bool(passed and swirl[-1]['wall_passed'] and min(ratios)>criteria['minimum_refinement_ratio'] and counterexample['strong_wall_rejected'] and counterexample['weak_momentum_residual']<1e-9 and coarse_refused and guard_restart_exact)
    refinement=dict(swirl=swirl,successive_wall_residual_ratios=ratios,weak_strong_counterexample=counterexample,coarse_candidate_refused_atomically=coarse_refused,finest_guarded_restart_exact=guard_restart_exact)
    atomic_json(out/'refinement.json',seal(refinement))
    if source!=identity():raise ValueError('Strong-wall audit source changed')
    report=dict(passed=passed,source=source,stationary_cases=len(cases),swirl_refinements=len(swirl),selected_linear_wall_benchmarks_qualified=passed,
        maximum_stationary_momentum_residual=max(v['strong_audit']['maximum_momentum_residual'] for v in cases.values()),
        maximum_stationary_wall_residual=max(v['strong_audit']['wall_momentum_residual'] for v in cases.values()),
        maximum_traction_error=max(v['traction_error'] for v in cases.values()),finest_swirl_wall_residual=swirl[-1]['audit']['wall_momentum_residual'],
        minimum_refinement_ratio=float(min(ratios)),weak_strong_counterexample_preserved=True,
        coarse_candidate_refused_atomically=coarse_refused,finest_guarded_restart_exact=guard_restart_exact,
        general_wall_momentum_qualified=False,nonlinear_advection_supported=False,thermal_model_supported=False,
        native_openpipeflow_backend_changed=False,station_usable=False,validated=False,closed_tasks=[])
    atomic_json(out/'report.json',seal(report))
    if not passed:raise ValueError('Selected wall benchmark qualification failed; evidence retained')
    if reference and {p.name:sha(p) for p in Path(reference).glob('*.json')}!={p.name:sha(p) for p in out.glob('*.json')}:raise ValueError('Installed wall scientific JSON differ')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--reference',type=Path)
    a=p.parse_args();print(run(a.output,a.reference)['passed'])

if __name__=='__main__':main()
