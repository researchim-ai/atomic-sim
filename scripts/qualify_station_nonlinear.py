"""Retained-mode nonlinear midpoint dynamics, energy, pressure and restart."""
import argparse,platform
from pathlib import Path
import numpy as np
import scipy
from scipy.integrate import solve_ivp
from scipy.linalg import solve
from atomic_model_station_nonlinear import NonlinearSession,NonlinearConvergenceError
from atomic_model_station_regular import regular_mode
from atomic_model_station_advection import signed_spectrum,convolution_advection
from atomic_model_multiphysics.provenance import sha,seal
from atomic_model_training.spec import atomic_json
from scripts.qualify_station_advection import modes


def identity():
    import atomic_model_station_nonlinear as package
    from scripts.qualify_station_advection import identity as base_identity
    root=Path(package.__file__).parent
    return dict(files={p.name:sha(p) for p in sorted(root.glob('*.py'))},driver=sha(__file__),base=base_identity(),python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__)


def reference_solution(state,end,rtol,atol):
    state=sorted(state,key=lambda pair:(pair[0].k,pair[0].m));models=[m for m,q in state]
    cuts=np.cumsum([0]+[m.dimension for m in models]);initial=np.concatenate([q for m,q in state]);r=models[0].radius
    keys=[(m.k,m.m) for m in models];targets=sorted({key for k,m in keys for key in [(k,m),(-k,-m)]})
    def rhs(time,value):
        q=[value[cuts[i]:cuts[i+1]] for i in range(len(models))]
        velocity,radial=signed_spectrum(list(zip(models,q)),r,.628319)
        nonlinear=convolution_advection(r,.628319,velocity,radial,targets)
        return np.concatenate([solve(model.mass,-model.transform.conj().T@model.load(nonlinear[(model.k,model.m)])-.05*model.stiffness@v,assume_a='pos') for model,v in zip(models,q)])
    result=solve_ivp(rhs,(0,end),initial,method='DOP853',rtol=rtol,atol=atol)
    if not result.success or abs(result.t[-1]-end)>1e-14:raise ValueError('Independent adaptive reference failed')
    return result.y[:,-1],result.nfev


def run(output,reference=None):
    out=Path(output).absolute();source=identity()
    if out.exists() or any(p.is_symlink() for p in (out,*out.parents)):raise ValueError('Fresh ordinary output required')
    out.mkdir(parents=True);criteria=dict(nonlinear_residual=1e-12,energy=1e-9,pressure=1e-9,advection_power=1e-10,method_agreement=1e-10,
        minimum_time_order=1.99,reference_agreement=1e-10,swirl_pressure=1e-10)
    atomic_json(out/'protocol.json',seal(dict(source=source,criteria=criteria,
        model='Fixed retained-mode regular incompressible Galerkin dynamics; implicit midpoint advection and viscosity',
        pressure='Retained weak momentum including midpoint nonlinear load',
        reference='Independent adaptive DOP853 with direct convolution, same semidiscrete ODE',
        exclusions=['Strong-wall acceptance','Spatial mode convergence','Thermal and plant dynamics'])))
    passed=True;cases={};finals={}
    specifications=[('inviscid',1,0.,'fft',False),('viscous-fft',1,.05,'fft',False),('viscous-convolution',1,.05,'convolution',False),('wider-band',2,.05,'fft',False),('external-work',1,.05,'fft',True)]
    for name,extent,nu,method,forced in specifications:
        state=[(model,q*10) for model,q in modes(extent,3,32)]
        if forced:state=[(model,np.zeros_like(q)) for model,q in state]
        session=NonlinearSession(state,viscosity=nu,method=method);force=None
        if forced:
            force={}
            for model,q in state:
                f=np.zeros((3,len(model.radius)),complex)
                if (model.k,model.m) in [(0,0),(1,1)]:f[2]=.1*model.radius**model.m*(1-model.radius**2)
                force[(model.k,model.m)]=model.load(f)
        maxima=dict(nonlinear=0.,energy=0.,pressure=0.,advection=0.);restart=None;exact=True;monotone=True;last=session.initial_energy
        for i in range(10):
            result,fields=session.advance(.01,force)
            maxima['nonlinear']=max(maxima['nonlinear'],result['nonlinear_relative_residual'])
            maxima['energy']=max(maxima['energy'],abs(result['total_energy_residual']))
            maxima['pressure']=max(maxima['pressure'],max(v['proof']['weak_momentum_relative_residual'] for v in result['pressures']))
            maxima['advection']=max(maxima['advection'],abs(result['advection_power']))
            if not forced:monotone=bool(monotone and result['kinetic_energy']<=last+1e-12)
            last=result['kinetic_energy']
            if restart is not None:
                other,op=restart.advance(.01,force)
                exact=bool(exact and result==other and all(np.array_equal(fields[key].coefficients,op[key].coefficients) for key in fields))
            if i==4:restart=NonlinearSession.restore(session.checkpoint())
        checks=dict(nonlinear=maxima['nonlinear']<=criteria['nonlinear_residual'],energy=maxima['energy']<criteria['energy'],pressure=maxima['pressure']<criteria['pressure'],
            advection=maxima['advection']<criteria['advection_power'],restart=exact,decay=monotone,external_work=not forced or result['external_work']>0)
        passed=bool(passed and all(checks.values()));finals[name]=np.concatenate([q for model,q in session.modes])
        cases[name]=dict(checks=checks,maxima=maxima,final_step=result,checkpoint=session.checkpoint(),extent=extent,viscosity=nu,method=method)
    atomic_json(out/'cases.json',seal(cases))
    method_error=float(np.linalg.norm(finals['viscous-fft']-finals['viscous-convolution']))
    state=[(model,q*20) for model,q in modes(1,3,32)]
    coarse,nfev=reference_solution(state,.1,1e-11,1e-13);fine,fine_nfev=reference_solution(state,.1,2e-13,2e-15)
    reference_error=float(np.linalg.norm(coarse-fine));temporal=[]
    for dt in (.02,.01,.005):
        session=NonlinearSession(state,viscosity=.05)
        for i in range(round(.1/dt)):session.advance(dt)
        error=float(np.linalg.norm(np.concatenate([q for model,q in session.modes])-fine));temporal.append(dict(dt=dt,error=error))
    orders=np.log2(np.array([v['error'] for v in temporal[:-1]])/[v['error'] for v in temporal[1:]])
    model=regular_mode(0,0,.628319,order=6);r=model.radius;u=np.zeros((3,len(r)),complex);u[1]=r*(1-r*r);q=model.fit(u)
    session=NonlinearSession([(model,q)],viscosity=0);result,fields=session.advance(.01);p,g=fields[(0,0)].evaluate(r)
    swirl_error=float(np.max(abs(p-(r*r/2-r**4/2+r**6/6-1/8))))
    stationary_exact=np.array_equal(session.modes[0][1],q)
    refused=NonlinearSession(state,max_iterations=1);before=refused.checkpoint();atomic_failure=False;failure_residual=None
    try:refused.advance(.01)
    except NonlinearConvergenceError as error:
        atomic_failure=refused.checkpoint()==before;failure_residual=error.residual
    refinement=dict(temporal=temporal,orders=orders.tolist(),adaptive_reference_agreement=reference_error,reference_nfev=[nfev,fine_nfev],
        fft_convolution_trajectory_error=method_error,stationary_swirl_pressure_error=swirl_error,stationary_swirl_velocity_exact=bool(stationary_exact),
        iteration_exhaustion_atomic=atomic_failure,iteration_exhaustion_residual=failure_residual)
    atomic_json(out/'refinement.json',seal(refinement))
    passed=bool(passed and method_error<criteria['method_agreement'] and min(orders)>criteria['minimum_time_order'] and reference_error<criteria['reference_agreement'] and swirl_error<criteria['swirl_pressure'] and stationary_exact and atomic_failure)
    if source!=identity():raise ValueError('Nonlinear evolution source changed')
    report=dict(passed=passed,source=source,trajectory_cases=len(cases),retained_nonlinear_midpoint_qualified=passed,
        maximum_nonlinear_residual=max(v['maxima']['nonlinear'] for v in cases.values()),maximum_energy_residual=max(v['maxima']['energy'] for v in cases.values()),
        maximum_pressure_residual=max(v['maxima']['pressure'] for v in cases.values()),maximum_advection_power=max(v['maxima']['advection'] for v in cases.values()),
        minimum_time_order=float(min(orders)),adaptive_reference_agreement=reference_error,fft_convolution_trajectory_error=method_error,
        all_native_restarts_exact=all(v['checks']['restart'] for v in cases.values()),stationary_swirl_pressure_error=swirl_error,iteration_exhaustion_atomic=atomic_failure,
        spatial_mode_convergence_qualified=False,strong_wall_momentum_qualified=False,thermal_model_supported=False,native_backend_changed=False,station_usable=False,validated=False,closed_tasks=[])
    atomic_json(out/'report.json',seal(report))
    if not passed:raise ValueError('Retained nonlinear midpoint qualification failed; evidence preserved')
    if reference and {p.name:sha(p) for p in Path(reference).glob('*.json')}!={p.name:sha(p) for p in out.glob('*.json')}:raise ValueError('Installed nonlinear evolution scientific JSON differ')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--reference',type=Path)
    a=p.parse_args();print(run(a.output,a.reference)['passed'])

if __name__=='__main__':main()
