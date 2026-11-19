"""Physical weak-pressure recovery, gauges and coupled Stokes acceptance."""
import argparse,platform
from pathlib import Path
import numpy as np
import scipy
from scipy.linalg import solve
from atomic_model_station_regular import regular_mode,StokesSession
from atomic_model_station_stokes_pressure import PressureStokesSession,pressure_basis,recover_pressure
from atomic_model_multiphysics.provenance import sha,seal
from atomic_model_training.spec import atomic_json


def identity():
    import atomic_model_station_stokes_pressure as package
    from scripts.qualify_station_regular import identity as base_identity
    root=Path(package.__file__).parent
    return dict(files={p.name:sha(p) for p in sorted(root.glob('*.py'))},driver=sha(__file__),base=base_identity(),
        python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__)


def manufactured(model):
    r=model.radius;s=r*r;f=1-1.8*s+s*s;df=-1.8+2*s
    pressure=r**model.m*f;gradient=np.zeros((3,len(r)),complex)
    gradient[0]=(model.m*r**(model.m-1)*f if model.m else 0)+2*r**(model.m+1)*df
    if model.m:gradient[1]=1j*model.m*r**(model.m-1)*f
    gradient[2]=1j*model.alpha*model.k*pressure
    if model.k==model.m==0:pressure-=13/30
    return pressure,gradient


def monolithic(model,before,force,dt,nu):
    _,gradient=pressure_basis(model,model.radius)
    mapping=np.column_stack([model.load(gradient[:,:,j]) for j in range(model.order+1)])
    start=1 if model.k==model.m==0 else 0;g=mapping[:,start:];n=3*model.order
    a=model.mass_full+.5*dt*nu*model.stiffness_full
    system=np.block([[a,dt*g],[dt*g.conj().T,np.zeros((g.shape[1],g.shape[1]))]])
    rhs=np.r_[(model.mass_full-.5*dt*nu*model.stiffness_full)@model.coefficients(before)+dt*force,np.zeros(g.shape[1])]
    solution=solve(system,rhs,assume_a='her');coefficients=np.r_[np.zeros(start),solution[n:]]
    return solution[:n],coefficients


def run(output,reference=None):
    out=Path(output).absolute();source=identity()
    if out.exists() or any(p.is_symlink() for p in (out,*out.parents)):raise ValueError('Fresh ordinary output required')
    out.mkdir(parents=True)
    criteria=dict(weak_momentum=1e-9,manufactured_pressure_l2=1e-10,manufactured_gradient_l2=1e-9,
        spurious_velocity=1e-10,monolithic_velocity=1e-8,monolithic_pressure_l2=1e-8)
    atomic_json(out/'protocol.json',seal(dict(source=source,criteria=criteria,
        scope='Dimensionless kinematic pressure recovered in regular finite-dimensional weak Stokes momentum',
        mean_gauge='zero radial-area mean for k=m=0',wall_scope='Pressure trace and derivative reported; strong wall momentum not qualified',
        native_openpipeflow_backend_changed=False)))
    cases={};passed=True
    for m in (0,1,2,4):
        k=0 if m==0 else 2
        for order in (4,6,8):
            model=regular_mode(k,m,.628319,order=order);exact,gradient=manufactured(model)
            load=model.load(gradient);field,proof=recover_pressure(model,load);p,g=field.evaluate(model.radius)
            p_error=float(np.sqrt(np.sum(model.weights*abs(p-exact)**2)))
            g_error=float(np.sqrt(np.sum(model.weights*abs(g-gradient)**2)))
            rest=PressureStokesSession(model,np.zeros(model.dimension));rest.advance(.005,load)
            q0=np.arange(model.dimension)*.0001;session=PressureStokesSession(model,q0);base=StokesSession(model,q0)
            force=np.arange(3*order)*.00001*(1+.2j)+.01*load
            maxima=dict(weak_momentum=0.);restart=None;restart_exact=True;base_exact=True
            for i in range(20):
                before=session.q;result,pressure=session.advance(.005,force);base.advance(.005,force)
                base_exact=bool(base_exact and np.array_equal(session.q,base.q))
                if restart is not None:
                    replay,rp=restart.advance(.005,force)
                    restart_exact=bool(restart_exact and replay==result and np.array_equal(rp.coefficients,pressure.coefficients))
                if i==9:restart=PressureStokesSession.restore(session.checkpoint())
                maxima['weak_momentum']=max(maxima['weak_momentum'],result['pressure']['weak_momentum_relative_residual'])
            direct_velocity,direct_pressure=monolithic(model,before,force,.005,.1)
            basis,_=pressure_basis(model,model.radius);actual,_=pressure.evaluate(model.radius)
            direct_p=basis@direct_pressure
            velocity_error=float(np.max(abs(model.coefficients(session.q)-direct_velocity)))
            pressure_error=float(np.sqrt(np.sum(model.weights*abs(actual-direct_p)**2)))
            checks=dict(manufactured_pressure=p_error<criteria['manufactured_pressure_l2'],manufactured_gradient=g_error<criteria['manufactured_gradient_l2'],
                no_spurious_velocity=np.linalg.norm(rest.q)<criteria['spurious_velocity'],monolithic_velocity=velocity_error<criteria['monolithic_velocity'],
                monolithic_pressure=pressure_error<criteria['monolithic_pressure_l2'],weak_momentum=maxima['weak_momentum']<criteria['weak_momentum'],restart=restart_exact,base_dynamics_exact=base_exact)
            checks={name:bool(value) for name,value in checks.items()}
            passed=bool(passed and all(checks.values()))
            cases[f'm{m}-order{order}']=dict(spec=model.spec,checks=checks,manufactured_pressure_l2_error=p_error,manufactured_gradient_l2_error=g_error,
                spurious_velocity=float(np.linalg.norm(rest.q)),monolithic_velocity_error=velocity_error,monolithic_pressure_l2_error=pressure_error,
                maximum_weak_momentum_residual=maxima['weak_momentum'],manufactured_pressure=proof,final_step=result,final_checkpoint=session.checkpoint())
    atomic_json(out/'cases.json',seal(cases))
    if source!=identity():raise ValueError('Pressure reconstruction source changed')
    report=dict(passed=passed,source=source,cases=len(cases),linear_weak_physical_pressure_qualified=passed,
        maximum_manufactured_pressure_error=max(v['manufactured_pressure_l2_error'] for v in cases.values()),
        maximum_manufactured_gradient_error=max(v['manufactured_gradient_l2_error'] for v in cases.values()),
        maximum_monolithic_velocity_error=max(v['monolithic_velocity_error'] for v in cases.values()),
        maximum_monolithic_pressure_error=max(v['monolithic_pressure_l2_error'] for v in cases.values()),
        maximum_weak_momentum_residual=max(v['maximum_weak_momentum_residual'] for v in cases.values()),
        all_restarts_exact=all(v['checks']['restart'] for v in cases.values()),base_v72_dynamics_exact=all(v['checks']['base_dynamics_exact'] for v in cases.values()),
        strong_wall_momentum_qualified=False,nonlinear_pressure_qualified=False,thermal_model_supported=False,native_openpipeflow_backend_changed=False,
        station_usable=False,validated=False,closed_tasks=[])
    atomic_json(out/'report.json',seal(report))
    if not passed:raise ValueError('Physical weak-pressure qualification failed; evidence retained')
    if reference and {p.name:sha(p) for p in Path(reference).glob('*.json')}!={p.name:sha(p) for p in out.glob('*.json')}:raise ValueError('Installed pressure scientific JSON differ')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--reference',type=Path)
    a=p.parse_args();print(run(a.output,a.reference)['passed'])

if __name__=='__main__':main()
