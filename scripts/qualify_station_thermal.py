"""Qualify selected coupled thermal/Boussinesq research cases, not a plant."""
import argparse
import platform
from pathlib import Path

import numpy as np
import scipy
from numpy.polynomial.legendre import leggauss
from scipy.special import jv, jn_zeros

from atomic_model_multiphysics.provenance import sha, seal
from atomic_model_training.spec import atomic_json
from atomic_model_station_refined import refined_mode, compare_refined
from atomic_model_station_thermal import scalar_mode, ThermalSystem, evolve_thermal, audit_thermal
from scripts.qualify_station_advection import modes


def identity():
    import atomic_model_station_thermal as package
    from scripts.qualify_station_refined import identity as base_identity
    return dict(files={p.name:sha(p) for p in sorted(Path(package.__file__).parent.glob('*.py'))},
                driver=sha(__file__),base=base_identity(),python=platform.python_version(),
                numpy=np.__version__,scipy=scipy.__version__)


def nonlinear_system(order=6, *, buoyancy=.2):
    seed={(m.k,m.m):(m,q*10) for m,q in modes(1,3,96)}
    velocity=[];temperature=[]
    for key,(old,q) in seed.items():
        m=refined_mode(*key,.628319,order=order)
        velocity.append((m,m.fit(old.evaluate(q,m.radius))))
        s=scalar_mode(*key,.628319,order=order)
        amplitude=.1 if key==(0,0) else .02*(1+.3j)
        temperature.append((s,s.fit(amplitude*s.radius**s.m*(1-s.radius**2))))
    return ThermalSystem(velocity,temperature,buoyancy=buoyancy)


def difference(a,b):
    if abs(a.report['time']-b.report['time'])>1e-12:raise ValueError('Same thermal comparison time required')
    x,w=leggauss(128);r=np.sqrt((x+1)/2);w=w/4
    v=compare_refined(a.velocity,b.velocity,r,w,first_time=a.report['time'],second_time=b.report['time'])['velocity_l2']
    aa={(s.k,s.m):(s,q) for s,q in a.temperature};bb={(s.k,s.m):(s,q) for s,q in b.temperature}
    if set(aa)!=set(bb):raise ValueError('Same retained thermal spectrum required')
    total=0.
    for key in sorted(aa):
        s,q=aa[key];t,p=bb[key]
        total+=(1 if key==(0,0) else 2)*float(np.sum(w*abs(s.evaluate(q,r)-t.evaluate(p,r))**2))
    return dict(velocity_l2=v,temperature_l2=float(np.sqrt(total)))


def transient(order,nu=.05,kappa=.03,buoyancy=.2):
    m=refined_mode(0,0,.628319,order=order);s=scalar_mode(0,0,.628319,order=order)
    root=jn_zeros(0,1)[0]
    system=ThermalSystem([(m,np.zeros(m.dimension))],[(s,s.fit(.1*jv(0,root*s.radius)))],
                         viscosity=nu,diffusivity=kappa,buoyancy=buoyancy)
    result=evolve_thermal(system)
    r=np.linspace(0,1,129);time=result.report['time']
    factor=(time*np.exp(-nu*root**2*time) if nu==kappa else
            (np.exp(-kappa*root**2*time)-np.exp(-nu*root**2*time))/((nu-kappa)*root**2))
    ue=buoyancy*.1*factor*jv(0,root*r);te=.1*np.exp(-kappa*root**2*time)*jv(0,root*r)
    return dict(velocity_error=float(np.max(abs(m.evaluate(result.velocity[0][1],r)[2]-ue))),
                temperature_error=float(np.max(abs(s.evaluate(result.temperature[0][1],r)-te))),
                solver=result.report,audit=audit_thermal(system,result,radius=r[1:]))


def run(output,reference=None):
    out=Path(output).absolute();source=identity()
    if out.exists() or any(p.is_symlink() for p in (out,*out.parents)):raise ValueError('Fresh ordinary output required')
    out.mkdir(parents=True)
    criteria=dict(analytic=1e-9,heat=1e-10,strong_analytic=1e-8,ode_l2=1e-8,
                  weak_budget=1e-9,weak_pressure=1e-9,nonlinear_power=1e-11)
    atomic_json(out/'protocol.json',seal(dict(source=source,criteria=criteria,
        equations=['u_t+u.grad(u)=-grad(p)+nu*lap(u)+b*theta*e_z','theta_t+u.grad(theta)=kappa*lap(theta)+Q'],
        boundaries='No slip and theta=0 on wall; regular axis; axial periodicity',
        units='Nondimensional constant-property Boussinesq anomaly; +z buoyancy',
        heat='H=integral mean(theta) r dr; dH/dt=integral mean(Q) r dr+kappa*d_r mean(theta)(1)',
        ledgers=['Kinetic energy with viscous dissipation and buoyant work','Scalar variance with diffusion and source correlation','Physical heat content with independently evaluated wall flux'],
        reference='DOP853 with direct pair convolutions vs real-split sparse BDF with FFT',
        exclusions=['General heated 3D strong convergence','Boiling/CHF','Variable-density EOS','Experimental validation','Plant acceptance','Bit-exact BDF restart'])))
    analytic={}
    for label,p,nu,kappa,b in [('coarse',3,.05,.03,.2),('medium',6,.05,.03,.2),('fine',10,.05,.03,.2),('equal_diffusivities',10,.03,.03,.2),('reverse_buoyancy',10,.05,.03,-.2)]:
        analytic[label]=transient(p,nu,kappa,b)
        print(label,'errors',analytic[label]['velocity_error'],analytic[label]['temperature_error'],flush=True)
    m=refined_mode(0,0,.628319,order=4);s=scalar_mode(0,0,.628319,order=4);r=s.radius
    velocity=np.zeros((3,len(r)),complex);velocity[2]=.2*.1/(16*.05)*(3-4*r*r+r**4)
    steady_system=ThermalSystem([(m,m.fit(velocity))],[(s,s.fit(.1*(1-r*r)))],sources={(0,0):[.012]})
    steady=evolve_thermal(steady_system)
    steady_evidence=dict(solver=steady.report,audit=audit_thermal(steady_system,steady,radius=np.linspace(.005,1,129)),
                         state_change=float(np.max(abs(np.array(steady.report['final'])-steady.report['initial']))))
    system=nonlinear_system();result=evolve_thermal(system,end_time=.05)
    tight=evolve_thermal(system,end_time=.05,rtol=2e-12,atol=2e-15)
    reference_result=evolve_thermal(system,end_time=.05,method='DOP853',products='direct',rtol=2e-12,atol=2e-15)
    nonbuoyant=evolve_thermal(nonlinear_system(buoyancy=0.),end_time=.05)
    high_system=nonlinear_system(order=10);high=evolve_thermal(high_system,end_time=.05)
    refined=difference(result,high)
    time_difference=difference(result,tight);ode_difference=difference(result,reference_result)
    feedback=difference(result,nonbuoyant)
    high_audit=audit_thermal(high_system,high,radius=np.linspace(.005,1,129))
    nonlinear=dict(solver=result.report,tight=tight.report,independent_reference=reference_result.report,
        time_difference=time_difference,independent_difference=ode_difference,buoyancy_feedback=feedback,
        radial_difference=refined,high_solver=high.report,high_audit=high_audit,
        general_spatial_gate_passed=bool(max(refined.values())<1e-7 and high_audit['maximum_wall_momentum_residual']<1e-8 and high_audit['maximum_wall_thermal_residual']<1e-8 and high.report['maximum_physical_heat_budget_residual']<1e-10))
    print('nonlinear time/reference',time_difference,ode_difference,'strong gate',nonlinear['general_spatial_gate_passed'],flush=True)
    all_reports=[case['solver'] for case in analytic.values()]+[steady.report,result.report,tight.report,reference_result.report,high.report]
    checks=dict(analytic=all(max(analytic[key]['velocity_error'],analytic[key]['temperature_error'])<criteria['analytic'] for key in ('fine','equal_diffusivities','reverse_buoyancy')),
        analytic_heat=all(analytic[key]['solver']['maximum_physical_heat_budget_residual']<criteria['heat'] for key in ('fine','equal_diffusivities','reverse_buoyancy')),
        analytic_strong=all(max(analytic[key]['audit']['maximum_momentum_residual'],analytic[key]['audit']['maximum_thermal_residual'])<criteria['strong_analytic'] for key in ('fine','equal_diffusivities','reverse_buoyancy')),
        coarse_refusal=analytic['coarse']['solver']['maximum_physical_heat_budget_residual']>criteria['heat'],
        steady=steady_evidence['state_change']<criteria['analytic'] and steady.report['maximum_physical_heat_budget_residual']<criteria['heat'],
        budgets=all(max(case['maximum_kinetic_budget_residual'],case['maximum_variance_budget_residual'])<criteria['weak_budget'] for case in all_reports),
        pressure=all(case['maximum_weak_pressure_residual']<criteria['weak_pressure'] for case in all_reports),
        power=all(max(abs(case['endpoint_advection_power']),abs(case['endpoint_advection_variance']))<criteria['nonlinear_power'] for case in all_reports),
        time=max(time_difference.values())<criteria['ode_l2'],independent_ode=max(ode_difference.values())<criteria['ode_l2'],
        feedback=feedback['velocity_l2']>1e-7 and feedback['temperature_l2']>1e-10)
    atomic_json(out/'physics.json',seal(dict(analytic=analytic,steady=steady_evidence)))
    atomic_json(out/'dynamics.json',seal(dict(nonlinear=nonlinear,checks=checks)))
    if source!=identity():raise ValueError('Thermal qualification source changed')
    report=dict(passed=bool(all(checks.values())),source=source,checks=checks,selected_coupled_ode_qualified=bool(all(checks.values())),
        selected_analytic_heat_qualified=bool(checks['analytic_heat'] and checks['analytic_strong']),
        independent_difference=ode_difference,buoyancy_feedback=feedback,
        general_heated_spatial_gate_passed=nonlinear['general_spatial_gate_passed'],
        general_heated_strong_pde_qualified=False,physical_heat_gate_enforced_by_engine=False,
        station_usable=False,validated=False,native_backend_changed=False,closed_tasks=[])
    atomic_json(out/'report.json',seal(report))
    if not report['passed']:raise ValueError('Selected thermal qualification failed; evidence preserved')
    if reference and {p.name:sha(p) for p in Path(reference).glob('*.json')}!={p.name:sha(p) for p in out.glob('*.json')}:raise ValueError('Installed thermal scientific JSON differ')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference',type=Path)
    args=parser.parse_args();print(run(args.output,args.reference)['passed'])

if __name__=='__main__':main()
