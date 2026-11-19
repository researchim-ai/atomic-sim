"""Close the selected v77 temporal target with local control and independent ODE reference."""
import argparse,platform
from pathlib import Path
import numpy as np
import scipy
from numpy.polynomial.legendre import leggauss
from atomic_model_station_nonlinear import NonlinearSession
from atomic_model_station_adaptive import AdaptiveSession,AccuracyError
from atomic_model_station_resolution import compare_states,audit_nonlinear_momentum
from atomic_model_multiphysics.provenance import sha,seal
from atomic_model_training.spec import atomic_json
from scripts.qualify_station_advection import modes
from scripts.qualify_station_resolution import initial_state
from scripts.qualify_station_nonlinear import reference_solution


def identity():
    import atomic_model_station_adaptive as package
    from scripts.qualify_station_resolution import identity as base_identity
    root=Path(package.__file__).parent
    return dict(files={p.name:sha(p) for p in sorted(root.glob('*.py'))},driver=sha(__file__),base=base_identity(),python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__)


def run(output,reference=None):
    out=Path(output).absolute();source=identity()
    if out.exists() or any(p.is_symlink() for p in (out,*out.parents)):raise ValueError('Fresh ordinary output required')
    out.mkdir(parents=True);criteria=dict(global_velocity_l2=1e-7,adaptive_reference_agreement=1e-10,minimum_accuracy_gain=5.,local_error_ratio=1.,strong_wall=1e-8)
    atomic_json(out/'protocol.json',seal(dict(source=source,criteria=criteria,
        case='v77 fixed physical seed,extent2/order9,quadrature64,nu0.05,time0.1',
        controller=dict(atol=1e-11,rtol=1e-8,min_step=1e-7,max_step=.0025),
        error_estimator='L2 mass norm of full vs two half steps divided by3; accept actual half-step trajectory',
        reference='Two independently tightened DOP853 runs of same semidiscrete ODE with direct convolution',
        exclusions=['Spatial convergence','General strong wall qualification','Thermal/plant applicability'])))
    seed={(m.k,m.m):(m,q*10) for m,q in modes(1,3,64)};state=initial_state(2,9,seed)
    coarse,nfev=reference_solution(state,.1,1e-11,1e-13);fine,fine_nfev=reference_solution(state,.1,2e-13,2e-15)
    reference_difference=float(np.linalg.norm(coarse-fine))
    ordered=sorted(state,key=lambda pair:(pair[0].k,pair[0].m));cuts=np.cumsum([0]+[m.dimension for m,q in ordered])
    reference_modes=[(model,fine[cuts[i]:cuts[i+1]]) for i,(model,q) in enumerate(ordered)]
    x,w=leggauss(96);r=np.sqrt((x+1)/2);w=w/4
    fixed=NonlinearSession(state,viscosity=.05)
    for i in range(80):fixed.advance(.00125)
    baseline=compare_states(fixed.modes,reference_modes,r,w,first_time=fixed.time,second_time=.1)
    print('fixed-step reference difference',baseline['total_rms'],flush=True)
    adaptive=AdaptiveSession(NonlinearSession(state,viscosity=.05));result,pressure=adaptive.advance_to(.1)
    error=compare_states(adaptive.modes,reference_modes,r,w,first_time=adaptive.time,second_time=.1)
    last=result['last_step'];before=NonlinearSession.restore(last['core_last_step_start'])
    wall=audit_nonlinear_momentum(before.modes,adaptive.modes,pressure,dt=last['actual_half_step'],viscosity=.05,radius=np.linspace(.005,1,101),tolerance=criteria['strong_wall'])
    print('adaptive reference difference',error['total_rms'],'steps',result['accepted_steps'],'wall',wall['maximum_retained_wall_residual'],flush=True)
    accuracy_gain=baseline['total_rms']/max(error['total_rms'],1e-30)
    endpoint_checkpoint=adaptive.checkpoint()
    # Verify exact continuation of controller state, next-step choice and pressure.
    replay=AdaptiveSession.restore(adaptive.checkpoint());a,pa=adaptive.step(.001);b,pb=replay.step(.001)
    restart_exact=bool(a==b and all(np.array_equal(pa[key].coefficients,pb[key].coefficients) for key in pa))
    failed=AdaptiveSession(NonlinearSession([(m,q*10) for m,q in modes(1,3,32)],viscosity=.05),atol=1e-14,rtol=1e-12,max_step=.01,max_rejections=0)
    checkpoint=failed.checkpoint();atomic=False
    try:failed.step(.01)
    except AccuracyError:atomic=failed.checkpoint()==checkpoint
    checks=dict(reference=reference_difference<criteria['adaptive_reference_agreement'],accuracy=error['total_rms']<criteria['global_velocity_l2'],
        gain=accuracy_gain>criteria['minimum_accuracy_gain'],local=result['maximum_local_error_ratio']<=criteria['local_error_ratio'],
        previous_fixed_gate_failed=baseline['total_rms']>criteria['global_velocity_l2'],restart=restart_exact,rejection_atomic=atomic)
    atomic_json(out/'accuracy.json',seal(dict(checks=checks,baseline=baseline,adaptive=error,accuracy_gain=accuracy_gain,
        reference_tolerance_agreement=reference_difference,reference_nfev=[nfev,fine_nfev],reference_real=fine.real.tolist(),reference_imag=fine.imag.tolist())))
    atomic_json(out/'controller.json',seal(dict(interval=result,wall_audit=wall,
        endpoint_checkpoint=endpoint_checkpoint,continuation_checkpoint=adaptive.checkpoint(),restart_exact=restart_exact,atomic_failure=atomic)))
    if source!=identity():raise ValueError('Adaptive qualification source changed')
    passed=bool(all(checks.values()))
    report=dict(passed=passed,source=source,selected_temporal_target_qualified=bool(checks['accuracy'] and checks['reference']),
        adaptive_velocity_error=error['total_rms'],fixed_velocity_error=baseline['total_rms'],accuracy_gain=accuracy_gain,
        adaptive_reference_agreement=reference_difference,accepted_intervals=result['accepted_steps'],rejected_trials=result['rejected_trials'],maximum_local_error_ratio=result['maximum_local_error_ratio'],
        controller_restart_exact=restart_exact,rejection_atomic=atomic,remaining_strong_wall_residual=wall['maximum_retained_wall_residual'],
        measured_last_wall_gate_passed=wall['retained_wall_passed'],global_error_bound_claimed=False,spatial_convergence_qualified=False,
        general_strong_wall_qualified=False,thermal_model_supported=False,native_backend_changed=False,station_usable=False,validated=False,closed_tasks=[])
    atomic_json(out/'report.json',seal(report))
    if not passed:raise ValueError('Adaptive temporal qualification failed; evidence preserved')
    if reference and {p.name:sha(p) for p in Path(reference).glob('*.json')}!={p.name:sha(p) for p in out.glob('*.json')}:raise ValueError('Installed adaptive scientific JSON differ')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--reference',type=Path)
    a=p.parse_args();print(run(a.output,a.reference)['passed'])

if __name__=='__main__':main()
