"""Qualify short DHR command replays without changing native controller cadence."""
import argparse
from pathlib import Path
from atomic_model_multiphysics.provenance import seal,sha
from atomic_model_training.spec import atomic_json
from atomic_model_station_env import StationEnv,EnvConfig
from atomic_model_station_event_heat import HeatEventError
from atomic_model_station_refined_heat import step_refined_heat
from atomic_model_station_refined_heat.model import identity


def run(output):
    out=Path(output).absolute()
    if out.exists() or any(p.is_symlink() for p in (out,*out.parents)):
        raise ValueError('Fresh ordinary output required')
    source=identity();driver=sha(__file__);out.mkdir(parents=True)
    config=EnvConfig(max_steps=4,tick_s=.01,automatic=False)
    env=StationEnv(config);env.reset();proofs=[];checks={}
    actions=[{'operator':{'dhr_requested':True}},
             {'operator':{'dhr_requested':False}},
             {'operator':{'dhr_requested':True,'load_fraction':.95}},{}]
    try:
        for i,action in enumerate(actions):
            saved=env.checkpoint()
            counts=(4,8,16) if i==0 else (8,16,32)
            transition,proof=step_refined_heat(env,action,subdivisions=counts);proofs.append(proof)
            atomic_json(out/f'step-{i}.json',seal(proof))
            if i==1:
                restored=StationEnv.restore(saved)
                again,replayed=step_refined_heat(restored,action,subdivisions=counts)
                checks['checkpoint_continuation']=again==transition and replayed==proof and restored.checkpoint()==env.checkpoint()
        fine=StationEnv(config);fine.reset()
        _,extra=step_refined_heat(fine,actions[0],subdivisions=(8,16,32))
        atomic_json(out/'extra-refinement.json',seal(extra))
        a=proofs[0]['audit']['sampled_integrals_j']['nuclear_heat_w']
        b=extra['audit']['sampled_integrals_j']['nuclear_heat_w']
        checks['extra_refinement']=abs(a-b)/max(1.,abs(b))<2e-6
        checks['all_steps']=len(proofs)==4 and all(p['passed'] for p in proofs)
        checks['dhr_coarse_negative']=not proofs[0]['coarse_audit']['passed']
        checks['native_controller_cadence']=all(not p['controller_substepped'] for p in proofs)
        failure=None
    except Exception as error:
        failure=dict(type=type(error).__name__,message=str(error),receipt=getattr(error,'receipt',None))
    if source!=identity() or driver!=sha(__file__):raise ValueError('Qualification source changed')
    report=dict(source=source,driver_sha256=driver,checks=checks,failure=failure,
        passed=failure is None and bool(checks) and all(checks.values()),completed_steps=len(proofs),
        horizon_s=.04,tick_s=.01,subdivisions_by_step=[[4,8,16]]+[[8,16,32]]*3,extra_subdivisions=[8,16,32],
        internal_event_subdivision_supported=False,isolated_quadrature_error_bound=False,
        stationary_efficiency_certified=False,closed_tasks=[])
    atomic_json(out/'report.json',seal(report))
    if not report['passed']:raise ValueError('Refined heat qualification failed; evidence retained')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    report=run(parser.parse_args().output)
    print(report['checks'])
