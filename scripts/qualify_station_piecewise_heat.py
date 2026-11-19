"""Short analytic rod-stop heat qualification, with retained coarse-solver refusal."""
import argparse
from pathlib import Path
from atomic_model_multiphysics.provenance import seal,sha
from atomic_model_training.spec import atomic_json
from atomic_model_station_env import StationEnv,EnvConfig
from atomic_model_station_event_heat import HeatEventError
from atomic_model_station_piecewise_heat import step_piecewise_heat
from atomic_model_station_piecewise_heat.model import identity


def run(output):
    out=Path(output).absolute()
    if out.exists() or any(p.is_symlink() for p in (out,*out.parents)):
        raise ValueError('Fresh ordinary output required')
    source=identity();driver=sha(__file__);out.mkdir(parents=True)
    checks={};results={};failure=None
    try:
        for name,offset in [('insert',.000037),('withdraw',-.000037),('near_endpoint',.0001)]:
            env=StationEnv(EnvConfig(max_steps=3,tick_s=.01,automatic=False,rtol=1e-10,max_step_s=.01));env.reset()
            action={'operator':{'rod_target':float(env._session.control.plant.state[12])+offset}}
            _,proof=step_piecewise_heat(env,action,subdivisions=(8,16,32) if name=='near_endpoint' else (4,8,16));results[name]=proof
            atomic_json(out/(name+'.json'),seal(proof))
            checks[name]=proof['passed'] and all(any(s['energy']['time_s']==proof['motion']['arrival_s'] for s in r['samples']) for r in proof['replays'])
            if name=='insert':
                saved=env.checkpoint();restored=StationEnv.restore(saved)
                actual,p=step_piecewise_heat(env,{})
                repeated,q=step_piecewise_heat(restored,{})
                checks['restart']=actual==repeated and p==q and env.checkpoint()==restored.checkpoint()
                checks['no_duplicate_stop']=all(not r['event_boundaries_s'] for r in p['replays'])
                atomic_json(out/'continuation.json',seal(p))
        coarse=StationEnv(EnvConfig(max_steps=3,tick_s=.01,automatic=False));coarse.reset();before=coarse.checkpoint()
        try:
            step_piecewise_heat(coarse,{'operator':{'rod_target':float(coarse._session.control.plant.state[12])+.000037}})
            checks['coarse_refused']=False
        except HeatEventError as error:
            checks['coarse_refused']=error.receipt['reason']=='refinement_failed' and not error.receipt['candidate']['checks']['endpoint_agreement'] and coarse.checkpoint()==before
            atomic_json(out/'coarse-refusal.json',seal(error.receipt))
    except Exception as error:
        failure=dict(type=type(error).__name__,message=str(error),receipt=getattr(error,'receipt',None))
    if source!=identity() or driver!=sha(__file__):raise ValueError('Source changed')
    report=dict(source=source,driver_sha256=driver,checks=checks,failure=failure,
        passed=failure is None and len(results)==3 and all(checks.values()),
        tick_s=.01,rtol=1e-10,max_step_s=.01,subdivisions_by_case=dict(insert=[4,8,16],withdraw=[4,8,16],near_endpoint=[8,16,32]),
        supported_internal_events=['rod_arrived'],discontinuous_flux_events_supported=False,
        stationary_efficiency_certified=False,closed_tasks=[])
    atomic_json(out/'report.json',seal(report))
    if not report['passed']:raise ValueError('Piecewise qualification failed; evidence retained')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    print(run(parser.parse_args().output)['checks'])
