"""Local qualification of atomic agent sequences and checkpoint continuation."""
import argparse
from copy import deepcopy
from pathlib import Path
from atomic_model_station_env import StationEnv,EnvConfig
from atomic_model_station_hybrid_heat import advance_hybrid_actions
from atomic_model_station_hybrid_heat.model import identity
from atomic_model_multiphysics.provenance import seal,sha
from atomic_model_training.spec import atomic_json


def run(output):
    out=Path(output).absolute()
    if out.exists() or any(p.is_symlink() for p in (out,*out.parents)):
        raise ValueError('Fresh ordinary output required')
    source=identity();driver=sha(__file__);out.mkdir(parents=True)
    checks={};failure=None
    actions=[{'operator':{'feed_pump_on':False}}, {'operator':{'feed_pump_on':True}},{}]
    try:
        env=StationEnv(EnvConfig(max_steps=3,tick_s=.02,automatic=False,rtol=1e-10))
        env.reset();native=deepcopy(env);expected=[]
        for i,action in enumerate(actions):
            expected.append(native.step(action))
            if i==0:midpoint=native.checkpoint()
        transitions,proof=advance_hybrid_actions(env,actions)
        atomic_json(out/'sequence.json',seal(dict(transitions=transitions,proof=proof)))
        atomic_json(out/'final-checkpoint.json',env.checkpoint())
        checks['native_transitions']=transitions==expected
        checks['native_checkpoint']=env.checkpoint()==native.checkpoint()
        checks['complete_horizon']=transitions[-1][3] and len(transitions)==3
        checks['internal_closure']=all([e['kind'] for e in r['events']]==['feed_check_closed'] for r in proof['steps'][0]['replays'])
        restored=StationEnv.restore(midpoint)
        tail,continued=advance_hybrid_actions(restored,actions[1:])
        atomic_json(out/'restart.json',seal(dict(transitions=tail,proof=continued)))
        checks['restart']=tail==transitions[1:] and continued['steps']==proof['steps'][1:] and restored.checkpoint()==env.checkpoint()
        checks['accounting']=proof['passed'] and continued['passed']
    except Exception as error:
        failure=dict(type=type(error).__name__,message=str(error),receipt=getattr(error,'receipt',None))
    if source!=identity() or driver!=sha(__file__):raise ValueError('Source changed')
    report=dict(source=source,driver_sha256=driver,actions=actions,checks=checks,failure=failure,
        passed=failure is None and bool(checks) and all(checks.values()),
        duration_s=.06,industrial_qualification=False,closed_tasks=[])
    atomic_json(out/'report.json',seal(report))
    if not report['passed']:raise ValueError('Agent sequence qualification failed; evidence retained')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    print(run(parser.parse_args().output)['checks'])
