"""Atomic StationEnv feed-valve command qualification with two-sided roots."""
import argparse
from copy import deepcopy
from pathlib import Path
from atomic_model_multiphysics.provenance import seal,sha
from atomic_model_training.spec import atomic_json
from atomic_model_station import PWRPowerSession,BackupConfig
from atomic_model_station_env import StationEnv,EnvConfig
from atomic_model_station_hybrid_heat import step_hybrid_heat,record_hybrid_window
from atomic_model_station_hybrid_heat.model import identity


def run(output):
    out=Path(output).absolute()
    if out.exists() or any(p.is_symlink() for p in (out,*out.parents)):raise ValueError('Fresh ordinary output required')
    source=identity();driver=sha(__file__);out.mkdir(parents=True)
    checks={};failure=None;rows=[]
    try:
        env=StationEnv(EnvConfig(max_steps=4,tick_s=.02,automatic=False,rtol=1e-10));env.reset()
        for i,action in enumerate([{'operator':{'feed_pump_on':False}},{'operator':{'feed_pump_on':True}},{}]):
            native=deepcopy(env);expected=native.step(action);saved=env.checkpoint()
            actual,proof=step_hybrid_heat(env,action,subdivisions=(4,8,16));rows.append(proof)
            checks[f'native_step_{i}']=actual==expected and env.checkpoint()==native.checkpoint()
            atomic_json(out/f'step-{i}.json',seal(proof))
            if i==1:
                restored=StationEnv.restore(saved)
                again,p=step_hybrid_heat(restored,action,subdivisions=(4,8,16))
                checks['restart']=again==actual and p==proof and restored.checkpoint()==env.checkpoint()
        checks['feed_closed_root']=all([e['kind'] for e in r['events']]==['feed_check_closed'] for r in rows[0]['replays'])
        checks['all_audits']=all(p['passed'] for p in rows)
        session=PWRPowerSession(backup=BackupConfig(initial_energy_j=1e4),rtol=1e-10)
        session.advance(.08,{'grid_available':False});saved=session.checkpoint()
        backup=record_hybrid_window(session,.04,subdivisions=16)
        atomic_json(out/'backup.json',seal(backup))
        checks['backup_regression']=backup['passed'] and [e['kind'] for e in backup['events']]==['backup_exhausted']
        checks['backup_readonly']=session.checkpoint()==saved
    except Exception as error:
        failure=dict(type=type(error).__name__,message=str(error),receipt=getattr(error,'receipt',None))
    if source!=identity() or driver!=sha(__file__):raise ValueError('Source changed')
    report=dict(source=source,driver_sha256=driver,checks=checks,failure=failure,
        passed=failure is None and len(rows)==3 and all(checks.values()),completed_steps=len(rows),
        tick_s=.02,rtol=1e-10,subdivisions=[4,8,16],station_env_integration=True,
        qualified_internal_valves=['feed_check_closed'],relief_valves_qualified=False,
        condensate_root_qualified=False,simultaneous_events_supported=False,closed_tasks=[])
    atomic_json(out/'report.json',seal(report))
    if not report['passed']:raise ValueError('Hybrid heat qualification failed; evidence retained')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    print(run(parser.parse_args().output)['checks'])
