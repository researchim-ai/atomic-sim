"""Local passport/configuration qualification with actual transitions and restart."""
from copy import deepcopy
from dataclasses import asdict
import argparse
from pathlib import Path
from atomic_model_station_env import StationEnv,EnvConfig
from atomic_model_station_faults import Fault
from atomic_model_station_capabilities import passport,migrate_config,load_config,RAW_FORMAT,CONFIG_FORMAT
from atomic_model_multiphysics.provenance import seal,unseal,sha
from atomic_model_training.spec import atomic_json


def run(output):
    out=Path(output).absolute()
    if out.exists() or any(p.is_symlink() for p in (out,*out.parents)):raise ValueError('Fresh ordinary output required')
    source=passport();driver=sha(__file__);out.mkdir(parents=True);checks={};failure=None
    try:
        atomic_json(out/'passport.json',seal(source))
        config=EnvConfig(max_steps=3,tick_s=.02,automatic=False,rtol=1e-10,
            faults=(Fault('drive_loss','pump_off','feed',1),))
        raw=asdict(config);atomic_json(out/'legacy-config.json',raw)
        document,audit=migrate_config(raw,from_format=RAW_FORMAT)
        atomic_json(out/'configuration.json',document);atomic_json(out/'migration.json',seal(audit))
        checks['configuration_preserved']=load_config(document)==config
        a=StationEnv(config);b=StationEnv(load_config(document));checks['initial_observation_exact']=a.reset()==b.reset()
        actions=[{}, {'operator':{'feed_pump_on':True}},{}]
        records=[];exact=True
        for i,action in enumerate(actions):
            x=a.step(deepcopy(action));y=b.step(deepcopy(action));exact=exact and x==y;records.append(y)
            if i==0:midpoint=b.checkpoint()
        restored=StationEnv.restore(midpoint);tail=[restored.step(action) for action in actions[1:]]
        checks['transitions_exact']=exact
        checks['restart_exact']=tail==records[1:] and a.checkpoint()==b.checkpoint()==restored.checkpoint()
        atomic_json(out/'trajectory.json',seal(dict(actions=actions,transitions=records,midpoint=midpoint,final=b.checkpoint())))
        stale=unseal(document);stale['model_source']='0'*64;stale=seal(stale)
        try:load_config(stale);checks['stale_refused']=False
        except ValueError:checks['stale_refused']=True
        try:migrate_config(stale,from_format=CONFIG_FORMAT);checks['implicit_rebind_refused']=False
        except ValueError:checks['implicit_rebind_refused']=True
        rebound,rebind_audit=migrate_config(stale,from_format=CONFIG_FORMAT,allow_rebind=True)
        checks['explicit_rebind']=load_config(rebound)==config and rebind_audit['requalification_required']
        atomic_json(out/'rebound.json',rebound);atomic_json(out/'rebind-audit.json',seal(rebind_audit))
        checks['evidence_current']=all(r['current_source_matches'] for r in source['qualifications'])
        checks['scope_preserved']=not any(source[k] for k in ('industrial_plant_qualified','full_r1_accepted','experimental_validation'))
    except Exception as error:failure=dict(type=type(error).__name__,message=str(error))
    checks['source_unchanged']=source==passport() and driver==sha(__file__)
    report=dict(source=source,driver_sha256=driver,checks=checks,failure=failure,
        passed=failure is None and len(checks)==10 and all(checks.values()),
        configuration_only_migration=True,full_library_suite_run=False,closed_tasks=[])
    atomic_json(out/'report.json',seal(report))
    if not report['passed']:raise ValueError('Station capability qualification failed; evidence retained')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    print(run(parser.parse_args().output)['checks'])
