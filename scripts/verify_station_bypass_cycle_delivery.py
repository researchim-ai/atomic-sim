"""Offline wheel replay of the upper bypass cycle only; no task closure."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import venv

from atomic_model_multiphysics.provenance import seal,sha,unseal
from atomic_model_station.bypass import bypass_signature
from atomic_model_training.spec import atomic_json
from scripts import qualify_station_bypass as study

WORKER = '''
import json,sys
from pathlib import Path
import atomic_model_station as station
from atomic_model_station import PWRBypassSession,BypassConfig
from atomic_model_multiphysics.provenance import sha,unseal
from scripts import qualify_station_bypass as study
root=Path(sys.argv[1])
assert 'site-packages' in Path(station.__file__).parts
assert 'site-packages' in Path(study.__file__).parts
protocol=unseal(json.loads((root/'protocol.json').read_text()))
assert sha(study.__file__)==protocol['driver_sha256']
trial=unseal(json.loads((root/'upper_cycle.json').read_text()))['trials'][-1]
s=PWRBypassSession(bypass=BypassConfig(relief_open_pa=6.4e6,relief_close_pa=6.38e6),
    max_step_s=trial['step_s'],rtol=trial['rtol'])
assert s.advance(.1,dict(scram=True,turbine_trip=True))==trial['before']
saved=json.loads(json.dumps(s.checkpoint()))
assert saved==trial['checkpoint']
t=PWRBypassSession.restore(saved)
assert s.advance(2.9)==trial['final']
assert t.advance(2.9)==trial['final']
assert s.state.tolist()==trial['state']
print('Installed upper cycle, checkpoint and restart match source exactly',flush=True)
'''


def run(output,*,wheel,evidence):
    output=Path(output).absolute();wheel=Path(wheel).resolve();evidence=Path(evidence).resolve()
    if any(p.is_symlink() for p in (output,*output.parents)):
        raise ValueError('Symlink output refused')
    def read(name):return unseal(json.loads((evidence/name).read_text()))
    protocol=read('protocol.json');report=read('report.json')
    if not (report['passed'] and report['source_unchanged'] and report['upper_cycle_qualified']
            and protocol['source']==bypass_signature() and protocol['driver_sha256']==sha(study.__file__)):
        raise ValueError('Stale or failed upper-cycle evidence')
    if wheel.suffix!='.whl' or not wheel.is_file():raise ValueError('Wheel required')
    output.mkdir(parents=True,exist_ok=False)
    signature=bypass_signature();driver=sha(__file__);wheel_hash=sha(wheel)
    evidence_hashes={n:sha(evidence/n) for n in ('protocol.json','report.json','upper_cycle.json')}
    with tempfile.TemporaryDirectory(prefix='atomic-bypass-cycle-') as temporary:
        temporary=Path(temporary).resolve()
        checkout=Path(__file__).resolve().parents[1]
        if temporary==checkout or checkout in temporary.parents:raise ValueError('Replay must run outside checkout')
        venv.EnvBuilder(with_pip=True,system_site_packages=True).create(temporary/'venv')
        python=temporary/'venv'/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
        environment=dict(os.environ)
        for name,command in [('install',[str(python),'-I','-m','pip','install','--no-index','--no-deps',str(wheel)]),
                             ('replay',[str(python),'-I','-c',WORKER,str(evidence)])]:
            print(name,flush=True)
            with (output/(name+'.log')).open('w') as log:
                subprocess.run(command,cwd=temporary,env=environment,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800)
    unchanged=(signature==bypass_signature() and driver==sha(__file__) and wheel_hash==sha(wheel)
        and protocol['driver_sha256']==sha(study.__file__)
        and all(value==sha(evidence/name) for name,value in evidence_hashes.items()))
    result=dict(passed=unchanged,source_unchanged=unchanged,upper_cycle_only=True,
        upper_cycle_duration_s=3.,restart_from_s=.1,installed_import=True,
        outside_checkout=True,all_compared_values_exact=True,source=signature,
        driver_sha256=driver,wheel_sha256=wheel_hash,evidence_sha256=evidence_hashes,
        closed_tasks=[],full_component_replay=False,full_duration_qualified=False,
        complete_package_b=False,r1_accepted=False,validated=False)
    atomic_json(output/'report.json',seal(result))
    if not unchanged:raise ValueError('Evidence changed during replay')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--wheel',type=Path,required=True)
    parser.add_argument('--evidence',type=Path,required=True)
    args=parser.parse_args();run(args.output,wheel=args.wheel,evidence=args.evidence)
