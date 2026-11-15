"""Offline installed replay of station instrumentation; local dependencies only."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import venv

from atomic_model_multiphysics.provenance import seal,unseal,sha
from atomic_model_training.spec import atomic_json
from scripts import qualify_station_signals as study

WORKER='''
from pathlib import Path
import sys
import atomic_model_station_signals as instruments
import atomic_model_station as station
from scripts import qualify_station_signals as study
from atomic_model_multiphysics.provenance import sha
assert 'site-packages' in Path(instruments.__file__).parts
assert 'site-packages' in Path(station.__file__).parts
assert 'site-packages' in Path(study.__file__).parts
assert sha(study.__file__)==sys.argv[3]
r=study.run(Path(sys.argv[1]),Path(sys.argv[2]))
assert r['passed'] and r['restart_exact']
print('All installed instrument evidence matches source exactly',flush=True)
'''


def read(path):return unseal(json.loads(Path(path).read_text()))


def run(output,*,source_evidence,library_evidence):
    output=Path(output).absolute();source=Path(source_evidence).resolve();library=Path(library_evidence).resolve()
    if any(p.is_symlink() for p in (output,*output.parents)):raise ValueError('Symlink output refused')
    result=read(source/'report.json');protocol=read(source/'protocol.json');delivery=read(library/'report.json')
    identity=study.signals_signature();driver=sha(__file__)
    if not (result['passed'] and result['source_unchanged'] and result['closed_tasks']==study.TASKS
            and protocol['source']==identity and protocol['driver_sha256']==sha(study.__file__)):
        raise ValueError('Stale or failed instrument evidence')
    if not (delivery['completed'] and len(delivery['phases'])==8 and all(p['passed'] for p in delivery['phases'])
            and not delivery['tests']['gate_failures'] and delivery['tests']['counts']['failed']==0
            and delivery['tests']['counts']['errors']==0):raise ValueError('Local delivery not accepted')
    wheels=list((library/'build').glob('*.whl'))
    if len(wheels)!=1:raise ValueError('Expected one instrument delivery wheel')
    wheel=wheels[0];wheel_hash=sha(wheel)
    bound={str(p):sha(p) for p in [source/'protocol.json',source/'report.json',library/'report.json']}
    output.mkdir(parents=True,exist_ok=False)
    with tempfile.TemporaryDirectory(prefix='atomic-instruments-install-') as temporary:
        temporary=Path(temporary).resolve();checkout=Path(__file__).resolve().parents[1]
        if temporary==checkout or checkout in temporary.parents:raise ValueError('Installed replay must be outside checkout')
        venv.EnvBuilder(with_pip=True,system_site_packages=True).create(temporary/'venv')
        python=temporary/'venv'/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
        for name,command in [('install',[str(python),'-I','-m','pip','install','--no-index','--no-deps',str(wheel)]),
            ('replay',[str(python),'-I','-c',WORKER,str(output/'installed'),str(source),sha(study.__file__)])]:
            print(name,flush=True)
            with (output/(name+'.log')).open('w') as log:
                subprocess.run(command,cwd=temporary,env=dict(os.environ),stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800)
    unchanged=(identity==study.signals_signature() and driver==sha(__file__) and wheel_hash==sha(wheel)
        and protocol['driver_sha256']==sha(study.__file__) and all(sha(p)==value for p,value in bound.items()))
    receipt=dict(passed=unchanged,source_unchanged=unchanged,installed_import=True,outside_checkout=True,
        all_evidence_exact=True,paired_restart_exact=True,station_duration_s=30.,
        wheel_sha256=wheel_hash,verifier_sha256=driver,source=identity,evidence_sha256=bound,
        library_tests=delivery['tests']['counts']['passed'],closed_tasks=study.TASKS if unchanged else [],
        complete_package_c=False,r1_accepted=False,validated=False)
    atomic_json(output/'report.json',seal(receipt))
    if not unchanged:raise ValueError('Instrument evidence changed during replay')
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source-evidence',type=Path,required=True)
    parser.add_argument('--library-evidence',type=Path,required=True)
    args=parser.parse_args();run(args.output,source_evidence=args.source_evidence,library_evidence=args.library_evidence)
