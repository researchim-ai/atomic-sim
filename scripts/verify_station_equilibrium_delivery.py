"""Installed full-hour refinement/native replay, offline and outside checkout."""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile
import venv
from atomic_model_multiphysics.provenance import seal,sha
from atomic_model_training.spec import atomic_json
from scripts.qualify_station_inventory import read
from scripts.qualify_station_equilibrium import identity


WORKER='''
import sys
from pathlib import Path
from scripts import qualify_station_equilibrium as study
import atomic_model_station_equilibrium as equilibrium
from atomic_model_multiphysics.provenance import seal,sha
from atomic_model_training.spec import atomic_json
out,reference,receipt=map(Path,sys.argv[1:4])
assert 'site-packages' in Path(study.__file__).parts
assert 'site-packages' in Path(equilibrium.__file__).parts
r=study.run(out,reference)
assert r['passed']
atomic_json(receipt,seal(dict(passed=True,source=study.identity(),outside_checkout=True,all_mathematical_json_exact=True,
  full_installed_fine_s=3600.,full_native_prefix_s=1800.,full_native_suffix_s=1800.,
  native_exact=True,driver_sha256=sha(study.__file__),stationary_accepted=False,r1_accepted=False)))
'''


def run(output,source,library):
    out=Path(output).absolute();source=Path(source).absolute();library=Path(library).absolute()
    if any(p.is_symlink() for p in (out,*out.parents)):raise ValueError('Symlink output refused')
    study=read(source/'mathematics/report.json');protocol=read(source/'mathematics/protocol.json');delivery=read(library/'report.json')
    expected=identity()
    if not study['passed'] or not study['source_unchanged'] or any(protocol[k]!=v for k,v in expected.items()):
        raise ValueError('Equilibrium source evidence is stale or unaccepted')
    if not delivery['completed'] or len(delivery['phases'])!=8 or not all(p['passed'] for p in delivery['phases']) or delivery['tests']['gate_failures']:
        raise ValueError('Library delivery not accepted')
    wheels=list((library/'build').glob('*.whl'))
    if len(wheels)!=1:raise ValueError('One delivery wheel required')
    out.mkdir(parents=True,exist_ok=False);own_sha=sha(__file__)
    with tempfile.TemporaryDirectory(prefix='atomic-station-equilibrium-install-') as directory:
        work=Path(directory).resolve();root=Path(__file__).resolve().parents[1]
        if work==root or root in work.parents:raise ValueError('Installed work must be outside checkout')
        venv.EnvBuilder(with_pip=True,system_site_packages=True).create(work/'venv')
        python=work/'venv'/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
        commands=[('install',[str(python),'-I','-m','pip','install','--no-index','--no-deps',str(wheels[0])]),
            ('installed-study',[str(python),'-I','-c',WORKER,str(out/'installed'),str(source),str(out/'worker.json')])]
        for name,command in commands:
            print(name,flush=True)
            with (out/(name+'.log')).open('w') as log:
                subprocess.run(command,cwd=work,env=dict(os.environ),stdout=log,stderr=subprocess.STDOUT,check=True,timeout=14400)
    worker=read(out/'worker.json')
    if worker['source']!=identity() or expected!=identity() or own_sha!=sha(__file__):
        raise ValueError('Equilibrium source changed during installed delivery')
    receipt=dict(worker,wheel_sha256=sha(wheels[0]),verifier_sha256=own_sha,
        library_tests=delivery['tests']['counts']['passed'],library_report_sha256=sha(library/'report.json'),
        source_report_sha256=sha(source/'mathematics/report.json'),
        mathematical_json_count=len(list((source/'mathematics').rglob('*.json'))),
        native_report_sha256=sha(out/'installed/native/report.json'),ci_required=False,external_runner_required=False)
    atomic_json(out/'report.json',seal(receipt));return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--source-evidence',type=Path,required=True);parser.add_argument('--library-evidence',type=Path,required=True)
    args=parser.parse_args();r=run(args.output,args.source_evidence,args.library_evidence);print('passed='+str(r['passed']),flush=True)


if __name__=='__main__':main()
