"""Offline installed-wheel repeat of the short load qualification, outside checkout."""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile
import venv
import zipfile
import hashlib
from atomic_model_multiphysics.provenance import seal,sha
from atomic_model_training.spec import atomic_json
from scripts.qualify_station_load_cycle import identity
from scripts.qualify_station_inventory import read

WORKER='''
import sys
from pathlib import Path
from scripts import qualify_station_load_short as study
import atomic_model_station_load
assert 'site-packages' in Path(study.__file__).parts
assert 'site-packages' in Path(atomic_model_station_load.__file__).parts
study.run(Path(sys.argv[1]),Path(sys.argv[2]))
'''


def run(output,source,library):
    out=Path(output).absolute();source=Path(source).absolute();library=Path(library).absolute()
    if any(p.is_symlink() for p in (out,*out.parents)):raise ValueError('Symlink output refused')
    result=read(source/'report.json');delivery=read(library/'report.json')
    initial=identity();own_sha=sha(__file__)
    if not result['passed'] or not result['native_exact'] or result['source']!=initial:raise ValueError('Short load source not accepted or stale')
    if not delivery['completed'] or len(delivery['phases'])!=8 or not all(p['passed'] for p in delivery['phases']) or delivery['tests']['gate_failures']:
        raise ValueError('Library delivery not accepted')
    wheels=list((library/'build').glob('*.whl'))
    if len(wheels)!=1:raise ValueError('One wheel required')
    root=Path(__file__).resolve().parents[1]
    with zipfile.ZipFile(wheels[0]) as wheel:
        files=[(str(p.relative_to(root/'src')),p) for p in (root/'src/atomic_model_station_load').glob('*.py')]
        files += [('scripts/'+name,root/'scripts'/name) for name in ('qualify_station_load_short.py','qualify_station_load_cycle.py')]
        if any(hashlib.sha256(wheel.read(n)).hexdigest()!=sha(p) for n,p in files):raise ValueError('Stale load wheel')
    hashes={str(p.relative_to(source)):sha(p) for p in source.rglob('*.json')}
    out.mkdir(parents=True,exist_ok=False)
    with tempfile.TemporaryDirectory(prefix='atomic-load-install-') as directory:
        work=Path(directory).resolve()
        if work==root or root in work.parents:raise ValueError('Installed load study must be outside checkout')
        venv.EnvBuilder(with_pip=True,system_site_packages=True).create(work/'venv')
        python=work/'venv'/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
        for name,command in [('install',[str(python),'-I','-m','pip','install','--no-index','--no-deps',str(wheels[0])]),
            ('study',[str(python),'-I','-c',WORKER,str(out/'installed'),str(source)])]:
            with (out/(name+'.log')).open('w') as log:
                subprocess.run(command,cwd=work,env=dict(os.environ),stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800)
    if initial!=identity() or own_sha!=sha(__file__) or hashes!={str(p.relative_to(source)):sha(p) for p in source.rglob('*.json')}:
        raise ValueError('Load source changed during delivery')
    actual={str(p.relative_to(out/'installed')):sha(p) for p in (out/'installed').rglob('*.json')}
    if actual!=hashes:raise ValueError('Installed scientific JSON differs')
    receipt=dict(passed=True,outside_checkout=True,source=initial,all_scientific_json_exact=True,
        scientific_json_count=len(hashes),native_exact=True,duration_s=.12,closed_tasks=[],r1_accepted=False,
        library_tests=delivery['tests']['counts']['passed'],library_report_sha256=sha(library/'report.json'),
        wheel_sha256=sha(wheels[0]),verifier_sha256=own_sha,source_report_sha256=sha(source/'report.json'),
        ci_required=False,external_runner_required=False,
        scope='Short three-setting numerical/refinement and native restart repeat only; not the full 1800s load cycle')
    atomic_json(out/'report.json',seal(receipt));return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source-evidence',type=Path,required=True);p.add_argument('--library-evidence',type=Path,required=True)
    a=p.parse_args();run(a.output,a.source_evidence,a.library_evidence)


if __name__=='__main__':main()
