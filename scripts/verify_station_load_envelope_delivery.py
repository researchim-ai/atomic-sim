"""Offline full-length fine/native wheel execution; reconcile against source proof."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import venv
import zipfile
from atomic_model_multiphysics.provenance import seal,sha
from atomic_model_training.spec import atomic_json
from atomic_model_station_load import LoadCycle
from scripts.qualify_station_inventory import read
from scripts import qualify_station_load_envelope as study

WORKER='''
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from atomic_model_multiphysics.provenance import seal
from atomic_model_training.spec import atomic_json
from scripts import qualify_station_load_envelope as study
import atomic_model_station_load
assert 'site-packages' in Path(study.__file__).parts
assert 'site-packages' in Path(atomic_model_station_load.__file__).parts
out,source=map(Path,sys.argv[1:3]);out.mkdir(parents=True,exist_ok=False)
identity=study.identity()
with ThreadPoolExecutor(max_workers=2) as pool:
    jobs=[pool.submit(study.launch,role,out/role,source/'coarse') for role in ('fine','native')]
    for job in jobs:job.result()
if identity!=study.identity():raise ValueError('Installed load implementation changed')
atomic_json(out/'worker.json',seal(dict(passed=True,source=identity,outside_checkout=True)))
'''


def finalize(output,source,library):
    out=Path(output).absolute();source=Path(source).absolute();library=Path(library).absolute()
    manifest=read(out/'manifest.json');expected=study.identity()
    if manifest['source']!=expected or manifest['verifier_sha256']!=sha(__file__):raise ValueError('Load envelope verifier/source changed')
    cycle=LoadCycle(**read(source/'protocol.json')['cycle'])
    summary=study.summarize(source,cycle);published=read(source/'report.json')
    if seal(summary)['sha256']!=seal(published)['sha256'] or not summary['passed']:raise ValueError('Full source qualification not accepted')
    worker=read(out/'installed/worker.json');native=read(out/'installed/native/report.json')
    fine,_,_=study.inspect_case(out/'installed/fine',cycle,study.SETTINGS['fine'])
    if not worker['passed'] or worker['source']!=expected or not worker['outside_checkout'] or not native['passed'] or native['source']!=expected or not fine['passed']:
        raise ValueError('Installed full fine/native qualification failed')
    files={str(p.relative_to(source/'fine')):sha(p) for p in (source/'fine').glob('*.json')}
    actual={str(p.relative_to(out/'installed/fine')):sha(p) for p in (out/'installed/fine').glob('*.json')}
    if files!=actual or len(files)!=6:raise ValueError('Installed full fine scientific JSON differs')
    coarse={str(p.relative_to(source/'coarse')):sha(p) for p in (source/'coarse').glob('*.json')}
    if coarse!=manifest['coarse_json_sha256']:raise ValueError('Coarse inputs changed during installed replay')
    delivery=read(library/'report.json')
    if sha(library/'report.json')!=manifest['library_report_sha256']:raise ValueError('Library evidence changed')
    receipt=dict(passed=True,source=expected,outside_checkout=True,full_installed_fine_s=cycle.duration_s,
        native_prefix_s=native['prefix_s'],native_suffix_s=native['suffix_s'],native_exact=native['prefix_exact'] and native['suffix_and_final_exact'],
        installed_fine_json_count=len(files),installed_fine_json_exact=True,coarse_json_sha256=coarse,
        wheel_sha256=manifest['wheel_sha256'],library_tests=delivery['tests']['counts']['passed'],
        library_report_sha256=manifest['library_report_sha256'],verifier_sha256=manifest['verifier_sha256'],
        source_report_sha256=sha(source/'report.json'),native_report_sha256=sha(out/'installed/native/report.json'),
        ci_required=False,external_runner_required=False,closed_tasks=[],r1_accepted=False,npp_11_02_accepted=False,
        scope='Independent full installed fine trajectory and full coarse native replay; coarse source files are declared inputs, source finer is not presented as installed execution')
    atomic_json(out/'report.json',seal(receipt));return receipt


def run(output,source,library):
    out=Path(output).absolute();source=Path(source).absolute();library=Path(library).absolute()
    if any(p.is_symlink() for p in (out,*out.parents)):raise ValueError('Symlink output refused')
    expected=study.identity();protocol=read(source/'protocol.json');cycle=LoadCycle(**protocol['cycle'])
    if protocol['source']!=expected or not study.inspect_case(source/'coarse',cycle,study.SETTINGS['coarse'])[0]['passed']:
        raise ValueError('Accepted coarse inputs and current source protocol required')
    delivery=read(library/'report.json')
    if not delivery['completed'] or len(delivery['phases'])!=8 or not all(p['passed'] for p in delivery['phases']) or delivery['tests']['gate_failures']:
        raise ValueError('Accepted library delivery required')
    wheels=list((library/'build').glob('*.whl'))
    if len(wheels)!=1:raise ValueError('One wheel required')
    root=Path(__file__).resolve().parents[1]
    with zipfile.ZipFile(wheels[0]) as wheel:
        required=[('scripts/'+p.name,p) for p in [Path(study.__file__),Path(__file__),root/'scripts/qualify_station_load_cycle.py',root/'scripts/qualify_station_hourly_replay.py']]
        required += [('atomic_model_station_load/'+p.name,p) for p in (root/'src/atomic_model_station_load').glob('*.py')]
        if any(hashlib.sha256(wheel.read(n)).hexdigest()!=sha(p) for n,p in required):raise ValueError('Stale full load wheel')
    out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'manifest.json',seal(dict(source=expected,verifier_sha256=sha(__file__),
        wheel_sha256=sha(wheels[0]),library_report_sha256=sha(library/'report.json'),
        coarse_json_sha256={str(p.relative_to(source/'coarse')):sha(p) for p in (source/'coarse').glob('*.json')})))
    with tempfile.TemporaryDirectory(prefix='atomic-full-load-install-') as directory:
        work=Path(directory).resolve()
        if work==root or root in work.parents:raise ValueError('Installed full load work must be outside checkout')
        venv.EnvBuilder(with_pip=True,system_site_packages=True).create(work/'venv')
        python=work/'venv'/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
        for name,command in [('install',[str(python),'-I','-m','pip','install','--no-index','--no-deps',str(wheels[0])]),
            ('study',[str(python),'-I','-c',WORKER,str(out/'installed'),str(source)])]:
            print(name,flush=True)
            with (out/(name+'.log')).open('w') as log:
                subprocess.run(command,cwd=work,env=dict(os.environ),stdout=log,stderr=subprocess.STDOUT,check=True,timeout=14400)
    return finalize(out,source,library)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source-evidence',type=Path,required=True);p.add_argument('--library-evidence',type=Path,required=True);p.add_argument('--finalize-only',action='store_true')
    a=p.parse_args();function=finalize if a.finalize_only else run;function(a.output,a.source_evidence,a.library_evidence)


if __name__=='__main__':main()
