"""Offline full SCRAM and short lifecycle wheel repeat outside checkout."""
import argparse
import os
import zipfile
import hashlib
from pathlib import Path
import subprocess
import tempfile
import venv
from atomic_model_station_shutdown import ShutdownCase
from atomic_model_multiphysics.provenance import seal,sha
from atomic_model_training.spec import atomic_json
from scripts.qualify_station_inventory import read
from scripts.qualify_station_shutdown import identity
from scripts.station_process import execute

WORKER='''
import sys
from pathlib import Path
from scripts import qualify_station_shutdown_short as short
from scripts import qualify_station_shutdown as full
from atomic_model_station_shutdown import ShutdownCase
import atomic_model_station_shutdown
assert 'site-packages' in Path(atomic_model_station_shutdown.__file__).parts
assert 'site-packages' in Path(full.__file__).parts
out,source,case=map(Path,sys.argv[1:4]);out.mkdir(parents=True,exist_ok=False)
short.run(out/'short',source)
'''


def run(output,source,full,library,*,timeout_s=7200.,idle_s=600.):
    out=Path(output).absolute();source=Path(source).absolute();full=Path(full).absolute();library=Path(library).absolute()
    if any(p.is_symlink() for p in (out,*out.parents)):raise ValueError('Symlink output refused')
    expected=identity();sr=read(source/'report.json');fr=read(full/'report.json');delivery=read(library/'report.json')
    if not sr['passed'] or not sr['native_exact'] or sr['source']!=expected:raise ValueError('Short shutdown source not accepted')
    p=read(full/'protocol.json');case=ShutdownCase(**p['case'])
    if not fr['passed'] or not fr['completed'] or not fr['source_unchanged'] or p['source']!=expected or case!=ShutdownCase():raise ValueError('Full 600s SCRAM required')
    if not delivery['completed'] or len(delivery['phases'])!=8 or not all(p['passed'] for p in delivery['phases']) or delivery['tests']['gate_failures']:raise ValueError('Library delivery not accepted')
    wheels=list((library/'build').glob('*.whl'))
    if len(wheels)!=1:raise ValueError('One wheel required')
    own_sha=sha(__file__)
    import scripts.qualify_station_shutdown as full_driver
    import scripts.qualify_station_shutdown_short as short_driver
    with zipfile.ZipFile(wheels[0]) as wheel:
        for driver in (full_driver,short_driver):
            if hashlib.sha256(wheel.read('scripts/'+Path(driver.__file__).name)).hexdigest()!=sha(driver.__file__):raise ValueError('Stale shutdown wheel')
    hashes={label+'/'+str(p.relative_to(root)):sha(p) for label,root in [('short',source),('scram',full)] for p in root.rglob('*.json')}
    out.mkdir(parents=True,exist_ok=False)
    with tempfile.TemporaryDirectory(prefix='atomic-shutdown-install-') as directory:
        work=Path(directory).resolve();root=Path(__file__).resolve().parents[1]
        if work==root or root in work.parents:raise ValueError('Installed work must be outside checkout')
        venv.EnvBuilder(with_pip=True,system_site_packages=True).create(work/'venv')
        python=work/'venv'/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
        for name,command in [('install',[str(python),'-I','-m','pip','install','--no-index','--no-deps',str(wheels[0])]),
            ('short-study',[str(python),'-I','-c',WORKER,str(out/'installed'),str(source),str(full)]),
            ('full-scram',[str(python),'-I','-m','scripts.qualify_station_shutdown',
                '--output',str(out/'installed/scram'),'--max-step-s',str(p['max_step_s']),'--rtol',str(p['rtol'])])]:
            print(name,flush=True)
            execute(command,cwd=work,log=out/(name+'.log'),timeout_s=timeout_s,idle_s=idle_s)
    if hashes!={label+'/'+str(p.relative_to(root)):sha(p) for label,root in [('short',source),('scram',full)] for p in root.rglob('*.json')}:raise ValueError('Shutdown input evidence changed')
    if identity()!=expected or sha(__file__)!=own_sha:raise ValueError('Shutdown implementation changed during delivery')
    for label,original in [('short',source),('scram',full)]:
        files={str(p.relative_to(original)):sha(p) for p in original.rglob('*.json')}
        actual={str(p.relative_to(out/'installed'/label)):sha(p) for p in (out/'installed'/label).rglob('*.json')}
        if files!=actual:raise ValueError('Installed shutdown scientific JSON differs')
    receipt=dict(passed=True,source=expected,outside_checkout=True,all_scientific_json_exact=True,
        full_scram_s=600.,full_scram_json_count=len(list(full.glob('*.json'))),short_json_count=len(list(source.rglob('*.json'))),
        short_native_exact=True,library_tests=delivery['tests']['counts']['passed'],library_report_sha256=sha(library/'report.json'),
        wheel_sha256=sha(wheels[0]),verifier_sha256=own_sha,closed_tasks=[],npp_11_03_accepted=False,r1_accepted=False,
        harness_scope='Source qualification harness at verifier_sha256; installed wheel supplies unchanged physics and scientific drivers',
        process_isolation='Short lifecycle and full SCRAM execute in separate fresh Python -I processes',
        scope='Independent installed 600s SCRAM and short native/lifecycle proof; full coastdown is not accepted',ci_required=False,external_runner_required=False)
    atomic_json(out/'report.json',seal(receipt));return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('output','source-evidence','full-evidence','library-evidence'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--timeout-s',type=float,default=7200.);p.add_argument('--idle-s',type=float,default=600.)
    a=p.parse_args();run(a.output,a.source_evidence,a.full_evidence,a.library_evidence,timeout_s=a.timeout_s,idle_s=a.idle_s)


if __name__=='__main__':main()
