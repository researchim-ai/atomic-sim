"""Offline isolated wheel replay of declared shutdown cases and native suffix.

A reproducible physical refusal is recorded as such, never as horizon acceptance.
Wall time and watchdog receipts are operational metadata outside scientific JSON.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import venv
import zipfile
from atomic_model_multiphysics.provenance import seal,sha
from atomic_model_training.spec import atomic_json
from scripts.qualify_station_inventory import read
from scripts.qualify_station_shutdown import identity
from scripts.station_process import execute


def run(output,cases,library,*,timeout_s=7200.,idle_s=300.,native=False):
    out=Path(output).absolute();paths=[Path(p).absolute() for p in cases];library=Path(library).absolute()
    if not paths or len(set(p.name for p in paths))!=len(paths):raise ValueError('Distinct case names required')
    if any(p.is_symlink() for p in (out,*out.parents)):raise ValueError('Symlink output refused')
    source=identity();own=sha(__file__);delivery=read(library/'report.json')
    if not delivery['completed'] or len(delivery['phases'])!=8 or not all(p['passed'] for p in delivery['phases']) or delivery['tests']['gate_failures']:raise ValueError('Complete local library delivery required')
    specs=[];hashes={}
    for path in paths:
        protocol=read(path/'protocol.json');report=read(path/'report.json')
        if protocol['source']!=source or not report['source_unchanged']:raise ValueError('Source identity differs')
        if not report['passed'] and (report['failure'] is None or not report['rollback_exact']):raise ValueError('Accepted case or exact rejected-step proof required')
        if report['native_prefix_exact']:raise ValueError('Use plain independent source case as input')
        specs.append((path,protocol,report))
        hashes[path.name]={str(p.relative_to(path)):sha(p) for p in path.rglob('*.json')}
    wheels=list((library/'build').glob('*.whl'))
    if len(wheels)!=1:raise ValueError('One wheel required')
    root=Path(__file__).resolve().parents[1]
    with zipfile.ZipFile(wheels[0]) as wheel:
        # All source modules and this delivery harness must correspond to the wheel.
        modules={str(p.relative_to(root/'src')):sha(p) for p in (root/'src').rglob('*.py')}
        modules.update({'scripts/'+p.name:sha(p) for p in (root/'scripts').glob('*.py')})
        for name,digest in modules.items():
            if hashlib.sha256(wheel.read(name)).hexdigest()!=digest:raise ValueError('Stale wheel module: '+name)
    out.mkdir(parents=True,exist_ok=False);results={}
    with tempfile.TemporaryDirectory(prefix='atomic-low-flow-install-') as directory:
        work=Path(directory).resolve()
        if work==root or root in work.parents:raise ValueError('Outside-checkout execution required')
        venv.EnvBuilder(with_pip=True,system_site_packages=True).create(work/'venv')
        python=work/'venv'/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
        execute([str(python),'-I','-m','pip','install','--no-index','--no-deps',str(wheels[0])],cwd=work,log=out/'install.log',timeout_s=timeout_s,idle_s=idle_s)
        for path,protocol,expected in specs:
            target=out/path.name;case=protocol['case']
            command=[str(python),'-I','-m','scripts.qualify_station_shutdown','--output',str(target),
                '--family',case['family'],'--duration-s',str(case['duration_s']),'--film-model',case['film_model'],
                '--max-step-s',str(protocol['max_step_s']),'--rtol',str(protocol['rtol'])]
            if case['tick_s']!=5. or not case['dhr_requested']:raise ValueError('This delivery profile uses 5s ticks and DHR')
            # The wrapper distinguishes an expected model refusal from an OS failure.
            wrapper='import runpy,sys;sys.argv=sys.argv[1:];\ntry:runpy.run_module("scripts.qualify_station_shutdown",run_name="__main__")\nexcept SystemExit as e:\n if e.code not in (0,1):raise\n'
            print('installed',path.name,flush=True)
            execute([str(python),'-I','-c',wrapper,*command[3:]],cwd=work,log=out/(path.name+'.log'),timeout_s=timeout_s,idle_s=idle_s)
            actual={str(p.relative_to(target)):sha(p) for p in target.rglob('*.json')}
            if actual!=hashes[path.name]:raise ValueError('Installed scientific JSON differs: '+path.name)
            result=dict(all_scientific_json_exact=True,json_count=len(actual),horizon_accepted=expected['passed'],
                completed_time_s=expected['completed_time_s'],native_suffix_exact=False)
            if native and expected['passed']:
                print('installed native',path.name,flush=True)
                execute([str(python),'-I','-m','scripts.qualify_station_low_flow','native','--output',str(out/(path.name+'-native')),'--reference',str(path)],
                    cwd=work,log=out/(path.name+'-native.log'),timeout_s=timeout_s,idle_s=idle_s)
                r=read(out/(path.name+'-native')/'report.json')
                if not r['passed'] or not r['suffix_and_final_exact']:raise ValueError('Installed native suffix differs')
                result['native_suffix_exact']=True
            results[path.name]=result
    if source!=identity() or own!=sha(__file__):raise ValueError('Delivery implementation changed')
    for path in paths:
        if hashes[path.name]!={str(p.relative_to(path)):sha(p) for p in path.rglob('*.json')}:raise ValueError('Input evidence changed')
    report=dict(passed=True,source=source,verifier_sha256=own,wheel_sha256=sha(wheels[0]),all_module_bytes_identical=True,
        outside_checkout=True,isolated_fresh_process_per_case=True,cases=results,library_tests=delivery['tests']['counts']['passed'],
        library_report_sha256=sha(library/'report.json'),input_json_sha256=hashes,
        closed_tasks=[],npp_11_03_accepted=False,r1_accepted=False,validated=False,ci_required=False,
        scope='Exact installed repeat of declared horizons/refusals; a refused horizon remains unaccepted')
    atomic_json(out/'report.json',seal(report));return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--cases',nargs='+',type=Path,required=True)
    p.add_argument('--library-evidence',type=Path,required=True);p.add_argument('--native',action='store_true');p.add_argument('--timeout-s',type=float,default=7200.);p.add_argument('--idle-s',type=float,default=300.)
    a=p.parse_args();run(a.output,a.cases,a.library_evidence,timeout_s=a.timeout_s,idle_s=a.idle_s,native=a.native)

if __name__=='__main__':main()
