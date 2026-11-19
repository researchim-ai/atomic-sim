"""Repeat inventory or hot-lifecycle studies from a wheel outside the checkout, offline."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import venv
import zipfile

from atomic_model_multiphysics.provenance import seal, sha
from atomic_model_training.spec import atomic_json
from scripts.qualify_station_inventory import read
from atomic_model_station_campaign.engine import signature


WORKER = '''
import json, sys
from pathlib import Path
import atomic_model_station_inventory as inventory
from atomic_model_station_inventory import model
import importlib
from atomic_model_multiphysics.provenance import seal, sha
from atomic_model_station_campaign.engine import signature
out, reference, receipt = map(Path, sys.argv[1:4])
name=sys.argv[4]
assert name in ('inventory','inventory_neutral','hot_state')
study=importlib.import_module('scripts.qualify_station_'+name)
assert 'site-packages' in Path(inventory.__file__).parts
assert 'site-packages' in Path(model.__file__).parts
assert 'site-packages' in Path(study.__file__).parts
if name=='hot_state':
    import atomic_model_station_hot as hot
    assert 'site-packages' in Path(hot.__file__).parts
result = study.run(out, reference)
assert result['passed'] and result['restart_exact']
receipt.write_text(json.dumps(seal(dict(passed=True, outside_checkout=True,
    source=signature(), driver_sha256=sha(study.__file__), inventory_sha256=sha(model.__file__),
    restart_exact=True, all_scientific_json_exact=True, closed_tasks=[],
    stationary_accepted=False, r1_accepted=False)), sort_keys=True))
'''


def run(output, source, library, *, timeout_s=1800, study_name='inventory'):
    if study_name not in ('inventory','inventory_neutral','hot_state'):raise ValueError('Inventory study name')
    source=Path(source).absolute(); library=Path(library).absolute(); out=Path(output).absolute()
    if type(timeout_s) is not int or not 1<=timeout_s<=7200:
        raise ValueError('Inventory installed timeout domain')
    if any(p.is_symlink() for p in (out,*out.parents)):
        raise ValueError('Symlink output refused')
    report=read(source/'report.json'); protocol=read(source/'protocol.json'); delivery=read(library/'report.json')
    if not report['passed'] or not report['source_unchanged'] or report['closed_tasks'] or protocol['source']!=signature():
        raise ValueError('Source diagnostic study is stale or not accepted')
    if not delivery['completed'] or len(delivery['phases'])!=8 or not all(p['passed'] for p in delivery['phases']) or delivery['tests']['gate_failures']:
        raise ValueError('Local library delivery is not accepted')
    wheels=list((library/'build').glob('*.whl'))
    if len(wheels)!=1:
        raise ValueError('One delivery wheel required')
    wheel=wheels[0]
    # Reject a stale package before running a physical experiment.
    import hashlib
    root=Path(__file__).resolve().parents[1]
    with zipfile.ZipFile(wheel) as archive:
        driver_name='qualify_station_'+study_name+'.py'
        implementation=[('scripts/'+driver_name,Path(__file__).with_name(driver_name))]
        implementation.extend(('atomic_model_station_inventory/'+p.name,p) for p in sorted((root/'src/atomic_model_station_inventory').glob('*.py')))
        if study_name=='hot_state':
            implementation.extend(('atomic_model_station_hot/'+p.name,p) for p in sorted((root/'src/atomic_model_station_hot').glob('*.py')))
        for name, local in implementation:
            if hashlib.sha256(archive.read(name)).hexdigest()!=sha(local):
                raise ValueError('Wheel diagnostic implementation differs: '+name)
    out.mkdir(parents=True,exist_ok=False)
    own_sha=sha(__file__)
    source_hashes={str(p.relative_to(source)):sha(p) for p in source.rglob('*.json')}
    with tempfile.TemporaryDirectory(prefix='atomic-inventory-install-') as directory:
        work=Path(directory).resolve()
        if work==root or root in work.parents:
            raise ValueError('Installed replay must run outside checkout')
        venv.EnvBuilder(with_pip=True,system_site_packages=True).create(work/'venv')
        python=work/'venv'/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
        commands=[('install',[str(python),'-I','-m','pip','install','--no-index','--no-deps',str(wheel)]),
                  ('installed-study',[str(python),'-I','-c',WORKER,str(out/'installed'),str(source),str(out/'worker.json'),study_name]),
                  ('installed-cli',[str(python),'-I','-m','atomic_model_station_inventory','--output',str(out/'cli'),
                                    '--duration-s','.1','--tick-s','.05'])]
        if study_name=='hot_state':
            commands.append(('installed-hot-cli',[str(python),'-I','-m','atomic_model_station_hot',
                '--resume',str(source/'warm-checkpoint.json'),'--output',str(out/'cli-hot')]))
        for name,command in commands:
            print(name,flush=True)
            with (out/(name+'.log')).open('w') as log:
                subprocess.run(command,cwd=work,env=dict(os.environ),stdout=log,stderr=subprocess.STDOUT,
                               check=True,timeout=timeout_s)
    worker=read(out/'worker.json')
    cli=read(out/'cli/report.json')
    if not cli['completed'] or not cli['source_unchanged'] or abs(cli['completed_time_s']-.1)>1e-10:
        raise ValueError('Installed inventory CLI did not complete its physical steps')
    if study_name=='hot_state':
        hot_cli=read(out/'cli-hot/report.json')
        if not hot_cli['hold_passed'] or not hot_cli['source_unchanged'] or (out/'cli-hot/checkpoint.json').read_bytes()!=(source/'coarse-final.json').read_bytes():
            raise ValueError('Installed hot CLI resume differs')
    current_hashes={str(p.relative_to(source)):sha(p) for p in source.rglob('*.json')}
    installed_hashes={str(p.relative_to(out/'installed')):sha(p) for p in (out/'installed').rglob('*.json')}
    if own_sha!=sha(__file__) or source_hashes!=current_hashes or source_hashes!=installed_hashes or worker['source']!=signature():
        raise ValueError('Diagnostic evidence or source changed during installed repeat')
    receipt=dict(worker,study_name=study_name,verifier_sha256=own_sha,wheel_sha256=sha(wheel),scientific_json_count=len(source_hashes),
                 installed_hot_cli_passed=study_name=='hot_state',
                 installed_cli_passed=True,installed_cli_report_sha256=sha(out/'cli/report.json'),
                 source_report_sha256=sha(source/'report.json'),library_report_sha256=sha(library/'report.json'),
                 library_tests=delivery['tests']['counts']['passed'],scientific_evidence_sha256=source_hashes,
                 ci_required=False,remote_execution_required=False)
    atomic_json(out/'report.json',seal(receipt))
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source-evidence',type=Path,required=True)
    parser.add_argument('--library-evidence',type=Path,required=True)
    parser.add_argument('--study',choices=['inventory','inventory_neutral','hot_state'],default='inventory')
    args=parser.parse_args()
    result=run(args.output,args.source_evidence,args.library_evidence,study_name=args.study)
    print(json.dumps(dict(passed=result['passed'],scientific_json_count=result['scientific_json_count'])))


if __name__=='__main__':
    main()
