"""Reusable offline isolated delivery for trusted station study entry points."""
import hashlib
import importlib
import os
from pathlib import Path
import tempfile
import venv
import zipfile
from atomic_model_multiphysics.provenance import seal,sha
from atomic_model_training.spec import atomic_json
from scripts.qualify_station_inventory import read
from scripts.station_process import execute

WORKER='''
import importlib,sys
from pathlib import Path
module,package,output,reference=sys.argv[1:]
study=importlib.import_module(module);model=importlib.import_module(package)
assert 'site-packages' in Path(study.__file__).parts
assert 'site-packages' in Path(model.__file__).parts
study.run(Path(output),Path(reference))
'''


def verify(output,source,library,*,module,package,scientific_files,entrypoint):
    out=Path(output).absolute();source=Path(source).absolute();library=Path(library).absolute()
    if out.exists() or any(p.is_symlink() for p in (out,*out.parents)):raise ValueError('Fresh ordinary output required')
    if not module.startswith('scripts.qualify_station_') or not package.startswith('atomic_model_station_'):raise ValueError('Trusted station study names required')
    study=importlib.import_module(module);expected=study.identity();own=sha(__file__);entry=sha(entrypoint)
    result=read(source/'report.json');delivery=read(library/'report.json')
    if not result['passed'] or result['source']!=expected or not result['source_unchanged']:raise ValueError('Accepted source study required')
    if not delivery['completed'] or len(delivery['phases'])!=8 or not all(p['passed'] for p in delivery['phases']) or delivery['tests']['gate_failures']:raise ValueError('Complete local library receipt required')
    wheels=list((library/'build').glob('*.whl'))
    if len(wheels)!=1:raise ValueError('One wheel required')
    root=Path(__file__).resolve().parents[1]
    paths={str(p.relative_to(root/'src')):p for p in (root/'src').rglob('*.py')}
    paths.update({'scripts/'+p.name:p for p in (root/'scripts').glob('*.py')})
    resources={str(p.relative_to(root/'src')):p for p in (root/'src'/package/'data').glob('*.json')}
    hashes={name:sha(path) for name,path in {**paths,**resources}.items()}
    with zipfile.ZipFile(wheels[0]) as wheel:
        for name,digest in hashes.items():
            if hashlib.sha256(wheel.read(name)).hexdigest()!=digest:raise ValueError('Stale study module/resource: '+name)
    inputs={p.name:sha(p) for p in source.glob('*.json')}
    if set(inputs)!=set(scientific_files):raise ValueError('Complete declared scientific JSON set required')
    out.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix='atomic-station-study-') as directory:
        work=Path(directory).resolve()
        if work==root or root in work.parents:raise ValueError('Outside-checkout execution required')
        venv.EnvBuilder(with_pip=True,system_site_packages=True).create(work/'venv')
        python=work/'venv'/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
        execute([str(python),'-I','-m','pip','install','--no-index','--no-deps',str(wheels[0])],cwd=work,log=out/'install.log',timeout_s=300,idle_s=120)
        execute([str(python),'-I','-c',WORKER,module,package,str(out/'installed'),str(source)],cwd=work,log=out/'study.log',timeout_s=600,idle_s=120)
    if inputs!={p.name:sha(p) for p in source.glob('*.json')} or inputs!={p.name:sha(p) for p in (out/'installed').glob('*.json')}:raise ValueError('Scientific JSON changed or installed output differs')
    if expected!=study.identity() or own!=sha(__file__) or entry!=sha(entrypoint):raise ValueError('Study or delivery source changed')
    if hashes!={name:sha(path) for name,path in {**paths,**resources}.items()}:raise ValueError('Source modules changed during delivery')
    report=dict(passed=True,source=expected,harness_sha256=own,entrypoint_sha256=entry,outside_checkout=True,cold_python_isolated=True,
        all_scientific_json_exact=True,scientific_json_count=len(inputs),all_module_bytes_identical=True,module_count=len(paths),packaged_resources_exact=True,
        library_tests=delivery['tests']['counts']['passed'],phases=8,wheel_sha256=sha(wheels[0]),library_report_sha256=sha(library/'report.json'),source_json_sha256=inputs,
        module=module,package=package,closed_tasks=[],station_usable=False,dns_qualified=False,npp_11_03_accepted=False,r1_accepted=False,validated=False,ci_required=False)
    atomic_json(out/'report.json',seal(report));return report
