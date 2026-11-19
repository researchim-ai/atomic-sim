"""Offline isolated wheel repeat of mixed-convection research primitives."""
import argparse
import hashlib
import os
from pathlib import Path
import tempfile
import venv
import zipfile
from atomic_model_multiphysics.provenance import seal,sha
from atomic_model_training.spec import atomic_json
from scripts.qualify_station_inventory import read
from scripts.qualify_station_mixed_convection import identity
from scripts.station_process import execute

WORKER='''
import sys
from pathlib import Path
import atomic_model_station_mixed
from scripts import qualify_station_mixed_convection as study
assert 'site-packages' in Path(atomic_model_station_mixed.__file__).parts
assert 'site-packages' in Path(study.__file__).parts
study.run(Path(sys.argv[1]),Path(sys.argv[2]))
'''


def run(output,source,library):
    out=Path(output).absolute();source=Path(source).absolute();library=Path(library).absolute()
    if out.exists() or any(p.is_symlink() for p in (out,*out.parents)):raise ValueError('Fresh ordinary output required')
    expected=identity();own=sha(__file__);report=read(source/'report.json');delivery=read(library/'report.json')
    if not report['passed'] or report['source']!=expected or not report['source_unchanged']:raise ValueError('Accepted source study required')
    if not delivery['completed'] or len(delivery['phases'])!=8 or not all(p['passed'] for p in delivery['phases']) or delivery['tests']['gate_failures']:raise ValueError('Complete local library receipt required')
    wheels=list((library/'build').glob('*.whl'))
    if len(wheels)!=1:raise ValueError('One wheel required')
    root=Path(__file__).resolve().parents[1]
    modules={str(p.relative_to(root/'src')):sha(p) for p in (root/'src').rglob('*.py')}
    modules.update({'scripts/'+p.name:sha(p) for p in (root/'scripts').glob('*.py')})
    resources={str(p.relative_to(root/'src')):sha(p) for p in (root/'src/atomic_model_station_mixed/data').glob('*.json')}
    with zipfile.ZipFile(wheels[0]) as wheel:
        for name,digest in {**modules,**resources}.items():
            if hashlib.sha256(wheel.read(name)).hexdigest()!=digest:raise ValueError('Stale installed study module/resource: '+name)
    inputs={p.name:sha(p) for p in source.glob('*.json')}
    if len(inputs)!=5:raise ValueError('Five complete source scientific JSON required')
    out.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix='atomic-mixed-install-') as directory:
        work=Path(directory).resolve()
        if work==root or root in work.parents:raise ValueError('Outside-checkout execution required')
        venv.EnvBuilder(with_pip=True,system_site_packages=True).create(work/'venv')
        python=work/'venv'/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
        execute([str(python),'-I','-m','pip','install','--no-index','--no-deps',str(wheels[0])],cwd=work,log=out/'install.log',timeout_s=300,idle_s=120)
        execute([str(python),'-I','-c',WORKER,str(out/'installed'),str(source)],cwd=work,log=out/'study.log',timeout_s=300,idle_s=120)
    if inputs!={p.name:sha(p) for p in source.glob('*.json')}:raise ValueError('Source evidence changed')
    if inputs!={p.name:sha(p) for p in (out/'installed').glob('*.json')}:raise ValueError('Installed scientific JSON differs')
    if expected!=identity() or own!=sha(__file__):raise ValueError('Study source changed during delivery')
    result=dict(passed=True,source=expected,verifier_sha256=own,outside_checkout=True,cold_python_isolated=True,
        all_scientific_json_exact=True,scientific_json_count=len(inputs),all_module_bytes_identical=True,module_count=len(modules),
        packaged_resources_exact=True,library_tests=delivery['tests']['counts']['passed'],phases=len(delivery['phases']),
        wheel_sha256=sha(wheels[0]),library_report_sha256=sha(library/'report.json'),source_json_sha256=inputs,
        closed_tasks=[],candidate_station_usable=False,npp_11_03_accepted=False,r1_accepted=False,validated=False,ci_required=False)
    atomic_json(out/'report.json',seal(result));return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('output','source-evidence','library-evidence'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.output,a.source_evidence,a.library_evidence)

if __name__=='__main__':main()
