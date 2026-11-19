"""Offline full-hour source/wheel requalification after a scientific source change."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import venv

from atomic_model_multiphysics.provenance import seal,sha
from atomic_model_training.spec import atomic_json
from scripts import qualify_station_equilibrium as study
from scripts.qualify_station_inventory import read


WORKER='''
import sys
from pathlib import Path
import atomic_model_station_equilibrium as model
from scripts import qualify_station_equilibrium as study
assert 'site-packages' in Path(model.__file__).parts
assert 'site-packages' in Path(study.__file__).parts
result=study.run(Path(sys.argv[1]),Path(sys.argv[2]))
assert result['passed']
'''


def tree_hashes(root):
    """Hash scientific JSON only; execution logs are deliberately separate."""
    root=Path(root)
    if root.is_symlink() or not root.is_dir():raise ValueError('Ordinary evidence directory required')
    paths=sorted(root.rglob('*'))
    if any(p.is_symlink() for p in paths):raise ValueError('Evidence symlinks refused')
    return {str(p.relative_to(root)):sha(p) for p in paths if p.is_file() and p.suffix=='.json'}


def validate_source(root,*,allow_running=False):
    root=Path(root);math=root/'mathematics';protocol=read(math/'protocol.json')
    if any(protocol.get(k)!=v for k,v in study.identity().items()):raise ValueError('Stale nominal source evidence')
    if protocol.get('duration_s')!=3600. or protocol.get('tick_s')!=5.:
        raise ValueError('Full nominal hour required')
    pending=allow_running and not (math/'report.json').exists()
    report=None if pending else read(math/'report.json')
    if not pending and (report.get('passed') is not True or report.get('source_unchanged') is not True or report.get('duration_s')!=3600.):
        raise ValueError('Accepted full source nominal required')
    for role in (('coarse',) if pending else ('coarse','fine')):
        evidence=read(math/role/'report.json')
        if (evidence.get('passed') is not True or evidence.get('completed') is not True or
            evidence.get('source_unchanged') is not True or evidence.get('duration_s')!=3600. or
            evidence.get('window_count')!=601 or evidence.get('all_600s_windows_ready') is not True):
            raise ValueError('Complete nominal role required: '+role)
        if len(study.rows(math/role))!=721:raise ValueError('Complete nominal trajectory required')
    return report


def launch(command,log,cwd,timeout_s=14400):
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    env.pop('PYTHONPATH',None)
    with Path(log).open('w') as stream:
        subprocess.run(command,cwd=cwd,env=env,stdout=stream,stderr=subprocess.STDOUT,
            check=True,timeout=timeout_s)


def pinned_hashes(source,allow_running):
    hashes=tree_hashes(Path(source)/'mathematics')
    return {k:v for k,v in hashes.items() if not allow_running or k=='protocol.json' or k.startswith('coarse/')}


def run(output,source=None,allow_running_source=False):
    out=Path(output).absolute()
    if out.exists() or any(p.is_symlink() for p in (out,*out.parents)):raise ValueError('Fresh ordinary output required')
    if type(allow_running_source) is not bool or (allow_running_source and source is None):
        raise ValueError('Running-source mode requires an explicit source directory')
    if source is not None:
        source=Path(source).absolute()
        if any(p.is_symlink() for p in (source,*source.parents)):raise ValueError('Source symlinks refused')
        validate_source(source,allow_running=allow_running_source)
    identity=study.identity();driver=sha(__file__);out.mkdir(parents=True)
    checks={};failure=None;wheel_hash=None;hashes=None;source_report=None;count=0
    try:
        if source is None:
            source=out/'source';study.run(source)
        source_report=validate_source(source,allow_running=allow_running_source)
        hashes=pinned_hashes(source,allow_running_source)
        atomic_json(out/'inputs.json',seal(dict(source=identity,source_files_sha256=hashes,
            reused_source_evidence=source!=out/'source',
            running_source_allowed=allow_running_source,
            pinned_scope='protocol and complete coarse only' if allow_running_source else 'all source mathematics',
            installed_execution='Full fine hour; native full coarse prefix and suffix',
            copied_inputs='Coarse mathematical files are declared references, not new execution')))
        root=Path(__file__).resolve().parents[1];build=out/'build'
        launch([sys.executable,'-m','pip','wheel','.', '--no-index','--no-deps','--no-build-isolation',
            '--wheel-dir',str(build)],out/'build.log',root)
        wheels=list(build.glob('*.whl'))
        if len(wheels)!=1:raise ValueError('One built wheel required')
        wheel=wheels[0];wheel_hash=sha(wheel)
        with tempfile.TemporaryDirectory(prefix='station-nominal-requalification-') as directory:
            work=Path(directory).resolve()
            if root==work or root in work.parents:raise ValueError('Installation must be outside checkout')
            venv.EnvBuilder(with_pip=True,system_site_packages=True).create(work/'env')
            python=work/'env'/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
            launch([str(python),'-I','-m','pip','install','--no-index','--no-deps',str(wheel)],out/'install.log',work)
            launch([str(python),'-I','-c',WORKER,str(out/'installed'),str(source)],out/'installed.log',work)
        source_report=validate_source(source)
        full_hashes=tree_hashes(source/'mathematics')
        atomic_json(out/'full-inputs.json',seal(dict(source_files_sha256=full_hashes)))
        checks['source_inputs_unchanged']=hashes==pinned_hashes(source,allow_running_source)
        checks['wheel_unchanged']=wheel_hash==sha(wheel)
        actual=tree_hashes(out/'installed/mathematics')
        checks['all_mathematical_json_exact']=actual==full_hashes
        count=len(full_hashes)
        native=read(out/'installed/native/report.json')
        checks['native_replay']=all(native.get(k) is True for k in
            ('completed','prefix_exact','full_suffix_and_final_exact','source_unchanged')) and native.get('duration_s')==3600.
        checks['installed_fine']=read(out/'installed/mathematics/report.json')['passed'] is True
    except Exception as error:
        failure=dict(type=type(error).__name__,message=str(error))
    checks['source_unchanged']=identity==study.identity() and driver==sha(__file__)
    result=dict(source=identity,driver_sha256=driver,checks=checks,failure=failure,
        passed=failure is None and len(checks)==6 and all(checks.values()),wheel_sha256=wheel_hash,
        mathematical_files_compared=count,source_report=source_report,
        scope='Declared nominal only; full library tests and other R1 scenarios are separate',
        full_library_suite_run=False,full_r1_accepted=False,closed_tasks=[],
        ci_required=False,external_runner_required=False)
    atomic_json(out/'report.json',seal(result))
    if not result['passed']:raise ValueError('Full nominal requalification failed; evidence retained')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source',type=Path,help='Optional completed source hour from the current version')
    parser.add_argument('--allow-running-source',action='store_true',
        help='Start after complete coarse; full source acceptance is still required before final success')
    result=run(**vars(parser.parse_args()))
    print(dict(passed=result['passed'],checks=result['checks']))
