"""Offline source/wheel station physics replay (primary, film, steam, stages or bypass); no CI or external runner.

By default run the library profile and primary study first. Existing sealed
runs may be reused explicitly with --library-evidence and --source-evidence.
The installed study always executes anew outside the checkout.
"""
import argparse
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import venv

from atomic_model_multiphysics.provenance import seal, sha, unseal
from atomic_model_training.spec import atomic_json
from scripts.verify_local_library import verify

ROOT = Path(__file__).resolve().parents[1]
WORKER = '''
import json, sys, os
import numpy, scipy
from pathlib import Path
import atomic_model_station as station
from atomic_model_station.core import core_signature
from atomic_model_multiphysics.provenance import seal, sha
import importlib
out, reference, receipt, driver = map(Path, sys.argv[1:5])
name = sys.argv[5]
assert name in ('primary', 'film', 'steam', 'stages', 'bypass')
study = importlib.import_module('scripts.qualify_station_' + name)
assert 'site-packages' in Path(station.__file__).parts
assert 'site-packages' in Path(study.__file__).parts
assert sha(study.__file__) == str(driver)
result = study.run(out, reference)
assert result['passed']
payload = dict(passed=True, installed_import=True, source_driver_equal=True,
    station_source=core_signature(), all_evidence_exact=True,
    source_driver_sha256=sha(study.__file__), restart_exact=result['restart_exact'],
    numerical_runtime=dict(python=sys.version, numpy=numpy.__version__, scipy=scipy.__version__,
        thread_environment={k: os.environ.get(k) for k in
            ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')}))
receipt.write_text(json.dumps(seal(payload), indent=2) + '\\n')
'''


def read(path):
    return unseal(json.loads(Path(path).read_text()))


def run(output, *, source_evidence=None, library_evidence=None, timeout_s=7200, study_name="primary"):
    if study_name not in ("primary", "film", "steam", "stages", "bypass"):
        raise ValueError("Study must be primary, film, steam, stages or bypass")
    study = importlib.import_module("scripts.qualify_station_" + study_name)
    if type(timeout_s) is not int or not 1 <= timeout_s <= 14400:
        raise ValueError('Timeout domain: 1..14400 seconds')
    output = Path(output).absolute()
    if any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError('Symlink output refused')
    output.mkdir(parents=True, exist_ok=False)
    verifier_sha = sha(__file__)
    source = Path(source_evidence).absolute() if source_evidence else output / 'source'
    library = Path(library_evidence).absolute() if library_evidence else output / 'library'
    if source_evidence is None:
        study.run(source)
    if library_evidence is None:
        verify(library, suite='library')
    result = read(source / 'report.json')
    protocol = read(source / 'protocol.json')
    delivery = read(library / 'report.json')
    if not (result['passed'] and result['source_unchanged']
            and result['closed_tasks'] == study.TASKS):
        raise ValueError('Source station study not accepted')
    if not (delivery['completed'] and len(delivery['phases']) == 8
            and not delivery['tests']['gate_failures']
            and delivery['tests']['counts']['failed'] == 0
            and delivery['tests']['counts']['errors'] == 0):
        raise ValueError('Local library delivery not accepted')
    from atomic_model_station.core import core_signature
    expected = core_signature()
    identity_name={'primary':'core_signature','film':'film_signature','steam':'steam_signature','stages':'stages_signature','bypass':'bypass_signature'}[study_name]
    qualification_source=getattr(study,identity_name)()
    if protocol['source'] != qualification_source or protocol['driver_sha256'] != sha(study.__file__):
        raise ValueError('Stale source evidence')
    wheels = list((library / 'build').glob('*.whl'))
    if len(wheels) != 1:
        raise ValueError('Expected one delivery wheel')
    wheel = wheels[0]
    # Preserve the source process's numerical runtime policy for bitwise replay.
    # Forcing a different BLAS thread count here can change roundoff.
    environment = dict(os.environ)
    with tempfile.TemporaryDirectory(prefix='atomic-primary-install-') as temp:
        temp = Path(temp).resolve()
        if temp == ROOT or ROOT in temp.parents:
            raise ValueError('Installed replay must run outside checkout')
        venv.EnvBuilder(with_pip=True, system_site_packages=True).create(temp / 'venv')
        python = temp / 'venv' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
        for name, command in [
            ('install', [str(python), '-I', '-m', 'pip', 'install', '--no-index', '--no-deps', str(wheel)]),
            ('installed-study', [str(python), '-I', '-c', WORKER, str(output / 'installed'),
                str(source), str(output / 'worker.json'), sha(study.__file__), study_name])]:
            print(name, flush=True)
            with (output / (name + '.log')).open('w') as log:
                subprocess.run(command, cwd=temp, env=environment, stdout=log,
                               stderr=subprocess.STDOUT, check=True, timeout=timeout_s)
    worker = read(output / 'worker.json')
    if worker['station_source'] != expected or expected != core_signature() or verifier_sha != sha(__file__):
        raise ValueError('Installed/source implementation differs or changed during verification')
    receipt = dict(worker, study_name=study_name, outside_checkout=True, wheel_sha256=sha(wheel),
        verifier_sha256=verifier_sha, source_report_sha256=sha(source / 'report.json'),
        library_report_sha256=sha(library / 'report.json'),
        library_tests=delivery['tests']['counts']['passed'],
        nominal_station_replayed_s=result['nominal_duration_s'], warmup_station_replayed_s=result.get('warmup_duration_s',0.), coastdown_replayed_s=result.get('coastdown_duration_s'),
        natural_study_replayed_s=result.get('natural_duration_s'), trip_replayed_s=result.get('trip_duration_s'), zero_head_from_s=60. if study_name == 'primary' else None,
        full_long_checkpoint_replay=False, complete_package_b=False,
        complete_package_c=False, r1_accepted=False, validated=False, plant_referenced=False)
    atomic_json(output / 'report.json', seal(receipt))
    print(json.dumps(dict(passed=True, output=str(output))), flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--study', choices=['primary', 'film', 'steam', 'stages', 'bypass'], default='primary')
    parser.add_argument('--source-evidence', type=Path)
    parser.add_argument('--library-evidence', type=Path)
    parser.add_argument('--timeout-s', type=int, default=7200)
    args = parser.parse_args()
    run(args.output, source_evidence=args.source_evidence,
        library_evidence=args.library_evidence, timeout_s=args.timeout_s, study_name=args.study)


if __name__ == '__main__':
    main()
