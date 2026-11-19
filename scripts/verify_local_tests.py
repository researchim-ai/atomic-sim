"""Resumable source-bound pytest batches. Completed results never cross source epochs."""
import argparse
import hashlib
from importlib.metadata import distributions
import os
import platform
from pathlib import Path
import subprocess
import sys
import time
import uuid

from atomic_model_multiphysics.provenance import seal,unseal,sha
from atomic_model_training.spec import atomic_json,read_json
from atomic_model_workspace.common import digest
from scripts import verify_local_library as local

ROOT=Path(__file__).resolve().parents[1]


def snapshot():
    versions={}
    for distribution in distributions():
        name=(distribution.metadata.get('Name') or '').lower().replace('_','-')
        if name and name!='atomic-sim':versions.setdefault(name,set()).add(distribution.version)
    runtime=dict(python=sys.version,platform=platform.platform(),machine=platform.machine(),
        distributions={k:sorted(v) for k,v in sorted(versions.items())},
        process_environment=dict(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONHASHSEED='0'))
    return dict(source=local.source_manifest(ROOT),evidence=local.evidence_manifest(ROOT),
        driver_sha256=sha(__file__),verifier_sha256=sha(local.__file__),runtime=runtime)


def same_snapshot(expected):
    now=snapshot()
    return (all(now[k]==expected[k] for k in ('source','driver_sha256','verifier_sha256','runtime'))
        and local.evidence_unchanged(expected['evidence'],ROOT))


def _ordinary(path):
    path=Path(path).absolute()
    if any(p.is_symlink() for p in (path,*path.parents)):raise ValueError('Ordinary output path required')
    return path


def _environment():
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONHASHSEED='0')
    for name in ('PYTHONPATH','PYTHONHOME','PYTEST_ADDOPTS','PYTEST_PLUGINS'):env.pop(name,None)
    return env


def _pytest(arguments,log,timeout_s):
    if type(timeout_s) is not int or not 1<=timeout_s<=14400:raise ValueError('Timeout in 1..14400 required')
    with Path(log).open('xb') as stream:
        try:
            return subprocess.run([sys.executable,'-m','pytest',*map(str,arguments),'-q','-p',
                'atomic_model_release.pytest_gate'],cwd=ROOT,env=_environment(),
                stdout=stream,stderr=subprocess.STDOUT,timeout=timeout_s).returncode
        except subprocess.TimeoutExpired:return 124


def prepare(output,*,suite='library',batch_files=4,timeout_s=300):
    if type(batch_files) is not int or not 1<=batch_files<=16:raise ValueError('One to sixteen files per batch required')
    chosen=local.selection(suite);out=_ordinary(output)
    if out.exists():raise ValueError('Fresh plan directory required')
    watched=('src','scripts','data','scenarios','validation','requirements','tests','examples','configs','docs/benchmarks')
    if out==ROOT or any(out==ROOT/x or ROOT/x in out.parents for x in watched):
        raise ValueError('Plan must be outside source and validation input trees')
    before=snapshot();out.mkdir(parents=True)
    code=_pytest([*chosen,'--collect-only','--atomic-nodeids-file',out/'collection.json'],out/'collection.log',timeout_s)
    if code:raise ValueError('Test collection failed; log retained')
    ids=read_json(out/'collection.json')
    if type(ids) is not list or not ids or len(set(ids))!=len(ids) or any(type(x) is not str or not x.startswith('tests/') or '::' not in x for x in ids):
        raise ValueError('Unique nonempty test collection required')
    files=list(dict.fromkeys(x.split('::',1)[0] for x in ids))
    batches=[]
    for start in range(0,len(files),batch_files):
        selected=set(files[start:start+batch_files]);batches.append([x for x in ids if x.split('::',1)[0] in selected])
    if not same_snapshot(before):raise ValueError('Source or evidence changed during collection')
    plan=dict(format='local-test-batches/v1',suite=suite,selection=chosen,batch_files=batch_files,
        snapshot=before,nodeids=ids,batches=batches,collection_sha256=sha(out/'collection.json'))
    atomic_json(out/'plan.json',seal(plan))
    return plan


def _plan(output,*,suite=None):
    out=_ordinary(output);p=unseal(read_json(out/'plan.json'))
    if p.get('format')!='local-test-batches/v1' or (suite is not None and p.get('suite')!=suite):raise ValueError('Matching declared test profile required')
    if p['selection']!=local.selection(p['suite']) or not same_snapshot(p['snapshot']):raise ValueError('Test plan source/evidence epoch changed')
    if sha(out/'collection.json')!=p['collection_sha256'] or read_json(out/'collection.json')!=p['nodeids']:
        raise ValueError('Original collection changed')
    flat=[x for batch in p['batches'] for x in batch]
    if len(flat)!=len(set(flat)) or set(flat)!=set(p['nodeids']) or any(not batch for batch in p['batches']):
        raise ValueError('Exact nonoverlapping batch coverage required')
    return out,p


def _counts(receipt,ids,*,suite):
    if (type(receipt) is not dict or type(receipt.get('schema_version')) is not int or receipt['schema_version']!=1
        or type(receipt.get('exit_code')) is not int):raise ValueError('Known typed pytest receipt schema required')
    expected={'passed','failed','skipped','errors','collected'}
    c=receipt.get('counts')
    if type(c) is not dict or set(c)!=expected or any(type(v) is not int or v<0 for v in c.values()):raise ValueError('Complete typed pytest counts required')
    if receipt.get('exit_code')!=0 or receipt.get('gate_failures')!=[] or c['failed'] or c['errors'] or c['collected']!=len(ids):
        raise ValueError('Successful complete pytest execution required')
    if c['passed']+c['skipped']!=len(ids) or c['passed']==0 or (suite=='library' and c['skipped']):
        raise ValueError('Every required test must execute; library skips refused')
    return c


def _accepted(out,p,index):
    folder=out/f'batch-{index:03d}';r=unseal(read_json(folder/'receipt.json'))
    if r['plan_sha256']!=digest(p) or r['index']!=index or r['nodeids_sha256']!=digest(p['batches'][index]):
        raise ValueError('Batch belongs to a different plan')
    attempt=r['attempt']
    if type(attempt) is not str or not attempt.startswith('attempt-') or len(attempt)!=40 or any(c not in '0123456789abcdef' for c in attempt[8:]):
        raise ValueError('Bounded batch attempt identity required')
    work=folder/attempt
    for name,h in r['files_sha256'].items():
        if name not in ('pytest.log','nodeids.json','pytest.json') or sha(work/name)!=h:raise ValueError('Batch artifact changed')
    if set(r['files_sha256'])!={'pytest.log','nodeids.json','pytest.json'}:raise ValueError('Complete batch artifacts required')
    if read_json(work/'nodeids.json')!=p['batches'][index] or read_json(work/'pytest.json')!=r['pytest']:
        raise ValueError('Batch collection/result changed')
    _counts(r['pytest'],p['batches'][index],suite=p['suite'])
    return r


def run_batch(output,index,*,timeout_s=1200):
    out,p=_plan(output)
    if type(index) is not int or not 0<=index<len(p['batches']):raise ValueError('Known batch index required')
    folder=out/f'batch-{index:03d}'
    if (folder/'receipt.json').exists():return _accepted(out,p,index)
    folder.mkdir(exist_ok=True);work=folder/('attempt-'+uuid.uuid4().hex);work.mkdir()
    arguments=[*p['batches'][index],'--atomic-require-executed','--atomic-receipt',work/'pytest.json',
        '--atomic-nodeids-file',work/'nodeids.json']
    if p['suite']=='library':arguments.append('--atomic-no-skips')
    start=time.monotonic();code=_pytest(arguments,work/'pytest.log',timeout_s)
    if code:raise ValueError(f'Batch {index} failed or interrupted ({code}); attempt retained')
    if read_json(work/'nodeids.json')!=p['batches'][index]:raise ValueError('Executed collection differs from plan')
    receipt=read_json(work/'pytest.json');_counts(receipt,p['batches'][index],suite=p['suite'])
    if not same_snapshot(p['snapshot']):raise ValueError('Source/evidence changed during test batch')
    result=dict(plan_sha256=digest(p),index=index,attempt=work.name,nodeids_sha256=digest(p['batches'][index]),
        pytest=receipt,wall_s=time.monotonic()-start,
        files_sha256={name:sha(work/name) for name in ('pytest.log','nodeids.json','pytest.json')})
    atomic_json(folder/'receipt.json',seal(result))
    return result


def accept_batches(output,*,suite):
    out,p=_plan(output,suite=suite);rows=[_accepted(out,p,i) for i in range(len(p['batches']))]
    attempts=0
    for i,row in enumerate(rows):
        for attempt in (out/f'batch-{i:03d}').glob('attempt-*'):
            attempts+=1
            if attempt.name==row['attempt'] or not (attempt/'pytest.json').exists():continue
            try:previous=read_json(attempt/'pytest.json')
            except (ValueError,OSError):continue # interrupted partial receipt; raw attempt is retained
            c=previous.get('counts',{}) if type(previous) is dict else {}
            if c.get('failed',0) or c.get('errors',0):
                raise ValueError('Earlier test failure cannot be hidden by a successful retry')
    counts={name:sum(r['pytest']['counts'][name] for r in rows) for name in ('passed','failed','skipped','errors','collected')}
    _counts(dict(schema_version=1,counts=counts,exit_code=0,gate_failures=[]),p['nodeids'],suite=suite)
    return dict(schema_version=1,counts=counts,exit_code=0,gate_failures=[],
        external_or_physical_validation_claimed=False,execution_mode='source-bound resumable batches',
        plan_sha256=digest(p),nodeids_sha256=digest(p['nodeids']),batch_count=len(rows),
        total_attempts=attempts,nonaccepted_attempts_retained=attempts-len(rows),
        total_test_wall_s=sum(r['wall_s'] for r in rows),
        wall_time_scope='Sum of accepted pytest batch process durations, not parallel makespan',
        batch_receipts_sha256=[sha(out/f'batch-{i:03d}/receipt.json') for i in range(len(rows))],
        reused_completed_test_executions=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--suite',choices=['library','portable','all'],default='library')
    parser.add_argument('--batch-files',type=int,default=4);parser.add_argument('--timeout-s',type=int,default=1200)
    action=parser.add_mutually_exclusive_group();action.add_argument('--prepare',action='store_true');action.add_argument('--batch',type=int);action.add_argument('--accept',action='store_true')
    args=parser.parse_args()
    if args.prepare:
        p=prepare(args.output,suite=args.suite,batch_files=args.batch_files,timeout_s=args.timeout_s);print(dict(batches=len(p['batches']),tests=len(p['nodeids'])));return
    if args.batch is not None:
        r=run_batch(args.output,args.batch,timeout_s=args.timeout_s);print(dict(batch=args.batch,passed=r['pytest']['counts']['passed']));return
    if not args.accept:
        if not (args.output/'plan.json').exists():prepare(args.output,suite=args.suite,batch_files=args.batch_files,timeout_s=args.timeout_s)
        _,p=_plan(args.output,suite=args.suite)
        for i in range(len(p['batches'])):run_batch(args.output,i,timeout_s=args.timeout_s)
    result=accept_batches(args.output,suite=args.suite)
    atomic_json(args.output/'summary.json',seal(result));print(result['counts'])


if __name__=='__main__':main()
