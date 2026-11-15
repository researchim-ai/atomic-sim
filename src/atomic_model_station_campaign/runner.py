"""Bounded local subprocess campaigns; no shared plant, CI or external runner."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from atomic_model_multiphysics.provenance import seal, unseal
from atomic_model_training.spec import atomic_json
from .spec import Episode,Limits

BOOTSTRAP = '''
import json,sys,resource,os,time
from pathlib import Path
request=Path(sys.argv[1]);output=Path(sys.argv[2]);root=sys.argv[3]
p=json.loads(request.read_text());limits=p['limits']
resource.setrlimit(resource.RLIMIT_CPU,(limits['cpu_s'],limits['cpu_s']))
memory=limits['memory_mb']*1024**2
resource.setrlimit(resource.RLIMIT_AS,(memory,memory))
size=limits['output_mb']*1024**2
resource.setrlimit(resource.RLIMIT_FSIZE,(size,size))
sys.path.insert(0,root)
from atomic_model_station_campaign.spec import Episode
from atomic_model_station_campaign.engine import run_episode
started=time.monotonic()
result=run_episode(Episode.from_payload(p['episode']),p['resume'])
usage=resource.getrusage(resource.RUSAGE_SELF)
payload=dict(scientific=result,execution=dict(pid=os.getpid(),started_monotonic=started,
    ended_monotonic=time.monotonic(),cpu_s=usage.ru_utime+usage.ru_stime,peak_rss_bytes=usage.ru_maxrss*(1 if sys.platform=='darwin' else 1024),
    thread_environment={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
    import_root=root,hard_limits=limits))
output.write_text(json.dumps(payload,sort_keys=True,allow_nan=False))
'''


def _execute(episode,limits,resume):
    started=time.monotonic()
    root=str(Path(__file__).resolve().parents[1])
    with tempfile.TemporaryDirectory(prefix='atomic-npp-episode-') as temp:
        temp=Path(temp)
        request=temp/'request.json';result=temp/'result.json'
        request.write_text(json.dumps(dict(episode=episode.payload(),limits=asdict(limits),resume=resume),allow_nan=False))
        env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        try:
            child=subprocess.run([sys.executable,'-I','-c',BOOTSTRAP,str(request),str(result),root],cwd=temp,
                env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=limits.wall_s,check=False)
        except subprocess.TimeoutExpired:
            return dict(episode=episode.name,accepted=False,status='timed_out',scientific=None,
                        execution=dict(wall_s=time.monotonic()-started,hard_limits=asdict(limits)))
        if child.returncode!=0 or not result.exists():
            return dict(episode=episode.name,accepted=False,status='worker_failed',scientific=None,
                execution=dict(wall_s=time.monotonic()-started,return_code=child.returncode,
                               stderr_tail=child.stderr[-2000:].decode(errors='replace'),hard_limits=asdict(limits)))
        if result.stat().st_size>limits.output_mb*1024**2:raise ValueError('Worker output exceeds budget')
        data=json.loads(result.read_text());science=unseal(data['scientific'])
        data['execution']['wall_s']=time.monotonic()-started
        return dict(episode=episode.name,accepted=science['accepted'],status=science['status'],**data)


def run_campaign(output,episodes,*,limits=None,resumes=None):
    if os.name!='posix':raise ValueError('Hard resource isolation currently requires POSIX')
    limits=limits if limits is not None else Limits()
    if type(limits) is not Limits:raise ValueError('Limits required')
    if type(episodes) not in (tuple,list) or not 1<=len(episodes)<=16 or any(type(e) is not Episode for e in episodes):
        raise ValueError('Campaign episode budget/type')
    episodes=[Episode.from_payload(e.payload()) for e in episodes]
    if len({e.name for e in episodes})!=len(episodes):raise ValueError('Duplicate campaign episode')
    resumes={} if resumes is None else resumes
    if type(resumes) is not dict or not set(resumes)<={e.name for e in episodes}:raise ValueError('Unknown resume target')
    # Validate the complete request before creating any result directory.
    json.dumps(dict(episodes=[e.payload() for e in episodes],resumes=resumes),allow_nan=False)
    output=Path(output).absolute()
    if any(p.is_symlink() for p in (output,*output.parents)):raise ValueError('Symlink output refused')
    output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    with ThreadPoolExecutor(max_workers=limits.workers) as pool:
        futures=[pool.submit(_execute,e,limits,resumes.get(e.name)) for e in episodes]
        results=[f.result() for f in futures]
    for result in results:atomic_json(output/(result['episode']+'.json'),seal(result))
    accepted=all(r['accepted'] for r in results)
    report=dict(completed=accepted,accepted=accepted,requested=len(episodes),results=[dict(episode=r['episode'],
        status=r['status'],accepted=r['accepted']) for r in results],limits=asdict(limits),
        wall_s=time.monotonic()-started,independent_processes=True,external_runner_required=False,
        ci_required=False,full_r1_matrix=False,validated=False)
    atomic_json(output/'report.json',seal(report))
    return report
