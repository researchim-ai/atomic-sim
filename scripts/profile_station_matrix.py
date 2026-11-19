"""Offline station cost matrix with exact profiled/unprofiled equivalence."""
import argparse
import cProfile
from copy import deepcopy
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import pstats
import time

from atomic_model_station_env import StationEnv,EnvConfig
from atomic_model_station_env.model import signature
from atomic_model_multiphysics.provenance import sha,seal
from atomic_model_training.spec import atomic_json

CASES=(
    ('hold',({},{})),
    ('feed-cycle',({'operator':{'feed_pump_on':False}},{'operator':{'feed_pump_on':True}},{})),
)


def category(filename,name):
    path=filename.replace('\\','/')
    if '/json/' in path or name in ('dumps','encode','iterencode'):return 'serialization'
    if name in ('jac','num_jac','_dense_num_jac','_sparse_num_jac','_validate_jac'):return 'jacobian'
    if name in ('solve_collocation_system','solve_bdf_system'):return 'nonlinear_solver'
    if name in ('lu_factor','lu_solve','splu','solve_lu'):return 'linear_algebra'
    if name in ('event','_custom_root','_resume_stop','solve_event_equation','handle_events','find_active_events'):
        return 'events'
    if '/eos.py' in path or '/iapws' in path or '/thermo/' in path:return 'eos'
    return 'other'


def summarize(stats):
    groups={};rows=[]
    for (filename,line,name),(primitive,calls,own,cumulative,_) in stats.items():
        group=category(filename,name)
        item=groups.setdefault(group,dict(own_s=0.,calls=0))
        item['own_s']+=own;item['calls']+=calls
        # Basename rather than absolute checkout/user directory in public evidence.
        rows.append(dict(file=Path(filename).name,line=line,function=name,category=group,
            primitive_calls=primitive,calls=calls,own_s=own,cumulative_s=cumulative))
    rows.sort(key=lambda r:(-r['own_s'],r['file'],r['line'],r['function']))
    return dict(groups=groups,total_own_s=sum(r['own_s'] for r in rows),functions=rows,
        aggregation='exclusive own time; cumulative rows must not be summed',
        call_counts_are_not_nonlinear_iteration_counts=True)


def measure(call,profiled):
    profiler=cProfile.Profile() if profiled else None
    started=time.perf_counter()
    if profiler:profiler.enable()
    try:value=call()
    finally:
        if profiler:profiler.disable()
    elapsed=time.perf_counter()-started
    return value,dict(wall_s=elapsed,profile=summarize(pstats.Stats(profiler).stats) if profiler else None)


def machine():
    cpu=platform.processor() or 'unreported'
    cpuinfo=Path('/proc/cpuinfo')
    if cpuinfo.is_file():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith('model name'):
                cpu=line.split(':',1)[1].strip();break
    return dict(system=platform.system(),release=platform.release(),architecture=platform.machine(),
        cpu=cpu,logical_cpus=os.cpu_count(),
        affinity_cpus=len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
        python=platform.python_version(),numpy=version('numpy'),scipy=version('scipy'),
        threads={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
        shared_machine_load_controlled=False)


def execute(actions,profiled):
    phases={};env=StationEnv(EnvConfig(max_steps=len(actions),tick_s=.02,automatic=False,rtol=1e-10))
    _,phases['initialize']=measure(env.reset,profiled)
    transitions,phases['advance']=measure(lambda:[env.step(deepcopy(a)) for a in actions],profiled)
    checkpoint,phases['checkpoint']=measure(env.checkpoint,profiled)
    serialized,phases['serialize']=measure(lambda:json.dumps(checkpoint,sort_keys=True,allow_nan=False),profiled)
    restored,phases['restore']=measure(lambda:StationEnv.restore(json.loads(serialized)),profiled)
    if restored.checkpoint()!=checkpoint:raise ValueError('Checkpoint continuation differs')
    return dict(transitions=transitions,checkpoint=checkpoint,phases=phases,
        serialization_bytes=len(serialized.encode()),
        restore_semantics='native replay of stored actions, not direct memory load')


def run(output,*,repeats=2):
    if type(repeats) is not int or not 1<=repeats<=5:raise ValueError('Repeats in [1,5] required')
    out=Path(output).absolute()
    if out.exists() or any(p.is_symlink() for p in (out,*out.parents)):
        raise ValueError('Fresh ordinary output required')
    source=signature();driver=sha(__file__);out.mkdir(parents=True);rows=[]
    for name,actions in CASES:
        for repeat in range(repeats):
            row=dict(case=name,repeat=repeat,actions=list(actions),passed=False,failure=None)
            try:
                baseline=execute(actions,False);profiled=execute(actions,True)
                row['checks']=dict(transitions=baseline['transitions']==profiled['transitions'],
                    checkpoint=baseline['checkpoint']==profiled['checkpoint'])
                row['scientific']=dict(checkpoint_sha256=baseline['checkpoint']['sha256'],
                    transitions=baseline['transitions'])
                duration=len(actions)*.02
                row.update(baseline=baseline['phases'],profiled=profiled['phases'],
                    duration_s=duration,serialization_bytes=profiled['serialization_bytes'],
                    restore_semantics=profiled['restore_semantics'],
                    baseline_wall_per_simulated_s=baseline['phases']['advance']['wall_s']/duration,
                    profiler_slowdown=profiled['phases']['advance']['wall_s']/max(baseline['phases']['advance']['wall_s'],1e-12))
                row['passed']=all(row['checks'].values())
            except Exception as error:
                row['failure']=dict(type=type(error).__name__,message=str(error))
            rows.append(row)
            atomic_json(out/f'{name}-{repeat}.json',seal(row))
    if source!=signature() or driver!=sha(__file__):raise ValueError('Source changed')
    passed=sum(r['passed'] for r in rows)
    report=dict(source=source,driver_sha256=driver,machine=machine(),repeats=repeats,
        attempts=len(rows),accepted=passed,failed=len(rows)-passed,passed=passed==len(rows),
        warmup_discarded=False,execution_order='case then repeat; baseline before profiled',
        files={p.name:sha(p) for p in sorted(out.glob('*.json'))},closed_tasks=[])
    atomic_json(out/'report.json',seal(report))
    if not report['passed']:raise ValueError('Cost matrix failed; every attempt retained')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--repeats',type=int,default=2)
    result=run(**vars(parser.parse_args()))
    print({k:result[k] for k in ('attempts','accepted','failed','passed')})
