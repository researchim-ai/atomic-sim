"""Exact-key water cache equivalence and paired local timing, in one exclusive process."""
import argparse
from contextlib import contextmanager
from pathlib import Path
import time

from atomic_model_plant_extension import water
from atomic_model_station_env import StationEnv,EnvConfig
from atomic_model_station_env.model import signature
from atomic_model_station_hybrid_heat.scenarios import prepare_relief_cycle
from atomic_model_station import PWRBypassSession
from atomic_model_multiphysics.provenance import seal,sha
from atomic_model_training.spec import atomic_json
from physics.materials import iapws
from scripts.profile_station_matrix import machine


@contextmanager
def cache_mode(enabled):
    """Qualification-only process-global switch; not a concurrent simulation API."""
    original=water._raw
    water._raw_values.cache_clear();water.saturation.cache_clear();iapws.saturation_properties.cache_clear()
    if not enabled:water._raw=water._raw_uncached
    try:yield
    finally:
        water._raw=original
        water._raw_values.cache_clear();water.saturation.cache_clear();iapws.saturation_properties.cache_clear()


def execute(case,enabled):
    with cache_mode(enabled):
        start=time.perf_counter()
        if case=='pressure-cycle':
            session,schedule,_=prepare_relief_cycle()
            commands=schedule;duration=.022
        else:
            session=StationEnv(EnvConfig(max_steps=3,tick_s=.02,automatic=False,rtol=1e-10))
            session.reset();duration=.06
            commands=[{}, {}, {}] if case=='hold' else [
                {'operator':{'feed_pump_on':False}}, {'operator':{'feed_pump_on':True}},{}]
        initialized=time.perf_counter();transitions=[]
        for command in commands:
            transitions.append(session.advance(command['duration_s'],command['patch']) if case=='pressure-cycle' else session.step(command))
        advanced=time.perf_counter();checkpoint=session.checkpoint();saved=time.perf_counter()
        restored=(PWRBypassSession if case=='pressure-cycle' else StationEnv).restore(checkpoint)
        finished=time.perf_counter()
        if restored.checkpoint()!=checkpoint:raise ValueError('Native checkpoint restart differs')
        info=water._raw_values.cache_info()
        return dict(scientific=dict(transitions=transitions,checkpoint=checkpoint),
            cache=dict(hits=info.hits,misses=info.misses,currsize=info.currsize,maxsize=info.maxsize),
            timing=dict(initialize_s=initialized-start,advance_s=advanced-initialized,
                checkpoint_s=saved-advanced,restore_s=finished-saved),duration_s=duration)


def run(output,*,repeats=2):
    if type(repeats) is not int or not 1<=repeats<=5:raise ValueError('Repeats in [1,5] required')
    out=Path(output).absolute()
    if out.exists() or any(p.is_symlink() for p in (out,*out.parents)):raise ValueError('Fresh ordinary output required')
    source=signature();driver=sha(__file__);out.mkdir(parents=True);rows=[]
    for case in ('hold','feed-cycle','pressure-cycle'):
        for repeat in range(repeats):
            order=('uncached','cached') if repeat%2==0 else ('cached','uncached')
            row=dict(case=case,repeat=repeat,order=list(order),passed=False,failure=None,runs={})
            try:
                for mode in order:row['runs'][mode]=execute(case,mode=='cached')
                a=row['runs']['uncached'];b=row['runs']['cached']
                row['checks']=dict(exact_scientific=a['scientific']==b['scientific'],
                    cache_used=b['cache']['hits']>0,uncached_no_raw_entries=a['cache']['currsize']==0)
                row['advance_speedup']=a['timing']['advance_s']/max(b['timing']['advance_s'],1e-12)
                row['passed']=all(row['checks'].values())
            except Exception as error:row['failure']=dict(type=type(error).__name__,message=str(error))
            rows.append(row);atomic_json(out/f'{case}-{repeat}.json',seal(row))
    if source!=signature() or driver!=sha(__file__):raise ValueError('Source changed')
    result=dict(source=source,driver_sha256=driver,machine=machine(),repeats=repeats,
        attempts=len(rows),accepted=sum(r['passed'] for r in rows),failed=sum(not r['passed'] for r in rows),
        passed=all(r['passed'] for r in rows),pair_order_alternated=True,
        caches_cleared_before_each_run=True,closed_tasks=[],
        files={p.name:sha(p) for p in sorted(out.glob('*.json'))})
    atomic_json(out/'report.json',seal(result))
    if not result['passed']:raise ValueError('Water cache qualification failed; attempts retained')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--repeats',type=int,default=2)
    result=run(**vars(parser.parse_args()))
    print({k:result[k] for k in ('attempts','accepted','failed','passed')})
