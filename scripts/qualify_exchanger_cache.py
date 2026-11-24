"""Scoped transition-cache correctness and separately measured local cost."""
import argparse
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import platform
import statistics
import time

import numpy as np
from atomic_model_multiphysics.provenance import seal
from atomic_model_training.spec import atomic_json
from physics.thermal import two_stream_exchanger as base
from physics.thermal import cached_exchanger as cached

MESHES=(1,4,8)
MODES=('counterflow','switching')
CRITERIA=dict(temperature_absolute_k=1e-9,integral_absolute_j=1e-6,integral_relative=1e-10,
    energy_absolute_j=1e-5,power_absolute_w=1e-6)


def source():
    return {name:hashlib.sha256(Path(path).read_bytes()).hexdigest() for name,path in
        (('driver',__file__),('base',base.__file__),('cached',cached.__file__))}


def inputs(n):
    cfg=dict(cells=n,a_cp_j_kg_k=4180.,b_cp_j_kg_k=3900.,a_cell_mass_kg=[10./n]*n,
        b_cell_mass_kg=[12./n]*n,wall_capacity_j_k=[1e5/n]*n,a_wall_ua_w_k=[4000./n]*n,
        b_wall_ua_w_k=[4000./n]*n,ambient_ua_w_k=[0.]*n,ambient_temperature_k=300.,wall_power_w=[0.]*n,
        energy_tolerance_j=CRITERIA['energy_absolute_j'])
    controls=dict(a_mass_flow_kg_s=1.,b_mass_flow_kg_s=-1.,a_inlet_temperature_k=560.,b_inlet_temperature_k=350.)
    initial=dict(mode='given',temperature_k=dict(a=[500.]*n,b=[360.]*n,wall=[430.]*n))
    return cfg,controls,initial


def schedule(mode,initial):
    if mode not in MODES:raise ValueError('Registered cache mode required')
    commands={}
    if mode=='switching':
        commands={4:dict(initial,a_inlet_temperature_k=520.,b_inlet_temperature_k=370.),
            8:dict(initial,a_mass_flow_kg_s=-2.),12:dict(initial),
            16:dict(initial,a_mass_flow_kg_s=0.,b_mass_flow_kg_s=0.)}
    return [dict(dt_s=.1 if i<16 else .2,controls=commands.get(i)) for i in range(20)]


def assessment(cases):
    expected={f'{mode}_{n}' for mode in MODES for n in MESHES}
    if set(cases)!=expected:raise ValueError('Complete cache mesh/control matrix required')
    gates=dict(temperature=True,integrals=True,ledgers=True,ports=True,controls=True,restart=True,
        cache_reuse=True,bounded=True,policy=True)
    temperature_error=integral_error=0.;totals=dict(hits=0,misses=0)
    for name,t in cases.items():
        if len(t['cached'])!=20 or len(t['legacy'])!=20:raise ValueError('Complete cache timeline required')
        gates['restart'] &= t['restart_exact'] is True
        info=t['cache'];gates['cache_reuse'] &= info['hits']>=15 and info['misses']>0
        gates['bounded'] &= info['entries']<=info['capacity']==2 and info['bytes']<=8*1024*1024
        gates['policy'] &= info['physical_states_cached'] is False and info['key_policy']=='Exact dt, matrix and capacity bytes; no rounding'
        totals['hits']+=info['hits'];totals['misses']+=info['misses']
        for a,b in zip(t['cached'],t['legacy']):
            at=np.array(list(a['temperature_k'].values()));bt=np.array(list(b['temperature_k'].values()))
            ae=np.asarray(a['integrated_net_cell_energy_j']);be=np.asarray(b['integrated_net_cell_energy_j'])
            temperature_error=max(temperature_error,float(np.max(abs(at-bt))));integral_error=max(integral_error,float(np.max(abs(ae-be))))
            gates['temperature'] &= np.allclose(at,bt,rtol=0,atol=CRITERIA['temperature_absolute_k'])
            gates['integrals'] &= np.allclose(ae,be,rtol=CRITERIA['integral_relative'],atol=CRITERIA['integral_absolute_j'])
            gates['controls'] &= a['controls']==b['controls'] and a['time_s']==b['time_s']
            for row in (a,b):
                gates['ledgers'] &= abs(row['energy_residual_j'])<=CRITERIA['energy_absolute_j'] and np.max(abs(np.asarray(row['cell_energy_residual_j'])))<=CRITERIA['energy_absolute_j']
                gates['ports'] &= abs(row['port_power_residual_w'])<=CRITERIA['power_absolute_w']
    return dict(passed=all(gates.values()),gates={k:bool(v) for k,v in gates.items()},
        maximum_temperature_difference_k=temperature_error,maximum_cell_integral_difference_j=integral_error,cache_calls=totals)


def negative_controls(cases):
    checks={}
    def reject(name,modify,gate):
        copied=deepcopy(cases);modify(copied['switching_4']);r=assessment(copied);checks[name]=not r['passed'] and not r['gates'][gate]
    reject('wrong_temperature',lambda t:t['cached'][-1]['temperature_k']['a'].__setitem__(0,1000.),'temperature')
    reject('wrong_integral',lambda t:t['cached'][-1]['integrated_net_cell_energy_j'][0].__setitem__(0,0.),'integrals')
    reject('energy_failure',lambda t:t['cached'][-1].__setitem__('energy_residual_j',1.),'ledgers')
    reject('port_failure',lambda t:t['cached'][-1].__setitem__('port_power_residual_w',1.),'ports')
    reject('lost_restart',lambda t:t.__setitem__('restart_exact',False),'restart')
    reject('no_reuse',lambda t:t['cache'].__setitem__('hits',0),'cache_reuse')
    return dict(passed=all(checks.values()),checks=checks,scope='Mutated copies; native traces retained')


def measured_cost():
    rows={}
    for n in (4,16):
        results={};final={}
        for mode in ('legacy','warm','cold'):
            elapsed=[]
            for repeat in range(3):
                cls=base.TwoStreamExchanger if mode=='legacy' else cached.CachedTwoStreamExchanger
                model=cls(*inputs(n));start=time.perf_counter()
                for _ in range(64):
                    if mode=='cold':model.clear_cache()
                    model.advance(.1)
                elapsed.append(time.perf_counter()-start);final[mode]=model.observation()
            results[mode]=dict(wall_seconds=elapsed,median_seconds=statistics.median(elapsed))
        errors=max(float(np.max(abs(np.array(list(final[m]['temperature_k'].values()))-np.array(list(final['legacy']['temperature_k'].values()))))) for m in ('warm','cold'))
        results.update(warm_speed_ratio_vs_legacy=results['legacy']['median_seconds']/results['warm']['median_seconds'],
            maximum_temperature_difference_k=errors)
        rows[str(n)]=results
    return dict(machine=dict(system=platform.system(),machine=platform.machine(),python=platform.python_version(),
        numpy=np.__version__,threads={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')}),
        protocol=dict(steps=64,dt_s=.1,repeats=3,construction_timed=False,warm_first_miss_timed=True),measurements=rows,
        scope='Local sequential warm-cache workload; no universal speed guarantee; timing excluded from scientific replay')


def run(output):
    out=Path(output).absolute()
    if out.exists() or any(p.is_symlink() for p in (out,*out.parents)):raise ValueError('Fresh ordinary output required')
    out.mkdir(parents=True);frozen=source();cases={}
    for mode in MODES:
        for n in MESHES:
            name=f'{mode}_{n}';print(name,flush=True);args=inputs(n)
            a=cached.CachedTwoStreamExchanger(*args,cache_entries=2);b=base.TwoStreamExchanger(*args)
            ca=[];cb=[];saved=None;steps=schedule(mode,args[1])
            for i,step in enumerate(steps):
                if step['controls'] is not None:a.set_controls(step['controls']);b.set_controls(step['controls'])
                ca.append(a.advance(step['dt_s']));cb.append(b.advance(step['dt_s']))
                if i==9:saved=a.snapshot()
            restored=cached.CachedTwoStreamExchanger.from_snapshot(saved);restart=True
            for step,row in zip(steps[10:],ca[10:]):
                if step['controls'] is not None:restored.set_controls(step['controls'])
                restart &= restored.advance(step['dt_s'])==row
            restart &= restored.snapshot()==a.snapshot()
            t=dict(cached=ca,legacy=cb,cache=a.cache_info(),restart_exact=bool(restart),checkpoint=a.snapshot())
            cases[name]=t;atomic_json(out/(name+'.json'),seal(t))
    result=assessment(cases);negative=negative_controls(cases)
    report=dict(passed=result['passed'] and negative['passed'] and frozen==source(),source=frozen,criteria=CRITERIA,
        assessment=result,negative_controls=negative,cached_trajectories=6,legacy_trajectories=6,restart_suffixes=6,
        experimental_validation=False,pwr_connected=False,closed_tasks=[],scope='Constant-property prescribed-flow exchanger only; no PWR or universal performance claim')
    atomic_json(out/'report.json',seal(report));atomic_json(out/'scientific.json',seal(dict(report=report,cases=cases)))
    atomic_json(out/'performance.json',seal(measured_cost()))
    if not report['passed']:raise ValueError('Cache qualification failed; attempts retained')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True,type=Path);print(run(parser.parse_args().output))
