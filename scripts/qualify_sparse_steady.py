"""Exact sparse factor reuse for steady exchanger inlet families."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import time

import numpy as np
import scipy
from atomic_model_multiphysics.provenance import seal
from atomic_model_training.spec import atomic_json
from core import sparse_linear
from physics.thermal import two_stream_exchanger as base
from physics.thermal import sparse_steady_exchanger as sparse
from scripts import qualify_exchanger_cache as thermal

MESHES=(1,4,16)
MODES=('counterflow','reverse','anchored_stop')
INLETS=((560.,350.),(520.,370.),(540.,360.))
CRITERIA=dict(temperature_absolute_k=1e-9,cell_energy_absolute_j=1e-5,steady_power_absolute_w=1e-6,backward_error=1e-12)


def source():
    return {name:hashlib.sha256(Path(path).read_bytes()).hexdigest() for name,path in
        (('driver',__file__),('linear',sparse_linear.__file__),('sparse',sparse.__file__),('base',base.__file__),('recipe',thermal.__file__))}


def inputs(n,mode):
    if mode not in MODES:raise ValueError('Registered steady mode required')
    cfg,boundary,_=thermal.inputs(n)
    if mode=='reverse':boundary.update(a_mass_flow_kg_s=-1.,b_mass_flow_kg_s=1.)
    if mode=='anchored_stop':
        boundary.update(a_mass_flow_kg_s=0.,b_mass_flow_kg_s=0.);cfg['ambient_ua_w_k']=[20./n]*n
    return cfg,boundary,dict(mode='steady')


def assess(cases):
    if set(cases)!={f'{mode}_{n}' for mode in MODES for n in MESHES}:raise ValueError('Complete steady family matrix required')
    gates=dict(temperature=True,initialization=True,ledgers=True,ports=True,controls=True,factor_reuse=True,
        residual_guard=True,bounded=True,restart=True)
    error=0.;max_steady=0.;total_hits=total_factors=0
    for case,t in cases.items():
        if len(t['initializations'])!=3 or len(t['sparse'])!=2 or len(t['dense'])!=2:raise ValueError('Three inlets and complete transient suffix required')
        info=t['cache'];total_hits+=info['hits'];total_factors+=info['factorizations']
        gates['factor_reuse'] &= info['hits']==2 and info['factorizations']==1 and info['misses']==1
        gates['residual_guard'] &= 0<=info['last_backward_error']<=CRITERIA['backward_error']
        gates['bounded'] &= info['entries']==1 and info['bytes']<=info['max_bytes']
        gates['restart'] &= t['restart_exact'] is True
        pairs=[]
        for index,initial in enumerate(t['initializations']):
            pairs.append((initial['sparse'],initial['dense']))
            power=np.max(abs(np.asarray(initial['steady_residual_w'])));max_steady=max(max_steady,float(power))
            gates['initialization'] &= power<=CRITERIA['steady_power_absolute_w']
            gates['controls'] &= initial['sparse']['controls']['a_inlet_temperature_k']==INLETS[index][0] and initial['sparse']['controls']['b_inlet_temperature_k']==INLETS[index][1]
        pairs.extend(zip(t['sparse'],t['dense']))
        for a,b in pairs:
            at=np.asarray(list(a['temperature_k'].values()));bt=np.asarray(list(b['temperature_k'].values()))
            error=max(error,float(np.max(abs(at-bt))));gates['temperature'] &= np.max(abs(at-bt))<=CRITERIA['temperature_absolute_k']
            gates['controls'] &= a['controls']==b['controls'] and a['time_s']==b['time_s']
            for row in (a,b):
                gates['ledgers'] &= abs(row['energy_residual_j'])<=CRITERIA['cell_energy_absolute_j'] and np.max(abs(np.asarray(row['cell_energy_residual_j'])))<=CRITERIA['cell_energy_absolute_j']
                gates['ports'] &= abs(row['port_power_residual_w'])<=CRITERIA['steady_power_absolute_w']
    return dict(passed=all(gates.values()),gates={k:bool(v) for k,v in gates.items()},
        maximum_temperature_difference_k=error,maximum_steady_cell_residual_w=max_steady,hits=total_hits,factorizations=total_factors)


def negative_controls(cases):
    checks={}
    def reject(name,change,gate):
        copied=deepcopy(cases);change(copied['counterflow_4']);r=assess(copied);checks[name]=not r['passed'] and not r['gates'][gate]
    reject('wrong_temperature',lambda t:t['sparse'][-1]['temperature_k']['a'].__setitem__(0,1000.),'temperature')
    reject('steady_residual_lost',lambda t:t['initializations'][0]['steady_residual_w'][0].__setitem__(0,1.),'initialization')
    reject('no_reuse',lambda t:t['cache'].__setitem__('factorizations',3),'factor_reuse')
    reject('bad_linear_residual',lambda t:t['cache'].__setitem__('last_backward_error',1.),'residual_guard')
    reject('failed_restart',lambda t:t.__setitem__('restart_exact',False),'restart')
    reject('energy_failure',lambda t:t['sparse'][-1].__setitem__('energy_residual_j',1.),'ledgers')
    return dict(passed=all(checks.values()),checks=checks,scope='Mutated copies; no extra native trajectories')


def measured_cost():
    results={}
    for n in (8,32):
        times={};observations={}
        for method in ('dense','reuse','cold'):
            elapsed=[]
            for _ in range(3):
                cache=sparse_linear.SparseFactorCache();start=time.perf_counter()
                for k in range(32):
                    cfg,boundary,initial=inputs(n,'counterflow');boundary['a_inlet_temperature_k']=520.+k
                    if method=='dense':model=base.TwoStreamExchanger(cfg,boundary,initial)
                    else:
                        if method=='cold':cache.clear_cache()
                        model=sparse.SparseSteadyExchanger(cfg,boundary,initial,factor_cache=cache)
                elapsed.append(time.perf_counter()-start);observations[method]=model.snapshot()['temperature_k']
            times[method]=dict(wall_seconds=elapsed,median_seconds=statistics.median(elapsed))
        times.update(reuse_speed_ratio_vs_dense=times['dense']['median_seconds']/times['reuse']['median_seconds'],
            maximum_temperature_difference_k=max(float(np.max(abs(np.asarray(observations[m])-np.asarray(observations['dense'])))) for m in ('reuse','cold')))
        results[str(n)]=times
    return dict(machine=dict(system=platform.system(),machine=platform.machine(),python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
        threads={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')}),
        protocol=dict(models_per_repeat=32,repeats=3,construction_timed=True,first_factorization_timed=True),measurements=results,
        scope='Local small steady inlet families; clocks excluded from scientific replay; no universal speed claim')


def run(output):
    out=Path(output).absolute()
    if out.exists() or any(p.is_symlink() for p in (out,*out.parents)):raise ValueError('Fresh ordinary output required')
    out.mkdir(parents=True);frozen=source();cases={}
    for mode in MODES:
        for n in MESHES:
            name=f'{mode}_{n}';print(name,flush=True);cache=sparse_linear.SparseFactorCache();initials=[]
            for hot,cold in INLETS:
                cfg,boundary,initial=inputs(n,mode);boundary.update(a_inlet_temperature_k=hot,b_inlet_temperature_k=cold)
                a=sparse.SparseSteadyExchanger(cfg,boundary,initial,factor_cache=cache);b=base.TwoStreamExchanger(cfg,boundary,initial)
                initials.append(dict(sparse=a.observation(),dense=b.observation(),steady_residual_w=a.initialization_report()['energy_rate_residual_w']))
            boundary=dict(a.boundary_controls,a_inlet_temperature_k=500.);a.set_controls(boundary);b.set_controls(boundary)
            left=[a.advance(.2)];right=[b.advance(.2)];saved=json.loads(json.dumps(a.snapshot(),allow_nan=False))
            left.append(a.advance(.3));right.append(b.advance(.3));restored=sparse.SparseSteadyExchanger.from_snapshot(saved)
            exact=restored.advance(.3)==left[-1] and restored.snapshot()==a.snapshot()
            t=dict(initializations=initials,sparse=left,dense=right,cache=cache.cache_info(),restart_exact=exact,checkpoint=a.snapshot())
            cases[name]=t;atomic_json(out/(name+'.json'),seal(t))
    result=assess(cases);negative=negative_controls(cases)
    report=dict(passed=result['passed'] and negative['passed'] and source()==frozen,source=frozen,criteria=CRITERIA,assessment=result,
        negative_controls=negative,steady_models_per_backend=27,transient_pairs=9,restart_suffixes=9,pwr_connected=False,
        experimental_validation=False,closed_tasks=[],scope='Small constant-property steady linear families; no sparse nonlinear PWR Jacobian claim')
    cost=measured_cost()
    if any(m['maximum_temperature_difference_k']>CRITERIA['temperature_absolute_k'] for m in cost['measurements'].values()):
        report['passed']=False
    atomic_json(out/'report.json',seal(report));atomic_json(out/'scientific.json',seal(dict(report=report,cases=cases)))
    atomic_json(out/'performance.json',seal(cost))
    if not report['passed']:raise ValueError('Sparse steady qualification failed; attempts retained')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path);print(run(p.parse_args().output))
