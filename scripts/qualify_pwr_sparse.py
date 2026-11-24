"""Matched dense/CSC Radau paths for the coupled single-phase research PWR."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import scipy
from atomic_model_multiphysics.provenance import seal
from atomic_model_training.spec import atomic_json
from coupling import pwr_sparse,refinement
from coupling.pwr_moderator import LocalModeratorPWRSession
from scripts import qualify_moderator_transients as rods
from scripts import qualify_pwr_distributed_fuel as budgets

BACKENDS=dict(dense=LocalModeratorPWRSession,sparse=pwr_sparse.SparseLocalModeratorPWRSession)
CASES=('insertion','return','scram')
PRESETS=dict(coarse=dict(max_step_s=.05,rtol=2e-9),fine=dict(max_step_s=.025,rtol=2e-10))
CRITERIA=dict(temporal_relative=1e-4,backend_relative=1e-5,initial_relative=1e-12)


def source():
    return dict(model=pwr_sparse.identity(),driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        helpers={m.__name__:hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest() for m in (refinement,rods,budgets)})


def protocol(case,position):
    if case not in CASES:raise ValueError('Registered sparse PWR scenario required')
    if case=='scram':return [dict(duration_s=.1,patch={'scram':True}),dict(duration_s=.4,patch=None),
        dict(duration_s=.5,patch=None),dict(duration_s=1.,patch=None),dict(duration_s=2.,patch=None)]
    return rods.protocol(case,position)


def assess(cases):
    expected={f'{backend}_{preset}' for backend in BACKENDS for preset in PRESETS}
    if set(cases)!=set(CASES) or any(set(t)!=expected for t in cases.values()):raise ValueError('Complete dense/sparse temporal matrix required')
    gates=dict(budgets=True,commands=True,timeline=True,initial_match=True,reference_frozen=True,stationary=True,
        sparse_coefficients=True,native_solver=True,restart=True,rollback=True,scram=True,events=True)
    time_errors=[];backend_errors=[];densities=[];solver_counts={}
    for case,trajectories in cases.items():
        for key,t in trajectories.items():
            backend,preset=key.split('_');initial=t['initial'];steps=protocol(case,initial['actuator_status']['rod_position'])
            if len(t['snapshots'])!=len(steps):raise ValueError('Complete PWR observations required')
            gates['commands'] &= t['steps']==steps and t['checkpoint']['payload']['history']==steps
            gates['timeline'] &= np.allclose([r['time_s'] for r in t['snapshots']],np.cumsum([s['duration_s'] for s in steps]),rtol=0,atol=1e-12)
            gates['rollback'] &= t['rollback'] is True
            if preset=='fine':gates['restart'] &= t['restart_exact'] is True
            report=initial['local_moderator_stationarity']
            gates['stationary'] &= report['constrained_rank']==report['constrained_unknowns'] and report['max_selected_tolerance_ratio']<=1.
            counts=dict(nlu=0,njev=0,rhs=0)
            for row in [initial,*t['snapshots']]:
                gates['budgets'] &= all(budgets.checks(row).values())
                gates['reference_frozen'] &= row['local_moderator']['reference']==initial['local_moderator']['reference']
                if row['time_s']>0:
                    stat=row['solver_statistics'];gates['native_solver'] &= stat['method']=='Radau' and stat['nlu']>0 and stat['njev']>0 and stat['rhs_evaluations']<=stat['rhs_limit']
                    counts['nlu']+=stat['nlu'];counts['njev']+=stat['njev'];counts['rhs']+=stat['rhs_evaluations']
                    if backend=='sparse':
                        jac=row['sparse_jacobian'];gates['sparse_coefficients'] &= jac['all_nonzero_coefficients_retained'] is True and jac['drop_tolerance'] is None and 0<jac['nnz']<jac['total_entries']
                        densities.append(jac['nnz']/jac['total_entries'])
            solver_counts[case+'_'+key]=counts
            if case=='scram':
                last=t['snapshots'][-1]
                gates['scram'] &= last['actuator_status']['rod_position']==1. and last['powers']['nuclear_fission_w']<.5*initial['powers']['nuclear_fission_w']
        for preset in PRESETS:
            dense=trajectories['dense_'+preset];sparse=trajectories['sparse_'+preset]
            gates['initial_match'] &= budgets.relative(dense['initial']['coupled_state'],np.asarray(sparse['initial']['coupled_state']))<=CRITERIA['initial_relative']
            for a,b in zip(dense['snapshots'],sparse['snapshots']):
                backend_errors.append(budgets.relative(a['coupled_state'],np.asarray(b['coupled_state'])))
                gates['events'] &= a['events']==b['events']
        for backend in BACKENDS:
            time_errors.extend(budgets.relative(a['coupled_state'],np.asarray(b['coupled_state'])) for a,b in
                zip(trajectories[backend+'_coarse']['snapshots'],trajectories[backend+'_fine']['snapshots']))
    gates.update(temporal=max(time_errors)<=CRITERIA['temporal_relative'],backend=max(backend_errors)<=CRITERIA['backend_relative'])
    return dict(passed=all(gates.values()),gates={k:bool(v) for k,v in gates.items()},temporal_maximum_relative=max(time_errors),
        backend_maximum_relative=max(backend_errors),sparse_density_range=[min(densities),max(densities)],solver_counts=solver_counts)


def negative_controls(cases):
    checks={}
    def reject(name,change,gate):
        copied=deepcopy(cases);change(copied['insertion']['sparse_fine']);r=assess(copied);checks[name]=not r['passed'] and not r['gates'][gate]
    reject('backend_state_corrupted',lambda t:t['snapshots'][-1]['coupled_state'].__setitem__(0,1e30),'backend')
    reject('threshold_introduced',lambda t:t['snapshots'][-1]['sparse_jacobian'].__setitem__('drop_tolerance',1e-9),'sparse_coefficients')
    reject('restart_failed',lambda t:t.__setitem__('restart_exact',False),'restart')
    reject('rollback_failed',lambda t:t.__setitem__('rollback',False),'rollback')
    reject('energy_failure',lambda t:t['snapshots'][-1].__setitem__('energy_relative',1.),'budgets')
    reject('lost_events',lambda t:t['snapshots'][-1].__setitem__('events',[]),'events')
    return dict(passed=all(checks.values()),checks=checks,scope='Mutated copies, not new physical trajectories')


def run(output):
    out=Path(output).absolute()
    if out.exists() or any(p.is_symlink() for p in (out,*out.parents)):raise ValueError('Fresh ordinary output required')
    out.mkdir(parents=True);frozen=source();cases={};cost={}
    for case in CASES:
        cases[case]={}
        for backend,cls in BACKENDS.items():
            for preset,options in PRESETS.items():
                key=backend+'_'+preset;print(case,key,flush=True)
                s=cls(**refinement.coupled_pwr_mesh(2),**options);initial=s.snapshot();steps=protocol(case,initial['actuator_status']['rod_position'])
                rows=[];saved=None;started=time.perf_counter()
                for i,step in enumerate(steps):
                    rows.append(s.advance(**step));print(case,key,s.time,flush=True)
                    if i==1:saved=json.loads(json.dumps(s.checkpoint(),allow_nan=False))
                elapsed=time.perf_counter()-started;restart=None
                if preset=='fine':
                    restored=cls.restore(saved)
                    restart=[restored.advance(**step) for step in steps[2:]]==rows[2:] and restored.checkpoint()==s.checkpoint()
                before=s.checkpoint()
                try:s.advance(.01,{'rod_target':2.})
                except ValueError:rollback=s.checkpoint()==before
                else:rollback=False
                t=dict(initial=initial,snapshots=rows,steps=steps,checkpoint=before,restart_exact=restart,rollback=rollback)
                cases[case][key]=t;cost[case+'_'+key]=elapsed;atomic_json(out/(case+'_'+key+'.json'),seal(t))
    result=assess(cases);negative=negative_controls(cases)
    report=dict(passed=result['passed'] and negative['passed'] and source()==frozen,source=frozen,criteria=CRITERIA,assessment=result,
        negative_controls=negative,trajectories=12,restart_suffixes=6,experimental_validation=False,long_horizon_accepted=False,
        station_env_connected=False,closed_tasks=[],scope='Single-phase 2/2/4 PWR, insertion/return and four-second SCRAM; no full plant shutdown qualification')
    atomic_json(out/'report.json',seal(report));atomic_json(out/'scientific.json',seal(dict(report=report,cases=cases)))
    atomic_json(out/'performance.json',seal(dict(integration_seconds=cost,construction_timed=False,restart_timed=False,repeats=1,
        machine=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__),
        scope='Single local observation per case/backend/preset; no statistical speed claim')))
    if not report['passed']:raise ValueError('Sparse PWR qualification failed; attempts retained')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path);print(run(p.parse_args().output))
