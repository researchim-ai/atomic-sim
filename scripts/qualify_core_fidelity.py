"""Measured two-resolution approximation with an executable protocol boundary."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
import platform
from pathlib import Path
import time

from atomic_model_multiphysics.provenance import seal
from atomic_model_training.spec import atomic_json
from coupling import core_fidelity as policy


def negative_controls(cases):
    checks={}
    def reject(name,change,gate):
        copied=deepcopy(cases);change(copied['insertion']['2_fine']);r=policy.assess(copied);checks[name]=not r['passed'] and not r['gates'][gate]
    reject('power_bias',lambda t:t['snapshots'][-1]['powers'].__setitem__('nuclear_fission_w',1e30),'powers')
    reject('sixth_precursor_bias',lambda t:t['snapshots'][-1]['core']['local_precursors'][5].__setitem__(0,1e30),'profiles')
    reject('energy_failure',lambda t:t['snapshots'][-1].__setitem__('energy_relative',1.),'budgets')
    reject('restart_failure',lambda t:t.__setitem__('restart_exact',False),'restart')
    reject('temperature_bias',lambda t:t['snapshots'][-1]['distributed_fuel']['temperature_k'].__setitem__(0,2500.),'temperature')
    reject('rollback_failure',lambda t:t.__setitem__('rollback',False),'rollback')
    return dict(passed=all(checks.values()),checks=checks)


def measured_cost():
    result={}
    for case in policy.CASES:
        result[case]={}
        for n in policy.MESHES:
            values=[]
            for repeat in range(3):
                start=time.perf_counter();s=policy.session(n);steps=policy.protocol(case,s.snapshot()['actuator_status']['rod_position'])
                for step in steps:s.advance(**step)
                values.append(time.perf_counter()-start)
                print('cost',case,n,repeat,flush=True)
            result[case][str(n)]=values
    return dict(format='core-fidelity-cost/v1',source=policy.identity(),measurements=result,
        construction_timed=True,repeats=3,machine=dict(system=platform.system(),machine=platform.machine(),python=platform.python_version(),threads={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')}),scope='Sequential local complete episode timing; no universal speed claim')


def run(output):
    out=Path(output).absolute()
    if out.exists() or any(p.is_symlink() for p in (out,*out.parents)):raise ValueError('Fresh ordinary output required')
    out.mkdir(parents=True);frozen=policy.identity();cases={}
    for case in policy.CASES:
        cases[case]={}
        for n in policy.MESHES:
            for preset in policy.PRESETS:
                key=f'{n}_{preset}';print(case,key,flush=True);s=policy.session(n,preset);initial=s.snapshot()
                steps=policy.protocol(case,initial['actuator_status']['rod_position']);rows=[];saved=None
                for i,step in enumerate(steps):
                    rows.append(s.advance(**step))
                    if i==1:saved=json.loads(json.dumps(s.checkpoint(),allow_nan=False))
                restart=None
                if preset=='fine':
                    restored=type(s).restore(saved)
                    restart=[restored.advance(**step) for step in steps[2:]]==rows[2:] and restored.checkpoint()==s.checkpoint()
                before=s.checkpoint()
                try:s.advance(.01,{'rod_target':2.})
                except ValueError:rollback=s.checkpoint()==before
                else:rollback=False
                t=dict(initial=initial,snapshots=rows,steps=steps,checkpoint=before,restart_exact=restart,rollback=rollback)
                cases[case][key]=t;atomic_json(out/(case+'_'+key+'.json'),seal(t))
    assessment=policy.assess(cases);negative=negative_controls(cases)
    report=dict(passed=assessment['passed'] and negative['passed'] and policy.identity()==frozen,
        assessment=assessment,negative_controls=negative,limits=policy.LIMITS,
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),trajectories=8,restart_suffixes=4,
        experimental_validation=False,continuum_error_estimated=False,long_horizon_accepted=False,
        scope='Fixed synthetic 2/4 neutron-node approximation; default thermal/fluid grids; fixed insertion/return protocols only')
    science=seal(dict(policy_source=frozen,cases=cases,report=report))
    # Delivery reader expects the standard report/cases envelope; policy source
    # is an additional explicit top-level field checked again by the runtime.
    atomic_json(out/'report.json',seal(report));atomic_json(out/'scientific.json',science)
    if not report['passed']:raise ValueError('Core fidelity accuracy failed; attempts retained')
    cost=seal(measured_cost());atomic_json(out/'performance.json',cost)
    checks={};decisions={}
    for case in policy.CASES:
        runtime=policy.QualifiedCoreRun(science,cost,case=case);decisions[case]=runtime.observation()['resolution']
        before=runtime.checkpoint()
        try:runtime.advance(.07,{'rod_target':0.})
        except ValueError:checks[case+'_off_protocol']=runtime.checkpoint()==before
        else:checks[case+'_off_protocol']=False
        position=runtime.observation()['snapshot']['actuator_status']['rod_position'];steps=policy.protocol(case,position)
        saved=None
        for i,step in enumerate(steps):
            runtime.advance(**step)
            if i==1:saved=json.loads(json.dumps(runtime.checkpoint(),allow_nan=False))
        restored=policy.QualifiedCoreRun.restore(saved,science,cost)
        for step in steps[2:]:restored.advance(**step)
        checks[case+'_restart']=restored.checkpoint()==runtime.checkpoint()
        before=runtime.checkpoint()
        try:runtime.advance(.1,{'rod_target':0.})
        except ValueError:checks[case+'_boundary']=runtime.checkpoint()==before
        else:checks[case+'_boundary']=False
    proof=dict(passed=all(checks.values()),checks=checks,decisions=decisions,scientific_sha256=hashlib.sha256((out/'scientific.json').read_bytes()).hexdigest())
    atomic_json(out/'policy_proof.json',seal(proof))
    if not proof['passed']:raise ValueError('Core fidelity executable boundary failed')
    return dict(passed=True,temporal=assessment['temporal_maximum_relative'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path);print(run(p.parse_args().output))
