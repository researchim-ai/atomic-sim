"""Heated liquid graph: source identities, finite-step balances and time oracle.

DOP853 reuses the same RHS/EOS: independent time integration, not independent physics.
"""
import argparse
from copy import deepcopy
from pathlib import Path
import math
import numpy as np
from scipy.integrate import solve_ivp
from atomic_model_hem_heated import HeatedHEMNetwork
from atomic_model_hem_heated.model import signature,canonical
from atomic_model_multiphysics.provenance import seal,sha
from atomic_model_training.spec import atomic_json
from scripts import verify_hllc_branching as branch

CASES=('uniform_heat','wall_relaxation','branched_pulse')
SCHEMES=('hllc','hllc-local')
CAPS=(2e-5,1e-5,5e-6)
TIMES=(0.,.0001,.00025,.0005)
BUDGETS=dict(pressure_pa=.1,temperature_k=1e-5,mass_kg=1e-9,total_energy_j=1e-4,momentum_kg_m_s=1e-8)


def request_for(case,cap):
    if case not in CASES or cap not in CAPS:raise ValueError('Declared heated matrix required')
    request=branch.request_for('branch_reversal',2,branch.CAPS[0]);spec=request['model'];spec['max_step_s']=cap
    wall=spec['walls'][0]
    if case=='uniform_heat':
        for node in spec['nodes']:
            node['initial'].update(pressure_pa=4e6,temperature_k=440.,velocity_m_s=0.)
            node['heat_power_w']=1e5*node['volume_m3']
        for edge in spec['edges']:
            for point in edge['initial']:point.update(pressure_pa=4e6,temperature_k=440.,velocity_m_s=0.)
            volume=np.pi*edge['diameter_m']**2/4*edge['length_m']/edge['cells']
            edge['heat_power_w']=[float(1e5*volume)]*edge['cells']
    else:
        wall.update(capacity_j_k=100.,initial_temperature_k=460.,conductance_w_k=10000.,ambient_ua_w_k=20.,ambient_temperature_k=430.)
        if case=='branched_pulse':
            spec['nodes'][0]['heat_power_w']=1e4;wall['heat_power_w']=5000.
    return request


def source_identity(model):
    y=model._y;point=model.primitive(y);rhs,aux=model.evaluate(y,model._controls)
    temperatures=np.array([s.temperature_k for s in point[4]]);tw=point[5]
    exchange=model.conductance*(tw-temperatures[model.ports])
    external=model._controls['walls']+model.ua*(model.ambient-tw)
    expected_energy=float(sum(model._controls['fluid'])+sum(external))
    actual_energy=float(sum(rhs[2*model.n:3*model.n])+sum(rhs[model.wall_slice]*model.capacity))
    entropy=float(sum(model._controls['fluid']/temperatures)+sum(model._controls['walls']/tw+model.ua*(model.ambient-tw)/model.ambient))
    production=float(sum(model.conductance*(tw-temperatures[model.ports])**2/(tw*temperatures[model.ports])+model.ua*(model.ambient-tw)**2/(model.ambient*tw)))
    checks=dict(energy=abs(actual_energy-expected_energy)<1e-7,
                exchange=np.allclose(aux['wall_to_fluid_w'],exchange,rtol=0,atol=1e-12),
                wall_rate=np.allclose(rhs[model.wall_slice],(external-exchange)/model.capacity,rtol=0,atol=1e-12),
                entropy=abs(rhs[model.entropy_index]-entropy)<1e-10,
                production=abs(rhs[model.production_index]-production)<1e-10,
                positive_production=production>=0.)
    return dict(checks={k:bool(v) for k,v in checks.items()},passed=all(checks.values()),energy_rate_error_w=abs(actual_energy-expected_energy))


def controls_at(model,case,index):
    controls=deepcopy(model.snapshot()['controls'])
    if case=='branched_pulse' and index==2:
        controls['fluid']['a']=-5000.;controls['walls']['wall']=0.
    return controls


def reference(request,scheme,case):
    model=HeatedHEMNetwork(request,scheme=scheme);y=model._y.copy();observations=[model.snapshot()];evaluations=0
    for index,(a,b) in enumerate(zip(TIMES,TIMES[1:]),1):
        controls=controls_at(model,case,index);mask=model._mask(controls)
        result=solve_ivp(lambda t,q:model.evaluate(q,mask)[0],(a,b),y,method='DOP853',rtol=2e-11,atol=1e-11,max_step=2.5e-6)
        if not result.success:raise ValueError(result.message)
        evaluations+=result.nfev;y=result.y[:,-1]
        observations.append(model._observe(y,b,mask,{},index))
        # Control schedule persists across windows, without integrating twice.
        model._controls=mask
    return dict(observations=observations,evaluations=evaluations,method='DOP853',shared_rhs=True)


def capture(request,scheme,case):
    model=HeatedHEMNetwork(request,scheme=scheme);initial=model.snapshot();observations=[initial];restart=False
    source=source_identity(model)
    for index,(a,b) in enumerate(zip(TIMES,TIMES[1:]),1):
        controls=controls_at(model,case,index)
        observations.append(model.step(controls,b-a,expected_revision=index-1,controller_state={'tick':index}))
        if index==2:
            checkpoint=model.checkpoint();restored=HeatedHEMNetwork(request,scheme=scheme);restored.restore(checkpoint)
            restart=canonical(restored.checkpoint())==canonical(checkpoint)
    last=restored.step(controls_at(restored,case,3),TIMES[3]-TIMES[2],expected_revision=2,controller_state={'tick':3})
    restart=restart and canonical(last)==canonical(observations[-1]) and canonical(restored.checkpoint())==canonical(model.checkpoint())
    checks=dict(source=source['passed'],restart=restart,
                mass=all(abs(s['global_ledger']['mass_residual_kg'])<=1e-9 for s in observations),
                energy=all(abs(s['global_ledger']['energy_residual_j'])<=1e-5 for s in observations),
                entropy=all(s['global_ledger']['numerical_entropy_residual_j_k']>=-2e-5 for s in observations),
                liquid=all(c['phase']=='liquid' and c['mass_kg']>0 for s in observations for c in s['cells']))
    return dict(request=request,observations=observations,source_identity=source,checks=checks,passed=all(checks.values()))


def compare(native,ref):
    if not native['observations'] or not ref['observations']:raise ValueError('Empty heated comparison')
    errors={k:0. for k in BUDGETS}
    for a,b in zip(native['observations'],ref['observations'],strict=True):
        if not math.isfinite(a['time_s']) or not math.isfinite(b['time_s']) or abs(a['time_s']-b['time_s'])>1e-12:raise ValueError('Heated time alignment')
        for x,y in zip(a['cells'],b['cells'],strict=True):
            if x['id']!=y['id']:raise ValueError('Heated cell alignment')
            for k in BUDGETS:
                delta=abs(x[k]-y[k])
                if not np.isfinite(delta):raise ValueError('Nonfinite heated comparison')
                errors[k]=max(errors[k],delta)
        for x,y in zip(a['walls'],b['walls'],strict=True):
            if x['id']!=y['id'] or x['cell']!=y['cell']:raise ValueError('Heated wall alignment')
            for field,budget in [('temperature_k',1e-5),('heat_to_fluid_j',1e-4),('external_heat_j',1e-4)]:
                delta=abs(x[field]-y[field])
                if not math.isfinite(delta):raise ValueError('Nonfinite heated wall')
                key='wall_'+field;errors[key]=max(errors.get(key,0.),delta)
        for field in ('external_entropy_j_k','thermal_entropy_production_j_k'):
            delta=abs(a['global_ledger'][field]-b['global_ledger'][field])
            if not math.isfinite(delta):raise ValueError('Nonfinite heated entropy ledger')
            errors[field]=max(errors.get(field,0.),delta)
    limits={**BUDGETS,'wall_temperature_k':1e-5,'wall_heat_to_fluid_j':1e-4,'wall_external_heat_j':1e-4,
            'external_entropy_j_k':1e-7,'thermal_entropy_production_j_k':1e-7}
    return dict(maximum_errors=errors,budgets=limits,passed=all(v<=limits[k] for k,v in errors.items()))


def run(output):
    output=Path(output).absolute()
    if any(p.is_symlink() for p in (output,*output.parents)):raise ValueError('Symlink output')
    output.mkdir(parents=True,exist_ok=False);source=signature();driver=sha(__file__)
    atomic_json(output/'protocol.json',seal(dict(source=source,driver=driver,cases=CASES,schemes=SCHEMES,caps_s=CAPS,times_s=TIMES,budgets=BUDGETS,required_accuracy_caps_s=[CAPS[-1]],pressure_refinement_ratio=.8,pressure_refinement_floor_pa=1e-3,reference='DOP853 shared RHS/EOS, rtol2e-11 atol1e-11 maxstep2.5us',validated=False)))
    records=[]
    for case in CASES:
        for scheme in SCHEMES:
            ref=reference(request_for(case,CAPS[-1]),scheme,case);runs=[]
            for cap in CAPS:
                native=capture(request_for(case,cap),scheme,case);comparison=compare(native,ref)
                runs.append(dict(cap_s=cap,checks=native['checks'],comparison=comparison,passed=native['passed'] and comparison['passed']))
                atomic_json(output/f'{case}-{scheme}-{cap}.json',seal(native))
            errors=[r['comparison']['maximum_errors']['pressure_pa'] for r in runs]
            refinement=all(b<=.8*a+1e-3 for a,b in zip(errors,errors[1:]))
            record=dict(case=case,scheme=scheme,runs=runs,pressure_refines=refinement,
                        passed=refinement and all(all(r['checks'].values()) for r in runs) and runs[-1]['comparison']['passed'])
            records.append(record);atomic_json(output/f'{case}-{scheme}-reference.json',seal(ref))
            print(case,scheme,record['passed'],[r['comparison']['maximum_errors']['pressure_pa'] for r in runs],flush=True)
    unchanged=source==signature() and sha(__file__)==driver
    report=dict(passed=unchanged and all(r['passed'] for r in records),source_unchanged=unchanged,records=records,native_paths=18,reference_paths=6,validated=False,plant_referenced=False)
    atomic_json(output/'report.json',seal(report));return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True,type=Path)
    return 0 if run(parser.parse_args().output)['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
