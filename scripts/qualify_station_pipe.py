"""Published base profiles and transient radial qualification, not DNS."""
import argparse
from dataclasses import replace
from importlib.resources import files
import json
from math import log
from pathlib import Path
import numpy as np
from atomic_model_station_pipe import PipeCase,PipeSession,exact_profiles,exponential_reference,relaxation_scales
from atomic_model_station_pipe.provenance import identity as model_identity
from atomic_model_multiphysics.provenance import seal,sha
from atomic_model_training.spec import atomic_json
from scripts.qualify_station_inventory import read


def identity():return dict(model=model_identity(),driver_sha256=sha(__file__))


def initial_pulse(case):
    s=PipeSession(case);x=s.x;w=s.w
    a=float(w@((1-x*x)*x*x)/(w@(1-x*x)));shape=(1-x*x)*(x*x-a)
    y=s.state;y[:case.cells]+=.02*shape;y[case.cells:]-=.05*shape
    return y.tolist()


def run(output,reference=None):
    out=Path(output).absolute();source=identity()
    if out.exists() or any(p.is_symlink() for p in (out,*out.parents)):raise ValueError('Fresh ordinary output required')
    fixture=json.loads(files('atomic_model_station_pipe').joinpath('data/jfm2021_base.json').read_text())
    protocol=dict(source=source,fixture=fixture,meshes=[32,64,128],temporal_steps=[10.,5.,2.5],prandtl_cases=[.7,1.],
        criteria=dict(profile_absolute=3e-4,temporal_absolute=2e-5,whole_path_absolute=5e-5,
                      pressure_beta_absolute=1e-3,sink_ratio_absolute=1e-4,
                      observed_order=1.8,ledger_relative=1e-10,first_mode_relative=2e-4),
        trajectory=dict(cells=32,max_step=2.5,interval=25.,blocks_c=[3.,5.,7.5,0.],samples_per_block=8),
        scope='Constrained radial invariant subspace; analytic base and diffusive modes only, no DNS or station acceptance')
    if reference and read(Path(reference)/'protocol.json')!=protocol:raise ValueError('Radial study reference identity differs')
    out.mkdir(parents=True);atomic_json(out/'protocol.json',seal(protocol))
    limits=protocol['criteria'];steady={};passed=True
    for c in fixture['cases_c']:
        rows=[];errors=[]
        for n in protocol['meshes']:
            s=PipeSession(PipeCase(cells=n,buoyancy_c=c));p=s.snapshot();exact=exact_profiles(s.x,c)
            error=max(float(np.max(np.abs(np.array(p[k])-exact[k]))) for k in ('velocity','temperature'))
            errors.append(error);rows.append(dict(snapshot=p,profile_absolute=error,
                pressure_beta_absolute=abs(p['pressure_beta']+c/3),sink_ratio_absolute=abs(p['sink_ratio_nu']-1)))
        orders=[log(x/y)/log(2) for x,y in zip(errors,errors[1:])]
        ok=errors[-1]<limits['profile_absolute'] and min(orders)>limits['observed_order'] and rows[-1]['pressure_beta_absolute']<limits['pressure_beta_absolute'] and rows[-1]['sink_ratio_absolute']<limits['sink_ratio_absolute']
        passed=passed and ok;steady[str(c)]=dict(passed=ok,results=rows,observed_orders=orders)
    atomic_json(out/'steady.json',seal(steady))
    temporal={}
    for pr in protocol['prandtl_cases']:
        case=PipeCase(cells=32,prandtl=pr);initial=initial_pulse(case)
        exact=exponential_reference(case,initial,200.,buoyancy_c=5.);errors=[];rows=[]
        for h in protocol['temporal_steps']:
            s=PipeSession(replace(case,max_step=h),initial_state=initial);p=s.advance(200.,buoyancy_c=5.)
            error=float(np.max(np.abs(s.state-exact)));errors.append(error);rows.append(dict(max_step=h,absolute_error=error,snapshot=p))
        orders=[log(x/y)/log(2) for x,y in zip(errors,errors[1:])]
        ok=errors[-1]<limits['temporal_absolute'] and min(orders)>limits['observed_order'] and all(r['snapshot']['checks']['relative_maximum']<limits['ledger_relative'] for r in rows)
        temporal[str(pr)]=dict(passed=ok,results=rows,observed_orders=orders);passed=passed and ok
    atomic_json(out/'temporal.json',seal(temporal))
    spectra=[relaxation_scales(PipeCase(cells=n)) for n in protocol['meshes']]
    mode_errors=[r['relative_errors'][0] for r in spectra];mode_orders=[log(a/b)/log(2) for a,b in zip(mode_errors,mode_errors[1:])]
    passed=passed and mode_errors[-1]<limits['first_mode_relative'] and min(mode_orders)>limits['observed_order']
    atomic_json(out/'spectrum.json',seal(dict(results=spectra,observed_orders=mode_orders)))
    path=protocol['trajectory'];case=PipeCase(cells=path['cells'],max_step=path['max_step']);initial=initial_pulse(case);owner=PipeSession(case,initial_state=initial)
    exact=initial;records=[];commands=[];maximum=0.;midpoint=None
    for c in path['blocks_c']:
        for _ in range(path['samples_per_block']):
            command=dict(duration=path['interval'],buoyancy_c=c);p=owner.advance(**command);commands.append(command)
            exact=exponential_reference(case,exact,path['interval'],buoyancy_c=c)
            error=float(np.max(np.abs(owner.state-exact)));maximum=max(maximum,error)
            records.append(dict(snapshot=p,exponential_reference_absolute=error))
            if len(records)==16:
                midpoint=owner.checkpoint();atomic_json(out/'midpoint.json',midpoint)
        print('radial trajectory',owner.time,'/800',flush=True)
    final=owner.checkpoint();atomic_json(out/'final.json',final)
    other=PipeSession.restore(midpoint);prefix_exact=other.checkpoint()==midpoint
    for index,command in enumerate(commands[16:],start=16):
        if other.advance(**command)!=records[index]['snapshot']:raise ValueError('Radial native suffix differs')
    native_exact=other.checkpoint()==final
    passed=bool(passed and prefix_exact and native_exact and maximum<limits['whole_path_absolute']
                and max(r['snapshot']['checks']['relative_maximum'] for r in records)<limits['ledger_relative'])
    atomic_json(out/'trajectory.json',seal(dict(records=records,commands=commands,maximum_absolute=maximum,
                native_prefix_exact=prefix_exact,native_suffix_and_final_exact=native_exact)))
    if source!=identity():raise ValueError('Radial source changed during execution')
    report=dict(passed=passed,source=source,source_unchanged=True,
        spatial_cases=len(steady),meshes=protocol['meshes'],maximum_profile_absolute=max(r['results'][-1]['profile_absolute'] for r in steady.values()),
        minimum_spatial_order=min(min(r['observed_orders']) for r in steady.values()),
        maximum_temporal_absolute=max(r['results'][-1]['absolute_error'] for r in temporal.values()),
        minimum_temporal_order=min(min(r['observed_orders']) for r in temporal.values()),
        whole_path_maximum_absolute=maximum,duration=800.,time_unit='R/(2Ub)',
        native_prefix_exact=prefix_exact,native_suffix_and_final_exact=native_exact,
        slowest_velocity_time=spectra[-1]['slowest_velocity_time'],slowest_temperature_time=spectra[-1]['slowest_temperature_time'],
        first_mode_relative=mode_errors[-1],minimum_mode_order=min(mode_orders),
        closed_tasks=[],station_usable=False,dns_qualified=False,npp_11_03_accepted=False,r1_accepted=False,validated=False,
        scope='Published laminar base, constrained radial evolution and diffusive modes qualified; no turbulent DNS comparison')
    atomic_json(out/'report.json',seal(report))
    if not passed:raise ValueError('Radial qualification failed; evidence preserved')
    if reference:
        expected={p.name:sha(p) for p in Path(reference).glob('*.json')};actual={p.name:sha(p) for p in out.glob('*.json')}
        if expected!=actual:raise ValueError('Installed radial scientific JSON differs')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--reference',type=Path)
    a=p.parse_args();r=run(a.output,a.reference);print(json.dumps(dict(passed=r['passed'],dns_qualified=False)))

if __name__=='__main__':main()
