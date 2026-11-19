"""Two heat constraints, FV/continuum convergence and independent mean replay."""
import argparse
from dataclasses import replace
import json
from math import log
from pathlib import Path
import numpy as np
from atomic_model_station_heated import HeatedCase,solve_heated,continuum,flux_to_delta
from atomic_model_station_heated.external import load_external
from atomic_model_station_heated.native_reference import native_prediction,load_quadrature
from atomic_model_station_heated.provenance import identity as model_identity
from atomic_model_multiphysics.provenance import seal,sha
from atomic_model_training.spec import atomic_json
from scripts.qualify_station_inventory import read


def identity():return dict(model=model_identity(),driver_sha256=sha(__file__))


def protocol():
    return dict(source=identity(),boundaries=['delta_temperature','heat_flux'],buoyancy_cases=[0.,2.,5.],meshes=[128,256,512],
        criteria=dict(fv_profile_absolute=2e-5,nu_relative=2e-5,spatial_order=1.8,external_profile_absolute=2e-5,
                      external_late_absolute=1e-7,external_time_absolute=2e-7,mapping_absolute=1e-8,balance=1e-8),
        model='Chu2025 velocity-weighted thermal sink; imposed steady laminar mean symmetry',
        temperature='signed gauge-relative field, not absolute kelvin',station_usable=False,turbulent=False,
        external_acceptance_cells=64,coarse_external_profiles='Grid diagnostics, with their individual strict verdicts retained',
        execution_gate='Numerical verification and native-constraint prediction; original strict external-continuum gate is retained separately')


def run(output,reference=None):
    out=Path(output).absolute();p=protocol();source=p['source'];limits=p['criteria']
    if out.exists() or any(q.is_symlink() for q in (out,*out.parents)):raise ValueError('Fresh ordinary output required')
    if reference and read(Path(reference)/'protocol.json')!=p:raise ValueError('Protocol identity differs')
    out.mkdir(parents=True);atomic_json(out/'protocol.json',seal(p));spatial={};normalizations={};external={};passed=True
    for bc in p['boundaries']:
        for c in p['buoyancy_cases']:
            case=HeatedCase(boundary=bc,buoyancy_c=c);ref=continuum(case);rows=[];errors=[]
            for n in p['meshes']:
                fv=solve_heated(replace(case,cells=n));exact=ref['solution'].sol(fv['radius'])[[0,2]]
                error=float(np.max(np.abs(exact-np.array([fv['velocity'],fv['temperature']]))));errors.append(error)
                rows.append(dict(cells=n,profile_absolute=error,nu_relative=abs(fv['nusselt_mixing']/ref['nusselt_mixing']-1),
                    checks=fv['checks'],axial_gradient_scaled=fv['axial_gradient_scaled']))
            orders=[log(a/b)/log(2) for a,b in zip(errors,errors[1:])]
            ok=bool(errors[-1]<limits['fv_profile_absolute'] and min(orders)>limits['spatial_order'] and rows[-1]['nu_relative']<limits['nu_relative'])
            spatial[f'{bc}-{c}']=dict(passed=ok,rows=rows,orders=orders);passed=bool(passed and ok)
            normalizations[f'{bc}-{c}']={k:fv[k] for k in ('area_mean_temperature','mixing_cup_temperature','nusselt_area','nusselt_mixing','wall_gradient','axial_gradient_scaled','pressure_beta')}
            if bc=='heat_flux':
                mapped=flux_to_delta(fv);other=solve_heated(replace(case,boundary='delta_temperature',buoyancy_c=mapped['buoyancy_c'],cells=p['meshes'][-1]))
                mapping_error=max(float(np.max(np.abs(np.array(mapped['temperature'])-other['temperature']))),
                    float(np.max(np.abs(np.array(fv['velocity'])-other['velocity']))),abs(mapped['pressure_beta']-other['pressure_beta']))
                normalizations[f'mapping-{c}']=dict(mapped_c=mapped['buoyancy_c'],scale_ratio=mapped['temperature_scale_ratio'],maximum_absolute=mapping_error)
                passed=bool(passed and mapping_error<limits['mapping_absolute'])
            print('heated continuum/FV',bc,c,'passed',ok,flush=True)
    fixture=load_external();quadrature=load_quadrature()
    from importlib.resources import files
    if quadrature['external_fixture_sha256']!=sha(str(files('atomic_model_station_heated').joinpath('data/openpipeflow_mean.json'))):
        raise ValueError('Quadrature and external fixture differ')
    for name,record in fixture['records'].items():
        bc=record['boundary'];c=record['c'];ref=continuum(HeatedCase(boundary=bc,buoyancy_c=c));final=record['final']
        exact=ref['solution'].sol(final['radius'])[[0,2]]
        error=float(np.max(np.abs(exact-np.array([final['velocity'],final['temperature']]))))
        native=native_prediction(bc,c,record['cells'])
        native_error=float(np.max(np.abs(np.array([native['velocity'],native['temperature']])-np.array([final['velocity'],final['temperature']]))))
        ok=bool(error<limits['external_profile_absolute'] and record['late_maximum']<limits['external_late_absolute'])
        native_ok=bool(native_error<limits['external_profile_absolute'] and record['late_maximum']<limits['external_late_absolute'])
        external[name]=dict(continuum_passed=ok,native_constraint_passed=native_ok,profile_absolute=error,native_profile_absolute=native_error,
            late_maximum=record['late_maximum'],time=final['time'],cells=record['cells'],step=record['step'])
        if record['cells']==p['external_acceptance_cells']:passed=bool(passed and native_ok)
    # Compare the same radial nodes after refining the external time step.
    time_errors={}
    for bc in p['boundaries']:
        records=[v for v in fixture['records'].values() if v['boundary']==bc and v['c']==2 and v['cells']==64]
        if len(records)!=2:raise ValueError('External time refinement pair required')
        error=float(np.max(np.abs(np.array([records[0]['final']['velocity'],records[0]['final']['temperature']])-np.array([records[1]['final']['velocity'],records[1]['final']['temperature']]))))
        time_errors[bc]=error;passed=bool(passed and error<limits['external_time_absolute'])
    for bc in p['boundaries']:
        for c in p['buoyancy_cases']:
            pair=[v for v in fixture['records'].values() if v['boundary']==bc and v['c']==c and v['step']==(.1 if bc=='delta_temperature' else .01)]
            values=[]
            for record in sorted(pair,key=lambda r:r['cells']):
                key=next(k for k,v in fixture['records'].items() if v is record)
                values.append(external[key]['profile_absolute'])
            if len(values)!=2 or values[-1]>max(values[0],1e-8):passed=False
    atomic_json(out/'spatial.json',seal(spatial));atomic_json(out/'normalizations.json',seal(normalizations))
    atomic_json(out/'external.json',seal(dict(cases=external,time_refinement_absolute=time_errors,fixture_harness_sha256=fixture['harness_sha256'])))
    if source!=identity():raise ValueError('Heated source changed during execution')
    report=dict(passed=passed,source=source,source_unchanged=True,boundaries=p['boundaries'],spatial_cases=len(spatial),external_cases=len(external),
        maximum_fv_profile_absolute=max(v['rows'][-1]['profile_absolute'] for v in spatial.values()),
        minimum_spatial_order=min(min(v['orders']) for v in spatial.values()),
        maximum_nu_relative=max(v['rows'][-1]['nu_relative'] for v in spatial.values()),
        maximum_external_profile_absolute=max(v['profile_absolute'] for v in external.values()),maximum_external_time_absolute=max(time_errors.values()),
        maximum_external_native_absolute=max(v['native_profile_absolute'] for v in external.values()),
        maximum_external_fine_native_absolute=max(v['native_profile_absolute'] for v in external.values() if v['cells']==64),
        maximum_external_late_absolute=max(v['late_maximum'] for v in external.values()),
        maximum_mapping_absolute=max(v['maximum_absolute'] for k,v in normalizations.items() if k.startswith('mapping')),
        forced_area_nusselt=normalizations['heat_flux-0.0']['nusselt_area'],forced_mixing_nusselt=normalizations['heat_flux-0.0']['nusselt_mixing'],
        external_continuum_comparison_passed=bool(all(v['continuum_passed'] for v in external.values())),
        external_native_comparison_passed=bool(all(v['native_constraint_passed'] for v in external.values() if v['cells']==64)),
        closed_tasks=[],station_usable=False,turbulent=False,npp_11_03_accepted=False,r1_accepted=False,validated=False,
        scope='Steady2025 mean subspace and two thermal constraints; no turbulent heat law, transient or plant qualification')
    atomic_json(out/'report.json',seal(report))
    if not passed:raise ValueError('Heated mean qualification failed; evidence preserved')
    if reference:
        if {q.name:sha(q) for q in out.glob('*.json')}!={q.name:sha(q) for q in Path(reference).glob('*.json')}:raise ValueError('Installed heated scientific JSON differs')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--reference',type=Path)
    a=p.parse_args();r=run(a.output,a.reference);print(json.dumps(dict(passed=r['passed'],external_cases=r['external_cases'])))

if __name__=='__main__':main()
