"""Forced smooth-pipe Cess momentum comparison against independent DNS."""
import argparse
import json
from math import log
from pathlib import Path
import numpy as np
from scipy.integrate import trapezoid
from atomic_model_station_turbulence import bulk_state,at_bulk_reynolds,velocity,finite_volume,load_profile
from atomic_model_station_turbulence.provenance import identity as model_identity
from atomic_model_multiphysics.provenance import seal,sha
from atomic_model_training.spec import atomic_json
from scripts.qualify_station_inventory import read


def identity():return dict(model=model_identity(),driver_sha256=sha(__file__))


def protocol():
    # Declared before running comparisons. Constants remain those of Eq B2.
    return dict(source=identity(),dns_cases=[180,360,550,1000],meshes=[256,512,1024],
        criteria=dict(friction_relative=.05,profile_relative_l2=.05,profile_max_over_dns_peak=.08,
                      fv_profile_absolute=.04,fv_bulk_relative=.001,observed_order=1.8,
                      bulk_inverse_residual=1e-10,work_relative=1e-9),
        constants=dict(kappa=.42,Aplus=27),
        primary_comparison='Fixed DNS bulk Re; predict Re_tau and Darcy friction without normalization or fitting',
        secondary_comparison='Fixed DNS Re_tau; wall-unit profile is diagnostic only',
        validation_kind='independent numerical DNS comparison, no experimental uncertainty qualification',
        execution_gate='passed requires complete comparisons and numerical verification; DNS acceptance is a separate gate',
        station_usable=False,thermal_qualified=False,buoyancy_qualified=False)


def run(output,reference=None):
    out=Path(output).absolute();p=protocol();source=p['source'];limits=p['criteria']
    if out.exists() or any(q.is_symlink() for q in (out,*out.parents)):raise ValueError('Fresh ordinary output required')
    if reference and read(Path(reference)/'protocol.json')!=p:raise ValueError('Protocol identity differs')
    out.mkdir(parents=True);atomic_json(out/'protocol.json',seal(p))
    spatial={};comparisons={};friction={};passed=True
    for label in p['dns_cases']:
        dns=load_profile(label);t=dns['re_tau'];ref=bulk_state(t);rows=[];errors=[]
        for n in p['meshes']:
            fv=finite_volume(t,n);v=velocity(fv['radius'],t)
            error=float(np.max(np.abs(np.array(fv['velocity_plus'])-v)));errors.append(error)
            rows.append(dict(cells=n,profile_absolute=error,
                bulk_relative=abs(fv['bulk_velocity_plus']/ref['bulk_velocity_plus']-1),
                work_relative=fv['work_relative_residual'],wall_stress_relative=fv['wall_stress_relative_residual'],
                linear_residual=fv['linear_residual']))
        orders=[log(a/b)/log(2) for a,b in zip(errors,errors[1:])]
        fv_ok=bool(errors[-1]<limits['fv_profile_absolute'] and rows[-1]['bulk_relative']<limits['fv_bulk_relative']
                   and min(orders)>limits['observed_order'] and max(r['work_relative'] for r in rows)<limits['work_relative'])
        spatial[str(label)]=dict(passed=fv_ok,rows=rows,orders=orders)
        prediction=at_bulk_reynolds(dns['re_bulk']);r=np.array(dns['radius']);u=np.array(dns['velocity_over_bulk'])
        modeled=velocity(r,prediction['re_tau'])/prediction['bulk_velocity_plus'];error=modeled-u
        l2=float(np.sqrt(trapezoid(2*r*error**2,r)/trapezoid(2*r*u**2,r)))
        peak=float(np.max(np.abs(error))/np.max(u))
        ferror=abs(prediction['darcy_friction']/dns['darcy_friction']-1)
        fixed_tau=velocity(r,t);tau_error=float(np.max(np.abs(fixed_tau-np.array(dns['velocity_plus']))))
        dns_ok=bool(ferror<limits['friction_relative'] and l2<limits['profile_relative_l2'] and peak<limits['profile_max_over_dns_peak']
                    and prediction['relative_bulk_residual']<limits['bulk_inverse_residual'])
        comparisons[str(label)]=dict(passed=dns_ok,dns=dns,prediction=prediction,
            predicted_velocity_over_bulk=modeled.tolist(),profile_relative_l2=l2,profile_max_over_dns_peak=peak,
            fixed_tau_velocity_plus=fixed_tau.tolist(),fixed_tau_maximum_absolute=tau_error)
        friction[str(label)]=dict(dns_re_bulk=dns['re_bulk'],dns_re_tau=t,predicted_re_tau=prediction['re_tau'],
            dns_darcy=dns['darcy_friction'],predicted_darcy=prediction['darcy_friction'],relative_error=ferror)
        passed=bool(passed and fv_ok);print('Cess/DNS case',label,'numerics',fv_ok,'DNS acceptance',dns_ok,flush=True)
    atomic_json(out/'spatial.json',seal(spatial));atomic_json(out/'dns.json',seal(comparisons));atomic_json(out/'friction.json',seal(friction))
    if identity()!=source:raise ValueError('Turbulence source changed during execution')
    report=dict(passed=passed,source=source,source_unchanged=True,dns_cases=len(comparisons),
        maximum_friction_relative=max(r['relative_error'] for r in friction.values()),
        maximum_profile_relative_l2=max(r['profile_relative_l2'] for r in comparisons.values()),
        maximum_profile_over_peak=max(r['profile_max_over_dns_peak'] for r in comparisons.values()),
        maximum_fv_profile_absolute=max(r['rows'][-1]['profile_absolute'] for r in spatial.values()),
        minimum_spatial_order=min(min(r['orders']) for r in spatial.values()),
        maximum_work_relative=max(row['work_relative'] for r in spatial.values() for row in r['rows']),
        dns_momentum_comparison_passed=bool(all(r['passed'] for r in comparisons.values())),
        dns_accepted_cases=[label for label in p['dns_cases'] if comparisons[str(label)]['passed']],
        dns_refused_cases=[label for label in p['dns_cases'] if not comparisons[str(label)]['passed']],
        model_is_dns=False,thermal_qualified=False,buoyancy_qualified=False,
        closed_tasks=[],station_usable=False,npp_11_03_accepted=False,r1_accepted=False,validated=False,
        scope='Cess forced smooth-pipe mean momentum only; no turbulent heat closure or station integration')
    atomic_json(out/'report.json',seal(report))
    if not passed:raise ValueError('Turbulent momentum qualification failed; evidence preserved')
    if reference:
        expected={q.name:sha(q) for q in Path(reference).glob('*.json')};actual={q.name:sha(q) for q in out.glob('*.json')}
        if expected!=actual:raise ValueError('Installed turbulent momentum JSON differs')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--reference',type=Path)
    a=p.parse_args();r=run(a.output,a.reference);print(json.dumps(dict(passed=r['passed'],dns_cases=r['dns_cases'])))

if __name__=='__main__':main()
