"""Full corrected mean-reference matrix against unchanged continuum equations."""
import argparse
import json
from math import log
from pathlib import Path
import numpy as np
from numpy.polynomial.legendre import legval
from scipy.integrate import quad
from scipy.linalg import lstsq
from atomic_model_station_quadrature import correct_quadrature
from atomic_model_station_quadrature.provenance import identity as model_identity
from atomic_model_station_quadrature.data import load_fixture
from atomic_model_station_heated import HeatedCase,continuum
from atomic_model_station_heated.external import load_external
from atomic_model_multiphysics.provenance import sha,seal
from atomic_model_training.spec import atomic_json
from scripts.qualify_station_inventory import read


def identity():return dict(model=model_identity(),driver_sha256=sha(__file__))


def protocol():
    return dict(source=identity(),boundaries=['delta_temperature','heat_flux'],buoyancy_cases=[0,2,5],meshes=[32,64],
        acceptance_cells=64,degree_in_radius_squared=10,
        criteria=dict(moment_absolute=5e-14,independent_correction_absolute=5e-14,external_profile_absolute=2e-5,
            external_late_absolute=1e-7,external_time_absolute=2e-7,meaningful_error=1e-8,minimum_radial_order=1.8),
        quadrature_uses_flow_data=False,source_variant='Numerical quadrature explicitly patched; original v67 unchanged',
        coarse_grid_role='All verdicts recorded; acceptance threshold applies to64point grid',
        physical_scope='Steady mean-only subspace at Re5300,Pr0.7; no turbulence or plant qualification')


def run(output,reference=None):
    out=Path(output).absolute();p=protocol();source=p['source'];limits=p['criteria']
    if out.exists() or any(q.is_symlink() for q in (out,*out.parents)):raise ValueError('Fresh ordinary output required')
    if reference and read(Path(reference)/'protocol.json')!=p:raise ValueError('Protocol identity differs')
    fixture=load_fixture();original=load_external()
    if set(fixture['records'])!=set(original['records']):raise ValueError('Matched original/corrected matrix required')
    out.mkdir(parents=True);atomic_json(out/'protocol.json',seal(p));moments={};passed=True
    for n,row in fixture['rules'].items():
        rule=correct_quadrature(row['radius'],row['original_weights'],degree=p['degree_in_radius_squared'])
        if rule!=row:raise ValueError('Recomputed quadrature rule differs from external input')
        r=np.array(rule['radius']);old=np.array(rule['original_weights']);new=np.array(rule['corrected_weights'])
        from numpy.polynomial.legendre import legvander
        a=legvander(2*r*r-1,p['degree_in_radius_squared']).T
        # Independent rank-revealing QR, rather than the production SVD.
        correction,_,rank,_=lstsq(a,np.array(rule['target_moments'])-a@old,lapack_driver='gelsy')
        independent=float(np.max(abs((new-old)-correction)));integrals=[]
        for j in range(p['degree_in_radius_squared']+1):
            coefficients=np.zeros(j+1);coefficients[-1]=1.
            exact=quad(lambda x:float(x*legval(2*x*x-1,coefficients)),0,1,epsabs=1e-13,epsrel=1e-13)[0]
            integrals.append(abs(float(a[j]@new)-exact))
        ok=bool(rank==len(a) and independent<limits['independent_correction_absolute'] and max(integrals)<limits['moment_absolute'] and min(new)>0)
        moments[n]=dict(passed=ok,independent_qr_absolute=independent,independent_integral_absolute=max(integrals),rule=rule);passed=bool(passed and ok)
    atomic_json(out/'moments.json',seal(moments));comparisons={};references={}
    for name,row in fixture['records'].items():
        key=(row['boundary'],row['c'])
        if key not in references:references[key]=continuum(HeatedCase(boundary=key[0],buoyancy_c=key[1]))
        final=row['final'];old=original['records'][name]['final']
        if final['radius']!=old['radius']:raise ValueError('Matched original/corrected radial nodes required')
        exact=references[key]['solution'].sol(final['radius'])[[0,2]]
        error=float(np.max(np.abs(exact-np.array([final['velocity'],final['temperature']]))))
        old_error=float(np.max(np.abs(exact-np.array([old['velocity'],old['temperature']]))))
        ok=bool(error<limits['external_profile_absolute'] and row['late_maximum']<limits['external_late_absolute']
            and abs(final['time']-8000)<1e-6 and abs(row['late']['time']-6000)<1e-6)
        comparisons[name]=dict(boundary=row['boundary'],c=row['c'],cells=row['cells'],step=row['step'],passed=ok,
            corrected_profile_absolute=error,original_profile_absolute=old_error,late_maximum=row['late_maximum'],
            improvement=old_error/max(error,1e-15),time=final['time'])
        if row['cells']==p['acceptance_cells']:passed=bool(passed and ok)
        print('corrected continuum',name,'error',error,'passed',ok,flush=True)
    atomic_json(out/'comparison.json',seal(comparisons));refinement={};time_errors={}
    for bc in p['boundaries']:
        for c in p['buoyancy_cases']:
            step=.1 if bc=='delta_temperature' else .01
            pair=sorted([v for v in comparisons.values() if v['boundary']==bc and v['c']==c and v['step']==step],key=lambda v:v['cells'])
            if len(pair)!=2:raise ValueError('Complete radial refinement pair required')
            errors=[v['corrected_profile_absolute'] for v in pair]
            meaningful=max(errors)>limits['meaningful_error']
            order=log(errors[0]/errors[1])/log(2) if min(errors)>0 and meaningful else None
            ok=bool(not meaningful or order>limits['minimum_radial_order'])
            refinement[f'{bc}-{c}']=dict(errors=errors,meaningful=meaningful,order=order,passed=ok);passed=bool(passed and ok)
        rows=[v for v in fixture['records'].values() if v['boundary']==bc and v['c']==2 and v['cells']==64]
        if len(rows)!=2:raise ValueError('Complete temporal refinement pair required')
        error=float(np.max(np.abs(np.array([rows[0]['final']['velocity'],rows[0]['final']['temperature']])-np.array([rows[1]['final']['velocity'],rows[1]['final']['temperature']]))))
        time_errors[bc]=error;passed=bool(passed and error<limits['external_time_absolute'])
    atomic_json(out/'refinement.json',seal(dict(radial=refinement,temporal_absolute=time_errors)))
    if source!=identity():raise ValueError('Quadrature study sources changed')
    fine=[v for v in comparisons.values() if v['cells']==p['acceptance_cells']]
    meaningful_orders=[v['order'] for v in refinement.values() if v['meaningful']]
    report=dict(passed=passed,source=source,source_unchanged=True,external_cases=len(comparisons),acceptance_cells=p['acceptance_cells'],
        maximum_corrected_fine_profile_absolute=max(v['corrected_profile_absolute'] for v in fine),
        maximum_corrected_all_meshes_profile_absolute=max(v['corrected_profile_absolute'] for v in comparisons.values()),
        maximum_original_fine_profile_absolute=max(v['original_profile_absolute'] for v in fine),
        maximum_time_refinement_absolute=max(time_errors.values()),minimum_meaningful_radial_order=min(meaningful_orders) if meaningful_orders else None,
        maximum_moment_residual=max(v['rule']['maximum_moment_residual'] for v in moments.values()),
        maximum_independent_qr_absolute=max(v['independent_qr_absolute'] for v in moments.values()),
        corrected_external_continuum_comparison_passed=bool(all(v['passed'] for v in fine)),
        original_external_continuum_gate_closed=False,original_v67_fixture_updated=False,
        station_usable=False,turbulent=False,validated=False,npp_11_03_accepted=False,r1_accepted=False,closed_tasks=[],
        scope='Complete corrected mean-only reference matrix at64points; not unmodified external, turbulent or plant acceptance')
    atomic_json(out/'report.json',seal(report))
    if not passed:raise ValueError('Corrected quadrature qualification failed; evidence preserved')
    if reference and {q.name:sha(q) for q in out.glob('*.json')}!={q.name:sha(q) for q in Path(reference).glob('*.json')}:
        raise ValueError('Installed corrected quadrature JSON differs')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--reference',type=Path)
    a=p.parse_args();r=run(a.output,a.reference);print(json.dumps(dict(passed=r['passed'],external_cases=r['external_cases'])))

if __name__=='__main__':main()
