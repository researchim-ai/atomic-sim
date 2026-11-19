"""Independent mixed-convection branches, radial benchmark and applicability study.

No station episode is advanced here and no plan task is accepted. Source v63
remains a frozen numerical sample, never experimental validation data.
"""
import argparse
from dataclasses import asdict
from decimal import Decimal,localcontext
from importlib.resources import files
import json
from math import log
from pathlib import Path
import numpy as np
from atomic_model_station_mixed import FOLD_BO,FOLD_RATIO,LaminarCase,solve_laminar,solve_ratio,tube_groups,assess_opposing
from atomic_model_station_mixed.reference import collocation
from atomic_model_station_mixed.provenance import identity as model_identity
from atomic_model_multiphysics.provenance import seal,sha
from atomic_model_training.spec import atomic_json
from physics.materials import iapws
from core.status import DomainError,ModelLimitError
from scripts.qualify_station_inventory import read


def identity():return dict(model=model_identity(),driver_sha256=sha(__file__))


def station_screen():
    f=json.loads(files('atomic_model_station_mixed').joinpath('data/station_v63_boundary.json').read_text())
    bulk=iapws.properties_pt(f['pressure_pa'],f['bulk_temperature_k'])
    tf=(f['bulk_temperature_k']+f['wall_temperature_k'])/2
    film=iapws.properties_pt(f['pressure_pa'],tf)
    def expansion(t):
        r=iapws._region1_raw(float(t),f['pressure_pa']*1e-6)
        return float(r['dv_dt']/r['v'])
    d=f['geometry']['diameter_m'];g=f['film']['mass_flux_kg_m2_s']
    groups=tube_groups(diameter_m=d,mass_flux_kg_m2_s=g,density_kg_m3=bulk.rho_kg_m3,
        viscosity_pa_s=bulk.viscosity_pa_s,conductivity_w_m_k=bulk.conductivity_w_m_k,cp_j_kg_k=bulk.cp_j_kg_k,
        expansion_per_k=expansion(bulk.T_k),heat_flux_w_m2=f['film']['heat_w']/f['geometry']['heated_area_m2'],direction='upward')
    if abs(groups['reynolds']/f['film']['reynolds']-1)>1e-12 or abs(groups['prandtl']/f['film']['prandtl']-1)>1e-12:
        raise ValueError('Frozen boundary properties differ from declared source state')
    nu_f=film.viscosity_pa_s/film.rho_kg_m3
    gr_f=9.80665*expansion(tf)*(f['wall_temperature_k']-f['bulk_temperature_k'])*d**3/nu_f**2
    pr_f=film.cp_j_kg_k*film.viscosity_pa_s/film.conductivity_w_m_k
    inputs=dict(reynolds_bulk=groups['reynolds'],prandtl_bulk=groups['prandtl'],
        reynolds_film=g*d/film.viscosity_pa_s,prandtl_film=pr_f,grashof_film=gr_f,rayleigh_film=gr_f*pr_f)
    screens={direction:assess_opposing(direction=direction,**inputs) for direction in ('upward','downward')}
    rejected=False
    try:LaminarCase(reynolds=groups['reynolds'],prandtl=groups['prandtl'],buoyancy=groups['laminar_buoyancy'])
    except DomainError:rejected=True
    ratios={branch:asdict(solve_ratio(groups['iaea_heat_flux_buoyancy'],direction='upward',branch=branch))
            for branch in ('upper','middle','inverted')}
    return dict(fixture=f,groups=groups,reference_temperature_groups=inputs,opposing_candidates=screens,
        laminar_reference_rejected=rejected,iaea_framework_roots=ratios,
        station_usable=False,validated=False,
        scope='Uses stored v63 heat flux, not a coupled new station solution. Orientation is a study hypothesis. Bulk is a lumped node, not measured inlet/outlet mean; marginal experimental boxes are not validation.')


def run(output,reference=None):
    out=Path(output).absolute();source=identity()
    if out.exists() or any(p.is_symlink() for p in (out,*out.parents)):raise ValueError('Fresh ordinary output required')
    protocol=dict(source=source,radial_meshes=[64,128,256],laminar_buoyancies=[-40.,0.,40.,200.],
                  criteria=dict(nusselt_relative=2e-4,observed_order_minimum=1.8,balance=1e-8,forced_relative=2e-5),
                  scope='Numerical research primitives; no station horizon, plant validation or task acceptance')
    if reference and read(Path(reference)/'protocol.json')!=protocol:raise ValueError('Study reference identity differs')
    out.mkdir(parents=True);atomic_json(out/'protocol.json',seal(protocol))
    roots=[]
    with localcontext() as ctx:
        ctx.prec=60
        for direction,branch,value in [('upward','upper','0.9'),('upward','middle','0.5'),('upward','inverted','0.4'),('downward','unique','1.2')]:
            r=Decimal(value);a=Decimal('.46');p=Decimal('2.1')
            factor=1-r**(1/a) if branch in ('upper','middle') else (1+r**(1/a) if branch=='inverted' else r**(1/a)-1)
            bo=float(r**p*factor/Decimal('100000'));actual=solve_ratio(bo,direction=direction,branch=branch)
            roots.append(dict(result=asdict(actual),reference_ratio=float(r),relative_error=abs(actual.ratio/float(r)-1)))
    fold=asdict(solve_ratio(FOLD_BO,direction='upward',branch='upper'))
    ended=False
    try:solve_ratio(2*FOLD_BO,direction='upward',branch='upper')
    except ModelLimitError as error:ended=error.criterion=='mixed_branch_fold'
    atomic_json(out/'roots.json',seal(dict(inverse_references=roots,fold=fold,fold_parameter=FOLD_BO,
        upper_branch_refusal_exact=ended,auto_branch_selection=False)))
    radial={};admitted=True
    for b in protocol['laminar_buoyancies']:
        independent=collocation(LaminarCase(buoyancy=b));results=[];errors=[]
        for n in protocol['radial_meshes']:
            case=solve_laminar(LaminarCase(buoyancy=b,cells=n))
            errors.append(abs(case['nusselt']/independent['nusselt']-1));results.append(case)
        orders=[log(x/y)/log(2) for x,y in zip(errors,errors[1:])]
        passed=errors[-1]<protocol['criteria']['nusselt_relative'] and min(orders)>1.8 and max(max(c['checks'].values()) for c in results)<1e-8
        admitted=admitted and passed
        radial[str(b)]=dict(passed=passed,independent=independent,finite_volume=results,relative_errors=errors,observed_orders=orders)
        print('radial',b,'finest Nu relative',errors[-1],flush=True)
    forced=solve_laminar(LaminarCase(cells=512));x=np.array(forced['radius_fraction'])
    analytic=dict(nusselt_relative=abs(forced['nusselt']/(48/11)-1),darcy_re_relative=abs(forced['darcy_friction_factor']*forced['reynolds']/64-1),
                  poiseuille_maximum_absolute=float(np.max(np.abs(np.array(forced['velocity_ratio'])-2*(1-x*x)))))
    reversals={}
    for b in (-100.,500.):
        independent=collocation(LaminarCase(buoyancy=b));rejected=False
        try:solve_laminar(LaminarCase(buoyancy=b))
        except ModelLimitError as error:rejected=error.criterion=='local_forward_velocity'
        reversals[str(b)]=dict(fv_rejected=rejected,independent_minimum_sampled_velocity=min(independent['velocity_ratio']))
    atomic_json(out/'laminar.json',seal(dict(cases=radial,forced=forced,analytic=analytic,reversals=reversals)))
    screen=station_screen();atomic_json(out/'station-screen.json',seal(screen))
    passed=bool(admitted and ended and max(r['relative_error'] for r in roots)<2e-12
        and max(analytic.values())<2e-5 and all(r['fv_rejected'] and r['independent_minimum_sampled_velocity']<0 for r in reversals.values())
        and screen['laminar_reference_rejected'] and not any(r['eligible_for_reported_box_comparison'] for v in screen['opposing_candidates'].values() for r in v.values()))
    if source!=identity():raise ValueError('Study source changed during execution')
    report=dict(passed=passed,source=source,source_unchanged=True,station_solver_modified=False,
        root_reference_count=len(roots),radial_cases=len(radial),meshes=protocol['radial_meshes'],
        maximum_nusselt_relative=max(c['relative_errors'][-1] for c in radial.values()),
        minimum_observed_order=min(min(c['observed_orders']) for c in radial.values()),
        analytic=analytic,fold_parameter=FOLD_BO,fold_ratio=FOLD_RATIO,
        candidate_station_usable=False,closed_tasks=[],npp_11_03_accepted=False,r1_accepted=False,validated=False,
        scope='Math and discretization qualified; no high-pressure turbulent mixed-convection closure accepted')
    atomic_json(out/'report.json',seal(report))
    if not passed:raise ValueError('Mixed convection qualification failed; evidence preserved')
    if reference:
        expected={str(p.relative_to(reference)):sha(p) for p in Path(reference).glob('*.json')}
        actual={str(p.relative_to(out)):sha(p) for p in out.glob('*.json')}
        if expected!=actual:raise ValueError('Installed mixed-convection scientific JSON differs')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--reference',type=Path)
    a=p.parse_args();r=run(a.output,a.reference);print(json.dumps(dict(passed=r['passed'],station_usable=False)))

if __name__=='__main__':main()
