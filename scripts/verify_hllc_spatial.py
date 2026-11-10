"""Actual liquid graph versus conservative nonlinear Riemann cell averages."""
import argparse
from copy import deepcopy
import math
from pathlib import Path
import numpy as np
from atomic_model_hem_contact import ContactHEMNetwork
from atomic_model_hem_contact.model import signature
from atomic_model_hem_qualified import eos
from atomic_model_multiphysics.provenance import seal,sha
from atomic_model_training.spec import atomic_json
from scripts import liquid_riemann_profile as exact
from scripts import liquid_riemann as oracle
from scripts import benchmark_modelica_network as network

MESHES=(16,32,64)
SCHEMES=('hllc','hllc-local')
CASES=('moderate','wide')
LENGTH=10.;TIME=.001;CAP=1e-5;EREF=7e5
MAX_SPATIAL_RATIO=.98
MAX_TIME_FRACTION=.05


def states(case):
    if case not in CASES:raise ValueError('Declared spatial case required')
    return (eos.pt(4.02e6,441.),eos.pt(3.98e6,439.)) if case=='moderate' else (eos.pt(14e6,441.),eos.pt(2e6,439.))


def request_for(case,cells,cap=CAP):
    if cells not in MESHES or cap not in (CAP,CAP/2):raise ValueError('Declared spatial matrix required')
    left,right=states(case);request=network.request_for('pressure_thermal_step',2,1e-5)
    def initial(s):return dict(kind='pt',pressure_pa=s.pressure_pa,temperature_k=s.temperature_k,velocity_m_s=0.)
    spec=request['model'];spec.update(max_step_s=cap,wave_speed_bound_m_s=2500.)
    for node,state in zip(spec['nodes'],(left,right),strict=True):node['initial']=initial(state)
    for edge in spec['edges']:
        edge.update(cells=cells,length_m=LENGTH,heat_power_w=[0.]*cells)
        edge['initial']=[initial(left if (j<cells//2)==(edge['from_node']=='a') else right) for j in range(cells)]
    return request


def reference(case,cells):
    left,right=states(case)
    profile=exact.Profile(left,right,pressure_bracket=(1e6,15e6))
    bound=max(abs(profile.solution.left_outer_speed_m_s),abs(profile.solution.right_outer_speed_m_s))
    if bound*TIME>=LENGTH/2:raise ValueError('Physical wave reaches graph boundary')
    travel=min(abs(profile.solution.left_outer_speed_m_s),abs(profile.solution.right_outer_speed_m_s))*TIME
    if travel/(LENGTH/min(MESHES))<2:raise ValueError('Coarsest grid does not resolve two cells of acoustic travel')
    dx=LENGTH/cells
    values=[profile.cell_average(j*dx,(j+1)*dx,TIME,origin=LENGTH/2,energy_reference=EREF) for j in range(cells)]
    pressures=[]
    for rho,momentum,energy in values:
        internal=EREF+energy/rho-.5*(momentum/rho)**2
        pressures.append(eos.rho_u(float(rho),float(internal),8e6,440.).pressure_pa)
    return dict(conserved=[list(map(float,v)) for v in values],pressure_pa=pressures)


def execute(case,cells,scheme,cap,ref):
    request=request_for(case,cells,cap);model=ContactHEMNetwork(request,scheme=scheme)
    initial=model.snapshot();final=model.advance(TIME)
    cell_by_id={c['id']:c for c in final['cells']};curves=[]
    for edge in request['model']['edges']:
        ordered=[cell_by_id[f'{edge["id"]}/{j}'] for j in range(cells)]
        forward=edge['from_node']=='a'
        if not forward:ordered=ordered[::-1]
        values=[]
        for cell in ordered:
            rho=cell['mass_kg']/cell['volume_m3'];momentum=cell['momentum_kg_m_s']/cell['volume_m3']
            energy=cell['total_energy_j']/cell['volume_m3']-EREF*rho
            values.append([rho,momentum if forward else -momentum,energy])
        errors=np.abs(np.asarray(values)-np.asarray(ref['conserved']))
        pressure=np.array([c['pressure_pa'] for c in ordered])
        curves.append(dict(edge=edge['id'],conserved=values,pressure_pa=list(map(float,pressure)),
            l1=dict(density_kg_m3=float(np.mean(errors[:,0])),momentum_kg_m2_s=float(np.mean(errors[:,1])),
                    centered_energy_j_m3=float(np.mean(errors[:,2])),pressure_pa=float(np.mean(abs(pressure-ref['pressure_pa']))))))
    errors={k:max(c['l1'][k] for c in curves) for k in curves[0]['l1']}
    ledger=final['global_ledger']
    checks=dict(positive_mass=all(c['mass_kg']>0 for c in final['cells']),liquid=all(c['phase']=='liquid' for c in final['cells']),
                mass_balance=abs(ledger['mass_residual_kg'])<=1e-9,energy_balance=abs(ledger['energy_residual_j'])<=1e-5,
                entropy=ledger['numerical_entropy_residual_j_k']>=-2e-5,
                boundary_drift=max(abs(a['pressure_pa']-b['pressure_pa']) for a,b in zip(initial['cells'][:2],final['cells'][:2],strict=True))<=.001*abs(states(case)[0].pressure_pa-states(case)[1].pressure_pa))
    return dict(case=case,cells=cells,scheme=scheme,cap_s=cap,l1=errors,checks=checks,passed=all(checks.values()),
                request=request,initial=initial,final=final,curves=curves,reference=ref)


def assess(rows):
    checks={};ratios=[]
    expected={(case,cells,scheme,CAP) for case in CASES for cells in MESHES for scheme in SCHEMES}
    expected|={(case,MESHES[-1],'hllc-local',CAP/2) for case in CASES}
    keys=[(r['case'],r['cells'],r['scheme'],r['cap_s']) for r in rows]
    if len(keys)!=len(set(keys)) or set(keys)!=expected:raise ValueError('Incomplete or duplicate spatial matrix')
    if any(not math.isfinite(v) or v<=0 for r in rows for v in r['l1'].values()):
        raise ValueError('Finite positive spatial errors required')
    for case in CASES:
        for scheme in SCHEMES:
            series=sorted((r for r in rows if r['case']==case and r['scheme']==scheme and r['cap_s']==CAP),key=lambda r:r['cells'])
            if [r['cells'] for r in series]!=list(MESHES):raise ValueError('Incomplete spatial matrix')
            for metric in ('pressure_pa','momentum_kg_m2_s'):
                values=[r['l1'][metric] for r in series]
                rs=[b/a for a,b in zip(values,values[1:])]
                checks[f'{case}-{scheme}-{metric}-refines']=all(r<MAX_SPATIAL_RATIO for r in rs)
                ratios.append(dict(case=case,scheme=scheme,metric=metric,ratios=rs,orders=[-math.log2(r) for r in rs]))
        local=next(r for r in rows if r['case']==case and r['scheme']=='hllc-local' and r['cells']==MESHES[-1] and r['cap_s']==CAP)
        fixed=next(r for r in rows if r['case']==case and r['scheme']=='hllc' and r['cells']==MESHES[-1])
        fine=next(r for r in rows if r['case']==case and r['cap_s']==CAP/2)
        for metric in ('pressure_pa','momentum_kg_m2_s'):
            checks[f'{case}-{metric}-improves-fixed']=local['l1'][metric]<fixed['l1'][metric]
        # Compare solutions, not the difference between their error norms.
        drift=max(float(np.mean(abs(np.array(a['pressure_pa'])-np.array(b['pressure_pa'])))) for a,b in zip(local['curves'],fine['curves'],strict=True))
        checks[f'{case}-time-error-small']=drift<=MAX_TIME_FRACTION*local['l1']['pressure_pa']
    return dict(checks=checks,ratios=ratios,passed=all(checks.values()))


def run(output):
    output=Path(output).absolute()
    if any(p.is_symlink() for p in (output,*output.parents)):raise ValueError('Symlink output')
    source=signature();drivers={str(Path(p).absolute()):sha(p) for p in (__file__,exact.__file__,oracle.__file__,network.__file__)}
    output.mkdir(parents=True,exist_ok=False)
    atomic_json(output/'protocol.json',seal(dict(source=source,drivers=drivers,meshes=MESHES,schemes=SCHEMES,cases=CASES,
        length_m=LENGTH,horizon_s=TIME,cap_s=CAP,min_coarse_acoustic_travel_cells=2.,max_boundary_pressure_drift_fraction=.001,max_spatial_ratio=MAX_SPATIAL_RATIO,max_time_fraction=MAX_TIME_FRACTION,
        reference='Conservative cell averages of nonlinear IF97 Riemann solution before physical boundary arrival',
        validated=False,plant_referenced=False)))
    rows=[]
    for case in CASES:
        for cells in MESHES:
            ref=reference(case,cells)
            for scheme in SCHEMES:
                row=execute(case,cells,scheme,CAP,ref);rows.append(row)
                atomic_json(output/f'{case}-{cells}-{scheme}.json',seal(row))
                print(case,cells,scheme,row['passed'],row['l1'],flush=True)
        ref=reference(case,MESHES[-1]);row=execute(case,MESHES[-1],'hllc-local',CAP/2,ref);rows.append(row)
        atomic_json(output/f'{case}-time-refinement.json',seal(row))
    assessment=assess(rows)
    unchanged=source==signature() and all(sha(p)==h for p,h in drivers.items())
    report=dict(completed=True,passed=unchanged and all(r['passed'] for r in rows) and assessment['passed'],source_unchanged=unchanged,
        rows=[{k:r[k] for k in ('case','cells','scheme','cap_s','l1','checks','passed')} for r in rows],assessment=assessment,
        validated=False,plant_referenced=False,whole_plant_compared=False)
    atomic_json(output/'report.json',seal(report));return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path)
    return 0 if run(p.parse_args().output)['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
