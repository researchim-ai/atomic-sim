"""Independent temporal references and stationary-contact mesh error for HEM."""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path

from atomic_model_hem_qualified.model import QualifiedHEMNetwork,signature
from atomic_model_hem_qualified.reference import solve
from atomic_model_multiphysics.provenance import seal,sha
from atomic_model_training.spec import atomic_json
from scripts import benchmark_modelica_network as bench

REFERENCE_BUDGETS=dict(pressure_pa=.1,temperature_k=1e-5,internal_energy_j_kg=.01,
    mass_kg=1e-9,total_energy_j=1e-5,momentum_kg_m_s=1e-8,velocity_m_s=1e-6,
    face_mass_kg=1e-9,face_energy_j=1e-5,support_kg_m_s=1e-8)
SPATIAL_MESHES=(2,4,8,16,32)
MAX_L1_RATIO=.95
ENERGY_REFINEMENT_FLOOR_J=1e-6
MAX_TIME_ERROR_RATIO=.8


def contact_request(cells):
    if cells not in SPATIAL_MESHES:raise ValueError('Declared contact meshes only')
    request=bench.request_for('thermal_step',2,1e-5)
    for edge in request['model']['edges']:
        blocks=deepcopy(edge['initial']);edge['cells']=cells
        edge['initial']=[deepcopy(blocks[0 if i<cells//2 else 1]) for i in range(cells)]
        edge['heat_power_w']=[0.]*cells
    return request


def as_peer(observations):
    """Only public observations cross the independent-reference boundary."""
    peer=[]
    for obs in observations:
        row=dict(time=obs['time_s'])
        for i,c in enumerate(obs['cells'],1):
            for field,column in bench.COLUMNS.items():
                row[f'{column}[{i}]']=c[field] if field!='velocity_m_s' else c['momentum_kg_m_s']/c['mass_kg']
        for column,key in [('faceMass','integrated_face_mass_kg'),('faceEnergy','integrated_face_energy_j'),('support','node_support_impulse_kg_m_s')]:
            for i,value in enumerate(obs['local_ledgers'][key],1):row[f'{column}[{i}]']=value
        peer.append(row)
    return peer


def spatial():
    records=[];stocks=[]
    for cells in SPATIAL_MESHES:
        request=contact_request(cells);model=QualifiedHEMNetwork(request)
        first=model.snapshot();last=model.advance(bench.TIMES[-1])
        initial={c['id']:c for c in first['cells']}
        edge=[c for c in last['cells'] if '/' in c['id']]
        denominator=math.fsum(c['volume_m3'] for c in edge)
        error=math.fsum(c['volume_m3']*abs(c['temperature_k']-initial[c['id']]['temperature_k']) for c in edge)/denominator
        stocks.append(dict(mass=math.fsum(c['mass_kg'] for c in first['cells']),energy=math.fsum(c['total_energy_j'] for c in first['cells'])))
        records.append(dict(cells=cells,volume_weighted_l1_temperature_k=error,
            pressure_departure_pa=max(abs(c['pressure_pa']-4e6) for c in last['cells']),
            native_mass_residual_kg=last['global_ledger']['mass_residual_kg'],
            native_energy_residual_j=last['global_ledger']['energy_residual_j']))
    ratios=[b['volume_weighted_l1_temperature_k']/a['volume_weighted_l1_temperature_k'] for a,b in zip(records,records[1:])]
    checks=dict(same_initial_mass=all(abs(s['mass']-stocks[0]['mass'])<=1e-9 for s in stocks),
        same_initial_energy=all(abs(s['energy']-stocks[0]['energy'])<=1e-5 for s in stocks),
        decreasing_contact_error=all(0<r<=MAX_L1_RATIO for r in ratios),
        closed_mass=all(abs(r['native_mass_residual_kg'])<=1e-9 for r in records),
        closed_energy=all(abs(r['native_energy_residual_j'])<=1e-5 for r in records))
    return dict(passed=all(checks.values()),checks=checks,runs=records,ratios=ratios,initial_stocks=stocks,
        scope='Stationary isobaric Euler contact; Rusanov numerical smearing, not physical heat conduction')


def run(output):
    output=Path(output).absolute()
    if any(p.is_symlink() for p in (output,*output.parents)):raise ValueError('Symlink output')
    source=signature();drivers={str(Path(p).absolute()):sha(p) for p in (__file__,bench.__file__)}
    output.mkdir(parents=True,exist_ok=False)
    atomic_json(output/'protocol.json',seal(dict(source=source,drivers=drivers,budgets=REFERENCE_BUDGETS,
        cases=bench.CASES,meshes=bench.MESHES,caps=bench.QUALIFICATION_CAPS,spatial_meshes=SPATIAL_MESHES,
        maximum_contact_error_ratio=MAX_L1_RATIO,reference='Independent absolute-stock graph RHS, shared normative EOS; DOP853/Radau',
        energy_refinement_floor_j=ENERGY_REFINEMENT_FLOOR_J,maximum_temporal_error_ratio=MAX_TIME_ERROR_RATIO,
        validated=False,plant_referenced=False)))
    records=[]
    for case in bench.CASES:
        for cells in bench.MESHES:
            refs=[]
            request=bench.request_for(case,cells,bench.QUALIFICATION_CAPS[0])
            for method in ('DOP853','Radau'):
                result=solve(request,[dict(duration_s=b-a,command=None) for a,b in zip(bench.TIMES,bench.TIMES[1:])],method,max_step=5e-6)
                refs.append(result)
            natives={cap:bench.execute(case,cells,cap) for cap in bench.QUALIFICATION_CAPS}
            comparisons=[]
            for cap,native in natives.items():
                for ref in refs:
                    assessment=bench.compare(native,as_peer(ref['observations']))
                    assessment['checks']['strict_reference_agreement']=all(assessment['maximum_errors'][k]<=v for k,v in REFERENCE_BUDGETS.items())
                    assessment['passed']=all(assessment['checks'].values())
                    record=dict(case=case,cells=cells,cap_s=cap,method=ref['method'],**assessment)
                    records.append(record);comparisons.append(record)
                    print(case,cells,cap,ref['method'],record['passed'],flush=True)
            atomic_json(output/f'{case}-{cells}.json',seal(dict(references=refs,native={str(k):v for k,v in natives.items()},comparisons=comparisons)))
    refinement=[]
    for case in bench.CASES:
        for cells in bench.MESHES:
            for method in ('DOP853','Radau'):
                pair=[next(r for r in records if r['case']==case and r['cells']==cells and r['method']==method and r['cap_s']==cap) for cap in bench.QUALIFICATION_CAPS]
                coarse,fine=[max(r['maximum_errors'][k] for k in ('total_energy_j','face_energy_j')) for r in pair]
                ratio=fine/coarse if coarse else None
                evaluated=coarse>ENERGY_REFINEMENT_FLOOR_J
                refinement.append(dict(case=case,cells=cells,method=method,coarse_energy_error_j=coarse,
                    fine_energy_error_j=fine,ratio=ratio,evaluated=evaluated,passed=not evaluated or ratio<=MAX_TIME_ERROR_RATIO))
    contact=spatial();atomic_json(output/'contact.json',seal(contact))
    unchanged=signature()==source and all(sha(p)==h for p,h in drivers.items())
    report=dict(completed=True,source_unchanged=unchanged,passed=unchanged and all(r['passed'] for r in records) and all(r['passed'] for r in refinement) and contact['passed'],
        temporal=records,temporal_refinement=refinement,contact=contact,validated=False,plant_referenced=False)
    atomic_json(output/'report.json',seal(report));return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True,type=Path)
    return 0 if run(parser.parse_args().output)['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
