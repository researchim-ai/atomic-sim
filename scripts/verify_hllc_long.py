"""Forty-millisecond graph transients, late restart and temporal refinement.

Independent absolute-stock Radau/BDF references; same spatial law and EOS.
No physical/experimental validation or whole-plant claim.
"""
import argparse
from copy import deepcopy
import math
from pathlib import Path
from atomic_model_hem_contact import ContactHEMNetwork
from atomic_model_hem_contact.model import signature
from scripts import long_hllc_reference as reference_solver
from atomic_model_hem_qualified import eos
from atomic_model_multiphysics.provenance import seal,sha
from atomic_model_training.spec import atomic_json
from scripts import graph_history as history
from scripts import benchmark_modelica_hllc as cases
from scripts import verify_heterogeneous_network as peer

TIMES=(0.,.00025,.0005,.001,.0025,.005,.0075,.01,.0125,.015,.0175,.02,.0225,.025,.0275,.03,.0325,.035,.0375,.04)
RESTART_INDEX=11
MATRIX=(('pressure_thermal_step',2),('pressure_thermal_step',4),('reverse_pressure_step',2),('wide_pressure_step',2))
CAPS=(4e-5,2e-5,1e-5)
REFERENCE_CAP=5e-5
REFERENCE_METHODS=('Radau','BDF')
ERROR_RATIO=.8
ERROR_FLOORS=dict(pressure_pa=1e-3,momentum_kg_m_s=1e-10)


def request_for(case,cells,cap):
    if (case,cells) not in MATRIX or cap not in CAPS:raise ValueError('Declared long graph matrix required')
    request=cases.request_for(case,cells,cases.CAPS[0]);request['model']['max_step_s']=cap
    return request


def capture(request,*,times=TIMES,restart_index=RESTART_INDEX):
    times=history.validate_times(times)
    if isinstance(restart_index,bool) or not isinstance(restart_index,int) or not 0<restart_index<len(times)-1:
        raise ValueError('Interior restart index required')
    model=ContactHEMNetwork(request,scheme='hllc-local');observations=[model.snapshot()];checkpoint=None
    for index,(a,b) in enumerate(zip(times,times[1:]),1):
        observations.append(model.advance(b-a))
        if index==restart_index:checkpoint=model.checkpoint()
    restored=ContactHEMNetwork(request,scheme='hllc-local');restored.restore(checkpoint)
    for a,b in zip(times[restart_index:],times[restart_index+1:]):last=restored.advance(b-a)
    checks=dict(restart_exact=last==observations[-1] and restored.checkpoint()==model.checkpoint(),
        liquid_positive=all(c['phase']=='liquid' and c['mass_kg']>0 for s in observations for c in s['cells']),
        mass_balance=all(abs(s['global_ledger']['mass_residual_kg'])<=1e-9 for s in observations),
        energy_balance=all(abs(s['global_ledger']['energy_residual_j'])<=1e-5 for s in observations),
        entropy=all(s['global_ledger']['numerical_entropy_residual_j_k']>=-2e-5 for s in observations),
        adiabatic=all(s['global_ledger'][k]==0. for s in observations for k in ('external_fluid_heat_j','external_wall_heat_j','external_entropy_j_k')))
    return dict(request=request,observations=observations,checkpoint=checkpoint,
        record=dict(scheme='hllc-local',restart_time_s=times[restart_index],checks=checks,passed=all(checks.values())))


def reference_packet(request,reference):
    observations=deepcopy(reference['observations'])
    volumes={n['id']:n['volume_m3'] for n in request['model']['nodes']}
    for edge in request['model']['edges']:
        volume=math.pi*edge['diameter_m']**2/4*edge['length_m']/edge['cells']
        volumes.update({f'{edge["id"]}/{j}':volume for j in range(edge['cells'])})
    for observation in observations:
        for cell in observation['cells']:
            cell['volume_m3']=volumes[cell['id']]
            cell['velocity_m_s']=cell['momentum_kg_m_s']/cell['mass_kg']
    return dict(request=request,observations=observations,record=dict(passed=True))


def refinement(comparisons):
    expected={(cap,method) for cap in CAPS for method in REFERENCE_METHODS}
    keys=[(r['cap_s'],r['method']) for r in comparisons]
    if len(set(keys))!=len(keys) or set(keys)!=expected:raise ValueError('Incomplete or duplicated temporal matrix')
    checks={};records=[]
    if any(not math.isfinite(r['maximum_errors'][metric]) or r['maximum_errors'][metric]<0 for r in comparisons for metric in ERROR_FLOORS):
        raise ValueError('Finite nonnegative temporal errors required')
    for metric,floor in ERROR_FLOORS.items():
        values=[max(r['maximum_errors'][metric] for r in comparisons if r['cap_s']==cap) for cap in CAPS]
        if any(not math.isfinite(v) or v<0 for v in values):raise ValueError('Finite nonnegative temporal errors required')
        flags=[b<=ERROR_RATIO*a+floor for a,b in zip(values,values[1:])]
        checks[metric+'_refines']=all(flags)
        records.append(dict(metric=metric,errors=values,ratios=[b/a if a>0 else None for a,b in zip(values,values[1:])],floor=floor))
    checks['finest_agreement']=all(r['passed'] for r in comparisons if r['cap_s']==CAPS[-1])
    return dict(checks=checks,metrics=records,passed=all(checks.values()))


def reference_accuracy(case,method):
    # Tighten the independently observed wide-case BDF energy discrepancy.
    return dict(rtol=1e-11,atol=5e-11) if case=='wide_pressure_step' and method=='BDF' else dict(rtol=2e-11,atol=1e-10)


def run(output):
    output=Path(output).absolute()
    if any(p.is_symlink() for p in (output,*output.parents)):raise ValueError('Symlink output')
    source=signature();drivers={str(Path(p).absolute()):sha(p) for p in (__file__,history.__file__,cases.__file__,peer.__file__,reference_solver.__file__)}
    output.mkdir(parents=True,exist_ok=False)
    atomic_json(output/'protocol.json',seal(dict(source=source,drivers=drivers,times_s=TIMES,restart_time_s=TIMES[RESTART_INDEX],matrix=MATRIX,caps_s=CAPS,
        reference_cap_s=REFERENCE_CAP,reference_window_budget=20000,reference_total_budget=reference_solver.TOTAL_EVALUATIONS,reference_methods=REFERENCE_METHODS,reference_first_step_s=1e-6,reference_accuracy_profiles={case:{method:reference_accuracy(case,method) for method in REFERENCE_METHODS} for case,_ in MATRIX},error_ratio=ERROR_RATIO,error_floors=ERROR_FLOORS,
        finest_budgets=history.BUDGETS,reference_cross_budgets=peer.REFERENCE_BUDGETS,accuracy_required_caps_s=[CAPS[-1]],
        scope='40ms closed adiabatic liquid graph; same spatial law, independent time/RH assembly; coarse caps are convergence probes',validated=False,plant_referenced=False)))
    records=[]
    for case,cells in MATRIX:
        folder=output/f'{case}-{cells}';folder.mkdir()
        try:
            request=request_for(case,cells,CAPS[-1])
            references=[reference_solver.solve(request,TIMES,method,max_step=REFERENCE_CAP,**reference_accuracy(case,method),on_window=lambda w:atomic_json(folder/f'{method}-progress.json',seal(w))) for method in REFERENCE_METHODS]
            atomic_json(folder/'references.json',seal(dict(references=references)))
            cross=history.compare(reference_packet(request,references[0]),peer.as_peer(references[1]['observations']),times=TIMES,budgets=peer.REFERENCE_BUDGETS)
            comparisons=[];runs=[]
            for cap in CAPS:
                native=capture(request_for(case,cells,cap));atomic_json(folder/f'native-{cap}.json',seal(native))
                for reference in references:
                    result=history.compare(native,peer.as_peer(reference['observations']),times=TIMES)
                    comparisons.append(dict(cap_s=cap,method=reference['method'],**result))
                runs.append(dict(cap_s=cap,**native['record']))
                print(case,cells,cap,native['record']['passed'],flush=True)
            refined=refinement(comparisons)
            initial_states=[eos.pt(n['initial']['pressure_pa'],n['initial']['temperature_k']) for n in request['model']['nodes']]
            transits=TIMES[-1]*min(eos.characteristic_speeds(s)[0] for s in initial_states)/max(e['length_m'] for e in request['model']['edges'])
            record=dict(case=case,cells=cells,native_runs=runs,comparisons=comparisons,reference_cross=cross,refinement=refined,
                reference_evaluations={r['method']:r['evaluations'] for r in references},acoustic_transits_lower=transits,
                passed=cross['passed'] and refined['passed'] and all(r['passed'] for r in runs))
        except (ValueError,RuntimeError,OSError,KeyError) as error:record=dict(case=case,cells=cells,passed=False,error=str(error))
        records.append(record);atomic_json(folder/'report.json',seal(record))
        atomic_json(output/'progress.json',seal(dict(completed=False,records=records)))
        print('case',case,cells,record['passed'],record.get('error'),flush=True)
    unchanged=source==signature() and all(sha(p)==h for p,h in drivers.items())
    report=dict(completed=True,passed=unchanged and len(records)==len(MATRIX) and all(r['passed'] for r in records),source_unchanged=unchanged,
        records=records,validated=False,plant_referenced=False,whole_plant_compared=False,experimental_validation=False)
    atomic_json(output/'report.json',seal(report));return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path)
    return 0 if run(p.parse_args().output)['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
