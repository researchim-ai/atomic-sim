"""Finite wall exchange versus a forward-EOS scalar energy ODE.

The reference has no HEM network, production inverse EOS or restart machinery.
"""
import argparse
import json
import math
from pathlib import Path

from scipy.integrate import solve_ivp
from atomic_model_hem_qualified.model import signature
from atomic_model_multiphysics.provenance import seal, sha
from atomic_model_training.spec import atomic_json
from scripts import phase_wall_case as wall
from scripts.verify_phase_cycles import CASES, CAPS, MESHES, forward_reference

BUDGETS = dict(pressure_pa=.1, temperature_k=1e-5,
               specific_energy_j_kg=2e-4, wall_temperature_k=1e-7)


def reference(case, times):
    endpoint, offset = CASES[case]
    initial = forward_reference(endpoint, offset)
    tw0 = initial.temperature_k + (10. if offset < 0 else -10.)
    def rhs(t, y):
        state = forward_reference(endpoint, float(y[0]))
        tw = tw0 - (float(y[0])-offset)/wall.WALL_SPECIFIC_CAPACITY
        return [wall.WALL_SPECIFIC_CONDUCTANCE*(tw-state.temperature_k)]
    outputs=[]
    for method in ('DOP853','Radau'):
        result=solve_ivp(rhs,(0.,times[-1]),[offset],method=method,
                         rtol=1e-11,atol=1e-10,max_step=2e-4,t_eval=times)
        if not result.success or len(result.t)!=len(times):
            raise ValueError('Independent wall reference incomplete')
        observations=[]
        for t,value in zip(result.t,result.y[0],strict=True):
            state=forward_reference(endpoint,float(value))
            observations.append(dict(time_s=float(t),pressure_pa=state.pressure_pa,
                temperature_k=state.temperature_k,specific_energy_j_kg=state.internal_energy_j_kg,
                wall_temperature_k=tw0-(float(value)-offset)/wall.WALL_SPECIFIC_CAPACITY,
                phase=state.phase))
        outputs.append(dict(method=method,observations=observations,nfev=result.nfev))
    return outputs


def errors(observations, reference_rows):
    if not observations or len(observations)!=len(reference_rows):raise ValueError('Incomplete wall histories')
    maxima={k:0. for k in BUDGETS}
    for obs,ref in zip(observations,reference_rows,strict=True):
        if any(not math.isfinite(v) for v in (obs['time_s'],ref['time_s'])) or abs(obs['time_s']-ref['time_s'])>1e-12:raise ValueError('Nonidentical wall times')
        if not obs['cells'] or len(obs['cells'])!=len(obs['walls']):raise ValueError('Incomplete wall cells')
        for c,w in zip(obs['cells'],obs['walls'],strict=True):
            values=dict(pressure_pa=c['pressure_pa'],temperature_k=c['temperature_k'],
                        specific_energy_j_kg=c['internal_energy_j_kg'],wall_temperature_k=w['temperature_k'])
            for key in BUDGETS:
                if not math.isfinite(values[key]) or not math.isfinite(ref[key]):raise ValueError('Nonfinite wall state')
                maxima[key]=max(maxima[key],abs(values[key]-ref[key]))
    return maxima


def run(output):
    output=Path(output).absolute()
    if any(p.is_symlink() for p in (output,*output.parents)):raise ValueError('Symlink output')
    source=signature()
    drivers={str(Path(p).absolute()):sha(p) for p in (__file__,wall.__file__,Path(__file__).with_name('verify_phase_cycles.py'))}
    output.mkdir(parents=True,exist_ok=False)
    atomic_json(output/'protocol.json',seal(dict(source=source,drivers=drivers,budgets=BUDGETS,
                methods=['DOP853','Radau'],meshes=MESHES,caps=CAPS,
                diameter_m=.08,
                reference='Scalar closed energy ODE and forward Gibbs roots, no production rho_u',
                validated=False,plant_referenced=False)))
    records=[]
    for case in CASES:
        refs=reference(case,[0.,.0025,.005,.0075,.01])
        atomic_json(output/f'{case}-reference.json',seal(refs))
        for cells in MESHES:
            for cap in CAPS:
                # Same geometry on BOTH meshes. At .05m the four-cell vapor
                # walls fall below the existing 100 J/K parser floor.
                result=wall.execute(case,cells=cells,cap=cap,diameter_m=.08)
                comparisons=[dict(method=ref['method'],maximum_errors=errors(result['observations'],ref['observations'])) for ref in refs]
                phase=all(obs['cells'][0]['phase']==ref['phase'] for obs,ref in zip(result['observations'],refs[0]['observations'],strict=True))
                passed=result['record']['passed'] and phase and all(c['maximum_errors'][k]<=BUDGETS[k] for c in comparisons for k in BUDGETS)
                record=dict(case=case,cells=cells,cap_s=cap,passed=passed,phase_matches=phase,
                            native_checks=result['record']['checks'],comparisons=comparisons)
                records.append(record)
                atomic_json(output/f'{case}-{cells}-{cap:g}.json',seal(dict(record=record,native=result)))
                print(case,cells,cap,passed,flush=True)
    unchanged=signature()==source and all(sha(p)==h for p,h in drivers.items())
    report=dict(completed=True,source_unchanged=unchanged,passed=unchanged and all(r['passed'] for r in records),
                runs=records,validated=False,plant_referenced=False)
    atomic_json(output/'report.json',seal(report))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    result=run(parser.parse_args().output)
    return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
