"""Repeat sealed native MSL wall executables with smaller internal time steps.

Reuses compiled equations, never reuses CSV results or changes phase budgets.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess

from atomic_model_hem_qualified.model import signature
from atomic_model_multiphysics.provenance import seal, sha, unseal
from atomic_model_training.spec import atomic_json
from scripts import benchmark_modelica_cycles as bench
from scripts.verify_phase_cycles import CASES

CAPS = (1e-4, 2e-5, 1e-5)


def assess(native, peer):
    errors=bench.compare(native['observations'],peer,times=bench.WALL_TIMES,wall=True)
    wall=native['request']['model']['walls'][0]
    initial=peer[0]
    mass=max(abs(r['mass']-initial['mass']) for r in peer)
    energy=max(abs(r['energy']-initial['energy']+wall['capacity_j_k']*(r['wallTemperature']-initial['wallTemperature'])) for r in peer)
    phase=all((peer[i]['phase']==2.)==(native['observations'][i]['cells'][0]['phase']=='two_phase') for i in (0,2,4))
    checks=dict(atomic_passed=native['record']['passed'],peer_agreement=all(errors[k]<=bench.BUDGETS[k] for k in bench.BUDGETS) and errors['wall_temperature_k']<=1e-3,
                phase_paths_match=phase,peer_mass_balance=mass<=bench.BUDGETS['mass_kg'],peer_energy_balance=energy<=bench.BUDGETS['energy_j'])
    return dict(passed=all(checks.values()),checks=checks,maximum_errors=errors,
                peer_mass_error_kg=mass,peer_energy_error_j=energy)


def run(campaign,output,*,tolerance=None):
    if tolerance is not None and (type(tolerance) not in (float,int) or not math.isfinite(tolerance) or not 1e-14<=tolerance<=1e-9):
        raise ValueError('Declared solver tolerance1e-14..1e-9')
    campaign,output=Path(campaign).absolute(),Path(output).absolute()
    if any(p.is_symlink() for p in (output,*output.parents)):raise ValueError('Symlink output')
    protocol=unseal(json.loads((campaign/'protocol.json').read_text()))
    if protocol['source']!=signature() or not protocol['include_wall']:
        raise ValueError('Matching physical source and wall campaign required')
    inputs={str(campaign/'protocol.json'):sha(campaign/'protocol.json')}
    jobs=[]
    for case in CASES:
        folder=campaign/f'p-1e+06-{case}-wall'
        native_path=folder/'atomic.json'
        native=unseal(json.loads(native_path.read_text()))
        if not native['record']['passed']:raise ValueError('Native wall qualification failed')
        inputs[str(native_path)]=sha(native_path)
        for method in bench.METHODS:
            comparison_path=folder/f'{method}-comparison.json'
            original=unseal(json.loads(comparison_path.read_text()))
            binary=folder/'AtomicHeatCycle'
            if original['record']['native_model_sha256']!=sha(binary):raise ValueError('Native executable changed')
            for p in (comparison_path,binary,folder/'AtomicHeatCycle_init.xml'):
                inputs[str(p)]=sha(p)
            jobs.append((case,folder,method,native))
    drivers={str(Path(p).absolute()):sha(p) for p in (__file__,bench.__file__)}
    output.mkdir(parents=True,exist_ok=False)
    atomic_json(output/'protocol.json',seal(dict(inputs=inputs,drivers=drivers,source=signature(),
                caps_s=CAPS,tolerance=protocol['rtol'] if tolerance is None else tolerance,
                scope='Fixed compiled MSL equations, reduced integrator step caps; original phase acceptance',
                source_protocol_sha256=sha(campaign/'protocol.json'),validated=False,plant_referenced=False)))
    records=[]
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    for case,folder,method,native in jobs:
        for cap in CAPS:
            stem=f'{case}-{method}-{cap:g}'
            csv_path=output/f'{stem}.csv'
            try:
                with (output/f'{stem}.log').open('x') as log:
                    command=[str(folder/'AtomicHeatCycle'),f'-s={method}',f'-maxStepSize={cap:.17g}',f'-r={csv_path}']
                    if tolerance is not None:command.append(f'-tolerance={tolerance:.17g}')
                    subprocess.run(command,
                                   cwd=folder,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=120,check=True)
                peer=bench.load_csv(csv_path,times=bench.WALL_TIMES,wall=True)
                record=dict(case=case,method=method,cap_s=cap,**assess(native,peer),csv_sha256=sha(csv_path))
                atomic_json(output/f'{stem}.json',seal(dict(record=record,peer_observations=peer)))
            except (OSError,ValueError,KeyError,subprocess.SubprocessError) as error:
                record=dict(case=case,method=method,cap_s=cap,passed=False,error=str(error))
            records.append(record)
            print(case,method,cap,record['passed'],flush=True)
            atomic_json(output/'progress.json',seal(dict(completed=False,runs=records)))
    unchanged=signature()==protocol['source'] and all(sha(p)==h for p,h in {**drivers,**inputs}.items())
    report=dict(completed=True,source_unchanged=unchanged,runs=records,
                passed=unchanged and len(records)==24 and all(r['passed'] for r in records),
                validated=False,plant_referenced=False,original_dynamic_report_promoted=False)
    atomic_json(output/'report.json',seal(report))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--tolerance',type=float)
    args=parser.parse_args()
    return 0 if run(args.campaign,args.output,tolerance=args.tolerance)['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
