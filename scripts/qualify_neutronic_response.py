"""Synthetic 1D flux-weighted temperature response, not a BEAVRS core model."""
import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
from atomic_model_spatial import Model
from atomic_model_spatial.cases import slab
from atomic_model_spatial.reference import dense_k
from atomic_model_multiphysics.absorber import insert
from atomic_model_multiphysics.provenance import seal,sha
from atomic_model_training.spec import atomic_json
from atomic_model_neutronic_response import absorption_temperature_response,signature

CELLS=(12,24,48)
TIPS=(0.,.37,.71)
STEPS_K=(2.,1.,.5)
CRITERIA=dict(k_dense_relative=1e-8,coefficient_dense_relative=1e-6,rod_effect_minimum_pcm_per_k=1e-3)


def fixture(cells,tip):
    spec=insert(slab(cells,heterogeneous=True),tip,.2)
    d=np.zeros((2,cells));x=(np.arange(cells)+.5)/cells
    d[1]=1e-5*(1+3*x)
    return spec,d


def run(output):
    output=Path(output).absolute()
    if any(p.is_symlink() for p in (output,*output.parents)):raise ValueError('Symlink output')
    source=signature();driver=sha(__file__)
    from atomic_model_multiphysics.provenance import identity
    helpers=identity()
    output.mkdir(parents=True,exist_ok=False)
    atomic_json(output/'protocol.json',seal(dict(source=source,helpers=helpers,driver_sha256=driver,cells=CELLS,
        tips_m=TIPS,temperature_steps_k=STEPS_K,criteria=CRITERIA,coefficient_family='numerical',
        scope='Synthetic heterogeneous slab; imposed absorption derivative, all other coefficients fixed',
        beavrs_data_used=False,whole_plant_compared=False)))
    rows=[]
    for cells in CELLS:
        for tip in TIPS:
            spec,d=fixture(cells,tip)
            response=absorption_temperature_response(spec,d,derivative_provenance='Synthetic thermal-group derivative 1e-5*(1+3*x), fast derivative zero; not evaluated nuclear data')
            reference_k=dense_k(Model(spec));differences=[]
            for h in STEPS_K:
                plus=deepcopy(spec);minus=deepcopy(spec)
                plus['absorption_per_m']=(np.asarray(spec['absorption_per_m'])+h*d).tolist()
                minus['absorption_per_m']=(np.asarray(spec['absorption_per_m'])-h*d).tolist()
                finite=(1./dense_k(Model(minus))-1./dense_k(Model(plus)))/(2*h)
                relative=abs(finite-response['temperature_coefficient_per_k'])/max(abs(response['temperature_coefficient_per_k']),1e-30)
                differences.append(dict(step_k=h,dense_coefficient_per_k=finite,relative_error=relative))
            rows.append(dict(cells=cells,tip_m=tip,response=response,dense_k=reference_k,
                k_relative_error=abs(reference_k-response['k'])/reference_k,finite_differences=differences))
            print('cells',cells,'tip',tip,flush=True)
    effects=[dict(cells=n,spread_pcm_per_k=max(r['response']['temperature_coefficient_pcm_per_k'] for r in rows if r['cells']==n)
                 -min(r['response']['temperature_coefficient_pcm_per_k'] for r in rows if r['cells']==n)) for n in CELLS]
    unchanged=source==signature() and helpers==identity() and driver==sha(__file__)
    passed=(unchanged and all(r['k_relative_error']<CRITERIA['k_dense_relative'] for r in rows)
            and all(x['relative_error']<CRITERIA['coefficient_dense_relative'] for r in rows for x in r['finite_differences'])
            and all(x['spread_pcm_per_k']>CRITERIA['rod_effect_minimum_pcm_per_k'] for x in effects))
    report=dict(completed=True,passed=passed,source_unchanged=unchanged,rows=rows,rod_effects=effects,
        coefficient_family='numerical',beavrs_data_used=False,spatial_order_established=False,
        validated=False,plant_referenced=False,whole_plant_compared=False,
        limitations=['Supplied absorption derivatives only; fission, diffusion, scattering and geometry fixed',
                     'Synthetic 1D coefficients and geometry; no evaluated BEAVRS derivative data',
                     'Standalone steady neutronic response; not connected to dynamic PWR runtime'])
    atomic_json(output/'report.json',seal(report));return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.output)
