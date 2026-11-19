"""Prepare a separate, explicitly changed spectral initial checkpoint."""
import argparse
from pathlib import Path
import shutil
import numpy as np
from scipy.io import netcdf_file
from atomic_model_station_initialization import project_velocity
from atomic_model_station_initialization import projection
from atomic_model_station_spectral import read_state, audit_state
from atomic_model_multiphysics.provenance import seal, sha
from atomic_model_training.spec import atomic_json


def run(output, initial, weights, maximum_relative_correction):
    out = Path(output).absolute(); initial = Path(initial).absolute(); weights = Path(weights).absolute()
    if out.exists() or any(p.is_symlink() for p in (out, *out.parents)):
        raise ValueError('Fresh ordinary output directory required')
    source = dict(driver=sha(__file__), projection=sha(projection.__file__))
    input_sha = sha(initial); weights_sha = sha(weights)
    state = read_state(initial); rule = np.loadtxt(weights)
    if rule.shape != (len(state['radius']), 2) or not np.array_equal(rule[:, 0], state['radius']):
        raise ValueError('Quadrature coordinates must match checkpoint exactly')
    fields, report = project_velocity(*(state['fields'][name] for name in ('Ur', 'Ut', 'Uz')),
        state['radius'], rule[:, 1], *state['dimensions'][:2], mp=state['dimensions'][2],
        alpha=state['metadata']['alpha'], maximum_relative_correction=maximum_relative_correction)
    if sha(initial) != input_sha or sha(weights) != weights_sha:
        raise ValueError('Initialization input changed')
    out.mkdir(parents=True); candidate = out / 'state.cdf.in'
    shutil.copyfile(initial, candidate)
    with netcdf_file(candidate, 'a', mmap=False) as f:
        for name, values in zip(('Ur', 'Ut', 'Uz'), fields):
            f.variables[name][0] = values.real; f.variables[name][1] = values.imag
    after = read_state(candidate)
    if after['metadata'] != state['metadata'] or not np.array_equal(after['fields']['T'], state['fields']['T']):
        raise ValueError('Thermal state or physical metadata changed')
    if source != dict(driver=sha(__file__), projection=sha(projection.__file__)):
        raise ValueError('Initialization source changed')
    receipt = dict(passed=True, source=source, initial_sha256=input_sha, weights_sha256=weights_sha,
        candidate_sha256=sha(candidate), initial_unchanged=sha(initial)==input_sha,
        thermal_field_unchanged=True, metadata_unchanged=True, projection=report,
        before=audit_state(initial,rule[:,1]), after=audit_state(candidate,rule[:,1]),
        scope='Explicit new initial data, not a repair during evolution or DNS acceptance',
        closed_tasks=[], station_usable=False, validated=False)
    atomic_json(out/'report.json', seal(receipt))
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('output','initial','weights'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--maximum-relative-correction',type=float,required=True)
    a=p.parse_args();r=run(a.output,a.initial,a.weights,a.maximum_relative_correction)
    print(r['projection'])

if __name__=='__main__':main()
