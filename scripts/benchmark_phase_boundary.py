"""Independent MSL phase classifier at identical rho/T; no time integration error.

This diagnostic never converts a failed dynamic comparison into acceptance.
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess

from atomic_model_hem_qualified import eos
from atomic_model_hem_qualified.model import signature
from atomic_model_multiphysics.provenance import seal, sha, unseal
from atomic_model_training.spec import atomic_json
from physics.materials import iapws
from scripts.benchmark_modelica_cycles import library_manifest

PRESSURES = (5e5, 1e6, 8e6, 15.5e6)
OFFSETS = (-1e-4, -1e-6, -1e-8, -1e-10, -1e-12, 0.,
           1e-12, 1e-10, 1e-8, 1e-6, 1e-4)


def reference(rho, temperature):
    """Direct Gibbs endpoints at the given T, independent of inverse rho/u."""
    if not math.isfinite(rho) or rho <= 0 or not 300 <= temperature < 623.15:
        raise ValueError('Finite positive density and sub-region3 saturation T required')
    p = float(iapws.saturation_pressure_mpa(temperature))
    liquid = 1 / float(iapws._region1_raw(temperature, p)['v'])
    vapor = 1 / float(iapws._region2_raw(temperature, p)['v'])
    region = 1 if rho > liquid else 2 if rho < vapor else 4
    # MSL phase_dT uses strict inequalities; region_dT includes endpoints.
    return dict(liquid_density=liquid, vapor_density=vapor, region=region,
                phase=2 if vapor < rho < liquid else 1,
                distance_to_boundary_kg_m3=min(abs(rho-liquid), abs(rho-vapor)))


def states():
    rows = []
    for pressure in PRESSURES:
        for side, endpoint in zip(('liquid', 'vapor'), eos.saturation(pressure), strict=True):
            for offset in OFFSETS:
                rows.append(dict(pressure_pa=pressure, side=side, relative_density_offset=offset,
                                 density=endpoint.density_kg_m3*(1+offset),
                                 temperature=endpoint.temperature_k))
    return rows


def render(rows):
    if not rows:
        raise ValueError('Nonempty boundary matrix required')
    lines = ['model AtomicPhaseBoundary',
             '  package IF97 = Modelica.Media.Water.IF97_Utilities;']
    for i, row in enumerate(rows, 1):
        reference(row['density'], row['temperature'])
        rho, t = row['density'], row['temperature']
        for name, expr in dict(liquid=f'IF97.rhol_T({t:.17g})',
                               vapor=f'IF97.rhov_T({t:.17g})',
                               phase=f'IF97.phase_dT({rho:.17g},{t:.17g})',
                               region=f'IF97.BaseIF97.Regions.region_dT({rho:.17g},{t:.17g})').items():
            lines.append(f'  output Real {name}{i} = {expr};')
    return '\n'.join(lines+['end AtomicPhaseBoundary;', ''])


def evaluate(rows, values):
    records = []
    for i, row in enumerate(rows, 1):
        ref = reference(row['density'], row['temperature'])
        peer = {key: float(values[f'{key}{i}']) for key in ('liquid', 'vapor', 'phase', 'region')}
        if any(not math.isfinite(v) for v in peer.values()) or not 0 < peer['vapor'] < peer['liquid'] or peer['phase'] not in (1., 2.) or peer['region'] not in (1., 2., 4.):
            raise ValueError('Invalid external boundary result')
        boundary_shift = max(abs(peer['liquid']-ref['liquid_density']),
                             abs(peer['vapor']-ref['vapor_density']))
        # A disagreement farther from BOTH boundaries than the measured shift
        # cannot be attributed to that shift. No tolerance changes acceptance.
        outside_shift = ref['distance_to_boundary_kg_m3'] > boundary_shift
        records.append(dict(**row, reference=ref, peer=peer,
                            phase_matches=ref['phase']==peer['phase'],
                            region_matches=ref['region']==peer['region'],
                            measured_boundary_shift_kg_m3=boundary_shift,
                            outside_measured_boundary_shift=outside_shift))
    return records


def run(output, *, omc, library, wall_trace=None, peer_traces=()):
    output, omc, library = map(lambda p: Path(p).absolute(), (output, omc, library))
    if any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError('Symlink output')
    if not omc.is_file() or not (library/'Modelica/package.mo').is_file():
        raise ValueError('Explicit compiler and library required')
    source, driver, libraries = signature(), sha(__file__), library_manifest(library)
    helper = Path(__file__).with_name('benchmark_modelica_cycles.py')
    helper_sha, compiler_sha = sha(helper), sha(omc)
    rows = states()
    inputs = {}
    if wall_trace is not None:
        wall_trace = Path(wall_trace)
        native = unseal(json.loads(wall_trace.read_text()))
        inputs[str(wall_trace.absolute())] = sha(wall_trace)
        for obs in native['observations']:
            c = obs['cells'][0]
            rows.append(dict(origin='native_wall_trace', time_s=obs['time_s'],
                             density=c['mass_kg']/c['volume_m3'], temperature=c['temperature_k'],
                             native_phase=c['phase']))
    for trace in peer_traces:
        trace = Path(trace)
        comparison = unseal(json.loads(trace.read_text()))
        inputs[str(trace.absolute())] = sha(trace)
        if not wall_trace:
            raise ValueError('Peer volume must come from the sealed native wall trace')
        volume = native['observations'][0]['cells'][0]['volume_m3']
        for obs in comparison['peer_observations']:
            rho = obs['mass']/volume
            reconstructed = eos.rho_u(rho, obs['specificEnergy'], obs['pressure'], obs['temperature'])
            region = reference(rho,obs['temperature'])['region']
            forward_residuals = {}
            if region in (1,2):
                raw = (iapws._region1_raw if region==1 else iapws._region2_raw)(obs['temperature'],obs['pressure']*1e-6)
                forward_residuals = dict(forward_density_relative_residual=(1/float(raw['v']))/rho-1,
                                         forward_energy_residual_j_kg=float(raw['u'])*1000-obs['specificEnergy'])
            rows.append(dict(origin='peer_wall_trace', method=comparison['record']['method'],
                             time_s=obs['time'], density=rho, temperature=obs['temperature'],
                             reported_phase=obs['phase'],
                             reconstructed_phase=reconstructed.phase,
                             reconstructed_temperature_k=reconstructed.temperature_k,
                             temperature_reconstruction_error_k=obs['temperature']-reconstructed.temperature_k,
                             **forward_residuals))
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output/'protocol.json', seal(dict(source=source, driver_sha256=driver,
                library_helper_sha256=helper_sha, compiler_wrapper_sha256=compiler_sha,
                libraries=libraries, inputs=inputs, states=rows,
                scope='Identical rho/T classifier diagnostic; no dynamic gate relaxation',
                validated=False, plant_referenced=False)))
    (output/'AtomicPhaseBoundary.mo').write_text(render(rows))
    commands = [f'setModelicaPath({json.dumps(str(library))});', 'setCompiler("gcc");', 'setCXXCompiler("g++");']
    commands += [f'loadFile({json.dumps(str(library/name))},uses=false);'
                 for name in ('Complex.mo', 'ModelicaServices/package.mo', 'Modelica/package.mo')]
    commands += ['loadFile("AtomicPhaseBoundary.mo",uses=false);',
                 'translateModel(AtomicPhaseBoundary,stopTime=0.001,numberOfIntervals=1,outputFormat="csv");', 'getErrorString();']
    (output/'run.mos').write_text('\n'.join(commands)+'\n')
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    version = subprocess.check_output([str(omc), '--version'], env=env, text=True, timeout=30).strip()
    report = dict(completed=False, diagnostic_reproduced=False, validated=False,
                  dynamic_comparison_promoted=False, openmodelica_version=version)
    try:
        for name, command in [('translate', [str(omc), '--loadMissingLibraries=false', 'run.mos']),
                              ('build', ['make', '-s', '-f', 'AtomicPhaseBoundary.makefile', 'CC=gcc', 'CXX=g++', '-j1']),
                              ('execute', [str(output/'AtomicPhaseBoundary'), '-r=boundary.csv'])]:
            with (output/f'{name}.log').open('x') as log:
                subprocess.run(command, cwd=output, env=env, stdout=log, stderr=subprocess.STDOUT,
                               timeout=120, check=True)
        with (output/'boundary.csv').open() as stream:
            raw = list(csv.DictReader(stream))
        if len(raw) < 2 or float(raw[0]['time']) != 0 or float(raw[-1]['time']) < .001:
            raise ValueError('Incomplete diagnostic trace')
        records = evaluate(rows, raw[0])
        # Constant expressions must remain identical at every output.
        if any(any(float(row[key]) != float(raw[0][key]) for key in raw[0] if key != 'time') for row in raw):
            raise ValueError('Static diagnostic changed with time')
        unchanged = source == signature() and driver == sha(__file__) and helper_sha == sha(helper) and compiler_sha == sha(omc) and libraries == library_manifest(library) and all(sha(p)==h for p,h in inputs.items())
        report.update(completed=True, source_unchanged=unchanged, records=records,
                      diagnostic_reproduced=unchanged,
                      phase_mismatches=sum(not r['phase_matches'] for r in records),
                      region_mismatches=sum(not r['region_matches'] for r in records),
                      unexplained_mismatches=sum(r['outside_measured_boundary_shift'] and (not r['phase_matches'] or not r['region_matches']) for r in records),
                      csv_sha256=sha(output/'boundary.csv'))
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        report['error'] = str(error)
    atomic_json(output/'report.json', seal(report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('output', 'omc', 'library'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--wall-trace', type=Path)
    parser.add_argument('--peer-traces', type=Path, nargs='*', default=[])
    args = parser.parse_args()
    result = run(args.output, omc=args.omc, library=args.library, wall_trace=args.wall_trace,
                 peer_traces=args.peer_traces)
    print(json.dumps({k:v for k,v in result.items() if k != 'records'}))
    return 0 if result['diagnostic_reproduced'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
