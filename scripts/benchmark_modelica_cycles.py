"""Actual OpenModelica ClosedVolume dynamics versus the uniform HEM heat cycle."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import time

from atomic_model_hem_qualified.model import signature
from atomic_model_multiphysics.provenance import seal,sha,unseal
from atomic_model_training.spec import atomic_json
from scripts.verify_phase_cycles import CASES,PRESSURES,execute

METHODS=('dassl','ida')
TIMES=(0.,.0005,.001,.0015,.002)
WALL_TIMES=(0.,.0025,.005,.0075,.01)
# External MSL dTX uses its own inverse tolerance1e-8; this is a separate
# predeclared comparison contract, not a change to Atomic qualification budgets.
BUDGETS=dict(pressure_pa=100.,temperature_k=1e-3,specific_energy_j_kg=.02,
             mass_kg=1e-9,energy_j=1e-3)
COLUMNS=dict(pressure_pa='pressure',temperature_k='temperature',
             specific_energy_j_kg='specificEnergy',mass_kg='mass',energy_j='energy')


def render_model(initial, offset, *, wall=None, stock_states=False):
    keys=('pressure_pa','enthalpy_j_kg','mass_kg','total_energy_j')
    if any(type(initial[k]) not in (float,int) or not math.isfinite(initial[k]) for k in keys) or offset not in (-50.,50.):
        raise ValueError('Finite predefined initial state only')
    model = f'''model AtomicHeatCycle
  package Medium
    extends Modelica.Media.Water.WaterIF97_base(
      ThermoStates=Modelica.Media.Interfaces.Choices.IndependentVariables.dTX,
      final ph_explicit=false, final dT_explicit=true, final pT_explicit=false,
      smoothModel=false, onePhase=false);
  end Medium;
  inner Modelica.Fluid.System system;
  Modelica.Fluid.Vessels.ClosedVolume volume(
    redeclare package Medium=Medium,V={initial['volume_m3']:.17g},nPorts=0,use_portsData=false,
    use_HeatTransfer=true,use_T_start=false,
    p_start={initial['pressure_pa']:.17g},h_start={initial['enthalpy_j_kg']:.17g},
    energyDynamics=Modelica.Fluid.Types.Dynamics.DynamicFreeInitial,
    massDynamics=Modelica.Fluid.Types.Dynamics.DynamicFreeInitial);
  Modelica.Thermal.HeatTransfer.Sources.PrescribedHeatFlow heater;
  output Real pressure=volume.medium.p;
  output Real temperature=volume.medium.T;
  output Real energy=volume.U;
  output Real mass=volume.m;
  output Real specificEnergy=volume.medium.u;
  output Integer phase=volume.medium.phase;
initial equation
  volume.m={initial['mass_kg']:.17g};
  volume.U={initial['total_energy_j']:.17g};
equation
  heater.Q_flow={initial['mass_kg']*(-2*offset/.001):.17g}*(if time<0.001 then 1 else -1);
  connect(heater.port,volume.heatPort);
end AtomicHeatCycle;
'''
    if wall is not None:
        model=model.replace('Modelica.Thermal.HeatTransfer.Sources.PrescribedHeatFlow heater;',
            f"Modelica.Thermal.HeatTransfer.Components.ThermalConductor conductor(G={wall['conductance_w_k']:.17g});\n  Modelica.Thermal.HeatTransfer.Components.HeatCapacitor capacitor(C={wall['capacity_j_k']:.17g},T(start={wall['initial_temperature_k']:.17g},fixed=true));\n  output Real wallTemperature=capacitor.T;")
        model='\n'.join(line for line in model.split('\n') if 'heater.Q_flow=' not in line)
        model=model.replace('connect(heater.port,volume.heatPort);',
            'connect(capacitor.port,conductor.port_a);\n  connect(conductor.port_b,volume.heatPort);')
    if stock_states:
        model=model.replace('redeclare package Medium=Medium,V=',
            'redeclare package Medium=Medium,medium(preferredMediumStates=false,\n'
            '      u(stateSelect=StateSelect.always),d(stateSelect=StateSelect.always)),V=')
    return model


def load_csv(path, *, times=TIMES, wall=False):
    with Path(path).open() as stream:
        raw=list(csv.DictReader(stream))
    if not raw: raise ValueError('Empty external trace')
    rows=[]
    previous=-1.
    for raw_row in raw:
        row={k:float(raw_row[k]) for k in ('time',*COLUMNS.values(),'phase',*(['wallTemperature'] if wall else []))}
        if any(not math.isfinite(v) for v in row.values()) or row['time']<previous:
            raise ValueError('Nonfinite/unordered external trace')
        if not .1e6<=row['pressure']<=16e6 or not 300<=row['temperature']<=900 or row['mass']<=0 or row['phase'] not in (1.,2.):
            raise ValueError('External state outside common domain')
        previous=row['time'];rows.append(row)
    selected=[]
    for t in times:
        candidates=[r for r in rows if abs(r['time']-t)<=1e-12]
        if not candidates:raise ValueError('Missing full common output grid')
        first=candidates[0]
        # Event output may duplicate times, but cannot hide incompatible states.
        if any(abs(r[column]-first[column])>BUDGETS[key] for r in candidates for key,column in COLUMNS.items()):
            raise ValueError('Inconsistent left/right event states')
        if any(r['phase'] != first['phase'] for r in candidates):
            raise ValueError('Ambiguous left/right event phase')
        if wall and any(abs(r['wallTemperature']-first['wallTemperature'])>1e-3 for r in candidates):
            raise ValueError('Inconsistent left/right wall temperature')
        selected.append(first)
    return selected


def compare(atomic, peer, *, times=TIMES, wall=False):
    if len(atomic)!=len(times) or len(peer)!=len(times):raise ValueError('Incomplete common histories')
    errors={key:0. for key in BUDGETS}
    for t,a,b in zip(times,atomic,peer,strict=True):
        if any(not math.isfinite(v) for v in (a['time_s'],b['time'])) or abs(a['time_s']-t)>1e-12 or abs(b['time']-t)>1e-12:raise ValueError('Nonidentical common times')
        c=a['cells'][0]
        expected=dict(pressure_pa=c['pressure_pa'],temperature_k=c['temperature_k'],
                      specific_energy_j_kg=c['internal_energy_j_kg'],mass_kg=c['mass_kg'],energy_j=c['total_energy_j'])
        for key,column in COLUMNS.items():
            values=(expected[key],b[column])
            if any(not math.isfinite(v) for v in values):raise ValueError('Nonfinite comparison')
            errors[key]=max(errors[key],abs(values[0]-values[1]))
        if wall:
            wall_values=(a['walls'][0]['temperature_k'],b['wallTemperature'])
            if any(not math.isfinite(v) for v in wall_values):raise ValueError('Nonfinite wall comparison')
            errors['wall_temperature_k']=max(errors.get('wall_temperature_k',0.),abs(wall_values[0]-wall_values[1]))
    return errors


def library_manifest(library):
    return {str(p.relative_to(library)):sha(p) for p in sorted(library.rglob('*'))
            if p.is_file() and p.suffix in ('.mo','.mos','.c','.h')}


def run(output, *, omc, library, methods=METHODS, tolerance=1e-11, include_wall=False, wall_only=False, stock_states=False):
    if not methods or len(set(methods))!=len(methods) or any(m not in METHODS for m in methods):raise ValueError('Choose declared independent integrators')
    if type(tolerance) not in (int,float) or not math.isfinite(tolerance) or not 1e-12<=tolerance<=1e-6:raise ValueError('Tolerance domain1e-12..1e-6')
    if wall_only and not include_wall:raise ValueError('Wall-only requires include-wall')
    from scripts import phase_wall_case
    wall_sha=sha(phase_wall_case.__file__)
    output=Path(output).absolute();library=Path(library).absolute();omc=Path(omc).absolute()
    if any(p.is_symlink() for p in (output,*output.parents)):raise ValueError('Symlink output')
    if not omc.is_file() or not (library/'Modelica/package.mo').is_file():raise ValueError('Compiler and explicit MSL required')
    source=signature();driver=sha(__file__)
    counterfactual_path=library/'counterfactual.json'
    counterfactual=unseal(json.loads(counterfactual_path.read_text())) if counterfactual_path.exists() else None
    counterfactual_sha=sha(counterfactual_path) if counterfactual is not None else None
    if counterfactual is not None and (counterfactual.get('counterfactual') is not True or
        sha(library/'Modelica/Media/Water/IF97_Utilities.mo')!=counterfactual['corrected_sha256']):
        raise ValueError('Counterfactual library manifest mismatch')
    library_hashes=library_manifest(library)
    cycle_sha=sha(Path(__file__).with_name('verify_phase_cycles.py'))
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    version=subprocess.check_output([str(omc),'--version'],env=env,text=True,timeout=30).strip()
    output.mkdir(parents=True,exist_ok=False)
    protocol=dict(schema_version=1,source=source,driver_sha256=driver,cycle_sha256=cycle_sha,
                  peer_library_variant='counterfactual' if counterfactual is not None else 'as_supplied',
                  wall_only=wall_only,state_selection='density/internal_energy' if stock_states else 'density/temperature',
                  counterfactual=counterfactual,counterfactual_manifest_sha256=counterfactual_sha,
                  openmodelica_version=version,compiler_wrapper_sha256=sha(omc),library_files=library_hashes,
                  include_wall=include_wall,wall_times_s=WALL_TIMES,wall_case_sha256=wall_sha,pressures_pa=PRESSURES,cases=CASES,methods=methods,rtol=tolerance,output_times_s=TIMES,
                  budgets=dict(BUDGETS,wall_temperature_k=1e-3),initialization='Exact equal mass and total internal energy, not merely equal p/h guesses',
                  medium='Configured MSL WaterIF97_base dTX; unmodified MSL functions with native inverse tolerance',
                  scope='Uniform HEM limit versus independent ClosedVolume mass/energy dynamics; no external network/whole-plant qualification',
                  validated=False,plant_referenced=False)
    atomic_json(output/'protocol.json',seal(protocol))
    records=[]
    scenarios=([] if wall_only else [(p,case,False) for p in PRESSURES for case in CASES])+([(1e6,case,True) for case in CASES] if include_wall else [])
    for p,case,coupled_wall in scenarios:
            offset=CASES[case][1]
            folder=output/(f'p-{p:g}-{case}'+('-wall' if coupled_wall else ''));folder.mkdir()
            atomic=phase_wall_case.execute(case) if coupled_wall else execute(case,2,2e-5,pressure_pa=p)
            times=WALL_TIMES if coupled_wall else TIMES
            wall=atomic['request']['model']['walls'][0] if coupled_wall else None
            atomic_json(folder/'atomic.json',seal(atomic))
            (folder/'AtomicHeatCycle.mo').write_text(render_model(atomic['observations'][0]['cells'][0],offset,wall=wall,stock_states=stock_states))
            lines=['setCompiler("gcc");','setCXXCompiler("g++");',f'setModelicaPath({json.dumps(str(library))});']
            for name in ('Complex.mo','ModelicaServices/package.mo','Modelica/package.mo'):
                lines.append(f'loadFile({json.dumps(str(library/name))},uses=false);')
            lines.extend(['loadFile("AtomicHeatCycle.mo",uses=false);',
                          f'translateModel(AtomicHeatCycle,startTime=0,stopTime={times[-1]:.17g},numberOfIntervals=4,tolerance={tolerance:.17g},method="dassl",outputFormat="csv");','getErrorString();'])
            (folder/'run.mos').write_text('\n'.join(lines)+'\n')
            started=time.perf_counter()
            try:
                with (folder/'translate.log').open('x') as log:
                    result=subprocess.run([str(omc),'--loadMissingLibraries=false','run.mos'],cwd=folder,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=120)
                if result.returncode or not (folder/'AtomicHeatCycle.makefile').is_file():raise ValueError('Model translation refused')
                translated=time.perf_counter()-started
                start=time.perf_counter()
                with (folder/'build.log').open('x') as log:
                    result=subprocess.run(['make','-s','-f','AtomicHeatCycle.makefile','CC=gcc','CXX=g++','-j1'],cwd=folder,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=120)
                if result.returncode:raise ValueError('Native model compilation refused')
                built=time.perf_counter()-start
                for method in methods:
                    try:
                        start=time.perf_counter()
                        with (folder/f'{method}.log').open('x') as log:
                            result=subprocess.run([str(folder/'AtomicHeatCycle'),f'-s={method}',f'-r={method}.csv'],cwd=folder,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=120)
                        if result.returncode:raise ValueError('External integration refused')
                        seconds=time.perf_counter()-start;peer=load_csv(folder/f'{method}.csv',times=times,wall=coupled_wall)
                        differences=compare(atomic['observations'],peer,times=times,wall=coupled_wall)
                        phase=all((r['phase']==2.)==(a['cells'][0]['phase']=='two_phase') for index,(a,r) in enumerate(zip(atomic['observations'],peer,strict=True)) if index in (0,2,4))
                        initial=peer[0];mass_error=max(abs(r['mass']-initial['mass']) for r in peer)
                        energy_error=(max(abs(r['energy']-initial['energy']+wall['capacity_j_k']*(r['wallTemperature']-initial['wallTemperature'])) for r in peer) if coupled_wall else max(abs(r['energy']-initial['energy']-initial['mass']*((offset if i in (0,4) else -offset if i==2 else 0)-offset)) for i,r in enumerate(peer)))
                        checks=dict(atomic_passed=atomic['record']['passed'],peer_agreement=all(differences[k]<=BUDGETS[k] for k in BUDGETS) and (not coupled_wall or differences['wall_temperature_k']<=1e-3),
                                    phase_paths_match=phase,peer_mass_balance=mass_error<=BUDGETS['mass_kg'],peer_energy_balance=energy_error<=BUDGETS['energy_j'])
                        record=dict(case=case,coupled_wall=coupled_wall,pressure_pa=p,method=method,passed=all(checks.values()),checks=checks,
                                    maximum_errors=differences,peer_mass_error_kg=mass_error,peer_energy_error_j=energy_error,
                                    translation_s=translated,build_s=built,integration_s=seconds,csv_sha256=sha(folder/f'{method}.csv'),
                                    native_model_sha256=sha(folder/'AtomicHeatCycle'),model_source_sha256=sha(folder/'AtomicHeatCycle.mo'))
                        atomic_json(folder/f'{method}-comparison.json',seal(dict(record=record,peer_observations=peer)))
                    except (OSError,ValueError,KeyError,subprocess.TimeoutExpired) as error:
                        record=dict(case=case,coupled_wall=coupled_wall,pressure_pa=p,method=method,passed=False,error=str(error))
                    records.append(record);print(case,p,method,record['passed'],flush=True)
            except (OSError,ValueError,subprocess.TimeoutExpired) as error:
                records.append(dict(case=case,pressure_pa=p,passed=False,error=str(error)))
                print(case,p,'build failed',str(error),flush=True)
            atomic_json(output/'progress.json',seal(dict(completed=False,runs=records)))
    unchanged=source==signature() and driver==sha(__file__) and cycle_sha==sha(Path(__file__).with_name('verify_phase_cycles.py')) and library_manifest(library)==library_hashes and wall_sha==sha(phase_wall_case.__file__) and (counterfactual_sha is None or counterfactual_sha==sha(counterfactual_path))
    report=dict(schema_version=1,completed=True,passed=unchanged and len(records)==len(scenarios)*len(methods) and all(r['passed'] for r in records),
                source_unchanged=unchanged,runs=records,openmodelica_version=version,
                peer_library_variant=protocol['peer_library_variant'],baseline_comparison_promoted=False,
                state_selection=protocol['state_selection'],wall_only=wall_only,
                scope=protocol['scope'],whole_plant_compared=False,validated=False,plant_referenced=False)
    atomic_json(output/'report.json',seal(report));return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--omc',type=Path,required=True)
    parser.add_argument('--library',type=Path,required=True)
    parser.add_argument('--methods',nargs='+',choices=METHODS,default=list(METHODS))
    parser.add_argument('--tolerance',type=float,default=1e-11)
    parser.add_argument('--include-wall',action='store_true')
    parser.add_argument('--wall-only',action='store_true')
    parser.add_argument('--stock-states',action='store_true')
    args=parser.parse_args(argv)
    result=run(args.output,omc=args.omc,library=args.library,methods=args.methods,tolerance=args.tolerance,include_wall=args.include_wall,wall_only=args.wall_only,stock_states=args.stock_states)
    print(json.dumps(dict(passed=result['passed'],whole_plant_compared=False)))
    return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
