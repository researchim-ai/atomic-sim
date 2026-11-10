"""Compiled native Modelica acceptance/rejection probes for local HLLC.

Shared input states, independent MSL star admission; not extra trajectories
in the physical comparison matrix. Rejection must name the intended guard.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
from atomic_model_hem_qualified import eos
from atomic_model_hem_contact.model import signature
from atomic_model_multiphysics.provenance import seal,sha
from atomic_model_training.spec import atomic_json
from scripts import benchmark_modelica_local_hllc as local
from scripts import benchmark_modelica_hllc as base
from scripts import benchmark_modelica_network as network

PROBES=(('accepted_compression',1.,2500.,None),
        ('star_outside',8.,2500.,'External local stars exceed signal envelope'),
        ('global_cap',0.,1000.,'External local envelope exceeds global CFL cap'))


def render(velocity,maximum):
    state=eos.pt(4e6,440.);rho=state.density_kg_m3;u=state.internal_energy_j_kg-7e5
    sound=eos.characteristic_speeds(state)[1]
    # A nonzero time coefficient prevents compile-time constant evaluation.
    return base.THERMO+local.FACE+f'''\nmodel AtomicProbe
  Real velocity; Real massFlux; Real momentumFlux; Real energyFlux;
equation
  velocity={velocity:.17g}+1e-6*time;
  (massFlux,momentumFlux,energyFlux)=AtomicHllc(
    {{{rho:.17g},{rho:.17g}*velocity,{rho:.17g}*({u:.17g}+0.5*velocity^2)}},
    {{{rho:.17g},-{rho:.17g}*velocity,{rho:.17g}*({u:.17g}+0.5*velocity^2)}},
    4e6,4e6,440,440,{sound:.17g},{sound:.17g},{maximum:.17g},7e5,1);
end AtomicProbe;
'''


def run(output,*,omc,library):
    output,omc,library=map(lambda p:Path(p).absolute(),(output,omc,library))
    if any(p.is_symlink() for p in (output,*output.parents)):raise ValueError('Symlink output')
    if not omc.is_file() or not (library/'Modelica/package.mo').is_file():raise ValueError('Explicit runtime required')
    if (library/'counterfactual.json').exists():raise ValueError('Unmodified supplied MSL required')
    source=signature();drivers={str(Path(p).absolute()):sha(p) for p in (__file__,local.__file__,base.__file__,network.__file__)}
    manifest=network.library_manifest(library);compiler=sha(omc)
    output.mkdir(parents=True,exist_ok=False)
    atomic_json(output/'protocol.json',seal(dict(source=source,drivers=drivers,library=manifest,compiler_wrapper_sha256=compiler,
        probes=PROBES,scope='Native local HLLC guard harness; shared forward input state; not independent experimental/trajectory evidence',validated=False,plant_referenced=False)))
    rows=[];env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    for name,velocity,maximum,expected in PROBES:
        folder=output/name;folder.mkdir();(folder/'AtomicProbe.mo').write_text(render(velocity,maximum))
        commands=['setCompiler("gcc");','setCXXCompiler("g++");',f'setModelicaPath({json.dumps(str(library))});']
        commands += [f'loadFile({json.dumps(str(library/p))},uses=false);' for p in ('Complex.mo','ModelicaServices/package.mo','Modelica/package.mo')]
        commands += ['loadFile("AtomicProbe.mo",uses=false);','translateModel(AtomicProbe,stopTime=1e-6,numberOfIntervals=1,tolerance=1e-11,outputFormat="csv");','getErrorString();']
        (folder/'run.mos').write_text('\n'.join(commands)+'\n')
        try:
            for phase,command in [('translate',[str(omc),'--loadMissingLibraries=false','run.mos']),('build',['make','-s','-f','AtomicProbe.makefile','CC=gcc','CXX=g++','-j1'])]:
                with (folder/f'{phase}.log').open('x') as log:
                    subprocess.run(command,cwd=folder,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=120,check=True)
            with (folder/'native.log').open('x') as log:
                result=subprocess.run([str(folder/'AtomicProbe'),'-s=dassl','-r=probe.csv'],cwd=folder,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=30)
            log=(folder/'native.log').read_text()
            checks=dict(native_executable=(folder/'AtomicProbe').is_file(),
                intended_outcome=(result.returncode==0 and (folder/'probe.csv').is_file()) if expected is None else (result.returncode!=0 and expected in log))
            row=dict(probe=name,velocity_m_s=velocity,maximum_m_s=maximum,expected_guard=expected,exit_code=result.returncode,
                checks=checks,passed=all(checks.values()),model_sha256=sha(folder/'AtomicProbe.mo'),native_sha256=sha(folder/'AtomicProbe'),log_sha256=sha(folder/'native.log'))
        except (ValueError,OSError,subprocess.SubprocessError) as error:row=dict(probe=name,passed=False,error=str(error))
        rows.append(row);print(name,row['passed'],flush=True)
    unchanged=source==signature() and manifest==network.library_manifest(library) and compiler==sha(omc) and all(sha(p)==h for p,h in drivers.items())
    report=dict(completed=True,passed=unchanged and len(rows)==3 and all(r['passed'] for r in rows),source_unchanged=unchanged,
        probes=rows,validated=False,plant_referenced=False,trajectory_comparison=False)
    atomic_json(output/'report.json',seal(report));return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('output','omc','library'):parser.add_argument('--'+key,required=True,type=Path)
    args=parser.parse_args();return 0 if run(args.output,omc=args.omc,library=args.library)['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
