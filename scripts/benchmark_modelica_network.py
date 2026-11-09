"""Heterogeneous HEM graph versus independently assembled Modelica balances.

Same Rusanov semidiscretization, independent MSL EOS and native DASSL/IDA.
This is not the off-the-shelf Modelica.Fluid pipe model or an experimental oracle.
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess

from atomic_model_hem_qualified.cases import baseline
from atomic_model_hem_qualified.model import QualifiedHEMNetwork,signature
from atomic_model_hem_qualified.qualification import CRITERIA
from atomic_model_multiphysics.provenance import seal,sha
from atomic_model_training.spec import atomic_json
from physics.materials import iapws
from scripts.benchmark_modelica_cycles import library_manifest

CASES=('thermal_contact','coupled_gradient','thermal_step','pressure_thermal_step')
MESHES=(2,4)
CAPS=(2e-5,1e-5)
TIMES=(0.,.00025,.0005,.00075,.001)
METHODS=('dassl','ida')
BUDGETS=dict(pressure_pa=100.,temperature_k=1e-3,internal_energy_j_kg=.1,
             mass_kg=1e-7,total_energy_j=.1,momentum_kg_m_s=1e-6,velocity_m_s=1e-5,
             face_mass_kg=1e-7,face_energy_j=.1,support_kg_m_s=1e-6)
COLUMNS=dict(pressure_pa='pressure',temperature_k='temperature',internal_energy_j_kg='u',
             mass_kg='mass',total_energy_j='energy',momentum_kg_m_s='momentum',velocity_m_s='velocity')


def request_for(case,cells=2,cap=2e-5):
    if case not in CASES or cells not in MESHES or cap not in CAPS:raise ValueError('Declared graph matrix only')
    request=baseline(0.,velocity=0.,cells=cells);request['friction']={}
    spec=request['model'];spec['max_step_s']=cap
    def point(x):
        # Two fixed physical blocks remain identical when refining2 ->4 cells.
        sign=1. if x<.5 else -1.
        large=case in ('thermal_step','pressure_thermal_step')
        dp=20000. if case=='pressure_thermal_step' else 100. if case=='coupled_gradient' else 0.
        return dict(kind='pt',pressure_pa=(4e6 if large else 1e6)+dp*sign,
                    temperature_k=440.+(1. if large else .001)*sign,velocity_m_s=0.)
    for node,x in zip(spec['nodes'],(0.,1.),strict=True):node['initial']=point(x)
    # Opposite axial orientation is an explicit check on junction signs.
    spec['edges'][1]['from_node'],spec['edges'][1]['to_node']='b','a'
    for edge in spec['edges']:
        forward=edge['from_node']=='a'
        edge['initial']=[point((j+.5)/cells if forward else 1-(j+.5)/cells) for j in range(cells)]
    for wall in spec['walls']:
        wall.update(conductance_w_k=0.,ambient_ua_w_k=0.,heat_power_w=0.,initial_temperature_k=440.,ambient_temperature_k=440.)
    return request


def execute(case,cells=2,cap=2e-5):
    request=request_for(case,cells,cap);model=QualifiedHEMNetwork(request)
    observations=[model.snapshot()];saved=None
    for i,(left,right) in enumerate(zip(TIMES,TIMES[1:])):
        observations.append(model.advance(right-left))
        if i==1:saved=model.checkpoint()
    restarted=QualifiedHEMNetwork(request);restarted.restore(saved)
    for left,right in zip(TIMES[2:],TIMES[3:]):repeat=restarted.advance(right-left)
    first,last=observations[0],observations[-1]
    checks=dict(restart_exact=repeat==last and restarted.checkpoint()==model.checkpoint(),
        positive_mass=all(c['mass_kg']>0 for s in observations for c in s['cells']),
        liquid_only=all(c['phase']=='liquid' for s in observations for c in s['cells']),
        mass_balance=all(abs(s['global_ledger']['mass_residual_kg'])<=CRITERIA['mass_balance_kg'] for s in observations),
        energy_balance=all(abs(s['global_ledger']['energy_residual_j'])<=CRITERIA['energy_balance_j'] for s in observations),
        entropy=all(s['global_ledger']['numerical_entropy_residual_j_k']>=CRITERIA['numerical_entropy_lower_j_k'] for s in observations),
        adiabatic=all(s['global_ledger'][k]==0. for s in observations for k in ('external_fluid_heat_j','external_wall_heat_j','external_entropy_j_k')),
        heterogeneous=max(c['temperature_k'] for c in first['cells'])-min(c['temperature_k'] for c in first['cells'])>.001,
        transfer_nonzero=max(abs(v) for v in last['local_ledgers']['integrated_face_mass_kg'])>1e-10,
        changed=any(abs(a['mass_kg']-b['mass_kg'])>1e-10 for a,b in zip(first['cells'],last['cells'],strict=True)))
    return dict(request=request,observations=observations,checkpoint=saved,
                record=dict(case=case,cells=cells,cap_s=cap,checks=checks,passed=all(checks.values())))


def topology(request,initial):
    """Reassemble topology/geometry from public inputs, not production faces."""
    spec=request['model'];names=[n['id'] for n in spec['nodes']]
    volumes=[n['volume_m3'] for n in spec['nodes']];faces=[];areas=[]
    for edge in spec['edges']:
        chain=[names.index(edge['from_node'])];area=math.pi*edge['diameter_m']**2/4
        for i in range(edge['cells']):
            chain.append(len(names));names.append(f'{edge["id"]}/{i}')
            volumes.append(area*edge['length_m']/edge['cells'])
        chain.append(names.index(edge['to_node']))
        for a,b in zip(chain,chain[1:]):faces.append((a,b));areas.append(area)
    if names!=[c['id'] for c in initial['cells']] or any(abs(v-c['volume_m3'])>1e-15 for v,c in zip(volumes,initial['cells'],strict=True)):
        raise ValueError('Initial cell identity/geometry mismatch')
    return names,volumes,faces,areas


def vector(values):return '{'+','.join(f'{float(v):.17g}' for v in values)+'}'


def render(native):
    request=native['request'];initial=native['observations'][0]
    names,volumes,faces,areas=topology(request,initial)
    cells=initial['cells'];n=len(names);nf=len(faces);eref=cells[0]['internal_energy_j_kg']
    incidence='['+';'.join(','.join('-1' if a==i else '1' if b==i else '0' for a,b in faces) for i in range(n))+']'
    thermo='''function AtomicLiquidState
  input Real rho; input Real u; input Real seedT;
  output Real state[2];
protected
  Real T; Real residual;
  Modelica.Media.Common.IF97BaseTwoPhase aux;
algorithm
  assert(rho>0,"Positive mass required");
  T := seedT;
  for iteration in 1:20 loop
    aux := Modelica.Media.Water.IF97_Utilities.waterBaseProp_dT(rho,T,1,1);
    residual := aux.h-aux.p/rho-u;
    if abs(residual)<=2e-7 then break; end if;
    assert(aux.cv>0,"Positive isochoric capacity required");
    T := T-residual/aux.cv;
    assert(T>300 and T<623.15,"Liquid inverse iteration outside chart");
  end for;
  assert(abs(residual)<=2e-7,"Liquid inverse did not converge");
  assert(aux.p>=1e5 and aux.p<=16e6 and T<Modelica.Media.Water.IF97_Utilities.BaseIF97.Basic.tsat(aux.p),"Outside declared single-phase liquid domain");
  state := {aux.p,T};
  annotation(Inline=false);
end AtomicLiquidState;
'''
    return thermo+f'''model AtomicNetwork
  parameter Integer n={n}, nf={nf};
  parameter Real V[n]={vector(volumes)}, A[nf]={vector(areas)};
  parameter Integer left[nf]={vector(a+1 for a,b in faces)}, right[nf]={vector(b+1 for a,b in faces)};
  parameter Real incidence[n,nf]={incidence};
  parameter Real m0[n]={vector(c['mass_kg'] for c in cells)};
  parameter Real P0[n]={vector(c['momentum_kg_m_s'] for c in cells)};
  parameter Real C0[n]={vector(c['total_energy_j']-eref*c['mass_kg'] for c in cells)};
  parameter Real seedT[n]={vector(c['temperature_k'] for c in cells)};
  parameter Real eref={eref:.17g}, wave={request['model']['wave_speed_bound_m_s']:.17g};
  output Real mass[n](start=m0,each fixed=true);
  output Real momentum[n](start=P0,each fixed=true);
  Real centeredEnergy[n](start=C0,each fixed=true);
  output Real pressure[n],temperature[n],u[n],energy[n],velocity[n];
  output Real faceMass[nf](each start=0,each fixed=true),faceEnergy[nf](each start=0,each fixed=true);
  output Real support[n](each start=0,each fixed=true);
  Real thermo[n,2],W[3,n],physicalFlux[3,n],flux[3,nf],net[3,n];
equation
  for i in 1:n loop
    velocity[i]=momentum[i]/mass[i];
    u[i]=eref+centeredEnergy[i]/mass[i]-0.5*velocity[i]^2;
    thermo[i,:]=AtomicLiquidState(mass[i]/V[i],u[i],seedT[i]);
    pressure[i]=thermo[i,1]; temperature[i]=thermo[i,2];
    energy[i]=centeredEnergy[i]+eref*mass[i];
    W[:,i]={{mass[i]/V[i],momentum[i]/V[i],centeredEnergy[i]/V[i]}};
    physicalFlux[:,i]={{momentum[i]/V[i],momentum[i]*velocity[i]/V[i]+pressure[i],velocity[i]*(centeredEnergy[i]/V[i]+pressure[i])}};
    for j in 1:3 loop
      net[j,i]=sum(incidence[i,k]*flux[j,k] for k in 1:nf);
    end for;
    der(mass[i])=net[1,i];
    der(momentum[i])=if i<=2 then 0 else net[2,i];
    der(centeredEnergy[i])=net[3,i];
    der(support[i])=if i<=2 then -net[2,i] else 0;
  end for;
  for k in 1:nf loop
    for j in 1:3 loop
      flux[j,k]=A[k]*(0.5*(physicalFlux[j,left[k]]+physicalFlux[j,right[k]])-0.5*wave*(W[j,right[k]]-W[j,left[k]]));
    end for;
    der(faceMass[k])=flux[1,k];
    der(faceEnergy[k])=flux[3,k]+eref*flux[1,k];
  end for;
end AtomicNetwork;
'''


def load_csv(path,n,nf):
    columns=[f'{name}[{i}]' for name in COLUMNS.values() for i in range(1,n+1)]
    columns += [f'{name}[{i}]' for name,count in (('faceMass',nf),('faceEnergy',nf),('support',n)) for i in range(1,count+1)]
    with Path(path).open() as stream:raw=list(csv.DictReader(stream))
    rows=[];previous=-1.
    for row in raw:
        values={k:float(row[k]) for k in ('time',*columns)}
        if any(not math.isfinite(v) for v in values.values()) or values['time']<previous or not -1e-12<=values['time']<=TIMES[-1]+1e-12:
            raise ValueError('Nonfinite/unordered/out-of-horizon graph trace')
        for i in range(1,n+1):
            p,t=values[f'pressure[{i}]'],values[f'temperature[{i}]']
            if not .1e6<=p<=16e6 or not 300<=t<623.15 or values[f'mass[{i}]']<=0 or t>=iapws.saturation_temperature_k(p*1e-6):
                raise ValueError('External graph outside domain')
        rows.append(values);previous=values['time']
    selected=[]
    for t in TIMES:
        matches=[r for r in rows if abs(r['time']-t)<=1e-12]
        if not matches:raise ValueError('Missing complete common graph grid')
        if any(r[k]!=matches[0][k] for r in matches for k in columns):raise ValueError('Ambiguous event output')
        selected.append(matches[0])
    return selected


def compare(native,peer):
    if len(peer)!=len(TIMES) or len(native['observations'])!=len(TIMES):raise ValueError('Incomplete graph histories')
    errors={k:0. for k in BUDGETS}
    names=topology(native['request'],native['observations'][0])[0]
    for t,a,b in zip(TIMES,native['observations'],peer,strict=True):
        if any(not math.isfinite(v) for v in b.values()) or not math.isfinite(a['time_s']) or abs(a['time_s']-t)>1e-12 or abs(b['time']-t)>1e-12:raise ValueError('Mismatched/nonfinite graph times or values')
        if [c['id'] for c in a['cells']]!=names:raise ValueError('Graph cell identity changed')
        for i,c in enumerate(a['cells'],1):
            for field,column in COLUMNS.items():
                error=abs(c[field]-b[f'{column}[{i}]'])
                if not math.isfinite(error):raise ValueError('Nonfinite graph comparison')
                errors[field]=max(errors[field],error)
        ledger=a['local_ledgers']
        for field,name,key in [('face_mass_kg','faceMass','integrated_face_mass_kg'),('face_energy_j','faceEnergy','integrated_face_energy_j'),('support_kg_m_s','support','node_support_impulse_kg_m_s')]:
            for i,value in enumerate(ledger[key],1):
                error=abs(value-b[f'{name}[{i}]'])
                if not math.isfinite(error):raise ValueError('Nonfinite face/support comparison')
                errors[field]=max(errors[field],error)
    n=len(native['observations'][0]['cells']);first=peer[0]
    mass_error=max(abs(sum(row[f'mass[{i}]']-first[f'mass[{i}]'] for i in range(1,n+1))) for row in peer)
    energy_error=max(abs(sum(row[f'energy[{i}]']-first[f'energy[{i}]'] for i in range(1,n+1))) for row in peer)
    momentum_error=max(abs(sum(row[f'momentum[{i}]']-first[f'momentum[{i}]']-row[f'support[{i}]'] for i in range(1,n+1))) for row in peer)
    checks=dict(native_passed=native['record']['passed'],peer_agreement=all(errors[k]<=BUDGETS[k] for k in errors),
        mass_balance=mass_error<=1e-9,energy_balance=energy_error<=1e-5,signed_axial_momentum_balance=momentum_error<=1e-9)
    return dict(passed=all(checks.values()),checks=checks,maximum_errors=errors,
                peer_mass_residual_kg=mass_error,peer_energy_residual_j=energy_error,peer_momentum_residual_kg_m_s=momentum_error)


def run(output,*,omc,library):
    output,omc,library=map(lambda p:Path(p).absolute(),(output,omc,library))
    if any(p.is_symlink() for p in (output,*output.parents)):raise ValueError('Symlink output')
    if not omc.is_file() or not (library/'Modelica/package.mo').is_file():raise ValueError('Explicit compiler/library required')
    source,driver,manifest,compiler=signature(),sha(__file__),library_manifest(library),sha(omc)
    version=subprocess.check_output([str(omc),'--version'],text=True,timeout=30).strip()
    output.mkdir(parents=True,exist_ok=False)
    protocol=dict(source=source,driver_sha256=driver,library=manifest,compiler_wrapper_sha256=compiler,openmodelica_version=version,
        cases=CASES,meshes=MESHES,caps_s=CAPS,times_s=TIMES,budgets=BUDGETS,
        scope='Same Rusanov finite-volume graph law, independently assembled Modelica balances and MSL liquid EOS',
        standard_modelica_fluid_network=False,phase_changes=False,validated=False,plant_referenced=False)
    atomic_json(output/'protocol.json',seal(protocol));records=[]
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    for case in CASES:
        for cells in MESHES:
            # Compile once per mesh; each native time cap is an independent trajectory.
            folder=output/f'{case}-{cells}';folder.mkdir()
            natives={cap:execute(case,cells,cap) for cap in CAPS}
            atomic_json(folder/'native.json',seal({str(k):v for k,v in natives.items()}))
            native=natives[CAPS[0]]
            (folder/'AtomicNetwork.mo').write_text(render(native))
            commands=['setCompiler("gcc");','setCXXCompiler("g++");',f'setModelicaPath({json.dumps(str(library))});']
            commands += [f'loadFile({json.dumps(str(library/p))},uses=false);' for p in ('Complex.mo','ModelicaServices/package.mo','Modelica/package.mo')]
            commands += ['loadFile("AtomicNetwork.mo",uses=false);',
                'translateModel(AtomicNetwork,stopTime=0.001,numberOfIntervals=4,tolerance=1e-11,outputFormat="csv");','getErrorString();']
            (folder/'run.mos').write_text('\n'.join(commands)+'\n')
            try:
                for name,command in [('translate',[str(omc),'--loadMissingLibraries=false','run.mos']),('build',['make','-s','-f','AtomicNetwork.makefile','CC=gcc','CXX=g++','-j1'])]:
                    with (folder/f'{name}.log').open('x') as log:
                        subprocess.run(command,cwd=folder,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=120,check=True)
                for method in METHODS:
                    try:
                        with (folder/f'{method}.log').open('x') as log:
                            subprocess.run([str(folder/'AtomicNetwork'),f'-s={method}',f'-r={method}.csv'],cwd=folder,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=120,check=True)
                        n=len(native['observations'][0]['cells']);nf=len(topology(native['request'],native['observations'][0])[2])
                        peer=load_csv(folder/f'{method}.csv',n,nf)
                        for cap,result in natives.items():
                            record=dict(case=case,cells=cells,cap_s=cap,method=method,**compare(result,peer),csv_sha256=sha(folder/f'{method}.csv'),model_source_sha256=sha(folder/'AtomicNetwork.mo'),native_executable_sha256=sha(folder/'AtomicNetwork'))
                            records.append(record);print(case,cells,cap,method,record['passed'],flush=True)
                        atomic_json(folder/f'{method}-comparison.json',seal(dict(records=records[-len(CAPS):],peer_observations=peer)))
                    except (OSError,ValueError,KeyError,subprocess.SubprocessError) as error:
                        records.append(dict(case=case,cells=cells,method=method,passed=False,error=str(error)))
            except (OSError,ValueError,subprocess.SubprocessError) as error:
                records.append(dict(case=case,cells=cells,passed=False,error=str(error)))
            atomic_json(output/'progress.json',seal(dict(completed=False,runs=records)))
    unchanged=source==signature() and driver==sha(__file__) and manifest==library_manifest(library) and compiler==sha(omc)
    report=dict(completed=True,source_unchanged=unchanged,passed=unchanged and len(records)==len(CASES)*len(MESHES)*len(CAPS)*len(METHODS) and all(r['passed'] for r in records),
        native_trajectories=len(CASES)*len(MESHES)*len(CAPS),external_trajectories=len(CASES)*len(MESHES)*len(METHODS),openmodelica_version=version,
        runs=records,scope=protocol['scope'],standard_modelica_fluid_network=False,whole_plant_compared=False,validated=False,plant_referenced=False)
    atomic_json(output/'report.json',seal(report));return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('output','omc','library'):parser.add_argument('--'+key,required=True,type=Path)
    args=parser.parse_args();return 0 if run(args.output,omc=args.omc,library=args.library)['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
