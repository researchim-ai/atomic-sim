"""Heated graph versus native OpenModelica/MSL, DASSL and IDA.

Same spatial law, independent heat/entropy equations and IF97 implementation.
No off-the-shelf Modelica.Fluid or whole-modeler superiority claim.
"""
import argparse
import csv
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import subprocess
import time
from atomic_model_hem_heated.model import signature
from atomic_model_multiphysics.provenance import seal,sha
from atomic_model_training.spec import atomic_json
from scripts import benchmark_modelica_local_hllc as local
from scripts import benchmark_modelica_network as network
from scripts import verify_heated_hllc as heated

METHODS=('dassl','ida')
LEDGER_BUDGETS=dict(face_mass_kg=1e-7,face_energy_j=.1,support_kg_m_s=1e-6)


def replace_once(text,old,new):
    if text.count(old)!=1:raise ValueError('Unrecognized Modelica template: '+old)
    return text.replace(old,new)


def render(native,case):
    if case not in heated.CASES:raise ValueError('Declared heated case required')
    packet={**native,'record':{'scheme':'hllc-local'}}
    text=local.render(packet);spec=native['request']['model'];cells=native['observations'][0]['cells'];names=[c['id'] for c in cells]
    walls=spec['walls'];n=len(cells);nw=len(walls)
    if text.count('if i<=2 then')!=2:raise ValueError('Unknown stationary-node template')
    text=text.replace('if i<=2 then',f"if i<={len(spec['nodes'])} then")
    powers=[node['heat_power_w'] for node in spec['nodes']]+[v for edge in spec['edges'] for v in edge['heat_power_w']]
    v=network.vector
    declarations=f'''  parameter Integer nw={nw};
  parameter Integer port[nw]={v(names.index(w['cell'])+1 for w in walls)};
  parameter Real capacity[nw]={v(w['capacity_j_k'] for w in walls)}, K[nw]={v(w['conductance_w_k'] for w in walls)};
  parameter Real UA[nw]={v(w['ambient_ua_w_k'] for w in walls)}, Ta[nw]={v(w['ambient_temperature_k'] for w in walls)};
  parameter Real fluidPower0[n]={v(powers)}, wallPower0[nw]={v(w['heat_power_w'] for w in walls)};
  output Real wallTemperature[nw](start={v(w['initial_temperature_k'] for w in walls)},each fixed=true);
  output Real wallHeat[nw](each start=0,each fixed=true),wallExternal[nw](each start=0,each fixed=true);
  output Real fluidHeat[n](each start=0,each fixed=true);
  output Real externalEntropy(start=0,fixed=true),thermalProduction(start=0,fixed=true);
  Real fluidPower[n],wallPower[nw],exchange[nw],wallInput[nw];
'''
    text=replace_once(text,'model AtomicNetwork\n','model AtomicNetwork\n'+declarations)
    energy='net[3,i]+fluidPower[i]+sum(if port[j]==i then exchange[j] else 0 for j in 1:nw)'
    text=replace_once(text,'der(centeredEnergy[i])=net[3,i];',f'der(centeredEnergy[i])={energy};\n    der(fluidHeat[i])=fluidPower[i];')
    fluid="if i==1 and time>=0.0001 then -5000 else fluidPower0[i]" if case=='branched_pulse' else 'fluidPower0[i]'
    wall="if time>=0.0001 then 0 else wallPower0[j]" if case=='branched_pulse' else 'wallPower0[j]'
    equations=f'''  for i in 1:n loop
    fluidPower[i]={fluid};
  end for;
  for j in 1:nw loop
    wallPower[j]={wall};
    exchange[j]=K[j]*(wallTemperature[j]-temperature[port[j]]);
    wallInput[j]=wallPower[j]+UA[j]*(Ta[j]-wallTemperature[j]);
    capacity[j]*der(wallTemperature[j])=wallInput[j]-exchange[j];
    der(wallHeat[j])=exchange[j]; der(wallExternal[j])=wallInput[j];
  end for;
  der(externalEntropy)=sum(fluidPower[i]/temperature[i] for i in 1:n)+sum(wallPower[j]/wallTemperature[j]+UA[j]*(Ta[j]-wallTemperature[j])/Ta[j] for j in 1:nw);
  der(thermalProduction)=sum(K[j]*(wallTemperature[j]-temperature[port[j]])^2/(wallTemperature[j]*temperature[port[j]])+UA[j]*(Ta[j]-wallTemperature[j])^2/(Ta[j]*wallTemperature[j]) for j in 1:nw);
'''
    return replace_once(text,'end AtomicNetwork;',equations+'end AtomicNetwork;')


def load(path,native):
    n=len(native['observations'][0]['cells']);nw=len(native['observations'][0]['walls'])
    nf=len(network.topology(native['request'],native['observations'][0])[2])
    columns=[f'{name}[{i}]' for name in network.COLUMNS.values() for i in range(1,n+1)]
    columns += [f'{name}[{i}]' for name,count in [('wallTemperature',nw),('wallHeat',nw),('wallExternal',nw),('fluidHeat',n),('faceMass',nf),('faceEnergy',nf),('support',n)] for i in range(1,count+1)]
    columns += ['time','externalEntropy','thermalProduction']
    with Path(path).open() as stream:rows=[{k:float(r[k]) for k in columns} for r in csv.DictReader(stream)]
    if not rows or any(not math.isfinite(v) for r in rows for v in r.values()):raise ValueError('Nonfinite/empty Modelica trace')
    if any(b['time']<a['time'] for a,b in zip(rows,rows[1:])):raise ValueError('Unordered Modelica trace')
    selected=[]
    for t in heated.TIMES:
        matches=[r for r in rows if abs(r['time']-t)<=1e-12]
        if not matches:raise ValueError('Missing common Modelica sample')
        nearest=min(abs(r['time']-t) for r in matches)
        matches=[r for r in matches if abs(r['time']-t)==nearest]
        if any(r[k]!=matches[0][k] for r in matches for k in columns if k!='time'):raise ValueError('Ambiguous event state')
        selected.append(matches[-1])
    return selected


def compare(native,rows):
    packet=deepcopy(native);errors={k:0. for k in LEDGER_BUDGETS};mass=energy=0.
    if len(rows)!=len(heated.TIMES):raise ValueError('Incomplete external heated trajectory')
    first=rows[0];n=len(native['observations'][0]['cells']);spec=native['request']['model']
    for observation,row,t in zip(packet['observations'],rows,heated.TIMES,strict=True):
        if not math.isfinite(row['time']) or abs(row['time']-t)>1e-12:raise ValueError('External time mismatch')
        for i,c in enumerate(observation['cells'],1):
            for field,column in network.COLUMNS.items():c[field]=row[f'{column}[{i}]']
            if c['mass_kg']<=0 or not 300<=c['temperature_k']<623.15 or not 1e5<=c['pressure_pa']<=16e6:raise ValueError('External fluid domain')
        for i,w in enumerate(observation['walls'],1):
            w.update(temperature_k=row[f'wallTemperature[{i}]'],heat_to_fluid_j=row[f'wallHeat[{i}]'],external_heat_j=row[f'wallExternal[{i}]'])
            if not 300<=w['temperature_k']<=900:raise ValueError('External wall domain')
        observation['global_ledger'].update(external_entropy_j_k=row['externalEntropy'],thermal_entropy_production_j_k=row['thermalProduction'])
        for field,column,key in [('face_mass_kg','faceMass','integrated_face_mass_kg'),('face_energy_j','faceEnergy','integrated_face_energy_j'),('support_kg_m_s','support','node_support_impulse_kg_m_s')]:
            for i,value in enumerate(observation['local_ledgers'][key],1):
                delta=abs(value-row[f'{column}[{i}]'])
                if not math.isfinite(delta):raise ValueError('Nonfinite external ledger difference')
                errors[field]=max(errors[field],delta)
        mass=max(mass,abs(math.fsum(row[f'mass[{i}]']-first[f'mass[{i}]'] for i in range(1,n+1))))
        residual=math.fsum(row[f'energy[{i}]']-first[f'energy[{i}]']-row[f'fluidHeat[{i}]'] for i in range(1,n+1))
        residual+=math.fsum(w['capacity_j_k']*(row[f'wallTemperature[{j}]']-first[f'wallTemperature[{j}]'])-row[f'wallExternal[{j}]'] for j,w in enumerate(spec['walls'],1))
        energy=max(energy,abs(residual))
    result=heated.compare(native,packet)
    checks=dict(states_and_heat=result['passed'],ledgers=all(errors[k]<=v for k,v in LEDGER_BUDGETS.items()),external_mass=mass<=1e-8,external_energy=energy<=1e-4)
    return dict(checks=checks,passed=all(checks.values()),comparison=result,ledger_errors=errors,external_mass_residual_kg=mass,external_energy_residual_j=energy)


def run(output,omc,library):
    output,omc,library=map(lambda p:Path(p).absolute(),(output,omc,library))
    if any(p.is_symlink() for p in (output,*output.parents)):raise ValueError('Symlink output')
    if not omc.is_file() or not (library/'Modelica/package.mo').is_file() or (library/'counterfactual.json').exists():raise ValueError('Explicit compiler and unmodified MSL required')
    output.mkdir(parents=True,exist_ok=False);source=signature();manifest=network.library_manifest(library)
    drivers={str(Path(p).absolute()):sha(p) for p in (__file__,heated.__file__,local.__file__,local.base.__file__,network.__file__)}
    protocol=dict(source=source,drivers=drivers,library=manifest,compiler=sha(omc),version=subprocess.check_output([str(omc),'--version'],text=True,timeout=30).strip(),
      cases=heated.CASES,methods=METHODS,times_s=heated.TIMES,scope='Independent Modelica heat/entropy/RH and unmodified MSL; same spatial law; not Modelica.Fluid',superiority_established=False)
    atomic_json(output/'protocol.json',seal(protocol));records=[]
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    for case in heated.CASES:
        folder=output/case;folder.mkdir();native=heated.capture(heated.request_for(case,heated.CAPS[-1]),'hllc-local',case)
        atomic_json(folder/'native.json',seal(native));(folder/'AtomicNetwork.mo').write_text(render(native,case))
        commands=['setCompiler("gcc");','setCXXCompiler("g++");',f'setModelicaPath({json.dumps(str(library))});']
        commands += [f'loadFile({json.dumps(str(library/p))},uses=false);' for p in ('Complex.mo','ModelicaServices/package.mo','Modelica/package.mo')]
        commands += ['loadFile("AtomicNetwork.mo",uses=false);','translateModel(AtomicNetwork,stopTime=0.0005,numberOfIntervals=10,tolerance=1e-11,outputFormat="csv");','getErrorString();']
        (folder/'run.mos').write_text('\n'.join(commands)+'\n')
        try:
            for name,command in [('translate',[str(omc),'--loadMissingLibraries=false','run.mos']),('build',['make','-s','-f','AtomicNetwork.makefile','CC=gcc','CXX=g++','-j1'])]:
                with (folder/f'{name}.log').open('x') as log:subprocess.run(command,cwd=folder,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=120,check=True)
            for method in METHODS:
                start=time.perf_counter()
                with (folder/f'{method}.log').open('x') as log:subprocess.run([str(folder/'AtomicNetwork'),f'-s={method}',f'-r={method}.csv'],cwd=folder,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=120,check=True)
                rows=load(folder/f'{method}.csv',native)
                record=dict(case=case,method=method,**compare(native,rows),execution_with_csv_s=time.perf_counter()-start,csv_sha256=sha(folder/f'{method}.csv'),model_sha256=sha(folder/'AtomicNetwork.mo'))
                records.append(record);atomic_json(folder/f'{method}-comparison.json',seal(dict(record=record,observations=rows)))
                print(case,method,record['passed'],flush=True)
        except (OSError,ValueError,KeyError,subprocess.SubprocessError) as error:
            records.append(dict(case=case,passed=False,error=str(error)));print(case,'failed',str(error),flush=True)
        atomic_json(output/'progress.json',seal(dict(records=records)))
    unchanged=source==signature() and manifest==network.library_manifest(library) and all(sha(p)==v for p,v in drivers.items()) and sha(omc)==protocol['compiler']
    report=dict(passed=unchanged and len(records)==6 and all(r['passed'] for r in records),source_unchanged=unchanged,records=records,
        superiority_established=False,whole_plant_compared=False,standard_modelica_fluid_network=False,validated=False,plant_referenced=False)
    atomic_json(output/'report.json',seal(report));return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('output','omc','library'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();return 0 if run(a.output,a.omc,a.library)['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
