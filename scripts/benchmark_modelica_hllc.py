"""Actual native Modelica HLLC graph with unmodified MSL liquid EOS.

Shares geometry/export/CSV infrastructure, not the Python HLLC face function.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess

from atomic_model_hem_contact.model import signature
from atomic_model_multiphysics.provenance import seal,sha
from atomic_model_training.spec import atomic_json
from scripts import benchmark_modelica_network as network
from scripts import verify_contact_flux as contact

CASES=(*network.CASES,'reverse_pressure_step','wide_pressure_step')
CAPS=contact.WAVE_CAPS
CONTACT_TEMPERATURE_BUDGET_K=1e-7

THERMO='''function AtomicLiquidThermo
  input Real rho; input Real u; input Real seedT;
  output Real state[4];
protected
  Real T; Real p; Real h; Real residual; Real vt; Real vp; Real cp; Real cv;
  Integer error;
  Modelica.Media.Common.GibbsDerivs g;
algorithm
  assert(rho>0,"Positive mass required");
  T := seedT;
  for iteration in 1:20 loop
    (p,error) := Modelica.Media.Water.IF97_Utilities.BaseIF97.Inverses.pofdt125(rho,T,1e-12,1);
    assert(error==0,"External pressure inverse failed");
    g := Modelica.Media.Water.IF97_Utilities.BaseIF97.Basic.g1(p,T);
    h := g.R_s*T*g.tau*g.gtau;
    vt := g.R_s/p*(g.pi*g.gpi-g.tau*g.pi*g.gtaupi);
    vp := g.R_s*T/(p*p)*g.pi*g.pi*g.gpipi;
    cp := -g.R_s*g.tau*g.tau*g.gtautau;
    cv := cp+T*vt*vt/vp;
    residual := h-p/rho-u;
    assert(cv>0 and vp<0,"Invalid external liquid derivatives");
    if abs(residual)<=2e-7 then break; end if;
    T := T-residual/cv;
    assert(T>300 and T<623.15,"External liquid trial outside chart");
  end for;
  assert(abs(residual)<=2e-7,"External liquid inverse did not converge");
  assert(p>=1e5 and p<=16e6 and T<Modelica.Media.Water.IF97_Utilities.BaseIF97.Basic.tsat(p),"Outside liquid scope");
  state := {p,T,cv,sqrt(-cp/(cv*rho*rho*vp))};
  annotation(Inline=false);
end AtomicLiquidThermo;
'''

FACE='''function AtomicHllc
  input Real UL[3]; input Real UR[3];
  input Real pL; input Real pR; input Real TL; input Real TR;
  input Real cL; input Real cR; input Real bound; input Real eref; input Real area;
  output Real massFlux; output Real momentumFlux; output Real energyFlux;
protected
  Real result[3];
  Real vL; Real vR; Real SL; Real SR; Real SM;
  Real FL[3]; Real FR[3]; Real starL[3]; Real starR[3];
  Real checkL[4]; Real checkR[4]; Real pStarL; Real pStarR;
  Real rhoStarL; Real rhoStarR; Real EStarL; Real EStarR;
algorithm
  assert(UL[1]>0 and UR[1]>0,"Nonpositive external inventory");
  vL := UL[2]/UL[1]; vR := UR[2]/UR[1];
  assert(abs(vL)+cL<=bound and abs(vR)+cR<=bound,"External wave envelope exceeded");
  assert(abs(vL)<=0.2*cL and abs(vR)<=0.2*cR,"External Mach envelope exceeded");
  FL := {UL[2],UL[2]*vL+pL,vL*(UL[3]+pL)};
  FR := {UR[2],UR[2]*vR+pR,vR*(UR[3]+pR)};
  if pL==pR and vL==vR then
    result := if vL>=0 then FL else FR;
    massFlux := area*result[1]; momentumFlux := area*result[2]; energyFlux := area*result[3];
    return;
  end if;
  SL := -bound; SR := bound;
  SM := (pR-pL+UL[1]*vL*(SL-vL)-UR[1]*vR*(SR-vR))/(UL[1]*(SL-vL)-UR[1]*(SR-vR));
  assert(SL<SM and SM<SR,"External contact outside fan");
  pStarL := pL+UL[1]*(SL-vL)*(SM-vL);
  pStarR := pR+UR[1]*(SR-vR)*(SM-vR);
  rhoStarL := UL[1]*(SL-vL)/(SL-SM);
  rhoStarR := UR[1]*(SR-vR)/(SR-SM);
  // Independent Rankine-Hugoniot energy form, no ideal-gas gamma.
  EStarL := ((SL-vL)*UL[3]-pL*vL+pStarL*SM)/(SL-SM);
  EStarR := ((SR-vR)*UR[3]-pR*vR+pStarR*SM)/(SR-SM);
  assert(rhoStarL>0 and rhoStarR>0 and pStarL>=1e5 and pStarL<=16e6 and pStarR>=1e5 and pStarR<=16e6,"Inadmissible external stars");
  checkL := AtomicLiquidThermo(rhoStarL,eref+EStarL/rhoStarL-0.5*SM*SM,TL);
  checkR := AtomicLiquidThermo(rhoStarR,eref+EStarR/rhoStarR-0.5*SM*SM,TR);
  assert(checkL[1]>0 and checkR[1]>0,"External star EOS admission failed");
  starL := {rhoStarL,rhoStarL*SM,EStarL};
  starR := {rhoStarR,rhoStarR*SM,EStarR};
  result := if SM>=0 then FL+SL*(starL-UL) else FR+SR*(starR-UR);
  massFlux := area*result[1]; momentumFlux := area*result[2]; energyFlux := area*result[3];
  annotation(Inline=false);
end AtomicHllc;
'''


def request_for(case,cells,cap):
    if case not in CASES or cells not in network.MESHES or cap not in CAPS:
        raise ValueError('Declared external HLLC matrix only')
    original=case if case in network.CASES else 'pressure_thermal_step'
    request=network.request_for(original,cells,1e-5)
    request['model']['max_step_s']=cap
    points=[n['initial'] for n in request['model']['nodes']]+[p for e in request['model']['edges'] for p in e['initial']]
    if case=='reverse_pressure_step':
        for p in points:p['pressure_pa']=8e6-p['pressure_pa']
    if case=='wide_pressure_step':
        for p in points:p['pressure_pa']=8e6+300*(p['pressure_pa']-4e6)
    return request


def transform_model(text):
    start=text.index('function AtomicLiquidState\n');end=text.index('end AtomicLiquidState;',start)+len('end AtomicLiquidState;')
    text=text[:start]+THERMO+FACE+text[end:]
    replacements={
        'Real thermo[n,2]':'Real thermo[n,4]',
        'thermo[i,:]=AtomicLiquidState(':'thermo[i,:]=AtomicLiquidThermo(',
        'pressure[i]=thermo[i,1]; temperature[i]=thermo[i,2];':'pressure[i]=thermo[i,1]; temperature[i]=thermo[i,2];\n    assert(thermo[i,3]>0,"External capacity must remain positive");',
        '''    for j in 1:3 loop
      flux[j,k]=A[k]*(0.5*(physicalFlux[j,left[k]]+physicalFlux[j,right[k]])-0.5*wave*(W[j,right[k]]-W[j,left[k]]));
    end for;''':'''    (flux[1,k],flux[2,k],flux[3,k])=AtomicHllc(W[:,left[k]],W[:,right[k]],pressure[left[k]],pressure[right[k]],temperature[left[k]],temperature[right[k]],thermo[left[k],4],thermo[right[k],4],wave,eref,A[k]);'''
    }
    for old,new in replacements.items():
        if text.count(old)!=1:raise ValueError('Unrecognized graph export template')
        text=text.replace(old,new)
    return text


def render(native):return transform_model(network.render(native))


def assess(native,peer,case):
    record=network.compare(native,peer)
    if case in ('thermal_contact','thermal_step'):
        initial=peer[0];n=len(native['observations'][0]['cells'])
        change=max(abs(row[f'temperature[{i}]']-initial[f'temperature[{i}]']) for row in peer for i in range(1,n+1))
        record['external_contact_temperature_change_k']=change
        record['checks']['stationary_contact_preserved']=change<=CONTACT_TEMPERATURE_BUDGET_K
    record['passed']=all(record['checks'].values());return record


def run(output,*,omc,library):
    output,omc,library=map(lambda p:Path(p).absolute(),(output,omc,library))
    if any(p.is_symlink() for p in (output,*output.parents)):raise ValueError('Symlink output')
    if not omc.is_file() or not (library/'Modelica/package.mo').is_file():raise ValueError('Explicit compiler/library required')
    if (library/'counterfactual.json').exists():raise ValueError('This protocol requires the as-supplied MSL, not a diagnostic patch')
    source=signature();drivers={str(Path(p).absolute()):sha(p) for p in (__file__,network.__file__,contact.__file__)}
    manifest=network.library_manifest(library);compiler=sha(omc)
    version=subprocess.check_output([str(omc),'--version'],text=True,timeout=30).strip()
    output.mkdir(parents=True,exist_ok=False)
    protocol=dict(source=source,drivers=drivers,library=manifest,compiler_wrapper_sha256=compiler,
        openmodelica_version=version,cases=CASES,meshes=network.MESHES,caps_s=CAPS,times_s=network.TIMES,
        budgets=network.BUDGETS,contact_temperature_budget_k=CONTACT_TEMPERATURE_BUDGET_K,
        msl_density_inverse_rtol=1e-12,energy_inverse_atol_j_kg=2e-7,integrator_rtol=1e-11,
        scope='Independent native Modelica HLLC graph, unmodified MSL liquid functions; same semidiscretization',
        standard_modelica_fluid_network=False,validated=False,plant_referenced=False)
    atomic_json(output/'protocol.json',seal(protocol));records=[];native_count=0;external_count=0
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    for case in CASES:
        for cells in network.MESHES:
            folder=output/f'{case}-{cells}';folder.mkdir()
            try:
                natives={cap:contact.capture(request_for(case,cells,cap),'hllc') for cap in CAPS}
                native_count+=len(natives);atomic_json(folder/'native.json',seal({str(k):v for k,v in natives.items()}))
                native=natives[CAPS[0]]
                (folder/'AtomicNetwork.mo').write_text(render(native))
                commands=['setCompiler("gcc");','setCXXCompiler("g++");',f'setModelicaPath({json.dumps(str(library))});']
                commands += [f'loadFile({json.dumps(str(library/p))},uses=false);' for p in ('Complex.mo','ModelicaServices/package.mo','Modelica/package.mo')]
                commands += ['loadFile("AtomicNetwork.mo",uses=false);','translateModel(AtomicNetwork,stopTime=0.001,numberOfIntervals=4,tolerance=1e-11,outputFormat="csv");','getErrorString();']
                (folder/'run.mos').write_text('\n'.join(commands)+'\n')
                for name,command in [('translate',[str(omc),'--loadMissingLibraries=false','run.mos']),('build',['make','-s','-f','AtomicNetwork.makefile','CC=gcc','CXX=g++','-j1'])]:
                    with (folder/f'{name}.log').open('x') as log:
                        subprocess.run(command,cwd=folder,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=120,check=True)
                n=len(native['observations'][0]['cells']);nf=len(network.topology(native['request'],native['observations'][0])[2])
                for method in network.METHODS:
                    try:
                        with (folder/f'{method}.log').open('x') as log:
                            subprocess.run([str(folder/'AtomicNetwork'),f'-s={method}',f'-r={method}.csv'],cwd=folder,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=120,check=True)
                        peer=network.load_csv(folder/f'{method}.csv',n,nf);external_count+=1
                        comparisons=[]
                        for cap,native in natives.items():
                            r=dict(case=case,cells=cells,cap_s=cap,method=method,**assess(native,peer,case),
                                csv_sha256=sha(folder/f'{method}.csv'),model_source_sha256=sha(folder/'AtomicNetwork.mo'),native_executable_sha256=sha(folder/'AtomicNetwork'))
                            records.append(r);comparisons.append(r);print(case,cells,cap,method,r['passed'],flush=True)
                        atomic_json(folder/f'{method}-comparison.json',seal(dict(records=comparisons,peer_observations=peer)))
                    except (OSError,ValueError,KeyError,subprocess.SubprocessError) as error:
                        records.append(dict(case=case,cells=cells,method=method,passed=False,error=str(error)))
            except (OSError,ValueError,KeyError,subprocess.SubprocessError) as error:
                records.append(dict(case=case,cells=cells,passed=False,error=str(error)))
            atomic_json(output/'progress.json',seal(dict(completed=False,runs=records)))
    unchanged=source==signature() and all(sha(p)==h for p,h in drivers.items()) and manifest==network.library_manifest(library) and compiler==sha(omc)
    expected=len(CASES)*len(network.MESHES)*len(CAPS)*len(network.METHODS)
    report=dict(completed=True,source_unchanged=unchanged,passed=unchanged and len(records)==expected and all(r['passed'] for r in records),
        native_trajectories=native_count,external_trajectories=external_count,runs=records,
        openmodelica_version=version,scope=protocol['scope'],standard_modelica_fluid_network=False,
        modified_msl=False,whole_plant_compared=False,validated=False,plant_referenced=False)
    atomic_json(output/'report.json',seal(report));return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('output','omc','library'):parser.add_argument('--'+key,required=True,type=Path)
    args=parser.parse_args();return 0 if run(args.output,omc=args.omc,library=args.library)['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
