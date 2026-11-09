"""Audit MSL two-phase tangent versus the actual forward IF97 saturation path."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
from scipy.integrate import solve_ivp

from atomic_model_hem_qualified import eos
from atomic_model_hem_qualified.model import signature
from atomic_model_multiphysics.provenance import seal,sha,unseal
from atomic_model_training.spec import atomic_json
from physics.materials import iapws
from scripts.audit_msl_if97 import FILE
from scripts.benchmark_modelica_cycles import library_manifest
from scripts.verify_phase_cycles import forward_reference,CASES
from scripts.phase_wall_case import WALL_SPECIFIC_CAPACITY,WALL_SPECIFIC_CONDUCTANCE

PRESSURES=(.5e6,1e6,8e6,15.5e6)
QUALITIES=(1e-6,.5,1-1e-6)
FD_STEPS=(2e-4,1e-4)
BUDGETS=dict(cv_fd_j_kg_k=.01,dpdt_fd_pa_k=.1)


def proxy_capacity(temperature,density):
    """MSL's Clapeyron tangent with normative forward endpoint properties."""
    path_values(temperature,density)  # Domain check; no projection of x.
    p_mpa=float(iapws.saturation_pressure_mpa(temperature));p=p_mpa*1e6
    f=iapws._region1_raw(temperature,p_mpa);g=iapws._region2_raw(temperature,p_mpa)
    vf,vg=float(f['v']),float(g['v'])
    x=(1/density-vf)/(vg-vf)
    slope=(float(g['s'])-float(f['s']))*1000/(vg-vf)
    data=[]
    for index,raw in enumerate((f,g)):
        v=float(raw['v']);vt=float(raw['dv_dt'])
        vp=float(raw['dv_dp_mpa'])*1e-6 if index==0 else -float(raw['cp'])/(float(raw['cv'])*float(raw['w'])**2)*v*v
        pt=-vt/vp;dv=vt+vp*slope
        du=float(raw['cv'])*1000+(temperature*pt-p)*dv
        data.append((dv,du))
    dx=-((1-x)*data[0][0]+x*data[1][0])/(vg-vf)
    cv=(1-x)*data[0][1]+x*data[1][1]+dx*(float(g['u'])-float(f['u']))*1000
    if not math.isfinite(cv) or cv<=0:raise ValueError('Invalid proxy capacity')
    return cv


def proxy_event(case):
    """No fitted parameters: independently integrate the identified MSL proxy."""
    if case not in ('liquid_exit','vapor_exit'):raise ValueError('Two-phase approach only')
    endpoint,offset=CASES[case]
    initial=forward_reference(endpoint,offset);saturated=eos.px(1e6,endpoint)
    t0=initial.temperature_k;tw0=t0+10.;target=saturated.temperature_k-t0
    def rhs(t,y):
        temperature=t0+y[0];tw=tw0+y[1]
        rate=WALL_SPECIFIC_CONDUCTANCE*(tw-temperature)
        return [rate/proxy_capacity(temperature,saturated.density_kg_m3),-rate/WALL_SPECIFIC_CAPACITY]
    def event(t,y):return y[0]-target
    event.terminal=True;event.direction=1
    outputs=[]
    for method in ('DOP853','Radau'):
        result=solve_ivp(rhs,(0.,.01),[0.,0.],method=method,rtol=1e-11,atol=1e-13,max_step=2e-4,events=event)
        if not result.success or len(result.t_events[0])!=1:raise ValueError('Incomplete proxy event')
        outputs.append(dict(method=method,time_s=float(result.t_events[0][0])))
    return dict(case=case,ode=outputs,scope='Clapeyron capacity proxy, no event-time fitting',
                integrators_agree=abs(outputs[0]['time_s']-outputs[1]['time_s'])<=2e-10)


def path_values(temperature,density):
    """Forward charts only; no production rho/u inverse or derivative call."""
    if not math.isfinite(temperature) or not 300<temperature<623.15 or not math.isfinite(density) or density<=0:
        raise ValueError('Declared sub-region3 mixture neighbourhood only')
    p=float(iapws.saturation_pressure_mpa(temperature))
    f=iapws._region1_raw(temperature,p);g=iapws._region2_raw(temperature,p)
    x=(1/density-f['v'])/(g['v']-f['v'])
    # These are derivative trial charts, including one-sided endpoint extensions;
    # no projected/extended state is admitted to the physical solver.
    return dict(pressure_pa=p*1e6,u_j_kg=float(((1-x)*f['u']+x*g['u'])*1000))


def forward_derivative(t,rho):
    values=[]
    for step in FD_STEPS:
        plus,minus=path_values(t+step,rho),path_values(t-step,rho)
        values.append(dict(cv=(plus['u_j_kg']-minus['u_j_kg'])/(2*step),
                           dpdt=(plus['pressure_pa']-minus['pressure_pa'])/(2*step)))
    return dict(cv=values[-1]['cv'],dpdt=values[-1]['dpdt'],
                cv_refinement=abs(values[0]['cv']-values[1]['cv']),
                dpdt_refinement=abs(values[0]['dpdt']-values[1]['dpdt']))


def corrected_copy(library,destination):
    """Explicit second ablation: keep forward states, differentiate their path."""
    library,destination=Path(library).absolute(),Path(destination).absolute()
    if destination.exists() or any(p.is_symlink() for p in (destination,*destination.parents)):
        raise ValueError('New nonsymlink tangent counterfactual required')
    receipt=unseal(json.loads((library/'counterfactual.json').read_text()))
    if receipt['corrected_sha256']!=sha(library/FILE) or not receipt['counterfactual']:
        raise ValueError('Admitted coefficient-corrected parent required')
    manifest=library_manifest(library);text=(library/FILE).read_text()
    start=text.index('  function waterBaseProp_dT\n');end=text.index('  end waterBaseProp_dT;',start)
    function=text[start:end]
    old='''      aux.dpT := if (liq.d <> vap.d) then (vap.s - liq.s)*liq.d*vap.d/(liq.d -
        vap.d) else BaseIF97.Basic.dptofT(aux.T);'''
    if function.count(old)!=1:raise ValueError('Unknown MSL saturation tangent expression')
    function=function.replace(old,'      aux.dpT := BaseIF97.Basic.dptofT(aux.T);')
    old_cv='''      aux.cv := Common.cv2Phase(
          liq,
          vap,
          aux.x,
          aux.T,
          aux.p);'''
    if function.count(old_cv)!=1:raise ValueError('Unknown MSL mixture capacity expression')
    function=function.replace(old_cv,'''      aux.cv := AtomicForwardCv(liq,vap,aux.x,aux.T,aux.p,aux.dpT);''')
    helper='''  function AtomicForwardCv
    "EXPLICIT COUNTERFACTUAL: derivative of forward region4, not Clapeyron proxy"
    input Common.PhaseBoundaryProperties liq;
    input Common.PhaseBoundaryProperties vap;
    input Real x; input Real T; input Real p; input Real dpT;
    output Real cv;
  protected
    Real dvl; Real dvv; Real dx;
  algorithm
    dvl := (liq.pt-dpT)/(liq.pd*liq.d*liq.d);
    dvv := (vap.pt-dpT)/(vap.pd*vap.d*vap.d);
    dx := -((1-x)*dvl+x*dvv)/(1/vap.d-1/liq.d);
    cv := (1-x)*(liq.cv+(T*liq.pt-p)*dvl)
        + x*(vap.cv+(T*vap.pt-p)*dvv) + dx*(vap.u-liq.u);
  end AtomicForwardCv;

'''
    modified=text[:start]+helper+function+text[end:]
    shutil.copytree(library,destination);(destination/FILE).write_text(modified)
    changed=[p for p,h in manifest.items() if sha(destination/p)!=h]
    if changed!=[FILE] or manifest!=library_manifest(library):raise ValueError('Unexpected mutation')
    receipt.update(parent_sha256=sha(library/FILE),corrected_sha256=sha(destination/FILE),
                   tangent_counterfactual=True,scope='Parent Table10 g2 correction plus waterBaseProp_dT exact forward region4 tangent; other media unchanged')
    atomic_json(destination/'counterfactual.json',seal(receipt))


def run(output,*,omc,library):
    output,omc,library=map(lambda p:Path(p).absolute(),(output,omc,library))
    if any(p.is_symlink() for p in (output,*output.parents)):raise ValueError('Symlink output')
    source,driver,manifest,compiler=signature(),sha(__file__),library_manifest(library),sha(omc)
    rows=[]
    for p in PRESSURES:
        for x in QUALITIES:
            state=eos.px(p,x)
            fd=forward_derivative(state.temperature_k,state.density_kg_m3)
            analytic=1/eos.derivatives(state)['temperature_wrt_energy']
            rows.append(dict(p=p,x=x,t=state.temperature_k,rho=state.density_kg_m3,forward=fd,atomic_cv=analytic))
    output.mkdir(parents=True,exist_ok=False)
    atomic_json(output/'protocol.json',seal(dict(source=source,driver_sha256=driver,library=manifest,
                compiler_wrapper_sha256=compiler,points=rows,fd_steps_k=FD_STEPS,budgets=BUDGETS,
                scope='Forward-EOS derivative diagnostic, not experimental validation')))
    probe='''function AtomicTangentProbe
  input Real rho; input Real T;
  output String result;
protected
  Modelica.Media.Common.IF97BaseTwoPhase aux;
algorithm
  aux := Modelica.Media.Water.IF97_Utilities.waterBaseProp_dT(rho,T,2,4);
  result := String(aux.cv,significantDigits=17)+","+String(aux.dpT,significantDigits=17)+","+String(aux.h-aux.p/rho,significantDigits=17);
end AtomicTangentProbe;
'''
    (output/'AtomicTangentProbe.mo').write_text(probe)
    commands=[f'setModelicaPath({json.dumps(str(library))});']
    commands += [f'loadFile({json.dumps(str(library/p))},uses=false);' for p in ('Complex.mo','ModelicaServices/package.mo','Modelica/package.mo')]
    commands += ['loadFile("AtomicTangentProbe.mo",uses=false);']
    commands += [f'writeFile("point-{i}.txt",AtomicTangentProbe({r["rho"]:.17g},{r["t"]:.17g}));' for i,r in enumerate(rows)]
    commands += ['getErrorString();'];(output/'run.mos').write_text('\n'.join(commands)+'\n')
    with (output/'external.log').open('x') as log:
        subprocess.run([str(omc),'--loadMissingLibraries=false','run.mos'],cwd=output,stdout=log,
            stderr=subprocess.STDOUT,timeout=120,check=True,env=dict(os.environ,OPENBLAS_NUM_THREADS='1'))
    for i,row in enumerate(rows):
        values=list(map(float,(output/f'point-{i}.txt').read_text().split(',')))
        if len(values)!=3 or any(not math.isfinite(v) for v in values):raise ValueError('Invalid MSL tangent response')
        row['peer']=dict(zip(('cv','dpdt','u'),values,strict=True))
        row['cv_difference']=values[0]-row['forward']['cv'];row['dpdt_difference']=values[1]-row['forward']['dpdt']
        row['atomic_cv_residual']=row['atomic_cv']-row['forward']['cv']
        row['proxy_cv_residual']=proxy_capacity(row['t'],row['rho'])-row['peer']['cv']
        row['tangent_matches']=abs(row['cv_difference'])<=BUDGETS['cv_fd_j_kg_k'] and abs(row['dpdt_difference'])<=BUDGETS['dpdt_fd_pa_k']
    unchanged=source==signature() and driver==sha(__file__) and manifest==library_manifest(library) and compiler==sha(omc)
    report=dict(completed=True,source_unchanged=unchanged,diagnostic_executed=unchanged,
        tangent_passed=unchanged and all(r['tangent_matches'] for r in rows),
        atomic_derivative_passed=all(abs(r['atomic_cv_residual'])<=BUDGETS['cv_fd_j_kg_k'] for r in rows),
        rows=rows,proxy_predictions=[proxy_event(case) for case in ('liquid_exit','vapor_exit')],
        validated=False,plant_referenced=False,baseline_promoted=False)
    atomic_json(output/'report.json',seal(report));return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('output','omc','library'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--counterfactual-copy',type=Path)
    args=parser.parse_args();result=run(args.output,omc=args.omc,library=args.library)
    if result['diagnostic_executed'] and args.counterfactual_copy:corrected_copy(args.library,args.counterfactual_copy)
    return 0 if result['diagnostic_executed'] and result['atomic_derivative_passed'] else 1


if __name__=='__main__':raise SystemExit(main())
