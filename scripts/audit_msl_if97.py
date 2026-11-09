"""Attribute MSL region-2 Gibbs differences to published ideal coefficients.

Optional corrected copies are explicitly counterfactual, never baseline peers.
"""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess

from atomic_model_hem_qualified import eos
from atomic_model_hem_qualified.model import signature
from atomic_model_multiphysics.provenance import seal, sha
from atomic_model_training.spec import atomic_json
from physics.materials import iapws
from scripts.benchmark_modelica_cycles import library_manifest

SOURCE = 'https://iapws.org/technical-guidance/release/IF97-Rev.download'
J = (0,1,-5,-4,-3,-2,-1,2,3)
# IAPWS R7-97(2012), Table10, Eq16, stable region2 (not metastable Eq18).
N = (-.96927686500217e1,.10086655968018e2,-.56087911283020e-2,
     .71452738081455e-1,-.40710498223928,.14240819171444e1,
     -.43839511319450e1,-.28408632460772,.21268463753307e-1)
# Literal strings from the g.g expression in the pinned MSL4.1.0 g2 function.
MSL_LITERALS = ('-9.6927686002170','10.0866556801800','-0.0056087911830200',
                '0.071452738814550','-0.40710498239280','1.42408197144400',
                '-4.3839511194500','-0.284086326077200','0.0212684635330700')
MSL = tuple(map(float,MSL_LITERALS))
FILE = 'Modelica/Media/Water/IF97_Utilities.mo'
BUDGETS = dict(h=2e-7,u=2e-7,s=1e-8,cp=1e-8,density_relative=1e-12)


def delta(temperature):
    """MSL minus Table10; pressure-independent, in physical SI units."""
    if type(temperature) not in (float,int) or not math.isfinite(temperature) or not 300<=temperature<=900:
        raise ValueError('Declared300..900K audit only')
    tau=540/temperature
    g=math.fsum((a-b)*tau**j for a,b,j in zip(MSL,N,J,strict=True))
    gt=math.fsum((a-b)*j*tau**(j-1) for a,b,j in zip(MSL,N,J,strict=True))
    gtt=math.fsum((a-b)*j*(j-1)*tau**(j-2) for a,b,j in zip(MSL,N,J,strict=True))
    return dict(h=461.526*540*gt,u=461.526*540*gt,
                s=461.526*(tau*gt-g),cp=-461.526*tau*tau*gtt)


def admitted_source(text):
    start=text.index('      function g2 "')
    end=text.index('      end g2;',start)
    function=text[start:end]
    ideal=function[function.index('g.g :='):function.index('g.gpi :=')]
    if any(literal not in ideal for literal in MSL_LITERALS):
        raise ValueError('Unrecognized MSL ideal coefficients; manual audit required')
    return start,end


def corrected_text(text):
    """Change only g2's ideal gamma and its two tau derivatives."""
    _,end=admitted_source(text)
    lines=['        // EXPLICIT COUNTERFACTUAL: Table10 ideal coefficients; original retained.']
    for name,order in (('g',0),('gtau',1),('gtautau',2)):
        terms=[]
        for normative,original,j in zip(N,MSL,J,strict=True):
            coefficient=normative-original
            for k in range(order):coefficient*=j-k
            if coefficient:terms.append(f'({coefficient:.17g})*g.tau^({j-order})')
        lines.append(f'        g.{name} := g.{name} + '+' + '.join(terms)+';')
    return text[:end]+'\n'.join(lines)+'\n'+text[end:]


def corrected_copy(library,destination):
    library,destination=Path(library).absolute(),Path(destination).absolute()
    if any(p.is_symlink() for p in (destination,*destination.parents)) or destination.exists():
        raise ValueError('New nonsymlink counterfactual destination required')
    original=library_manifest(library)
    text=(library/FILE).read_text()
    corrected=corrected_text(text)
    shutil.copytree(library,destination)
    (destination/FILE).write_text(corrected)
    changed=[p for p,h in original.items() if sha(destination/p)!=h]
    if changed != [FILE] or library_manifest(library)!=original:
        raise ValueError('Counterfactual changed unexpected files or baseline')
    atomic_json(destination/'counterfactual.json',seal(dict(counterfactual=True,
        baseline_sha256=sha(library/FILE),corrected_sha256=sha(destination/FILE),changed_files=changed,
        normative_source=SOURCE,exponents=J,table10=N,observed_msl=MSL,
        baseline_modified=False,validated=False,scope='g2 ideal gamma and first/second tau derivatives only')))


def points():
    rows=[dict(p=p,t=t) for p in (.5e6,1e6,8e6,15.5e6) for t in (650.,750.,900.)]
    rows += [dict(p=p,t=eos.saturation(p)[1].temperature_k) for p in (.5e6,1e6,8e6,15.5e6)]
    return rows


def run(output,*,omc,library):
    output,omc,library=map(lambda p:Path(p).absolute(),(output,omc,library))
    if any(p.is_symlink() for p in (output,*output.parents)):raise ValueError('Symlink output')
    admitted_source((library/FILE).read_text())
    source,driver,manifest,compiler=signature(),sha(__file__),library_manifest(library),sha(omc)
    output.mkdir(parents=True,exist_ok=False)
    atomic_json(output/'protocol.json',seal(dict(source=source,driver_sha256=driver,library=manifest,
                compiler_wrapper_sha256=compiler,normative_source=SOURCE,exponents=J,
                normative=N,msl=MSL,budgets=BUDGETS,points=points(),counterfactual=False)))
    # Direct external function evaluation eliminates integration and CSV interpolation.
    probe='''function AtomicGibbsProbe
  input Real p; input Real T;
  output String result;
protected
  Modelica.Media.Common.GibbsDerivs g;
  Real h; Real u; Real s; Real cp; Real d;
algorithm
  g := Modelica.Media.Water.IF97_Utilities.BaseIF97.Basic.g2(p,T);
  d := p/(g.R_s*T*g.pi*g.gpi);
  h := g.R_s*T*g.tau*g.gtau;
  u := h-p/d;
  s := g.R_s*(g.tau*g.gtau-g.g);
  cp := -g.R_s*g.tau*g.tau*g.gtautau;
  result := String(h,significantDigits=17)+","+String(u,significantDigits=17)+","+String(s,significantDigits=17)+","+String(cp,significantDigits=17)+","+String(d,significantDigits=17);
end AtomicGibbsProbe;
'''
    (output/'AtomicGibbsProbe.mo').write_text(probe)
    commands=[f'setModelicaPath({json.dumps(str(library))});']
    commands += [f'loadFile({json.dumps(str(library/p))},uses=false);' for p in ('Complex.mo','ModelicaServices/package.mo','Modelica/package.mo')]
    commands += ['loadFile("AtomicGibbsProbe.mo",uses=false);']
    commands += [f'writeFile("point-{i}.txt",AtomicGibbsProbe({r["p"]:.17g},{r["t"]:.17g}));' for i,r in enumerate(points())]
    commands += ['getErrorString();']
    (output/'run.mos').write_text('\n'.join(commands)+'\n')
    records=[]
    with (output/'external.log').open('x') as log:
        subprocess.run([str(omc),'--loadMissingLibraries=false','run.mos'],cwd=output,stdout=log,
                       stderr=subprocess.STDOUT,timeout=120,check=True,env=dict(os.environ,OPENBLAS_NUM_THREADS='1'))
    for i,row in enumerate(points()):
        values=list(map(float,(output/f'point-{i}.txt').read_text().split(',')))
        if len(values)!=5 or any(not math.isfinite(v) for v in values):raise ValueError('Invalid direct MSL response')
        peer=dict(zip(('h','u','s','cp','density'),values,strict=True))
        raw=iapws._region2_raw(row['t'],row['p']*1e-6)
        measured={k:peer[k]-float(raw[k])*1000 for k in ('h','u','s','cp')}
        expected=delta(row['t'])
        residual={k:abs(measured[k]-expected[k]) for k in measured}
        residual['density_relative']=abs(peer['density']*float(raw['v'])-1)
        records.append(dict(**row,measured_difference=measured,predicted_difference=expected,
                            attribution_residual=residual,passed=all(residual[k]<=BUDGETS[k] for k in BUDGETS)))
    unchanged=source==signature() and driver==sha(__file__) and manifest==library_manifest(library) and compiler==sha(omc)
    report=dict(completed=True,source_unchanged=unchanged,passed=unchanged and all(r['passed'] for r in records),
                records=records,validated=False,baseline_dynamic_promoted=False,
                conclusion='Direct g2 differences explained by ideal coefficients; no integration/output involved')
    atomic_json(output/'report.json',seal(report))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--omc',type=Path,required=True)
    parser.add_argument('--library',type=Path,required=True)
    parser.add_argument('--counterfactual-copy',type=Path)
    args=parser.parse_args()
    result=run(args.output,omc=args.omc,library=args.library)
    if result['passed'] and args.counterfactual_copy:
        corrected_copy(args.library,args.counterfactual_copy)
    return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
