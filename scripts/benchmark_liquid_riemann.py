"""Physical liquid-wave oracle versus approximate HLLC interface fluxes."""
import argparse
from dataclasses import asdict
import math
from pathlib import Path
from atomic_model_hem_qualified import eos
from atomic_model_hem_contact.flux import hllc
from atomic_model_hem_contact.model import signature
from atomic_model_multiphysics.provenance import seal,sha
from atomic_model_training.spec import atomic_json
from scripts import liquid_riemann as oracle


def cases():
    result=[]
    for name,pl,pr in [('moderate',4.02e6,3.98e6),('wide',14e6,2e6),('reversed',2e6,14e6)]:
        result.append((name,eos.pt(pl,441.),eos.pt(pr,439.),0.,0.,(1e6,15e6)))
    for name,v in [('compression',1.),('expansion',-1.),('contact',0.)]:
        result.append((name,eos.pt(8e6,441.),eos.pt(8e6,439.),v,-v,(1e6,15e6)))
    for p,t in [(1e6,350.),(4e6,440.),(14e6,550.)]:
        curve=oracle.WaveCurve(eos.pt(p,t))
        for delta in (100.,1000.,10000.):
            result.append((f'acoustic-{p:g}-{delta:g}',curve.isentrope(p+delta),curve.isentrope(p-delta),0.,0.,(p-20000.,p+20000.)))
    return result


def evaluate(case):
    name,left,right,vl,vr,bracket=case
    solution=oracle.solve(left,right,vl,vr,pressure_bracket=bracket)
    exact=solution.interface_flux()
    checks={'contact_velocity':solution.velocity_residual_m_s<=2e-8,
            'wave_order':solution.left_outer_speed_m_s<=solution.left_inner_speed_m_s<=solution.velocity_m_s<=solution.right_inner_speed_m_s<=solution.right_outer_speed_m_s}
    for side,initial,star,kind in [('left',left,solution.left,solution.left_wave),('right',right,solution.right,solution.right_wave)]:
        entropy=star.entropy_j_kg_k-initial.entropy_j_kg_k
        if kind=='shock':
            residual=star.internal_energy_j_kg-initial.internal_energy_j_kg+.5*(solution.pressure_pa+initial.pressure_pa)*(1/star.density_kg_m3-1/initial.density_kg_m3)
            checks[side+'_hugoniot']=abs(residual)<=2e-6
            checks[side+'_entropy']=entropy>=-2e-8
        else:checks[side+'_isentropic']=abs(entropy)<=2e-8
    impedance=None
    if name.startswith('acoustic-'):
        _,p,delta=name.split('-');p=float(p);delta=float(delta)
        base=oracle.WaveCurve(left).isentrope(p)
        impedance=base.density_kg_m3*eos.characteristic_speeds(base)[1]
        acoustic_velocity=delta/impedance
        checks['linear_acoustic_limit']=abs(solution.velocity_m_s/acoustic_velocity-1)<=2e-4
    candidates=[]
    # Local envelope is a face diagnostic only. No graph/CFL/restart qualification
    # is implied, and the production fixed envelope remains unchanged.
    local=1.01*max(abs(vl)+eos.characteristic_speeds(left)[1],abs(vr)+eos.characteristic_speeds(right)[1])
    for label,bound in [('fixed_2500',2500.),('local_acoustic_diagnostic',local)]:
        approximate=hllc(left,right,vl,vr,wave_bound=bound,energy_reference=0.)
        absolute={k:abs(float(a)-b) for k,a,b in zip(('mass_kg_m2_s','momentum_pa','energy_w_m2'),approximate,exact)}
        candidates.append(dict(envelope=label,wave_bound_m_s=bound,flux=list(map(float,approximate)),absolute_errors=absolute,
            relative_mass_flux_error=None if abs(exact[0])<1e-12 else absolute['mass_kg_m2_s']/abs(exact[0])))
    return dict(case=name,input=dict(left=asdict(left),right=asdict(right),velocity_left_m_s=vl,velocity_right_m_s=vr,pressure_bracket_pa=list(bracket)),
        oracle=asdict(solution),godunov_flux=list(exact),checks=checks,passed=all(checks.values()),candidates=candidates)


def run(output):
    output=Path(output).absolute()
    if any(p.is_symlink() for p in (output,*output.parents)):raise ValueError('Symlink output')
    source=signature();drivers={str(Path(p).absolute()):sha(p) for p in (__file__,oracle.__file__)}
    output.mkdir(parents=True,exist_ok=False)
    atomic_json(output/'protocol.json',seal(dict(source=source,drivers=drivers,reference=oracle.SOURCE,
        scope='Independent nonlinear single-phase IF97 shock/rarefaction oracle; shared EOS; subsonic star-region flux only',
        hllc_accuracy_gate=False,local_envelope_graph_qualified=False,validated=False,plant_referenced=False)))
    rows=[evaluate(case) for case in cases()]
    unchanged=source==signature() and all(sha(p)==value for p,value in drivers.items())
    report=dict(completed=True,passed=unchanged and all(r['passed'] for r in rows),source_unchanged=unchanged,
        cases=rows,oracle_cases=len(rows),hllc_accuracy_gate=False,local_envelope_graph_qualified=False,
        validated=False,plant_referenced=False)
    atomic_json(output/'report.json',seal(report));return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path)
    return 0 if run(p.parse_args().output)['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
