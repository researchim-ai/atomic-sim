"""Stock-defined phase event roots: native HEM, scalar ODE and energy quadrature."""
import argparse
import csv
import json
import math
from pathlib import Path

from scipy.integrate import quad,solve_ivp
from atomic_model_hem_qualified import eos
from atomic_model_hem_qualified.model import QualifiedHEMNetwork,signature
from atomic_model_multiphysics.provenance import seal,sha,unseal
from atomic_model_training.spec import atomic_json
from scripts import phase_wall_case as wall
from scripts.verify_phase_cycles import CASES,MESHES,CAPS,forward_reference

ROOT_WIDTH_S=5e-10
NATIVE_BUDGET_S=2e-8
REFERENCE_BUDGET_S=2e-10
# External phase event is measured and compared, never silently reclassified.
EXTERNAL_BUDGET_S=2e-8


def reference_event(case):
    endpoint,offset=CASES[case]
    state=forward_reference(endpoint,offset)
    tw0=state.temperature_k+(10. if offset<0 else -10.)
    def rate(value):
        s=forward_reference(endpoint,float(value))
        tw=tw0-(float(value)-offset)/wall.WALL_SPECIFIC_CAPACITY
        return wall.WALL_SPECIFIC_CONDUCTANCE*(tw-s.temperature_k)
    integral,error=quad(lambda value:1/rate(value),offset,0.,epsabs=1e-12,epsrel=1e-11)
    times=[]
    def event(t,y):return y[0]
    event.terminal=True
    event.direction=-1 if offset>0 else 1
    for method in ('DOP853','Radau'):
        result=solve_ivp(lambda t,y:[rate(y[0])],(0.,.01),[offset],method=method,
            rtol=1e-11,atol=1e-10,max_step=2e-4,events=event)
        if not result.success or len(result.t_events[0])!=1:raise ValueError('Missing independent event')
        times.append(dict(method=method,time_s=float(result.t_events[0][0])))
    passed=math.isfinite(integral) and 0<integral<.01 and error<=REFERENCE_BUDGET_S and all(abs(r['time_s']-integral)<=REFERENCE_BUDGET_S for r in times)
    return dict(case=case,quadrature_time_s=integral,quadrature_error_s=error,ode=times,passed=passed)


def native_event(case,cells,cap):
    request=wall.wall_request(case,cells=cells,cap=cap,diameter_m=.08)
    endpoint,offset=CASES[case]
    target=eos.px(1e6,endpoint).internal_energy_j_kg
    direction=-1 if offset>0 else 1
    base=QualifiedHEMNetwork(request);base.advance(.004)
    checkpoint=base.checkpoint()
    def at(t):
        model=QualifiedHEMNetwork(request);model.restore(checkpoint)
        snapshot=model.advance(t-.004)
        values=[direction*(c['internal_energy_j_kg']-target) for c in snapshot['cells']]
        if any(not math.isfinite(v) for v in values) or max(values)-min(values)>2e-4:
            raise ValueError('Nonuniform/nonfinite stock-defined event')
        return values[0],model,snapshot
    lo,hi=.0045,.0055
    left,_,_=at(lo);right,_,_=at(hi)
    if not left<0<right:raise ValueError('Event not bracketed')
    probes=2
    while hi-lo>ROOT_WIDTH_S:
        mid=(lo+hi)/2;value,_,_=at(mid);probes+=1
        if value<0:lo=mid
        else:hi=mid
    left,_,before=at(lo);right,model,after=at(hi)
    _,replayed,repeat=at(hi)
    exact=after==repeat and model.checkpoint()==replayed.checkpoint()
    return dict(case=case,cells=cells,cap_s=cap,lower_s=lo,upper_s=hi,width_s=hi-lo,
                signed_energy_bracket_j_kg=[left,right],before_phase=before['cells'][0]['phase'],
                after_phase=after['cells'][0]['phase'],restart_exact=exact,probes=probes+3)


def external_events(path):
    """Preserve both sides of native event output; no interpolated event time."""
    with Path(path).open() as stream:rows=list(csv.DictReader(stream))
    if not rows:raise ValueError('Empty external event trace')
    events=[];previous=-1.;previous_phase=None
    for row in rows:
        t,phase=float(row['time']),float(row['phase'])
        if not math.isfinite(t) or t<previous or phase not in (1.,2.):raise ValueError('Invalid external event trace')
        if previous_phase is not None and phase!=previous_phase:
            # An actual event must have left/right output at exactly one time.
            if t!=previous:raise ValueError('Phase change lacks explicit left/right event output')
            events.append(dict(time_s=t,before_phase=previous_phase,after_phase=phase))
        previous,previous_phase=t,phase
    if float(rows[0]['time'])!=0 or float(rows[-1]['time'])<.01 or len(events)!=1:
        raise ValueError('Missing complete single-event trajectory')
    return events[0]


def run(output,*,campaign=None):
    output=Path(output).absolute()
    if any(p.is_symlink() for p in (output,*output.parents)):raise ValueError('Symlink output')
    source=signature();drivers={str(Path(p).absolute()):sha(p) for p in (__file__,wall.__file__,Path(__file__).with_name('verify_phase_cycles.py'))}
    inputs={}
    if campaign:
        campaign=Path(campaign).absolute()
        protocol=unseal(json.loads((campaign/'protocol.json').read_text()))
        if protocol['source']!=source or not protocol['include_wall']:raise ValueError('Matching wall campaign required')
        inputs[str(campaign/'protocol.json')]=sha(campaign/'protocol.json')
        for case in CASES:
            for method in ('dassl','ida'):
                p=campaign/f'p-1e+06-{case}-wall' / f'{method}.csv'
                record=unseal(json.loads(p.with_name(f'{method}-comparison.json').read_text()))['record']
                if sha(p)!=record['csv_sha256']:raise ValueError('External CSV changed')
                inputs[str(p)]=sha(p)
    output.mkdir(parents=True,exist_ok=False)
    atomic_json(output/'protocol.json',seal(dict(source=source,drivers=drivers,inputs=inputs,
        root_width_s=ROOT_WIDTH_S,native_budget_s=NATIVE_BUDGET_S,reference_budget_s=REFERENCE_BUDGET_S,
        external_budget_s=EXTERNAL_BUDGET_S,event='Fixed initial density; u equals exact saturated endpoint energy',
        validated=False,plant_referenced=False)))
    native=[];external=[];references=[]
    for case in CASES:
        ref=reference_event(case);references.append(ref)
        for cells in MESHES:
            for cap in CAPS:
                r=native_event(case,cells,cap)
                difference=max(abs(r['lower_s']-ref['quadrature_time_s']),abs(r['upper_s']-ref['quadrature_time_s']))
                r.update(maximum_event_error_s=difference,passed=ref['passed'] and r['restart_exact'] and difference<=NATIVE_BUDGET_S)
                native.append(r);print(case,cells,cap,r['passed'],flush=True)
                atomic_json(output/'progress.json',seal(dict(completed=False,native=native,references=references)))
        if campaign:
            for method in ('dassl','ida'):
                p=campaign/f'p-1e+06-{case}-wall'/f'{method}.csv'
                r=external_events(p);error=abs(r['time_s']-ref['quadrature_time_s'])
                external.append(dict(case=case,method=method,**r,event_error_s=error,passed=error<=EXTERNAL_BUDGET_S))
    unchanged=source==signature() and all(sha(p)==h for p,h in {**drivers,**inputs}.items())
    report=dict(completed=True,source_unchanged=unchanged,native_passed=unchanged and all(r['passed'] for r in native),
                external_passed=unchanged and bool(external) and all(r['passed'] for r in external),
                native=native,external=external,references=references,validated=False,plant_referenced=False)
    atomic_json(output/'report.json',seal(report))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--campaign',type=Path)
    args=parser.parse_args();result=run(args.output,campaign=args.campaign)
    return 0 if result['native_passed'] and (not args.campaign or result['external_passed']) else 1


if __name__=='__main__':raise SystemExit(main())
