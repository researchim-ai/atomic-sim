"""Qualify an explicitly incomplete shutdown prefix and its rejected next step.

Separate from positive horizon qualification. No path here accepts NPP-11.03.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
from atomic_model_station_faults import FaultSession
from atomic_model_station_shutdown import ShutdownCase,capture,assess
from atomic_model_station_load import compare
from atomic_model_multiphysics.provenance import seal,sha
from atomic_model_training.spec import atomic_json
from core.status import DomainError,SolverError,ModelLimitError,classify_model_error
from scripts import qualify_station_shutdown as study
from scripts.qualify_station_inventory import read
from scripts.qualify_station_low_flow import SETTINGS,ACCOUNTING
from scripts.qualify_station_hourly_replay import physical_events
from scripts.qualify_station_load_envelope import event_agreement


def inspect(path):
    path=Path(path);protocol=read(path/'protocol.json');case=ShutdownCase(**protocol['case']);r=read(path/'report.json')
    rows=read(path/'trajectory.json')['records'];initial=read(path/'initial.json')
    if protocol['source']!=study.identity() or not r['source_unchanged']:raise ValueError('Source differs')
    if not case.full_horizon or case.family!='coastdown' or case.film_model!='gnielinski':raise ValueError('Full configured Gnielinski coastdown required')
    if (r['passed'] or r['completed'] or not r['rollback_exact'] or r['failure'] is None
        or r['failure'].get('details',{}).get('criterion')!='film_richardson_d'):raise ValueError('Exact buoyancy-screen refusal required')
    if not rows or len(rows)>=case.ticks or r['assessment']['reasons']!=['incomplete_horizon']:raise ValueError('Incomplete otherwise acceptable prefix required')
    if initial['layout']['accounting_indices']!=ACCOUNTING or initial['layout']['size']!=78:raise ValueError('Point-core accounting schema')
    partial=replace(case,duration_s=len(rows)*case.tick_s)
    verdict=assess(initial['record'],rows,partial)
    if not verdict['passed'] or r['completed_time_s']!=len(rows)*case.tick_s:raise ValueError('Prefix invariants')
    for row in [initial['record'],*rows]:
        film=row['physical']['film']
        if (film['model_id']!='gnielinski-bulk-smooth-tube' or not 3000<=film['reynolds']<=1e6
            or not .6<=film['prandtl']<=200 or not 0<=film['richardson_d']<=.01):raise ValueError('Accepted film outside bounds')
    return case,protocol,r,rows


def native(output,reference):
    out=Path(output);ref=Path(reference);case,protocol,expected,rows=inspect(ref)
    if out.exists() or any(p.is_symlink() for p in (out,*out.parents)):raise ValueError('Fresh ordinary output required')
    source=study.identity();own=sha(__file__);out.mkdir(parents=True)
    saved=json.loads((ref/'midpoint.json').read_text());owner=FaultSession.restore(saved)
    if owner.checkpoint()!=saved:raise ValueError('Native prefix differs')
    start=len(owner.history)
    if start!=case.ticks//2 or start>=len(rows):raise ValueError('Midpoint before refusal required')
    print('refusal native prefix checked',owner.control.plant.time,flush=True)
    commands=read(ref/'final.json')['history'];suffix=[]
    for tick in range(start,len(rows)):
        row=capture(owner,owner.step(**commands[tick]))
        if row!=rows[tick]:raise ValueError('Native accepted suffix differs')
        suffix.append(row)
        if (tick+1)%4==0:print('refusal native',owner.control.plant.time,flush=True)
    final=owner.checkpoint()
    if final!=json.loads((ref/'final.json').read_text()):raise ValueError('Native last accepted checkpoint differs')
    failure=None
    try:owner.step()
    except (ValueError,DomainError,SolverError) as error:
        failure=dict(kind=type(error).__name__,category=classify_model_error(error),message=str(error),attempted_start_s=owner.control.plant.time)
        if isinstance(error,ModelLimitError) or isinstance(error,SolverError) and hasattr(error,'details'):failure['details']=error.details()
    if failure!=expected['failure'] or owner.checkpoint()!=final:raise ValueError('Native refusal or rollback differs')
    if source!=study.identity() or own!=sha(__file__):raise ValueError('Native implementation changed')
    atomic_json(out/'suffix.json',seal(dict(start_tick=start,records=suffix)));atomic_json(out/'final.json',final)
    report=dict(passed=True,source=source,driver_sha256=own,prefix_exact=True,suffix_and_final_exact=True,refusal_and_rollback_exact=True,
        prefix_s=start*case.tick_s,accepted_suffix_s=(len(rows)-start)*case.tick_s,last_accepted_s=expected['completed_time_s'],failure=failure,
        full_horizon=False,closed_tasks=[],npp_11_03_accepted=False,r1_accepted=False,validated=False)
    atomic_json(out/'report.json',seal(report));return report


def refinement(output,paths):
    out=Path(output);source=study.identity();own=sha(__file__)
    if len(paths)!=3 or out.exists() or any(p.is_symlink() for p in (out,*out.parents)):raise ValueError('Three settings and fresh ordinary output required')
    values=[inspect(p) for p in paths];case=values[0][0]
    for (c,p,_,_),setting in zip(values,SETTINGS):
        if c!=case or (p['max_step_s'],p['rtol'])!=setting:raise ValueError('Case or settings differ')
    count=min(len(v[3]) for v in values);end=count*case.tick_s
    differences={n:compare(v[3][:count],values[2][3][:count],accounting_indices=ACCOUNTING) for n,v in zip(('coarse','fine'),values[:2])}
    events=[[e for e in physical_events(json.loads((Path(p)/'final.json').read_text())) if e['time_s']<=end] for p in paths]
    agreement={n:event_agreement(e,events[2]) for n,e in zip(('coarse','fine'),events[:2])}
    if source!=study.identity() or own!=sha(__file__):raise ValueError('Refinement implementation changed')
    report=dict(passed=all(r['passed'] for r in [*differences.values(),*agreement.values()]),source=source,driver_sha256=own,
        common_accepted_prefix_s=end,required_horizon_s=case.duration_s,full_horizon=False,
        refinement=differences,event_agreement=agreement,
        cases={n:dict(last_accepted_s=v[2]['completed_time_s'],failure=v[2]['failure'],rollback_exact=True) for n,v in zip(('coarse','fine','finer'),values)},
        inputs={Path(p).name:{f.name:sha(f) for f in Path(p).glob('*.json')} for p in paths},
        closed_tasks=[],npp_11_03_accepted=False,r1_accepted=False,validated=False,
        scope='Only the common accepted prefix is compared; no exact continuous crossing time or full horizon claimed')
    out.mkdir(parents=True);atomic_json(out/'report.json',seal(report));return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['native','refinement']);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reference',type=Path);p.add_argument('--cases',nargs=3,type=Path);a=p.parse_args()
    if a.mode=='native':
        if not a.reference:p.error('--reference required')
        r=native(a.output,a.reference)
    else:
        if not a.cases:p.error('--cases required')
        r=refinement(a.output,a.cases)
    print(json.dumps(dict(passed=r['passed'],full_horizon=False)));return 0 if r['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
