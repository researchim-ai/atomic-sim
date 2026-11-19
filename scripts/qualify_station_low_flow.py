"""Whole-path shutdown refinement and independent native suffix replay.

Does not close a plan task or imply experimental validation. Inputs must have
identical physics, settings declared in their protocols, and complete horizons.
"""
import argparse
import json
from math import isfinite
from pathlib import Path
from atomic_model_station_faults import FaultSession
from atomic_model_station_shutdown import ShutdownCase,capture,assess
from atomic_model_station_load import compare
from atomic_model_multiphysics.provenance import seal,sha
from atomic_model_training.spec import atomic_json
from scripts import qualify_station_shutdown as study
from scripts.qualify_station_inventory import read
from scripts.qualify_station_hourly_replay import physical_events
from scripts.qualify_station_load_envelope import event_agreement

SETTINGS=((1.,1e-7),(.5,2e-8),(.25,5e-9))
ACCOUNTING=[45,46,47,48,60,61,62]


def inspect(path):
    path=Path(path);protocol=read(path/'protocol.json');case=ShutdownCase(**protocol['case'])
    if protocol['source']!=study.identity():raise ValueError('Shutdown source differs')
    initial=read(path/'initial.json');rows=read(path/'trajectory.json')['records'];report=read(path/'report.json')
    layout=initial['layout']
    if layout['size']!=78 or layout['accounting_indices']!=ACCOUNTING:raise ValueError('Declared point-core accounting layout required')
    verdict=assess(initial['record'],rows,case)
    if not verdict['passed'] or not report['passed'] or not report['source_unchanged'] or len(rows)!=case.ticks:raise ValueError('Complete accepted shutdown required')
    if any(len(r['state'])!=78 or any(not isfinite(v) for v in r['state']) for r in rows):raise ValueError('Complete finite state required')
    films=[r['physical']['film'] for r in [initial['record'],*rows]]
    ranges={key:[min(f[key] for f in films),max(f[key] for f in films)] for key in ('reynolds','prandtl')}
    if case.film_model=='gnielinski':
        ranges['richardson_d']=[min(f['richardson_d'] for f in films),max(f['richardson_d'] for f in films)]
        if any(f['model_id']!='gnielinski-bulk-smooth-tube' or not 3000<=f['reynolds']<=1e6 or not .6<=f['prandtl']<=200 or not 0<=f['richardson_d']<=.01 for f in films):raise ValueError('Gnielinski applicability')
    final=json.loads((path/'final.json').read_text())
    return case,protocol,rows,dict(assessment=verdict,film_ranges=ranges,events=physical_events(final))


def native(output,reference):
    out=Path(output);ref=Path(reference);case,protocol,rows,_=inspect(ref)
    source=study.identity();own=sha(__file__)
    if any(p.is_symlink() for p in (out,*out.parents)):raise ValueError('Symlink output refused')
    out.mkdir(parents=True,exist_ok=False)
    saved=json.loads((ref/'midpoint.json').read_text());owner=FaultSession.restore(saved)
    if owner.checkpoint()!=saved:raise ValueError('Native complete prefix differs')
    start=len(owner.history)
    if start!=case.ticks//2:raise ValueError('Native midpoint epoch')
    print('native prefix checked',owner.control.plant.time,flush=True)
    commands=read(ref/'final.json')['history'];suffix=[]
    for tick in range(start,case.ticks):
        result=owner.step(**commands[tick]);row=capture(owner,result)
        if row!=rows[tick]:raise ValueError('Native suffix differs at tick '+str(tick))
        suffix.append(row)
        if (tick+1)%4==0:print('native',owner.control.plant.time,'/',case.duration_s,flush=True)
    final=owner.checkpoint()
    if final!=json.loads((ref/'final.json').read_text()):raise ValueError('Native final checkpoint differs')
    if source!=study.identity() or own!=sha(__file__):raise ValueError('Native implementation changed')
    atomic_json(out/'suffix.json',seal(dict(start_tick=start,records=suffix)))
    atomic_json(out/'final.json',final)
    report=dict(passed=True,source=source,driver_sha256=own,reference_protocol_sha256=sha(ref/'protocol.json'),
        prefix_exact=True,suffix_and_final_exact=True,prefix_s=start*case.tick_s,suffix_s=(case.ticks-start)*case.tick_s,
        duration_s=case.duration_s,full_horizon=case.full_horizon,closed_tasks=[],validated=False,
        scope='Restore replays complete native prefix; every newly executed suffix row and final checkpoint compared')
    atomic_json(out/'report.json',seal(report));return report


def refinement(output,paths):
    out=Path(output);source=study.identity();own=sha(__file__)
    if len(paths)!=3:raise ValueError('Three numerical settings required')
    inspected=[inspect(p) for p in paths];case=inspected[0][0]
    for entry,setting in zip(inspected,SETTINGS):
        c,p,_,_=entry
        if c!=case or (p['max_step_s'],p['rtol'])!=setting:raise ValueError('Physics or numerical settings differ')
    differences={name:compare(entry[2],inspected[2][2],accounting_indices=ACCOUNTING) for name,entry in zip(('coarse','fine'),inspected[:2])}
    events={name:event_agreement(entry[3]['events'],inspected[2][3]['events']) for name,entry in zip(('coarse','fine'),inspected[:2])}
    passed=all(v['passed'] for v in [*differences.values(),*events.values()])
    if source!=study.identity() or own!=sha(__file__):raise ValueError('Refinement implementation changed')
    report=dict(passed=passed,source=source,driver_sha256=own,full_horizon=case.full_horizon,duration_s=case.duration_s,
        settings=SETTINGS,cases={name:entry[3] for name,entry in zip(('coarse','fine','finer'),inspected)},
        refinement=differences,event_agreement=events,
        inputs={str(Path(p).absolute()):{f.name:sha(f) for f in Path(p).glob('*.json')} for p in paths},
        closed_tasks=[],npp_11_03_accepted=False,validated=False,r1_accepted=False,
        scope='Whole-path numerical agreement only; installed delivery and full native replay are separate gates')
    if out.exists() or out.is_symlink():raise ValueError('Existing output refused')
    out.mkdir(parents=True);atomic_json(out/'report.json',seal(report));return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['native','refinement']);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reference',type=Path);p.add_argument('--cases',nargs=3,type=Path)
    a=p.parse_args()
    if a.mode=='native':
        if not a.reference:p.error('--reference required')
        report=native(a.output,a.reference)
    else:
        if not a.cases:p.error('--cases required')
        report=refinement(a.output,a.cases)
    print(json.dumps(dict(passed=report['passed'],full_horizon=report['full_horizon'])));return 0 if report['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
