"""Full-duration three-setting load qualification; all acceptance gates stay explicit."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from math import isfinite
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import os

from atomic_model_multiphysics.provenance import seal,sha
from atomic_model_training.spec import atomic_json
from atomic_model_station_faults import FaultSession
from atomic_model_station_load import LoadCycle,assess,compare
from scripts import qualify_station_load_cycle as driver
from scripts import qualify_station_hourly_replay as event_helper
from scripts.qualify_station_inventory import read

SETTINGS={'coarse':(1.,1e-7),'fine':(.5,2e-8),'finer':(.25,5e-9)}


def identity():
    return dict(load=driver.identity(),envelope_sha256=sha(__file__),event_helper_sha256=sha(event_helper.__file__))


def event_agreement(a,b):
    if any(type(e) is not dict or type(e.get('kind')) is not str or type(e.get('time_s')) not in (int,float)
           or not isfinite(e['time_s']) or e['time_s']<0 for e in [*a,*b]):raise ValueError('Finite physical event sequence required')
    pairs=list(zip(a,b))
    exact_kinds=len(a)==len(b) and all(x['kind']==y['kind'] for x,y in pairs)
    error=max((abs(x['time_s']-y['time_s']) for x,y in pairs),default=0.)
    return dict(passed=exact_kinds and error<=.05,counts=[len(a),len(b)],same_kinds=exact_kinds,maximum_time_error_s=error)


def inspect_case(path,cycle,setting):
    path=Path(path);protocol=read(path/'protocol.json');report=read(path/'report.json')
    episode=cycle.episode(max_step_s=setting[0],rtol=setting[1]).payload()
    if (protocol['source']!=driver.identity() or LoadCycle(**protocol['cycle'])!=cycle or protocol['episode']!=episode):
        raise ValueError('Load case source, cycle or settings differ')
    data=read(path/'trajectory.json')['records'];initial=read(path/'initial.json');layout=initial['layout']
    if len(data)!=cycle.ticks or layout['size']!=78 or layout['accounting_indices']!=[45,46,47,48,60,61,62]:
        raise ValueError('Complete point-core trajectory and declared accounting schema required')
    if any(len(r['state'])!=layout['size'] or any(type(v) not in (float,int) or not isfinite(v) for v in r['state']) for r in data):
        raise ValueError('Finite complete physical states required')
    assessment=assess(data,cycle)
    bounds={k:[1.,0.] for k in ('rod','admission','bypass')};authority=Counter();valid=True;own=True
    for r in data:
        receipt=r['result']['receipt'];powers=r['physical']['powers']
        for k in bounds:
            value=receipt['actual_positions'][k]
            if type(value) not in (int,float) or not isfinite(value) or not 0<=value<=1:valid=False
            else:bounds[k]=[min(bounds[k][0],value),max(bounds[k][1],value)]
        for choice in receipt['decisions'].values():authority[choice['authority']]+=1
        gross=powers['gross_electrical_w'];needs=powers['own_needs_w']
        own=own and type(gross) in (int,float) and type(needs) in (int,float) and isfinite(gross) and isfinite(needs) and gross>needs>=0
    final=read(path/'final.json')
    state=final['control']['payload']['state']['plant']['payload']['snapshot']
    if state['time_s']!=cycle.duration_s or report['completed_time_s']!=cycle.duration_s:raise ValueError('Full load duration required')
    events=event_helper.physical_events(seal(final))
    passed=bool(assessment['passed'] and valid and own and report['passed'] and report['completed'] and report['source_unchanged'])
    return dict(passed=passed,assessment=assessment,actuator_bounds_valid=valid,actuator_ranges=bounds,
        authority_counts=dict(sorted(authority.items())),own_needs_supplied_every_tick=bool(own),physical_events=events),data,layout


def summarize(root,cycle):
    root=Path(root);protocol=read(root/'protocol.json')
    if protocol['source']!=identity() or LoadCycle(**protocol['cycle'])!=cycle:raise ValueError('Load envelope protocol differs')
    checks={};records={};layouts={}
    for name,setting in SETTINGS.items():checks[name],records[name],layouts[name]=inspect_case(root/name,cycle,setting)
    if any(layout!=layouts['finer'] for layout in layouts.values()):raise ValueError('Load schemas differ')
    differences={name:compare(records[name],records['finer'],accounting_indices=layouts['finer']['accounting_indices']) for name in ('coarse','fine')}
    events={name:event_agreement(checks[name]['physical_events'],checks['finer']['physical_events']) for name in ('coarse','fine')}
    passed=all(c['passed'] for c in checks.values()) and all(c['passed'] for c in differences.values()) and all(e['passed'] for e in events.values())
    return dict(passed=passed,source=identity(),source_unchanged=True,duration_s=cycle.duration_s,settings=SETTINGS,
        cases=checks,refinement=differences,event_agreement=events,closed_tasks=[],r1_accepted=False,validated=False,
        load_cycle_candidate=bool(passed),npp_11_02_accepted=False,
        scope='Full source three-setting physics and whole-path/event comparison; native and installed qualification remain separate')


def native(output,reference):
    out=Path(output);ref=Path(reference)
    if any(p.is_symlink() for p in (out,*out.parents)):raise ValueError('Symlink output refused')
    protocol=read(ref/'protocol.json');cycle=LoadCycle(**protocol['cycle']);inspect_case(ref,cycle,SETTINGS['coarse'])
    out.mkdir(parents=True,exist_ok=False);source=identity()
    expected=read(ref/'trajectory.json')['records'];saved=__import__('json').loads((ref/'midpoint.json').read_text())
    print('native restoring complete prefix',cycle.duration_s/2,flush=True)
    owner=FaultSession.restore(saved)
    if owner.checkpoint()!=saved:raise ValueError('Native load prefix differs')
    start=len(owner.history)
    if start!=cycle.ticks//2:raise ValueError('Native midpoint epoch')
    commands=read(ref/'final.json')['history'];suffix=[]
    for tick in range(start,cycle.ticks):
        result=owner.step(**commands[tick]);snap=owner.control.plant.snapshot()
        row=dict(result=result,load_fraction=owner.control.plant.controls['load_fraction'],state=owner.control.plant.state.tolist(),
                 physical={k:snap[k] for k in ('time_s','powers','energy_relative','mass_relative','boron_relative')})
        if row!=expected[tick]:raise ValueError('Native load suffix differs at tick '+str(tick))
        suffix.append(row)
        if (tick+1)%20==0:print('native',owner.control.plant.time,'/',cycle.duration_s,flush=True)
    final=owner.checkpoint()
    if final!=__import__('json').loads((ref/'final.json').read_text()):raise ValueError('Native load final checkpoint differs')
    if source!=identity():raise ValueError('Native load source changed')
    atomic_json(out/'suffix.json',seal(dict(start_tick=start,records=suffix)))
    atomic_json(out/'final.json',final)
    result=dict(passed=True,source=source,prefix_exact=True,suffix_and_final_exact=True,
        prefix_s=start*cycle.tick_s,suffix_s=(cycle.ticks-start)*cycle.tick_s,duration_s=cycle.duration_s,
        copied_prefix_rows=False,scope='Native owner re-executes full prefix; every suffix transition and final checkpoint checked')
    atomic_json(out/'report.json',seal(result));return result


def launch(role,output,reference):
    out=Path(output).absolute();ref=Path(reference).absolute()
    root=Path(__file__).resolve().parents[1];library=Path(__import__('atomic_model_station_load').__file__).resolve().parents[1]
    code='import sys;sys.path[:0]=sys.argv[1:3];from scripts.qualify_station_load_envelope import worker;worker(sys.argv[3],sys.argv[4],sys.argv[5])'
    with tempfile.TemporaryDirectory(prefix='atomic-load-envelope-') as cwd:
        with (out.parent/(role+'.log')).open('w') as log:
            subprocess.run([sys.executable,'-I','-c',code,str(root),str(library),role,str(out),str(ref)],cwd=cwd,
                env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'),
                stdout=log,stderr=subprocess.STDOUT,check=True,timeout=14400)


def worker(role,output,reference):
    if role=='native':return native(output,reference)
    if role not in ('fine','finer'):raise ValueError('Load envelope worker role')
    setting=SETTINGS[role];cycle=LoadCycle(**read(Path(reference)/'protocol.json')['cycle'])
    result=driver.run(output,cycle,max_step_s=setting[0],rtol=setting[1],reference=reference)
    if not result['passed']:raise ValueError('Full load refinement failed; evidence preserved')


def run(output,coarse,cycle=None):
    cycle=cycle or LoadCycle();out=Path(output).absolute();coarse=Path(coarse).absolute()
    if any(p.is_symlink() for p in (out,*out.parents)):raise ValueError('Symlink output refused')
    check,_,_=inspect_case(coarse,cycle,SETTINGS['coarse'])
    if not check['passed']:raise ValueError('Accepted complete coarse load case required')
    hashes={str(p.relative_to(coarse)):sha(p) for p in coarse.glob('*.json')}
    out.mkdir(parents=True,exist_ok=False);shutil.copytree(coarse,out/'coarse')
    atomic_json(out/'protocol.json',seal(dict(source=identity(),cycle=asdict(cycle),settings=SETTINGS,
        reused_coarse_json_sha256=hashes,coarse_scope='Declared immutable accepted source inputs, not a new execution',
        event_time_s=.05,relative_state=.001,closed_tasks=[],r1_accepted=False)))
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(launch,role,out/role,out/'coarse') for role in ('fine','finer')]
        errors=[]
        for future in futures:
            try:future.result()
            except Exception as error:errors.append(str(error))
    if hashes!={str(p.relative_to(coarse)):sha(p) for p in coarse.glob('*.json')}:raise ValueError('Coarse inputs changed during qualification')
    if errors:
        atomic_json(out/'failure.json',seal(dict(passed=False,errors=errors,closed_tasks=[])))
        raise ValueError('Full load workers failed; evidence preserved')
    report=summarize(out,cycle);atomic_json(out/'report.json',seal(report))
    if not report['passed']:raise ValueError('Full load envelope acceptance failed')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--coarse',type=Path,required=True)
    a=p.parse_args();run(a.output,a.coarse)


if __name__=='__main__':main()
