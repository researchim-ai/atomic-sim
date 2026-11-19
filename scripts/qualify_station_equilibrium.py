"""Prospective full nominal hour from a constrained time-zero equilibrium.

Source executes coarse+fine. Installed delivery re-executes fine and the
complete coarse history via native prefix restore plus suffix comparison.
Native evidence stays separate from the immutable mathematical tree.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from atomic_model_multiphysics.provenance import sha,seal
from atomic_model_training.spec import atomic_json
from atomic_model_station_faults import FaultSession
from atomic_model_station_faults import model as fault_model
from atomic_model_station_campaign.engine import signature
from atomic_model_station_equilibrium import signature as equilibrium_signature
from atomic_model_station_inventory import capture as inventory_capture, summarize
from atomic_model_station_inventory import model as inventory_model
from atomic_model_station_readiness import capture, assess, schema, ReadinessConfig
from atomic_model_station_readiness import model as readiness_model
from scripts.qualify_station_inventory import read
from scripts.qualify_station_hourly_replay import physical_events


CRITERIA=dict(refinement_relative=1e-3,event_time_s=.05,local_continuity_kg_s=1e-9,
              total_rate_kg_s=1e-9,energy_relative=1e-6,mass_relative=1e-10,boron_relative=1e-10)
BLOCKS=(120,240,360,480,600,720)


def identity():
    return dict(source=signature(),equilibrium=equilibrium_signature(),driver_sha256=sha(__file__),
                inventory_sha256=sha(inventory_model.__file__),readiness_sha256=sha(readiness_model.__file__))


def collect(owner,result):
    return dict(sample=capture(owner),inventory=inventory_capture(owner),result=result)


def rows(directory):
    root=Path(directory)
    initial=read(root/'initial.json')
    values=[initial['record']]
    for n in BLOCKS:values.extend(read(root/('block-'+str(n)+'.json'))['records'])
    return values


def worker(role,output,coarse=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    source=identity();normal=ReadinessConfig();stock=[];samples=[];window=[];ready=[]
    if role in ('coarse','fine'):
        owner=FaultSession(tick_s=5.,history_budget_steps=8192,equilibrium_start=True,inventory_deadband_m=.0002,
                           max_step_s=1. if role=='coarse' else .5,rtol=1e-7 if role=='coarse' else 2e-8)
        initial=collect(owner,None);samples.append(initial['sample']);stock.append(initial['inventory']);window.append(initial['sample'])
        atomic_json(output/'initial.json',seal(dict(record=initial,checkpoint=owner.checkpoint(),layout=schema(owner))))
        start=0;expected=None
    elif role=='native':
        expected=rows(coarse)
        saved=read(Path(coarse)/'block-360.json')['checkpoint']
        print('native restore: verifying full 1800s prefix',flush=True)
        owner=FaultSession.restore(saved)
        if owner.checkpoint()!=saved:raise ValueError('Equilibrium native prefix differs')
        start=360;print('native prefix exact; continuing to 3600s',flush=True)
    else:raise ValueError('Equilibrium worker role')
    exact=True;block=[]
    for index in range(start,720):
        result=owner.step(modes={k:'auto' for k in owner.control.loops} if index==0 else None)
        record=collect(owner,result);block.append(record)
        if role=='native':
            exact=record==expected[index+1] and exact
        else:
            samples.append(record['sample']);stock.append(record['inventory']);window.append(record['sample']);window=window[-121:]
            if index+1>=120:ready.append(assess(window,schema(owner),normal))
        if (index+1)%20==0:print(role,owner.control.plant.time,'/3600',flush=True)
        if (index+1)%120==0:
            atomic_json(output/('block-'+str(index+1)+'.json'),seal(dict(records=block,checkpoint=owner.checkpoint())))
            block=[]
    final=owner.checkpoint()
    if role=='native':
        exact=exact and final==read(Path(coarse)/'block-720.json')['checkpoint']
        report=dict(completed=True,prefix_exact=True,full_suffix_and_final_exact=bool(exact),
                    duration_s=owner.control.plant.time,source_unchanged=source==identity())
        atomic_json(output/'report.json',seal(report))
        if not exact or not report['source_unchanged']:raise ValueError('Native equilibrium replay failed')
        return
    full=assess(samples,schema(owner),ReadinessConfig(window_s=3600.,maximum_gap_s=5.))
    inventory=summarize(stock,maximum_gap_s=5.)
    physical=(inventory['maximum_local_continuity_residual_kg_s']<=CRITERIA['local_continuity_kg_s']
              and inventory['maximum_total_rate_kg_s']<=CRITERIA['total_rate_kg_s']
              and all(inventory['maximum_audits'][k]<=CRITERIA[k] for k in ('energy_relative','mass_relative','boron_relative')))
    report=dict(completed=True,duration_s=owner.control.plant.time,source_unchanged=source==identity(),
        full_hour_envelope=full,window_count=len(ready),all_600s_windows_ready=all(x['ready'] for x in ready),
        maximum_600s_span=max(x['maximum_normalized_span'] for x in ready),inventory=inventory,
        physical_audits_passed=physical,all_observables_met=all(s['goals_met'] and not s['trip'] for s in samples),
        initial_state_solved=True,operating_state_projected=False,clock_reset=False,pi_reset=False)
    report['passed']=bool(full['ready'] and report['all_600s_windows_ready'] and len(ready)==601 and physical
                          and report['all_observables_met'] and report['source_unchanged'])
    atomic_json(output/'report.json',seal(report))
    if not report['passed']:raise ValueError('Full nominal equilibrium hour failed; evidence preserved')


def launch(role,output,coarse=None):
    output=Path(output).absolute();root=str(Path(__file__).resolve().parents[1]);library_root=str(Path(fault_model.__file__).resolve().parents[1])
    code='import sys;sys.path.insert(0,sys.argv[1]);sys.path.insert(0,sys.argv[2]);from scripts.qualify_station_equilibrium import worker;worker(sys.argv[3],sys.argv[4],None if sys.argv[5]=="-" else sys.argv[5])'
    with tempfile.TemporaryDirectory(prefix='atomic-equilibrium-worker-') as cwd:
        with (output.parent/(role+'.log')).open('w') as log:
            subprocess.run([sys.executable,'-I','-c',code,root,library_root,role,str(output),str(coarse) if coarse else '-'],
                cwd=cwd,env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'),
                stdout=log,stderr=subprocess.STDOUT,check=True,timeout=14400)


def run(output,reference=None):
    out=Path(output).absolute()
    if any(p.is_symlink() for p in (out,*out.parents)):raise ValueError('Symlink output refused')
    out.mkdir(parents=True,exist_ok=False);mathematics=out/'mathematics';mathematics.mkdir()
    protocol=dict(**identity(),criteria=CRITERIA,readiness=asdict(ReadinessConfig()),duration_s=3600.,tick_s=5.,
        initial_setup='Constrained fresh time-zero solve, same equipment parameters and conserved masses; explicit initial energy declaration',
        mode='Six automatic loops, equilibrium_start=True, inventory_deadband_m=0.0002',
        restart_at_s=1800.,native_scope='Full prefix re-executed and full suffix compared in installed delivery',
        stationary_accepted=False,r1_accepted=False)
    if reference is not None and read(Path(reference)/'mathematics/protocol.json')!=protocol:
        raise ValueError('Installed equilibrium study identity differs before physics work')
    atomic_json(mathematics/'protocol.json',seal(protocol))
    if reference is None:
        jobs=[('coarse',mathematics/'coarse',None),('fine',mathematics/'fine',None)]
    else:
        reference=Path(reference)
        shutil.copytree(reference/'mathematics/coarse',mathematics/'coarse')
        jobs=[('fine',mathematics/'fine',None),('native',out/'native',mathematics/'coarse')]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(launch,*job) for job in jobs]
        for future in futures:future.result()
    coarse=rows(mathematics/'coarse');fine=rows(mathematics/'fine')
    excluded=set(read(mathematics/'coarse/initial.json')['layout']['accounting_indices'])
    difference=max(abs(a-b)/max(abs(b),1.) for x,y in zip(coarse,fine)
        for i,(a,b) in enumerate(zip(x['sample']['state'],y['sample']['state'])) if i not in excluded)
    coarse_cp=read(mathematics/'coarse/block-720.json')['checkpoint'];fine_cp=read(mathematics/'fine/block-720.json')['checkpoint']
    a=physical_events(coarse_cp);b=physical_events(fine_cp)
    events=len(a)==len(b) and all(x['kind']==y['kind'] and abs(x['time_s']-y['time_s'])<=CRITERIA['event_time_s'] for x,y in zip(a,b))
    coarse_report=read(mathematics/'coarse/report.json');fine_report=read(mathematics/'fine/report.json')
    unchanged=all(protocol[k]==v for k,v in identity().items())
    report=dict(passed=bool(coarse_report['passed'] and fine_report['passed'] and difference<=CRITERIA['refinement_relative'] and events and unchanged),
        source_unchanged=unchanged,duration_s=3600.,refinement_relative=difference,physical_events_agree=events,
        coarse=coarse_report,fine=fine_report,stationary_candidate=True,stationary_accepted=False,closed_tasks=[],r1_accepted=False,
        accepted_scope='Prospective coarse/fine full-hour stationarity; native qualification is separate delivery evidence')
    atomic_json(mathematics/'report.json',seal(report))
    if not report['passed']:raise ValueError('Equilibrium full-hour refinement gate failed')
    if reference is not None:
        expected={p.relative_to(reference/'mathematics') for p in (reference/'mathematics').rglob('*.json')}
        actual={p.relative_to(mathematics) for p in mathematics.rglob('*.json')}
        if expected!=actual or any((reference/'mathematics'/p).read_bytes()!=(mathematics/p).read_bytes() for p in actual):
            raise ValueError('Installed full-hour mathematical evidence differs')
        native=read(out/'native/report.json')
        if not native['prefix_exact'] or not native['full_suffix_and_final_exact'] or not native['source_unchanged']:
            raise ValueError('Installed full-hour native gate failed')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--reference',type=Path)
    args=parser.parse_args();r=run(args.output,args.reference);print(json.dumps(dict(passed=r['passed'],native_included=args.reference is not None)),flush=True)


if __name__=='__main__':main()
