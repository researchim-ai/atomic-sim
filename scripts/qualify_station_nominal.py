"""Prospective refinement/restart of the archived 3600s six-loop hot approach.

The coarse trace is explicitly retrospective and content-addressed. Criteria
come from the pre-existing R1 catalog, not from fitting the new fine result.
This accepts only NPP-11.01's selected hot approach/stationary tail, not R1.
"""
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from atomic_model_multiphysics.provenance import sha,seal,unseal
from atomic_model_training.spec import atomic_json
from atomic_model_station_faults import FaultSession
from atomic_model_station_faults import model as fault_model
from atomic_model_station_campaign.engine import signature
from atomic_model_station_campaign.matrix import catalog
from scripts import qualify_station_actuators as evidence
from scripts import qualify_station_bypass_full as physical
from scripts import qualify_station_stages as stages
from scripts import qualify_station_steam as steam
from scripts import qualify_station_primary as primary

TASKS=['NPP-11.01']
CRITERIA=dict(energy_relative=1e-6,mass_relative=1e-10,boron_relative=1e-10,
    continuous_relative=1e-3,event_time_s=.05,physical_tail_relative=1e-3,
    stored_power_relative=1e-3,tail_sg_level_m=.02,tail_hotwell_level_m=.005,
    tail_feed_level_m=.005,tail_power_relative=.02,tail_speed_relative=.01,tail_pressure_relative=.02)
BLOCKS=(120,240,360,480,600,720)


def nominal_signature():return signature()


def read(path):return unseal(json.loads(Path(path).read_text()))


def helper_sources():
    return {m.__name__:sha(m.__file__) for m in (evidence,physical,stages,steam,primary)}


def row(owner,result):
    snap=owner.control.plant.snapshot()
    record=dict(time_s=snap['time_s'],result=result,physical_state=owner.control.plant.state.tolist(),
                levels=snap['level_readouts'],powers=snap['powers'])
    return record,snap


def control_events(checkpoint):
    state=checkpoint['payload']['control']['payload']['state']
    ignored={'rod_arrived','bypass_position_arrival','steam_inlet_arrival','steam_outlet_arrival'}
    events=[dict(kind=e['kind'],time_s=e['time_s']) for e in state['plant']['payload']['snapshot']['events']
            if 'command' not in e['kind'] and e['kind'] not in ignored]
    events.extend(dict(kind=e['kind'],time_s=e['time_s']) for e in state['events']
                  if e['kind'] in ('protection_latched','pressure_interlock_entered','pressure_interlock_released'))
    return sorted(events,key=lambda e:(e['time_s'],e['kind']))


def worker(role,inputs,output):
    inputs=Path(inputs);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    source=signature();baseline=read(inputs/'baseline.json')
    assert baseline['source']==source
    coarse=[read(inputs/(str(n)+'.json')) for n in BLOCKS]
    expected=[r for b in coarse for r in b['trace']]
    prefix_exact=None
    if role=='fine':
        owner=FaultSession(tick_s=5.,history_budget_steps=8192,max_step_s=.5,rtol=2e-8);start=0
    elif role=='restart':
        print('native restore: 1800s verified prefix',flush=True)
        saved=coarse[2]['checkpoint'];owner=FaultSession.restore(json.loads(json.dumps(saved,sort_keys=True)))
        prefix_exact=owner.checkpoint()==saved;start=360
        print('native restore: prefix exact; continuing to 3600s',flush=True)
    else:raise ValueError('Unknown nominal worker role')
    trace=[];admitted=True;exact=True;maximum={k:0. for k in ('energy_relative','mass_relative','boron_relative')}
    for index in range(start,720):
        result=owner.step(modes={k:'auto' for k in owner.control.loops} if index==0 else None)
        record,snap=row(owner,result);trace.append(record)
        admitted=bool(physical.admit(snap) and admitted)
        for k in maximum:maximum[k]=max(maximum[k],snap[k])
        if role=='restart':exact=record==expected[index] and exact
        if (index+1)%20==0:print(role,owner.control.plant.time,'/3600',flush=True)
        if (index+1)%120==0:
            atomic_json(output/(str(index+1)+'.json'),seal(dict(source=source,trace=trace,checkpoint=owner.checkpoint())))
            trace=[]
    final=owner.checkpoint()
    if role=='restart':exact=bool(exact and final==coarse[-1]['checkpoint'] and prefix_exact)
    report=dict(role=role,completed=True,source_unchanged=source==signature(),duration_s=owner.control.plant.time,
        maximum_sampled_balances=maximum,physical_audits_passed=admitted,
        restart_prefix_exact=prefix_exact,restart_full_suffix_and_final_exact=exact if role=='restart' else None)
    atomic_json(output/'report.json',seal(report))


def _launch(role,inputs,output,driver):
    root=str(Path(__file__).resolve().parents[1])
    library_root=str(Path(fault_model.__file__).resolve().parents[1])
    code='import sys;sys.path.insert(0,sys.argv[1]);sys.path.insert(0,sys.argv[2]);from scripts.qualify_station_nominal import worker;worker(sys.argv[3],sys.argv[4],sys.argv[5])'
    with tempfile.TemporaryDirectory(prefix='atomic-nominal-worker-') as cwd:
        with (output.parent/(role+'.log')).open('w') as log:
            subprocess.run([sys.executable,'-I','-c',code,root,library_root,role,str(inputs),str(output)],cwd=cwd,
                env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'),
                stdout=log,stderr=subprocess.STDOUT,check=True,timeout=14400)
    if sha(__file__)!=driver:raise ValueError('Nominal driver changed during execution')


def materialize(output,reference,coarse_evidence,baseline_path):
    inputs=output/'inputs';inputs.mkdir()
    if reference is not None:
        source=Path(reference)/'inputs'
        for p in source.iterdir():
            if p.is_file():shutil.copyfile(p,inputs/p.name)
    else:
        baseline_path=Path(baseline_path);baseline=read(baseline_path)
        shutil.copyfile(baseline_path,inputs/'baseline.json')
        for n in BLOCKS:
            p=Path(coarse_evidence)/(str(n)+'.json')
            key='artifacts/station-nominal-exploration-v1/'+p.name
            if sha(p)!=baseline['evidence_sha256'][key]:raise ValueError('Archived coarse trace hash differs')
            shutil.copyfile(p,inputs/p.name)
    baseline=read(inputs/'baseline.json')
    if (not baseline['completed'] or baseline['completed_duration_s']!=3600. or baseline['source']!=signature()
            or baseline['formal_acceptance']):raise ValueError('Expected unchanged, explicitly exploratory coarse input')
    all_rows=[]
    for n in BLOCKS:
        p=inputs/(str(n)+'.json');key='artifacts/station-nominal-exploration-v1/'+p.name
        if sha(p)!=baseline['evidence_sha256'][key]:raise ValueError('Copied coarse trace hash differs')
        b=read(p)
        if b['source']!=signature():raise ValueError('Coarse model identity differs')
        all_rows.extend(b['trace'])
    if len(all_rows)!=720 or [r['time_s'] for r in all_rows]!=[5.*i for i in range(1,721)]:raise ValueError('Complete 3600s archive required')
    return inputs


def _family_metrics(folder,coarse=False):
    rows=[];snapshots=[];checkpoints=[]
    for n in BLOCKS:
        block=read(Path(folder)/(str(n)+'.json'));rows.extend(block['trace']);checkpoints.append(block['checkpoint'])
        snapshots.append(block['checkpoint']['payload']['control']['payload']['state']['plant']['payload']['snapshot'])
    targets=checkpoints[-1]['payload']['control']['payload']['state']['targets']
    tail=[r for r in rows if r['time_s']>=3000.]
    fields={'sg_collapsed_level_m':'feed_flow_fraction','hotwell_level_m':'condensate_flow_fraction','feed_tank_level_m':'feed_inventory_fraction'}
    errors={k:max(abs(r['levels'][k]-targets[v]) for r in tail) for k,v in fields.items()}
    spans={k:max(r['levels'][k] for r in tail)-min(r['levels'][k] for r in tail) for k in fields}
    measured={n:max(abs(r['result']['observation']['measurements'][n]['value']/targets[k]-1) for r in tail)
        for n,k in [('power','rod_target'),('pressure','steam_bypass'),('speed','steam_admission')]}
    vectors=[physical.physical_vector(r) for r in snapshots]
    drift=float(max(abs(vectors[-1]-vectors[-2])/np.maximum(abs(vectors[-2]),1.)))
    final=snapshots[-1];powers=final['powers']
    stored=abs(powers['external_net_w'])/max(abs(powers['nuclear_heat_w']),1.)
    physical_pass=all(physical.admit(r) for r in snapshots)
    good=bool(physical_pass and all(not r['result']['receipt']['trip_latched'] for r in rows)
        and measured['power']<=CRITERIA['tail_power_relative'] and measured['pressure']<=CRITERIA['tail_pressure_relative']
        and measured['speed']<=CRITERIA['tail_speed_relative']
        and errors['sg_collapsed_level_m']<=CRITERIA['tail_sg_level_m'] and spans['sg_collapsed_level_m']<=CRITERIA['tail_sg_level_m']
        and errors['hotwell_level_m']<=CRITERIA['tail_hotwell_level_m'] and spans['hotwell_level_m']<=CRITERIA['tail_hotwell_level_m']
        and errors['feed_tank_level_m']<=CRITERIA['tail_feed_level_m'] and spans['feed_tank_level_m']<=CRITERIA['tail_feed_level_m']
        and drift<=CRITERIA['physical_tail_relative'] and stored<=CRITERIA['stored_power_relative']
        and powers['gross_electrical_w']>powers['own_needs_w']>0 and powers['net_export_w']>0
        and final['steam_flow_kg_s']>0 and powers['condenser_heat_out_w']>0)
    summary=dict(accepted=good,level_errors_m=errors,level_spans_m=spans,measured_relative_errors=measured,
        physical_tail_relative=drift,stored_power_relative=stored,
        saved_checkpoint_physical_audits=physical_pass,final_snapshot=final)
    return rows,checkpoints[-1],summary


def run(output,reference=None,*,coarse_evidence='artifacts/station-nominal-exploration-v1',baseline='docs/benchmarks/station_nominal_exploration_v54.json'):
    output=Path(output).absolute()
    if any(p.is_symlink() for p in (output,*output.parents)):raise ValueError('Symlink output refused')
    output.mkdir(parents=True,exist_ok=False);source=signature();driver=sha(__file__);helpers=helper_sources()
    inputs=materialize(output,reference,coarse_evidence,baseline)
    evidence.save(output/'protocol.json',dict(source=source,driver_sha256=driver,helpers_sha256=helpers,criteria=CRITERIA,
        baseline_sha256=sha(inputs/'baseline.json'),coarse_input_sha256={p.name:sha(p) for p in inputs.iterdir()},
        candidate_tasks=TASKS,nominal_duration_s=3600.,controller_tick_s=5.,tracking_tau_s=10.,history_budget_steps=8192,
        integration_settings=[dict(max_step_s=1.,rtol=1e-7),dict(max_step_s=.5,rtol=2e-8)],restart_at_s=1800.,
        coarse_origin='Retrospective archived exploratory trace; prospective fine/restart checked against prior R1 catalog criteria',
        qualification_scope='Hot six-loop approach and stationary last 600s; no all-hour stationary initialization claim',
        full_r1_matrix=False,catalog=catalog()))
    print('launching full 3600s refinement and native 1800s restore + 1800s continuation',flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(_launch,role,inputs,output/role,driver) for role in ('fine','restart')]
        for future in futures:future.result()
    coarse,coarse_final,coarse_metrics=_family_metrics(inputs,True)
    fine,fine_final,fine_metrics=_family_metrics(output/'fine')
    relative=float(max(abs(a-b)/max(abs(b),1.) for x,y in zip(coarse,fine) for a,b in zip(x['physical_state'],y['physical_state'])))
    ce=control_events(coarse_final);fe=control_events(fine_final)
    events=bool(len(ce)==len(fe) and all(a['kind']==b['kind'] and abs(a['time_s']-b['time_s'])<=CRITERIA['event_time_s'] for a,b in zip(ce,fe)))
    restart=read(output/'restart/report.json');fine_report=read(output/'fine/report.json')
    restart_ok=bool(restart['completed'] and restart['restart_prefix_exact'] and restart['restart_full_suffix_and_final_exact'])
    unchanged=source==signature() and driver==sha(__file__) and helpers==helper_sources()
    passed=bool(unchanged and coarse_metrics['accepted'] and fine_metrics['accepted'] and relative<=CRITERIA['continuous_relative']
        and events and restart_ok and fine_report['physical_audits_passed'] and restart['physical_audits_passed'])
    evidence.save(output/'comparison.json',dict(maximum_all_coordinate_relative=relative,events_agree=events,
        coarse_events=ce,fine_events=fe,coarse=coarse_metrics,fine=fine_metrics))
    report=dict(completed=True,passed=passed,source_unchanged=unchanged,closed_tasks=TASKS if passed else [],
        nominal_duration_s=3600.,trip_duration_s=0.,restart_exact=restart_ok,full_horizon_restart=True,
        refinement_relative=relative,events_agree=events,coarse_stationary_tail=coarse_metrics['accepted'],
        fine_stationary_tail=fine_metrics['accepted'],complete_npp_11=False,full_r1_matrix=False,
        complete_package_c=False,r1_accepted=False,validated=False,plant_referenced=False,
        limitations=['Accepted hot approach/stationary tail, not an initially stationary 3600s hold',
            'Coarse data explicitly retrospective; refinements/restart are prospective with frozen criteria',
            'Other long R1 scenario families and full NPP-11.04 remain open',
            'Research geometry, closures and controllers; no experimental plant validation'])
    evidence.save(output/'report.json',report)
    if reference is not None:
        reference=Path(reference);files={p.relative_to(output) for p in output.rglob('*.json')}
        if files!={p.relative_to(reference) for p in reference.rglob('*.json')}:raise ValueError('Nominal evidence inventories differ')
        for name in sorted(files):
            if evidence.read(output/name)!=evidence.read(reference/name):raise ValueError('Installed nominal evidence differs:'+str(name))
        print('All installed full-hour nominal evidence matches exactly',flush=True)
    if not passed:raise ValueError('Full-hour nominal qualification failed')
    print(json.dumps(dict(passed=True,closed_tasks=TASKS)),flush=True);return report


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference',type=Path);args=parser.parse_args();run(args.output,args.reference)
