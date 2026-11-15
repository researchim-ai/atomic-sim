"""Station campaign isolation, extended journals and scientific replay proof."""
from argparse import ArgumentParser
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import time

from atomic_model_multiphysics.provenance import seal,unseal,sha
from atomic_model_training.spec import atomic_json
from atomic_model_station.journal import Journal
from atomic_model_station_faults import Fault,FaultSession
from atomic_model_station_campaign import Episode,Limits,run_campaign
from atomic_model_station_campaign.engine import run_episode,signature
from atomic_model_station_campaign.matrix import catalog
from scripts import qualify_station_actuators as evidence

TASKS=['NPP-15.03']


def campaign_signature():return signature()


def read(path):return unseal(json.loads(Path(path).read_text()))


def run(output,reference=None):
    output=Path(output).absolute()
    if any(p.is_symlink() for p in (output,*output.parents)):raise ValueError('Symlink output refused')
    output.mkdir(parents=True,exist_ok=False)
    source=signature();driver=sha(__file__);serializer=sha(evidence.__file__)
    execution=output/'execution';execution.mkdir()
    limits=Limits(workers=1,wall_s=180,cpu_s=180)
    episodes=[Episode('measurement',4,faults=(Fault('hidden','bias','pressure',2,value=150000.),),
                      react_on_alarm=True,checkpoint_ticks=(1,)),
              Episode('equipment',4,faults=(Fault('hidden','pump_off','cooling',1),),checkpoint_ticks=(1,))]
    bridge=Episode('beyond_256',257,tick_s=.01,automatic=False,react_on_alarm=True,
        faults=(Fault('measurement','bias','pressure',255,value=150000.),Fault('pump','pump_off','cooling',256)),
        record_every=32,checkpoint_ticks=(256,))
    evidence.save(output/'protocol.json',dict(source=source,driver_sha256=driver,serializer_sha256=serializer,
        candidate_tasks=TASKS,episodes=[e.payload() for e in episodes],bridge=bridge.payload(),
        limits=asdict(limits),parallel_workers=2,nominal_duration_s=2.57,trip_duration_s=4.,
        scope='Short local execution/restart family; 257 accepted ticks are not an R1 physical horizon',
        evidence_comparison_scope='scientific_root_json_only'))
    atomic_json(output/'r1_matrix_catalog.json',catalog())
    print('campaign extended journal: 257 ticks',flush=True)
    first=run_episode(bridge)
    atomic_json(output/'extended.json',first)
    anchor=unseal(first)['checkpoints']['256']
    print('campaign extended journal: resume across 256',flush=True)
    replay=run_episode(bridge,json.loads(json.dumps(anchor,sort_keys=True)))
    exact=first==replay
    evidence.save(output/'extended_restart.json',dict(exact=exact,final=replay))
    final=unseal(first)['final']
    print('campaign native checkpoint: 257 ticks',flush=True)
    restored=FaultSession.restore(final)
    native=restored.checkpoint()==final
    before=restored.checkpoint();blocked=False
    try:restored.step()
    except ValueError:blocked=restored.checkpoint()==before
    evidence.save(output/'native_restore.json',dict(exact=native,budget_rejected_atomically=blocked))
    science=unseal(first)
    bridge_ok=bool(science['accepted'] and science['completed_ticks']==257
        and len(science['final']['payload']['history'])==257 and exact and native and blocked
        and science['final']['payload']['control']['payload']['state']['trip_latched'])
    print('campaign sequential independent processes',flush=True)
    seq=run_campaign(execution/'sequential',episodes,limits=limits)
    print('campaign parallel independent processes',flush=True)
    parallel_limits=Limits(workers=2,wall_s=180,cpu_s=180)
    par=run_campaign(execution/'parallel',episodes,limits=parallel_limits)
    sequential={e.name:read(execution/'sequential'/(e.name+'.json')) for e in episodes}
    parallel={e.name:read(execution/'parallel'/(e.name+'.json')) for e in episodes}
    process_ok=all(r['accepted'] and r['scientific'] is not None for r in [*sequential.values(),*parallel.values()])
    pair_exact=process_ok and all(sequential[n]['scientific']==parallel[n]['scientific'] for n in sequential)
    pids=[r['execution'].get('pid') for r in parallel.values()]
    isolated=bool(process_ok and len(set(pids))==len(episodes)
        and all(r['execution']['thread_environment']==dict(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1') for r in parallel.values()))
    for name,result in sequential.items():
        if result['scientific'] is not None:atomic_json(output/(name+'.json'),result['scientific'])
    resumes={n:unseal(r['scientific'])['checkpoints']['1'] for n,r in sequential.items()} if process_ok else {}
    print('campaign parallel checkpoint continuations',flush=True)
    resumed=run_campaign(execution/'resumed',episodes,limits=parallel_limits,resumes=resumes)
    resumed_exact=process_ok and resumed['accepted'] and all(read(execution/'resumed'/(n+'.json'))['scientific']==r['scientific'] for n,r in sequential.items())
    evidence.save(output/'parallel_replay.json',dict(sequential_accepted=seq['accepted'],parallel_accepted=par['accepted'],
        all_scientific_exact=bool(pair_exact),separate_processes=isolated,resumed_exact=bool(resumed_exact)))
    print('campaign hard wall-time limit',flush=True)
    probe=run_campaign(execution/'wall_limit',[Episode('wall_probe',8192,tick_s=.01,automatic=False,record_every=32)],
        limits=Limits(workers=1,wall_s=1,cpu_s=120))
    timed_out=not probe['accepted'] and probe['results'][0]['status']=='timed_out'
    evidence.save(output/'resource_gate.json',dict(timeout_enforced=timed_out,partial_episode_not_accepted=not probe['accepted']))
    invalid=Episode('rejected',3,tick_s=.01,automatic=False,expected_rejection=True,
        commands=({'tick':1,'operator':{'unknown_valve':.5}},))
    rejection=run_episode(invalid);atomic_json(output/'rejection.json',rejection)
    reject_ok=unseal(rejection)['accepted'] and unseal(rejection)['rollback_exact'] and unseal(rejection)['completed_ticks']==1
    # Timing is measured separately; a claimed speedup is not an acceptance gate.
    costs=[]
    for count in (1024,4096,8192):
        rows=[dict(operator=dict(x=i)) for i in range(count)];journal=Journal(rows)
        start=time.perf_counter()
        for _ in range(20):deepcopy(rows)
        lists=time.perf_counter()-start;start=time.perf_counter()
        for _ in range(20):deepcopy(journal)
        structural=time.perf_counter()-start
        costs.append(dict(rows=count,list_copy_s=lists,journal_copy_s=structural,copies=20))
    atomic_json(execution/'journal_cost.json',seal(costs))
    unchanged=source==signature() and driver==sha(__file__) and serializer==sha(evidence.__file__)
    passed=bool(unchanged and bridge_ok and pair_exact and isolated and resumed_exact and timed_out and reject_ok)
    report=dict(completed=True,passed=passed,source_unchanged=unchanged,closed_tasks=TASKS if passed else [],
        nominal_duration_s=2.57,trip_duration_s=4.,restart_exact=bool(exact and native and resumed_exact),
        bridge_ticks=257,extended_journal_qualified=bridge_ok,parallel_scientific_exact=bool(pair_exact),
        independent_processes=isolated,hard_timeout_enforced=timed_out,invalid_input_rollback=bool(reject_ok),
        maximum_energy_relative=science['maxima']['energy_relative'],maximum_mass_relative=science['maxima']['mass_relative'],
        maximum_boron_relative=science['maxima']['boron_relative'],evidence_comparison_scope='scientific_root_json_only',
        complete_npp_11=False,full_r1_matrix=False,complete_package_c=False,r1_accepted=False,validated=False,
        limitations=['257 ticks at 10ms qualify operation-count/restart support, not 3600s physical stationarity',
            'R1 catalog is frozen but its long-duration physical cases are not accepted by this study',
            'Resource isolation uses POSIX RLIMITs; per-worker limits, not a cluster scheduler',
            'Restart deliberately replays the verified prefix; no constant-time restoration claim',
            'Scientific JSON is exact; PID, RSS and wall/CPU timing are separate nonidentical measurements'])
    evidence.save(output/'report.json',report)
    if reference is not None:
        reference=Path(reference);names={p.name for p in output.glob('*.json')}
        if names!={p.name for p in reference.glob('*.json')}:raise ValueError('Campaign scientific evidence sets differ')
        for name in sorted(names):
            if evidence.read(output/name)!=evidence.read(reference/name):raise ValueError('Installed campaign science differs:'+name)
        print('All installed campaign scientific evidence matches exactly',flush=True)
    if not passed:raise ValueError('Station campaign qualification failed')
    print(json.dumps(dict(passed=True,closed_tasks=TASKS)),flush=True);return report


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference',type=Path);args=parser.parse_args();run(args.output,args.reference)
