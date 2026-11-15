"""Full-hour numerical replay acceptance, separate from failed stationarity.

Preserves the original failed NPP-11.01 report. No task is closed by this
study. Installed execution repeats both full physical workers, not the audit
alone. The readiness verdict cannot be overridden by small sensor errors.
"""
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import shutil

import numpy as np
from atomic_model_multiphysics.provenance import seal,sha
from atomic_model_training.spec import atomic_json
from atomic_model_station_faults import FaultSession
from atomic_model_station_campaign.engine import signature
from atomic_model_station_readiness import schema,assess,ReadinessConfig
from atomic_model_station_readiness import model as readiness
from scripts import qualify_station_nominal as nominal
from scripts import qualify_station_actuators as evidence

TASKS=[]


def hourly_signature():return signature()


def read(path):return evidence.read(path)


def physical_events(checkpoint):
    state=checkpoint['payload']['control']['payload']['state']
    # Exact kinematic arrivals are checked by actuator-coordinate convergence.
    # These are the actual backend labels, not guessed inlet/outlet aliases.
    kinematic={'rod_arrived','steam_valve_arrived','bypass_position_arrival'}
    result=[dict(kind=e['kind'],time_s=e['time_s']) for e in state['plant']['payload']['snapshot']['events']
        if 'command' not in e['kind'] and e['kind'] not in kinematic]
    result.extend(dict(kind=e['kind'],time_s=e['time_s']) for e in state['events']
        if e['kind'] in ('protection_latched','pressure_interlock_entered','pressure_interlock_released'))
    return sorted(result,key=lambda e:(e['time_s'],e['kind']))


def readiness_of(rows,layout,audit_bounds):
    # Old archive stores raw state and power each tick, but balance residuals
    # only at 600s checkpoints. The backend accepted every original tick under
    # its global audit bounds; never invent zero residuals for missing records.
    samples=[]
    for r in rows:
        p=r['powers']
        samples.append(dict(time_s=r['time_s'],state=r['physical_state'],
            energy_relative=audit_bounds['energy_relative'],mass_relative=audit_bounds['mass_relative'],
            boron_relative=audit_bounds['boron_relative'],stored_power_relative=abs(p['external_net_w'])/max(abs(p['nuclear_heat_w']),1.),
            trip=r['result']['receipt']['trip_latched'],goals_met=readiness.goals_met(r['result']['observation'],r['result']['receipt']['targets'])))
    verdict=assess(samples,layout,ReadinessConfig())
    verdict['audit_evidence']='Conservative whole-run backend bounds, not per-sample measured maxima in the sparse legacy archive'
    verdict['goals_evidence']='Computed from every supplied measured frame and declared setpoints'
    return verdict


def run(output,reference=None,*,candidate='artifacts/station-nominal-qualified-v2'):
    output=Path(output).absolute()
    if any(p.is_symlink() for p in (output,*output.parents)):raise ValueError('Symlink output refused')
    output.mkdir(parents=True,exist_ok=False)
    source=signature();driver=sha(__file__);worker_sha=sha(nominal.__file__);readiness_sha=sha(readiness.__file__)
    study=output/'study'
    if reference is not None:
        expected=read(Path(reference)/'protocol.json')
        if expected['source']!=source or expected['worker_sha256']!=worker_sha or expected['readiness_sha256']!=readiness_sha:
            raise ValueError('Installed hourly helper/model identity differs before numerical work')
    if reference is None:
        candidate=Path(candidate)
        if not (candidate/'report.json').exists():raise ValueError('Complete candidate evidence required')
        original=read(candidate/'report.json');protocol=read(candidate/'protocol.json')
        if not original['source_unchanged'] or protocol['source']!=source or protocol['driver_sha256']!=worker_sha:
            raise ValueError('Stale original nominal study')
        shutil.copytree(candidate,study)
    else:
        reference=Path(reference);study.mkdir()
        inputs=nominal.materialize(study,reference/'study',None,None)
        shutil.copyfile(reference/'study/protocol.json',study/'protocol.json')
        print('installed full-hour refinement and full native restart',flush=True)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(nominal._launch,role,inputs,study/role,worker_sha) for role in ('fine','restart')]
            for future in futures:future.result()
        # Preserve the original negative qualification and comparison verbatim;
        # scientific worker data are all compared anew below.
        for name in ('comparison.json','report.json'):
            shutil.copyfile(reference/'study'/name,study/name)
    old=read(study/'protocol.json')
    if old['source']!=source or old['driver_sha256']!=worker_sha or old['helpers_sha256']!=nominal.helper_sources():raise ValueError('Original nominal worker identity')
    coarse,coarse_cp,coarse_metrics=nominal._family_metrics(study/'inputs',True)
    fine,fine_cp,fine_metrics=nominal._family_metrics(study/'fine')
    fine_report=read(study/'fine/report.json');restart_report=read(study/'restart/report.json')
    relative=float(max(abs(a-b)/max(abs(b),1.) for x,y in zip(coarse,fine) for a,b in zip(x['physical_state'],y['physical_state'])))
    a=physical_events(coarse_cp);b=physical_events(fine_cp)
    events=bool(len(a)==len(b) and all(x['kind']==y['kind'] and abs(x['time_s']-y['time_s'])<=nominal.CRITERIA['event_time_s'] for x,y in zip(a,b)))
    owner=FaultSession(tick_s=5.,history_budget_steps=8192);layout=schema(owner)
    bounds={k:nominal.CRITERIA[k] for k in ('energy_relative','mass_relative','boron_relative')}
    coarse_ready=readiness_of(coarse,layout,bounds)
    fine_ready=readiness_of(fine,layout,fine_report['maximum_sampled_balances'])
    restart=bool(restart_report['restart_prefix_exact'] and restart_report['restart_full_suffix_and_final_exact'])
    # The old source gate may fail stationarity; numerical correctness remains
    # a separate, explicitly labelled result. Never turn it into NPP-11 closure.
    passed=bool(relative<=nominal.CRITERIA['continuous_relative'] and events and restart
        and fine_report['physical_audits_passed'] and restart_report['physical_audits_passed']
        and fine_report['source_unchanged'] and restart_report['source_unchanged']
        and source==signature() and driver==sha(__file__) and worker_sha==sha(nominal.__file__)
        and readiness_sha==sha(readiness.__file__))
    evidence.save(output/'protocol.json',dict(source=source,driver_sha256=driver,worker_sha256=worker_sha,
        readiness_sha256=readiness_sha,criteria=nominal.CRITERIA,candidate_tasks=[],nominal_duration_s=3600.,
        restart_at_s=1800.,settings=old['integration_settings'],scope='Full-hour numerical reproducibility only; whole-inventory readiness is separate',
        original_nominal_protocol_sha256=sha(study/'protocol.json'),
        original_nominal_report_sha256=sha(study/'report.json')))
    evidence.save(output/'readiness.json',dict(schema=layout,coarse=coarse_ready,fine=fine_ready,
        original_coarse_metrics={k:v for k,v in coarse_metrics.items() if k!='final_snapshot'},
        original_fine_metrics={k:v for k,v in fine_metrics.items() if k!='final_snapshot'},
        full_stationary_accepted=bool(coarse_ready['ready'] and fine_ready['ready'] and coarse_metrics['accepted'] and fine_metrics['accepted'])))
    evidence.save(output/'comparison.json',dict(maximum_coordinate_relative=relative,physical_events_agree=events,
        coarse_events=a,fine_events=b,native_restart_exact=restart))
    report=dict(completed=True,passed=passed,source_unchanged=source==signature(),closed_tasks=[],
        nominal_duration_s=3600.,trip_duration_s=0.,restart_exact=restart,full_horizon_restart=True,
        refinement_relative=relative,physical_events_agree=events,
        stationary_accepted=bool(coarse_ready['ready'] and fine_ready['ready'] and coarse_metrics['accepted'] and fine_metrics['accepted']),
        complete_npp_11=False,r1_accepted=False,validated=False,plant_referenced=False,
        accepted_scope='Numerical replay, not hot stationary admission',
        limitations=['NPP-11.01 remains open when auxiliary stocks or other physical coordinates drift',
            'Coarse trace retrospectively archived; full fine/restart are prospective',
            'No full R1 matrix or experimental validation; no thresholds relaxed after failed stationarity'])
    evidence.save(output/'report.json',report)
    if reference is not None:
        names={p.relative_to(output) for p in output.rglob('*.json')}
        if names!={p.relative_to(reference) for p in reference.rglob('*.json')}:raise ValueError('Hourly scientific evidence sets differ')
        for name in sorted(names):
            if read(output/name)!=read(reference/name):raise ValueError('Installed hourly evidence differs:'+str(name))
        print('All installed hourly replay and readiness evidence matches exactly',flush=True)
    if not passed:raise ValueError('Full-hour numerical replay failed')
    print(json.dumps(dict(passed=True,closed_tasks=[],stationary_accepted=report['stationary_accepted'])),flush=True)
    return report


if __name__=='__main__':
    p=ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reference',type=Path);p.add_argument('--candidate',type=Path,default=Path('artifacts/station-nominal-qualified-v2'));args=p.parse_args();run(args.output,args.reference,candidate=args.candidate)
