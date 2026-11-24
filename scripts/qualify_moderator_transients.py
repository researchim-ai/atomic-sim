"""Seconds-scale rod transients and controlled legacy-law comparison.

Both laws are reduced research models in this library, not independent codes.
No coefficient, domain or solver work limit is changed by this qualification.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
from atomic_model_multiphysics.provenance import seal
from atomic_model_training.spec import atomic_json
from atomic_model_station import CoreConfig
from coupling.conservative import vector
from coupling.pwr_moderator import LocalModeratorPWRSession,identity
from coupling.pwr_stationary import StationaryChannelPWRSession
from scripts.qualify_pwr_distributed_fuel import checks as fuel_checks,relative

PRESETS=dict(coarse=dict(max_step_s=.1,rtol=2e-8),fine=dict(max_step_s=.05,rtol=2e-9))
CRITERIA=dict(refinement_relative=1e-4,initial_state_relative=1e-12,minimum_correction=1e-6,
    minimum_density_change_kg_m3=.01,minimum_power_effect_relative=1e-4,numerical_margin=20.,map_relative=1e-12)


def separation_metrics(local_coarse,local_fine,bulk_coarse,bulk_fine,reference_power_w):
    """Compare an internal law effect with both laws' numerical preset changes."""
    if type(reference_power_w) not in (int,float) or not np.isfinite(reference_power_w) or reference_power_w<=0:
        raise ValueError('Positive finite reference fission power required')
    arrays=[vector(v) for v in (local_coarse,local_fine,bulk_coarse,bulk_fine)]
    if any(a.shape!=arrays[0].shape or np.any(a<0) for a in arrays):raise ValueError('Matching nonnegative fission traces required')
    lc,lf,bc,bf=arrays
    effect=float(max(abs(lf-bf))/reference_power_w)
    noise=float(max(max(abs(lc-lf)),max(abs(bc-bf)))/reference_power_w)
    return dict(power_effect_relative=effect,numerical_preset_change_relative=noise,
        separation_ratio=None if noise==0 else effect/noise,
        resolved=bool(effect>=CRITERIA['minimum_power_effect_relative'] and effect>=CRITERIA['numerical_margin']*noise))


def protocol(case,position):
    if type(position) not in (int,float) or not np.isfinite(position) or not 0<=position<=.99:raise ValueError('Rod insertion reference in0..0.99 required')
    if case not in ('insertion','return'):raise ValueError('Known rod scenario required')
    steps=[dict(duration_s=.1,patch={'rod_target':position+.01}),dict(duration_s=.4,patch=None),dict(duration_s=.5,patch=None)]
    steps.append(dict(duration_s=1.,patch=None if case=='insertion' else {'rod_target':position}))
    if case=='return':steps.append(dict(duration_s=1.,patch=None))
    return steps


def _run(session,steps,restart):
    initial=session.snapshot();rows=[];saved=None
    for i,step in enumerate(steps):
        rows.append(session.advance(**step))
        print(type(session).__name__,'time',session.time,flush=True)
        if i==1:saved=json.loads(json.dumps(session.checkpoint()))
    replay=None
    if restart:
        restored=LocalModeratorPWRSession.restore(saved)
        suffix=[restored.advance(**step) for step in steps[2:]]
        replay=suffix==rows[2:] and restored.checkpoint()==session.checkpoint()
    before=session.checkpoint()
    try:session.advance(.01,{'rod_target':2.})
    except ValueError:rollback=session.checkpoint()==before
    else:rollback=False
    return dict(initial=initial,steps=deepcopy(steps),snapshots=rows,checkpoint=before,restart_exact=replay,
        invalid_command_rollback=rollback,checks=[fuel_checks(r) for r in rows])


def assess(trajectories,steps):
    """Recompute comparison/budget gates from the retained trajectory records."""
    if type(trajectories) is not dict or set(trajectories)!={'local_coarse','local_fine','bulk_coarse','bulk_fine'}:
        raise ValueError('Exactly four law/preset trajectories required')
    initial=trajectories['local_fine']['initial'];reference=initial['local_moderator']['reference']
    timeline=np.cumsum([s['duration_s'] for s in steps])
    gates=dict(commands=True,timeline=True,initial_state=True,budgets=True,rollback=True)
    power={}
    for name,t in trajectories.items():
        if len(t['snapshots'])!=len(steps):raise ValueError('Complete common observation timeline required')
        gates['commands']&=t['steps']==steps
        gates['timeline']&=all(abs(r['time_s']-time)<1e-12 for r,time in zip(t['snapshots'],timeline))
        gates['initial_state']&=relative(t['initial']['coupled_state'],np.asarray(initial['coupled_state']))<=CRITERIA['initial_state_relative']
        gates['budgets']&=all(all(fuel_checks(r).values()) for r in t['snapshots'])
        gates['rollback']&=t['invalid_command_rollback'] is True
        power[name]=[r['powers']['nuclear_fission_w'] for r in t['snapshots']]
    refined=max(relative(a['coupled_state'],np.asarray(b['coupled_state'])) for law in ('local','bulk')
        for a,b in zip(trajectories[law+'_coarse']['snapshots'],trajectories[law+'_fine']['snapshots']))
    local_rows=[r for key,t in trajectories.items() if key.startswith('local_') for r in t['snapshots']]
    correction=max(float(max(abs(np.r_[r['local_moderator']['density_reactivity_correction'],r['local_moderator']['boron_reactivity_correction']]))) for r in local_rows)
    density=max(float(max(abs(np.asarray(r['local_moderator']['density_kg_m3'])-r['local_moderator']['reference']['density_kg_m3']))) for r in local_rows)
    gates.update(refined=refined<=CRITERIA['refinement_relative'],noticeable_correction=correction>=CRITERIA['minimum_correction'],
        noticeable_density=density>=CRITERIA['minimum_density_change_kg_m3'],
        reference_frozen=all(r['local_moderator']['reference']==reference for r in local_rows),
        map_budgets=all(max(r['local_moderator']['mass_relative_defect'],r['local_moderator']['boron_relative_defect'])<=CRITERIA['map_relative'] for r in local_rows),
        exact_mid_motion_restart=trajectories['local_fine']['restart_exact'] is True)
    separation=separation_metrics(*(power[n] for n in ('local_coarse','local_fine','bulk_coarse','bulk_fine')),float(initial['powers']['nuclear_fission_w']))
    gates['effect_resolved']=separation['resolved']
    # The same physical rod drive must complete its requested stroke in each
    # law. Commands are held across the exact motion's arrival boundary.
    target=steps[-1]['patch']['rod_target'] if steps[-1]['patch'] else next(s['patch']['rod_target'] for s in reversed(steps) if s['patch'])
    gates['rod_target_reached']=all(abs(t['snapshots'][-1]['actuator_status']['rod_position']-target)<1e-12 for t in trajectories.values())
    expected_arrivals=[1.] if len(steps)==4 else [1.,2.]
    gates['rod_arrivals_complete']=all(len(events)==len(expected_arrivals) and all(abs(e['time_s']-time)<1e-12 for e,time in zip(events,expected_arrivals))
        for events in ([e for e in t['snapshots'][-1]['events'] if e['kind']=='rod_arrived'] for t in trajectories.values()))
    return dict(passed=all(gates.values()),gates={k:bool(v) for k,v in gates.items()},refinement_relative=refined,
        maximum_correction=correction,maximum_density_change_kg_m3=density,comparison=separation)


def run(output):
    out=Path(output).absolute()
    if out.exists() or any(p.is_symlink() for p in (out,*out.parents)):raise ValueError('Fresh ordinary output required')
    out.mkdir(parents=True);source=identity();driver=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    cases={};summaries=[]
    for case in ('insertion','return'):
        trajectories={};steps=None
        for preset,options in PRESETS.items():
            local=LocalModeratorPWRSession(core=CoreConfig(kind='diffusion',cells=2),**options)
            bulk=StationaryChannelPWRSession.restore(local.initial_stationary)
            if steps is None:steps=protocol(case,float(local.state[12]))
            for law,session in [('local',local),('bulk',bulk)]:
                print(case,preset,law,flush=True)
                trajectories[law+'_'+preset]=_run(session,steps,restart=law=='local' and preset=='fine')
        summary=dict(case=case,**assess(trajectories,steps));summaries.append(summary)
        cases[case]=dict(summary=summary,trajectories=trajectories)
        atomic_json(out/(case+'.json'),seal(cases[case]));print(case,'passed',summary['passed'],flush=True)
    report=dict(source=source,driver_sha256=driver,criteria=CRITERIA,cases=summaries,
        passed=all(c['passed'] for c in summaries) and source==identity() and driver==hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        trajectories=8,local_restart_suffixes=2,horizons_s=[2.,3.],
        comparison_scope='Controlled internal comparison of local anchored and inherited bulk moderator laws; same code family',
        seconds_scale_density_response_accepted=True,boron_transient_qualified=False,experimental_validation=False,
        long_horizon_accepted=False,station_env_connected=False,closed_tasks=[])
    # A failed assessment cannot advertise acceptance through a separate flag.
    report['seconds_scale_density_response_accepted']=report['passed']
    atomic_json(out/'report.json',seal(report));atomic_json(out/'scientific.json',seal(dict(report=report,cases=cases)))
    if not report['passed']:raise ValueError('Moderator transient qualification failed; attempts retained')
    return dict(passed=True,cases=2,minimum_effect_relative=min(c['comparison']['power_effect_relative'] for c in summaries))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    print(run(parser.parse_args().output))
