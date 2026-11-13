"""Finite backup/DHR local hot-station qualification; no plant-safety claim."""
from argparse import ArgumentParser
import json
from pathlib import Path
import numpy as np
from atomic_model_station import PWRPowerSession,BackupConfig,TrainConfig
from atomic_model_station.pwr_train import physical_signature
from core.status import DomainError,SolverError
from atomic_model_multiphysics.provenance import sha
from scripts import qualify_station_actuators as evidence

TASKS=['NPP-08.03','NPP-08.04']
SETTINGS=[(.2,1e-7),(.1,2e-8),(.05,2e-9)]
CRITERIA=dict(energy_relative=1e-6,mass_relative=1e-10,boron_relative=1e-10,instantaneous_relative=1e-8,
    transfer_residual_w=1e-5,refinement_relative=1e-4,tail_drift_relative=1e-5,exhaustion_time_error_s=1e-8,
    backup_use_error_j=1e-3,grid_energy_residual_j=1e-2,grid_energy_relative=1e-12,entropy_min_w_k=-1e-6)


def vector(r):return np.r_[r['pwr_state'],r['turbine_energy_j'],r['clad_temperature_k'],r['cooling_energy_j'],r['cooling']['flow_kg_s'],r['feed_inventory'],r['electrical']['backup_energy_j']]


def grid_energy_admitted(r):
    scale=max(abs(r['net_exported_j']),abs(r['net_generated_j']),abs(r['backup_used_j']),1.)
    tolerance=max(CRITERIA['grid_energy_residual_j'],CRITERIA['grid_energy_relative']*scale)
    return abs(r['net_exported_j']-r['net_generated_j']-r['backup_used_j'])<=tolerance


def admit(r):
    e=r['electrical']
    return bool(r['energy_relative']<=CRITERIA['energy_relative'] and r['mass_relative']<=CRITERIA['mass_relative']
        and r['boron_relative']<=CRITERIA['boron_relative'] and abs(r['instantaneous_energy_defect_w'])/max(abs(r['powers']['nuclear_heat_w']),1.)<=CRITERIA['instantaneous_relative']
        and abs(e['energy_transfer_residual_w'])<=CRITERIA['transfer_residual_w'] and e['dhr_entropy_generation_w_k']>=CRITERIA['entropy_min_w_k']
        and grid_energy_admitted(r))


def run(output,reference=None):
    output=Path(output).absolute()
    if any(p.is_symlink() for p in (output,*output.parents)):raise ValueError('Symlink output')
    output.mkdir(parents=True,exist_ok=False)
    source=physical_signature();driver=sha(__file__);serializer=sha(evidence.__file__)
    s=PWRPowerSession(max_step_s=1.);initial=s.snapshot()
    evidence.save(output/'protocol.json',dict(source=source,driver_sha256=driver,serializer_sha256=serializer,
        criteria=CRITERIA,settings=SETTINGS,parameters=s.parameter_registry(),initial=initial,nominal_duration_s=3600.,candidate_tasks=TASKS,
        scope='Hot research system,5s loss of external station supply and1s small-reserve exhaustion; no long accident admission',
        supply='Main auxiliaries grid-only; ideal finite dedicated backup for DHR only, no automatic recharge',
        trip='External availability loss immediately latches SCRAM/turbine trip and disables main drives; no sensing/switching delay model',
        heat='Capped heat exchanger from primary cold leg to existing finite cold cooling inventory; pump electric input becomes heat',
        check_mode='Declared1Pa opening band avoids zero-drive check-valve chatter; no manufacturer valve cracking map'))
    print('finite station power protocol sealed',flush=True)
    rows=[]
    for i in range(60):
        rows.append(s.advance(60.))
        if i%5==0:print('finite station power nominal',i+1,'/60',flush=True)
    tail=float(np.max(abs(vector(rows[-1])-vector(rows[-3]))/np.maximum(abs(vector(rows[-3])),1.)))
    evidence.save(output/'nominal.json',dict(trace=rows,tail_relative_drift=tail,checkpoint=s.checkpoint()))
    cases=[];observations=[initial,*rows]
    for step,rtol in SETTINGS:
        t=PWRPowerSession(backup=BackupConfig(initial_energy_j=1e5),max_step_s=step,rtol=rtol)
        active=t.advance(.5,{'grid_available':False});saved=json.loads(json.dumps(t.checkpoint()));restored=PWRPowerSession.restore(saved)
        exhausted=t.advance(.75);exact=restored.advance(.75)==exhausted
        returned=t.advance(.5,{'grid_available':True})
        event=[e for e in exhausted['events'] if e['kind']=='backup_exhausted']
        analytic=bool(len(event)==1 and abs(event[0]['time_s']-1.)<=CRITERIA['exhaustion_time_error_s']
            and abs(exhausted['backup_used_j']-1e5)<=CRITERIA['backup_use_error_j'] and exhausted['electrical']['backup_energy_j']==0.)
        drives=bool(not any(active['electrical']['effective_drives'].values()) and active['powers']['net_export_w']==0.
            and active['electrical']['dhr_active'] and not exhausted['electrical']['dhr_active']
            and returned['electrical']['main_bus_available'] and returned['actuator_status']['scram_latched']
            and returned['actuator_status']['turbine_trip_latched'] and returned['electrical']['backup_energy_j']==0.
            and returned['electrical']['dhr_active'] and not returned['electrical']['dhr_from_backup'])
        cases.append(dict(max_step_s=step,rtol=rtol,active=active,exhausted=exhausted,returned=returned,
            checkpoint=saved,restart_exact=exact,analytic_exhaustion_passed=analytic,drive_priority_passed=drives))
        observations.extend([active,exhausted,returned]);print('finite power switching refinement',step,flush=True)
    fine=vector(cases[-1]['returned']);errors=[float(np.max(abs(vector(c['returned'])-fine)/np.maximum(abs(fine),1.))) for c in cases]
    evidence.save(output/'switches.json',dict(cases=cases,refinement_errors=errors))
    long=PWRPowerSession(max_step_s=.2);loss=long.advance(5.,{'grid_available':False});observations.append(loss)
    a=PWRPowerSession(max_step_s=.2);b=PWRPowerSession(max_step_s=.2)
    with_backup=a.advance(.5,{'grid_available':False});without=b.advance(.5,{'grid_available':False,'backup_enabled':False})
    observations.extend([with_backup,without])
    causal=bool(with_backup['electrical']['dhr_donor_temperature_k']<without['electrical']['dhr_donor_temperature_k']
        and with_backup['electrical']['dhr_receiving_temperature_k']>without['electrical']['dhr_receiving_temperature_k']
        and without['backup_used_j']==0. and without['powers']['dhr_heat_w']==0.)
    limited=PWRPowerSession(TrainConfig(inertia_kg_m2=1e5));before=limited.checkpoint();rollback=False
    try:limited.advance(1.,{'grid_available':False})
    except (ValueError,DomainError,SolverError):rollback=limited.checkpoint()==before and not limited.emergency_latched
    evidence.save(output/'blackout.json',dict(five_seconds=loss,with_backup=with_backup,without_backup=without,causal=causal,trip_and_energy_rollback=rollback))
    all_admitted=all(admit(r) for r in observations)
    checks={'NPP-08.03':bool(all_admitted and rollback and all(c['drive_priority_passed'] and c['restart_exact'] for c in cases)),
        'NPP-08.04':bool(all_admitted and causal and all(c['analytic_exhaustion_passed'] for c in cases)
            and abs(loss['backup_used_j']-5e5)<=CRITERIA['backup_use_error_j'] and loss['pwr_state'][12]==1.)}
    unchanged=source==physical_signature() and driver==sha(__file__) and serializer==sha(evidence.__file__)
    passed=bool(unchanged and all(checks.values()) and max(errors)<=CRITERIA['refinement_relative'] and tail<=CRITERIA['tail_drift_relative'])
    report=dict(completed=True,passed=passed,source_unchanged=unchanged,task_checks=checks,closed_tasks=TASKS if passed else [],
        nominal_duration_s=rows[-1]['time_s'],nominal_tail_relative_drift=tail,refinement_errors=errors,blackout_duration_s=5.,
        max_energy_relative=max(r['energy_relative'] for r in observations),max_mass_relative=max(r['mass_relative'] for r in observations),
        max_boron_relative=max(r['boron_relative'] for r in observations),
        max_grid_energy_absolute_j=max(abs(r['net_exported_j']-r['net_generated_j']-r['backup_used_j']) for r in observations),
        max_grid_energy_relative=max(abs(r['net_exported_j']-r['net_generated_j']-r['backup_used_j'])/max(abs(r['net_exported_j']),abs(r['net_generated_j']),abs(r['backup_used_j']),1.) for r in observations),
        restart_exact=all(c['restart_exact'] for c in cases),
        analytic_exhaustion=all(c['analytic_exhaustion_passed'] for c in cases),causal_dhr_heat_transfer=causal,rollback=rollback,
        complete_package_b=False,complete_package_c=False,r1_accepted=False,validated=False,plant_referenced=False,full_nominal_checkpoint_replay=False,
        limitations=['Hypothetical research backup and DHR sizing; not licensed or measured station safety equipment',
            'Only5s coupled loss-of-power and1s small-reserve exhaustion qualified, no long blackout safety claim',
            'Main drives shed immediately; finite backup serves only lumped DHR, no diesel startup/AC voltage/frequency model',
            'DHR has capped lumped UA and ideal availability, no independent DHR pipe-flow transient or equipment failure map',
            'External cooling-water facility remains a prescribed boundary, not a regional power-failure model',
            'No automatic recharge, trip reset, islanding or grid resynchronization; no full sensor/protection admission',
            'Existing phase, density, NPSH, steam-path and inactive spray restrictions remain'])
    evidence.save(output/'report.json',report)
    if reference is not None:
        reference=Path(reference);files={p.relative_to(output) for p in output.rglob('*.json')}
        if files!={p.relative_to(reference) for p in reference.rglob('*.json')}:raise ValueError('Evidence file sets differ')
        for path in sorted(files):
            if evidence.read(output/path)!=evidence.read(reference/path):raise ValueError('Installed power evidence differs:'+str(path))
        print('All installed finite power evidence matches exactly',flush=True)
    if not passed:raise ValueError('Finite station power qualification failed; no tasks accepted')
    print(json.dumps(dict(passed=passed,closed_tasks=report['closed_tasks'])),flush=True)
    return report


if __name__=='__main__':
    p=ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--reference',type=Path)
    args=p.parse_args();run(args.output,args.reference)
