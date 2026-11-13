"""Local hot-station rod/trip/bypass qualification, including baseline 3600s.

Optional --reference compares every sealed result with a source-checkout run.
Run the installed command outside checkout to check the shipped wheel offline.
"""
from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np

from atomic_model_station import PWRTrainSession,TrainConfig
from atomic_model_station.pwr_train import physical_signature
from atomic_model_multiphysics.provenance import seal,unseal,sha
from atomic_model_training.spec import atomic_json
from scripts import qualify_physical_pwr_train as baseline

TASKS=['NPP-03.03','NPP-06.04','NPP-07.01']
CRITERIA=dict(refinement_relative=1e-4,energy_relative=1e-6,mass_relative=1e-10,boron_relative=1e-10,
              condenser_mass_rate_kg_s=1e-9,condenser_energy_rate_w=1e-5,trip_duration_s=10.)
SETTINGS=[(.2,1e-7),(.1,2e-8),(.05,2e-9)]


def native(value):
    """Preserve JSON values while converting NumPy scalar results explicitly."""
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {key:native(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)):return [native(item) for item in value]
    return value


def save(path,payload):return atomic_json(path,seal(native(payload)))


def read(path):return unseal(json.loads(Path(path).read_text()))


def run(output,reference=None):
    output=Path(output).absolute()
    if any(p.is_symlink() for p in (output,*output.parents)):raise ValueError('Symlink output')
    output.mkdir(parents=True,exist_ok=False)
    source=physical_signature();driver=sha(__file__)
    save(output/'protocol.json',dict(source=source,driver_sha256=driver,criteria=CRITERIA,
        settings=SETTINGS,candidate_tasks=TASKS,scope='Hot aggregate rod motion, explicit latched trips and passive pressure bypass; no autonomous plant protection admission',
        boron='Unavailable: no finite supply/letdown subsystem; inventory setters refused',
        drive='Exact bounded operating0.01/s and SCRAM0.2/s; arrival is analytical, no ODE state projection',
        bypass='Existing pressure target at7MPa, inertial lag0.2s, finite SG/hotwell inventories; small signed numerical band1e-9kg/s retained and reported',
        reset='No online trip reset; clearing the request cannot clear a latch'))
    base=baseline.run(output/'baseline')
    trials=[];rows=[]
    for step,rtol in SETTINGS:
        s=PWRTrainSession(max_step_s=step,rtol=rtol);initial=s.snapshot()
        early=s.advance(4.,{'rod_target':0.,'scram':True,'turbine_trip':True,'generator_connected':True,'valve_request':1.})
        saved=json.loads(json.dumps(s.checkpoint()));restored=PWRTrainSession.restore(saved)
        late=s.advance(6.);restart_exact=restored.advance(6.)==late
        conflict={'scram':False,'turbine_trip':False,'rod_target':0.,'generator_connected':True,'valve_request':1.}
        held=s.advance(.2,conflict)
        arrivals=[e for e in late['events'] if e['kind']=='rod_arrived']
        case=dict(max_step_s=step,rtol=rtol,initial=initial,early=early,late=late,held=held,
                  checkpoint=saved,restart_exact=restart_exact,
                  exact_stroke=(early['pwr_state'][12]==late['pwr_state'][12]==held['pwr_state'][12]==1. and len(arrivals)==1),
                  latch_priority=(held['actuator_status']['scram_latched'] and held['actuator_status']['turbine_trip_latched']
                    and held['effective_valve']==0. and held['powers']['gross_electrical_w']==0.),
                  coastdown=(late['turbine_speed_rad_s']<early['turbine_speed_rad_s'] and late['powers']['steam_work_w']<late['powers']['turbine_bearing_w']))
        trials.append(case);rows.extend([initial,early,late,held]);print('actuator trip refinement',step,flush=True)
    fine=baseline.vector(trials[-1]['late'])
    errors=[float(np.max(abs(baseline.vector(t['late'])-fine)/np.maximum(abs(fine),1.))) for t in trials]
    motion=PWRTrainSession(max_step_s=.2);pos=float(motion.state[12]);m0=motion.snapshot()
    moving=motion.advance(.5,{'rod_target':pos+.02});arrived=motion.advance(2.)
    requested_separate=moving['pwr_state'][12]==pos+.005 and moving['actuator_status']['rod_requested']==pos+.02 and arrived['pwr_state'][12]==pos+.02
    before=motion.checkpoint();unsupported=False
    try:motion.advance(.1,{'boron_ppm':900.})
    except ValueError:unsupported=motion.checkpoint()==before
    rejected=PWRTrainSession(TrainConfig(inertia_kg_m2=1e5));before=rejected.checkpoint();rollback=False
    try:rejected.advance(1.,{'scram':True,'generator_connected':False})
    except ValueError:rollback=rejected.checkpoint()==before and not rejected.scram_latched and not rejected.events
    rows.extend([m0,moving,arrived]);all_admitted=all(baseline.admit(r) for r in rows)
    condenser=all(abs(r['condenser_balance']['mass_residual_kg_s'])<=CRITERIA['condenser_mass_rate_kg_s']
        and abs(r['condenser_balance']['energy_residual_w'])<=CRITERIA['condenser_energy_rate_w']
        and r['minimum_sampled_dump_flow_kg_s']>=-r['dump_signed_roundoff_band_kg_s'] for r in rows)
    n=13+3*motion.plant.network.volume_names().index('sg_secondary')
    secondary_closed=all(abs((t['late']['pwr_state'][n]+t['late']['hotwell_mass_kg'])-(t['initial']['pwr_state'][n]+t['initial']['hotwell_mass_kg']))
        <=CRITERIA['mass_relative']*(t['initial']['pwr_state'][n]+t['initial']['hotwell_mass_kg']) for t in trials)
    checks={
        'NPP-03.03':bool(requested_separate and unsupported and rollback and all(t['exact_stroke'] and t['latch_priority'] and t['restart_exact'] for t in trials)),
        'NPP-06.04':bool(base['passed'] and rollback and all(t['coastdown'] for t in trials)),
        'NPP-07.01':bool(condenser and secondary_closed and all(t['late']['dump_flow_kg_s']>t['early']['dump_flow_kg_s']>0 for t in trials)),
    }
    save(output/'trips.json',dict(refinements=trials,refinement_errors=errors,condenser_balance_passed=condenser,secondary_mass_closed=secondary_closed))
    save(output/'motion.json',dict(initial=m0,moving=moving,arrived=arrived,request_separate=requested_separate,
        boron_refused_atomically=unsupported,pending_trip_rollback=rollback))
    passed=bool(base['passed'] and all_admitted and max(errors)<=CRITERIA['refinement_relative'] and all(checks.values())
        and source==physical_signature() and driver==sha(__file__))
    report=dict(completed=True,passed=passed,source_unchanged=source==physical_signature(),task_checks=checks,
        closed_tasks=TASKS if passed else [],baseline_passed=base['passed'],nominal_duration_s=base['nominal_duration_s'],
        refinement_errors=errors,max_energy_relative=max(r['energy_relative'] for r in rows),
        max_mass_relative=max(r['mass_relative'] for r in rows),max_boron_relative=max(r['boron_relative'] for r in rows),
        minimum_sampled_dump_flow_kg_s=min(r['minimum_sampled_dump_flow_kg_s'] for r in rows),
        trip_duration_s=CRITERIA['trip_duration_s'],trip_restart_exact=all(t['restart_exact'] for t in trials),
        full_nominal_checkpoint_replay=False,complete_package_b=False,r1_accepted=False,validated=False,plant_referenced=False,
        limitations=['Hot-domain acceptance only, aggregate rod; no bank-by-bank motion',
          'Explicit trips only; no new autonomous voting/delay protection qualification',
          'No trip reset or grid synchronization; generator reconnection after turbine trip is inhibited',
          'Pressure bypass retained as reduced target-flow law; no qualified nozzle/choking map',
          'No separate steam-line inventory, MSR or hardware moisture admission',
          'No finite chemical supply/letdown subsystem',
          'Inactive spray activation still refused; pump coastdown/natural circulation not newly qualified',
          'Signed bypass roundoff band1e-9kg/s is numerical, not reverse-flow physics; no clipping'])
    save(output/'report.json',report)
    if reference is not None:
        reference=Path(reference)
        files=[p.relative_to(output) for p in output.rglob('*.json')]
        if set(files)!={p.relative_to(reference) for p in reference.rglob('*.json')}:raise ValueError('Evidence file sets differ')
        for relative in files:
            if read(output/relative)!=read(reference/relative):raise ValueError('Installed evidence differs: '+str(relative))
        print('All installed actuator and baseline evidence matches exactly',flush=True)
    if not passed:raise ValueError('Station actuator qualification failed; no tasks accepted')
    print(json.dumps(dict(passed=passed,closed_tasks=report['closed_tasks'])),flush=True)
    return report


if __name__=='__main__':
    p=ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--reference',type=Path)
    args=p.parse_args();run(args.output,args.reference)
