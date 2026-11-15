"""Local, explicit fault/reaction qualification; no experimental validation."""
from argparse import ArgumentParser
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
from atomic_model_multiphysics.provenance import sha
from atomic_model_station_faults import Fault, FaultSession
from atomic_model_station_faults.model import signature
from scripts import qualify_station_actuators as evidence

TASKS=['NPP-10.01','NPP-10.02','NPP-10.03','NPP-10.04']
CRITERIA=dict(energy_relative=1e-6,mass_relative=1e-10,boron_relative=1e-10,
    pressure_pair_threshold_pa=100000.,minimum_bias_reaction_power_reduction=.1,
    maximum_detection_delay_s=11.,physical_refinement_relative=1e-4)


def faults_signature():return signature()


def scenario(faults=(), *, duration=8, react=False, step=1.,rtol=1e-7,auto=True):
    owner=FaultSession(faults,react_on_alarm=react,max_step_s=step,rtol=rtol)
    trace=[]
    for index in range(duration):
        kwargs=dict(modes={k:'auto' for k in owner.control.loops}) if index==0 and auto else {}
        result=owner.step(**kwargs)
        trace.append(dict(result=result,plant=owner.control.plant.snapshot()))
    return dict(trace=trace,final=owner.checkpoint(),physical_state=owner.control.plant.state.tolist())


def detection_metrics(case,fault):
    trace=case['trace'];onset=float(fault.start_tick) if fault else None
    times=[r['result']['detection']['time_s'] for r in trace if r['result']['detection']['alarms']]
    before=[t for t in times if onset is None or t<onset]
    after=[t for t in times if onset is not None and t>=onset]
    return dict(false_alarm_frames=len(before),detected=bool(after),missed=fault is not None and not after,
        onset_s=onset,first_alarm_s=min(after) if after else None,
        detection_delay_s=min(after)-onset if after else None,
        uniquely_diagnosed=False)


def run(output,reference=None):
    output=Path(output).absolute()
    if any(p.is_symlink() for p in (output,*output.parents)):raise ValueError('Symlink output refused')
    output.mkdir(parents=True,exist_ok=False)
    source=signature();driver=sha(__file__)
    definitions=[Fault('sensor_bias','bias','pressure',2,value=150000.),
        Fault('sensor_drift','drift','pressure',2,value=60000.),Fault('sensor_stuck','stuck','pressure',2),
        Fault('sensor_dropout','dropout','pressure',2),Fault('sensor_failure','failure','pressure',2),
        Fault('below_threshold','bias','pressure',2,value=1000.),
        Fault('drive_loss','drive_limit','feed',2,value=.72),Fault('pump_loss','pump_off','cooling',2),
        Fault('water_loss','cooling_loss','external_water',2,value=.5),Fault('valve_jam','bypass_stuck','bypass',2)]
    evidence.save(output/'protocol.json',dict(source=source,driver_sha256=driver,criteria=CRITERIA,
        candidate_tasks=TASKS,definitions=[asdict(f) for f in definitions],tick_s=1.,
        reference='Redundant pressure instrument, same declared lag; no common-mode-failure qualification',
        integration_settings=[dict(step_s=1.,rtol=1e-7),dict(step_s=.5,rtol=2e-8)],
        scope='Short hot research fault family, not full R1 or vendor protection/diagnostic qualification'))
    healthy=scenario(duration=16);evidence.save(output/'healthy.json',healthy)
    cases={};metrics={};all_rows=list(healthy['trace'])
    for fault in definitions:
        case=scenario([fault],duration=16 if fault.kind=='dropout' else 8)
        cases[fault.name]=case;metrics[fault.name]=detection_metrics(case,fault)
        all_rows.extend(case['trace']);evidence.save(output/(fault.name+'.json'),dict(fault=asdict(fault),case=case,metrics=metrics[fault.name]))
        print('fault case',fault.name,json.dumps(metrics[fault.name]),flush=True)
    # Alarm response uses only the preceding measured frame; compare consequences.
    fault=definitions[0];reaction=scenario([fault],react=True)
    all_rows.extend(reaction['trace']);held=cases[fault.name]
    held_power=held['trace'][-1]['plant']['core']['fission_total_w']
    stopped_power=reaction['trace'][-1]['plant']['core']['fission_total_w']
    reduction=1.-stopped_power/held_power
    requests=[r['result']['reaction'] for r in reaction['trace'] if r['result']['reaction'] is not None]
    causal=all(r['observed_at_s']<row['result']['receipt']['completed_at_s'] for row in reaction['trace'] if (r:=row['result']['reaction']) is not None)
    evidence.save(output/'reaction.json',dict(case=reaction,hold_final_power_w=held_power,reaction_final_power_w=stopped_power,
        power_reduction_fraction=reduction,causal=causal,reaction_requests=len(requests),
        consequence='Lower power after chosen stop; no global utility or universally better reaction claimed'))
    # Sorted file restore across an onset and pending queued samples is tested by full replay.
    owner=FaultSession([Fault('pending','failure','pressure',3),Fault('hardware','drive_limit','feed',3,value=.9)])
    rows=[]
    for _ in range(2):rows.append(owner.step())
    saved=json.loads(json.dumps(owner.checkpoint(),sort_keys=True));restored=FaultSession.restore(saved)
    restart=all(owner.step()==restored.step() for _ in range(3)) and owner.checkpoint()==restored.checkpoint()
    evidence.save(output/'restart.json',dict(saved=saved,final=owner.checkpoint(),exact=bool(restart)))
    before=owner.checkpoint();rollback=False
    try:owner.step(dict(feed_flow_fraction=True))
    except ValueError:rollback=owner.checkpoint()==before
    evidence.save(output/'rollback.json',dict(exact=rollback))
    # Same physical fault schedule and controller; only integration tolerances change.
    coarse=cases['drive_loss'];fine=scenario([definitions[6]],step=.5,rtol=2e-8)
    all_rows.extend(fine['trace'])
    x=np.array(coarse['physical_state']);y=np.array(fine['physical_state'])
    refined=float(max(abs(x-y)/np.maximum(abs(y),1.)))
    evidence.save(output/'refinement.json',dict(fine=fine,coarse_case='drive_loss.json',relative_error=refined,automatic=True))
    # Undisturbed subclass is required to be numerically identical to the base station.
    from atomic_model_station import PWRBypassSession
    from atomic_model_station.faults import PWRFaultSession
    a=PWRBypassSession(max_step_s=1.);b=PWRFaultSession(max_step_s=1.)
    for _ in range(2):a.advance(1.);b.advance(1.)
    legacy=bool(np.array_equal(a.state,b.state))
    equipment_checks={
        'drive_limit':all(r['plant']['feed_train']['feed_controller']['delivered_head_pa']<=12e6*.72**2 for r in cases['drive_loss']['trace'][2:]),
        'pump_off':all(r['plant']['cooling']['pump_head_pa']==0 for r in cases['pump_loss']['trace'][2:]),
        'cooling_loss':all(r['plant']['equipment_fault_truth']['effective_controls']['external_water_fraction']==.5 for r in cases['water_loss']['trace'][2:]),
        'stuck':len({r['plant']['bypass_position'] for r in cases['valve_jam']['trace'][2:]})==1}
    hidden=all(not any(key in json.dumps(r['result']) for key in ('equipment_fault_truth','start_tick','end_tick','fault_memory')) for r in all_rows)
    balances=all(r['plant']['energy_relative']<=CRITERIA['energy_relative'] and r['plant']['mass_relative']<=CRITERIA['mass_relative'] and r['plant']['boron_relative']<=CRITERIA['boron_relative'] for r in all_rows)
    required=[f.name for f in definitions[:5]]
    detection_ok=all(metrics[n]['detected'] and metrics[n]['detection_delay_s']<=CRITERIA['maximum_detection_delay_s'] for n in required)
    # Explicitly require visible limitations, not an oracle that identifies every planted fault.
    miss_ok=all(metrics[n]['missed'] for n in ('below_threshold','drive_loss','pump_loss','water_loss','valve_jam'))
    false_alarms=detection_metrics(healthy,None)['false_alarm_frames']+sum(m['false_alarm_frames'] for m in metrics.values())
    unchanged=source==signature() and driver==sha(__file__)
    passed=bool(unchanged and balances and hidden and restart and rollback and legacy and all(equipment_checks.values())
        and detection_ok and miss_ok and false_alarms==0 and causal and len(requests)==1
        and reduction>=CRITERIA['minimum_bias_reaction_power_reduction'] and refined<=CRITERIA['physical_refinement_relative'])
    report=dict(completed=True,passed=passed,source_unchanged=unchanged,closed_tasks=TASKS if passed else [],
        nominal_duration_s=16.,trip_duration_s=8.,restart_exact=bool(restart),rollback_exact=rollback,
        metrics=metrics,false_alarm_frames=false_alarms,equipment_checks=equipment_checks,
        public_fault_truth_hidden=hidden,legacy_subclass_states_exact=legacy,refinement_relative_error=refined,
        measured_reaction_causal=causal,reaction_power_reduction_fraction=reduction,
        maximum_energy_relative=max(r['plant']['energy_relative'] for r in all_rows),
        maximum_mass_relative=max(r['plant']['mass_relative'] for r in all_rows),
        maximum_boron_relative=max(r['plant']['boron_relative'] for r in all_rows),
        complete_npp_10=passed,complete_package_c=False,r1_accepted=False,validated=False,plant_referenced=False,
        limitations=['Single-clock boundaries, no mid-tick stochastic fault events',
            'Single fault per target at a time; no common-mode redundant sensor validation',
            'Pressure-pair monitor detects selected measurement faults, does not uniquely isolate cause',
            'Subthreshold bias and selected hardware faults are missed; no universal diagnostic coverage',
            'Eight/sixteen second hot cases, no full R1 fault horizon acceptance'])
    evidence.save(output/'report.json',report)
    if reference is not None:
        reference=Path(reference);files={p.name for p in output.glob('*.json')}
        if files!={p.name for p in reference.glob('*.json')}:raise ValueError('Fault evidence inventories differ')
        for name in sorted(files):
            if evidence.read(output/name)!=evidence.read(reference/name):raise ValueError('Installed fault evidence differs:'+name)
        print('All installed fault evidence matches exactly',flush=True)
    if not passed:raise ValueError('Station fault qualification failed')
    print(json.dumps(dict(passed=True,closed_tasks=TASKS)),flush=True)
    return report


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference',type=Path);args=parser.parse_args();run(args.output,args.reference)
