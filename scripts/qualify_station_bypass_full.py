"""Full-duration finite-header bypass qualification, research hot-station scope.

Explicit 3600 s settling and 3600 s hold, without reset or projection;
then all component experiments and an independent fresh-station 60 s trip.
Installed --reference replay must compare every sealed evidence file exactly.
"""
from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np
from atomic_model_station import PWRBypassSession,BypassConfig
from atomic_model_station.bypass import bypass_signature
from atomic_model_multiphysics.provenance import sha
from scripts import qualify_station_actuators as evidence
from scripts import qualify_station_bypass as component
from scripts import qualify_station_stages as stages
from scripts import qualify_station_steam as steam
from scripts import qualify_station_primary as primary

TASKS=['NPP-05.04']
CRITERIA=dict(energy_relative=1e-6,mass_relative=1e-10,boron_relative=1e-10,
    instantaneous_relative=1e-8,routing_energy_w=.01,mass_stream_kg_s=1e-8,
    entropy_roundoff_w_k=.01,nominal_tail_relative=1e-5,decay_reference_relative=1e-6)


def helpers():
    return {name:sha(module.__file__) for name,module in
            [('serializer',evidence),('component',component),('stages',stages),('steam',steam),('physical_vector',primary)]}


def physical_vector(row):
    # Cumulative external-energy ledgers grow at steady power; they are audited
    # separately, rather than mistaken for stationary physical inventories.
    return np.r_[steam.vector(row),row['bypass_position']]


def admit(row):
    bypass=row['header_bypass'];condenser=row['condenser_balance']
    return bool(stages.admit(row) and steam.admit(row)
        and row['energy_relative']<=CRITERIA['energy_relative']
        and row['mass_relative']<=CRITERIA['mass_relative']
        and row['boron_relative']<=CRITERIA['boron_relative']
        and abs(row['instantaneous_energy_defect_w'])/max(abs(row['powers']['nuclear_heat_w']),1.)<=CRITERIA['instantaneous_relative']
        and abs(condenser['energy_residual_w'])<=CRITERIA['routing_energy_w']
        and abs(condenser['mass_residual_kg_s'])<=CRITERIA['mass_stream_kg_s']
        and abs(row['bypass_port_audit']['energy_residual_w'])<=CRITERIA['routing_energy_w']
        and bypass['entropy_generation_w_k']>=-CRITERIA['entropy_roundoff_w_k']
        and 0<=row['bypass_position']<=1 and bypass['flow_kg_s']>=0
        and bypass['stagnation_enthalpy_j_kg']==bypass['recovered_enthalpy_j_kg'])


def run(output,reference=None):
    output=Path(output).absolute()
    if any(p.is_symlink() for p in (output,*output.parents)):raise ValueError('Symlink output refused')
    output.mkdir(parents=True,exist_ok=False)
    source=bypass_signature();driver=sha(__file__);helper_sources=helpers()
    s=PWRBypassSession(max_step_s=1.,rtol=1e-7);initial=s.snapshot()
    evidence.save(output/'protocol.json',dict(source=source,driver_sha256=driver,
        helpers_sha256=helper_sources,criteria=CRITERIA,initial=initial,
        parameters=s.parameter_registry(),candidate_tasks=TASKS,
        warmup_duration_s=3600.,nominal_duration_s=3600.,total_operating_duration_s=7200.,
        trip_duration_s=60.,nominal_step_s=1.,nominal_rtol=1e-7,
        physical_tail='All dynamic physical inventories and actuator coordinates; cumulative ledgers audited by conservation',
        stock_reset=False,state_projection=False,
        scope='Hot research station, near-dry header, equilibrium no-slip bypass, finite condenser; no hardware valve validation'))
    rows=[]
    for phase in ('warmup','nominal'):
        trace=[]
        for i in range(60):
            trace.append(s.advance(60.))
            if i%5==0:print('bypass full',phase,i+1,'/60',flush=True)
        tail=float(max(abs(physical_vector(trace[-1])-physical_vector(trace[-3]))/
                       np.maximum(abs(physical_vector(trace[-3])),1.)))
        evidence.save(output/(phase+'.json'),dict(trace=trace,tail_relative_drift=tail,checkpoint=s.checkpoint()))
        rows.extend(trace)
        if phase=='nominal':nominal=trace;nominal_tail=tail
    decay=stages.late_decay_reference(nominal)
    evidence.save(output/'decay_reference.json',decay)
    trip_session=PWRBypassSession(max_step_s=.5);trip=[]
    for i in range(6):
        trip.append(trip_session.advance(10.,dict(scram=True,turbine_trip=True) if i==0 else None))
        print('bypass full trip',i+1,'/6',flush=True)
    evidence.save(output/'trip_60s.json',dict(trace=trip,checkpoint=trip_session.checkpoint()))
    rows.extend(trip)
    fault=PWRBypassSession(bypass=BypassConfig(area_m2=.03,stroke_time_s=5.,trip_opening=1.),max_step_s=.5)
    before=fault.checkpoint();rejected=False;message=None
    try:fault.advance(10.,dict(scram=True,turbine_trip=True))
    except ValueError as error:
        message=str(error)
        rejected=('Near-dry equilibrium nozzle stagnation state' in message and fault.checkpoint()==before)
    evidence.save(output/'phase_fault.json',dict(config=dict(area_m2=.03,stroke_time_s=5.,trip_opening=1.),
        rejected_atomically=rejected,reason=message,checkpoint_unchanged=fault.checkpoint()==before))
    component_report=component.run(output/'components')
    unchanged=source==bypass_signature() and driver==sha(__file__) and helper_sources==helpers()
    conservation=all(admit(row) for row in [initial,*rows])
    final=trip[-1]['header_bypass']
    trip_ok=(any(e['kind']=='bypass_supply_close' for e in trip[-1]['events'])
        and final['flow_kg_s']==0 and not final['supply_permitted'])
    passed=bool(unchanged and conservation and trip_ok and rejected
        and component_report['passed'] and component_report['restart_exact']
        and nominal_tail<=CRITERIA['nominal_tail_relative']
        and decay['relative_error']<=CRITERIA['decay_reference_relative'])
    report=dict(completed=True,passed=passed,source_unchanged=unchanged,
        closed_tasks=TASKS if passed else [],task_checks={'NPP-05.04':passed},
        warmup_duration_s=3600.,nominal_duration_s=3600.,total_operating_duration_s=7200.,trip_duration_s=60.,
        nominal_tail_relative_drift=nominal_tail,decay_reference_relative_error=decay['relative_error'],
        restart_exact=component_report['restart_exact'],phase_fault_rejected_atomically=rejected,
        component_report_sha256=sha(output/'components'/'report.json'),
        max_energy_relative=max(row['energy_relative'] for row in [initial,*rows]),
        max_mass_relative=max(row['mass_relative'] for row in [initial,*rows]),
        max_boron_relative=max(row['boron_relative'] for row in [initial,*rows]),
        full_duration_qualified=passed,complete_package_b=False,complete_package_c=False,
        r1_accepted=False,validated=False,plant_referenced=False,full_long_checkpoint_replay=False,
        limitations=['NPP-05.04 only in the hot near-dry research scope; full B still requires NPP-04.04',
            'Equilibrium no-slip critical nozzle with illustrative geometry, discharge and pressure thresholds',
            'No manufacturer valve map, pipe waves, non-equilibrium critical flow or experimental validation',
            'Physical 7200s trace repeated, but full 7200s checkpoint restoration is not a separate accepted experiment'])
    evidence.save(output/'report.json',report)
    if reference is not None:
        reference=Path(reference)
        files={p.relative_to(output) for p in output.rglob('*.json')}
        if files!={p.relative_to(reference) for p in reference.rglob('*.json')}:raise ValueError('Evidence sets differ')
        for name in sorted(files):
            if evidence.read(output/name)!=evidence.read(reference/name):raise ValueError('Full bypass evidence differs:'+str(name))
        print('All installed full-duration bypass evidence matches exactly',flush=True)
    if not passed:raise ValueError('Full-duration bypass qualification failed; no task accepted')
    print(json.dumps(dict(passed=True,closed_tasks=TASKS)),flush=True)
    return report


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference',type=Path)
    args=parser.parse_args();run(args.output,args.reference)
