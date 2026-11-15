"""Six-loop research regulation with finite hydraulic feeding and geometric levels."""
from argparse import ArgumentParser
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
from atomic_model_station import PWRBypassSession
from atomic_model_station_control import feed_control_session,ControlSession,ProtectionConfig,control_signature
from atomic_model_station_signals import StationSignals,Sensor
from atomic_model_multiphysics.provenance import sha
from physics.materials import iapws
from scripts import qualify_station_actuators as evidence

TASKS=['NPP-09.02']
CRITERIA=dict(energy_relative=1e-6,mass_relative=1e-10,boron_relative=1e-10,refinement_relative=1e-4,
    tail_sg_level_m=.02,tail_hotwell_level_m=.005,tail_feed_level_m=.005,
    tail_power_relative=.02,tail_speed_relative=.01,tail_pressure_relative=.02,
    geometry_reference_relative=1e-10)


def feed_control_signature():return control_signature()


def schedule(index,refs,*,long):
    if index==0:return dict(modes={k:'auto' for k in refs})
    first=20 if long else 4;back=60 if long else 16
    if index==first:return dict(targets=dict(feed_flow_fraction=refs['feed_flow_fraction']+.005,
        condensate_flow_fraction=refs['condensate_flow_fraction']+.002,feed_inventory_fraction=refs['feed_inventory_fraction']-.002))
    if index==back:return dict(targets={k:refs[k] for k in ('feed_flow_fraction','condensate_flow_fraction','feed_inventory_fraction')})
    if long and index==40:return dict(modes=dict(feed_flow_fraction='manual'))
    if long and index==42:return dict(modes=dict(feed_flow_fraction='auto'))
    if long and index==44:return dict(modes=dict(feed_inventory_fraction='manual'))
    if long and index==46:return dict(modes=dict(feed_inventory_fraction='auto'))
    return {}


def observation(owner,result):
    return dict(result=result,plant=owner.plant.snapshot(),pi={k:v.state() for k,v in owner.loops.items()})


def run(output,reference=None):
    output=Path(output).absolute()
    if any(p.is_symlink() for p in (output,*output.parents)):raise ValueError('Symlink output refused')
    output.mkdir(parents=True,exist_ok=False)
    source=control_signature();driver=sha(__file__);serializer=sha(evidence.__file__)
    s=feed_control_session(tick_s=5.);refs=deepcopy(s.targets)
    evidence.save(output/'protocol.json',dict(source=source,driver_sha256=driver,serializer_sha256=serializer,
        criteria=CRITERIA,candidate_tasks=TASKS,nominal_duration_s=600.,trip_duration_s=5.,
        initial=s.checkpoint(),targets=refs,controller_tick_s=5.,tracking_tau_s=10.,
        integration_settings=[dict(step_s=1.,rtol=1e-7),dict(step_s=.5,rtol=2e-8)],
        scope='Six coupled hot research loops; collapsed SG liquid level and declared collecting-section geometry, not manufacturer instruments',
        feed='Measured steam/feed flow feed-forward, bounded hydraulic head, actual momentum and finite source/sink tanks',
        full_r1_matrix=False,full_600s_checkpoint_restore=False))
    vessel=s.plant.plant.network.volumes['sg_secondary'];liquid,vapor=iapws.saturation_properties(vessel.pressure_pa)
    expected=vessel.volume_m3/s.plant.feed_config.sg_area_m2*((vessel.mass_kg/vessel.volume_m3)-vapor.rho_kg_m3)/(liquid.rho_kg_m3-vapor.rho_kg_m3)
    actual=s.plant.snapshot()['level_readouts']['sg_collapsed_level_m']
    geometry_error=float(abs(actual-expected)/max(abs(expected),1.))
    evidence.save(output/'geometry_reference.json',dict(volume_fraction_level_m=float(expected),quality_inventory_level_m=float(actual),relative_error=geometry_error,
        source='Independent density-volume-fraction lever rule, compared with quality/liquid-mass readout'))
    all_rows=[];nominal=[]
    for index in range(120):
        row=observation(s,s.step(**schedule(index,refs,long=True)));nominal.append(row);all_rows.append(row)
        if index%10==0:print('six-loop regulation',s.plant.time,'/600',flush=True)
    evidence.save(output/'nominal.json',dict(trace=nominal,final=s.checkpoint()))
    tail=[r['plant'] for r in nominal if r['plant']['time_s']>=480.]
    errors=dict(sg=max(abs(r['level_readouts']['sg_collapsed_level_m']-refs['feed_flow_fraction']) for r in tail),
        hotwell=max(abs(r['level_readouts']['hotwell_level_m']-refs['condensate_flow_fraction']) for r in tail),
        feed=max(abs(r['level_readouts']['feed_tank_level_m']-refs['feed_inventory_fraction']) for r in tail),
        power=max(abs(r['core']['fission_total_w']/refs['rod_target']-1) for r in tail),
        speed=max(abs(r['turbine_speed_rad_s']/refs['steam_admission']-1) for r in tail),
        pressure=max(abs(r['steam_header']['pressure_pa']/refs['steam_bypass']-1) for r in tail))
    print('six-loop tail',json.dumps(errors),flush=True)
    refinement=[]
    for step,rtol in [(1.,1e-7),(.5,2e-8)]:
        owner=feed_control_session(tick_s=5.,integration_step_s=step,rtol=rtol);targets=deepcopy(owner.targets);trace=[]
        for index in range(24):
            row=observation(owner,owner.step(**schedule(index,targets,long=False)));trace.append(row);all_rows.append(row)
        vector=np.r_[owner.plant.state,[v.integral for v in owner.loops.values()]]
        refinement.append(dict(step_s=step,rtol=rtol,trace=trace,vector=vector.tolist(),final=owner.checkpoint()))
        print('six-loop integration refinement',step,flush=True)
    a,b=[np.array(r['vector']) for r in refinement]
    error=float(max(abs(a-b)/np.maximum(abs(b),1.)))
    evidence.save(output/'refinement.json',dict(cases=refinement,relative_error=error))
    owner=feed_control_session(tick_s=1.);targets=deepcopy(owner.targets);short=[];saved=None
    for index in range(12):
        kwargs=dict(modes={k:'auto' for k in owner.loops}) if index==0 else dict(modes=dict(feed_flow_fraction='manual')) if index==4 else dict(modes=dict(feed_flow_fraction='auto')) if index==6 else {}
        row=observation(owner,owner.step(**kwargs));short.append(dict(kwargs=kwargs,row=row));all_rows.append(row)
        if index==2:saved=json.loads(json.dumps(owner.checkpoint(),sort_keys=True))
    restored=ControlSession.restore(saved);restart=True
    for record in short[3:]:restart=restored.step(**record['kwargs'])==record['row']['result'] and restart
    restart=restart and restored.checkpoint()==owner.checkpoint()
    evidence.save(output/'restart.json',dict(trace=short,checkpoint=saved,final=owner.checkpoint(),exact=bool(restart)))
    failed=feed_control_session();specs=[Sensor(**dict(asdict(x),delay_ticks=2)) if x.name=='sg_level' else x for x in failed.instruments.sensors]
    failed=ControlSession(PWRBypassSession(max_step_s=1.),instruments=StationSignals(specs),loops=[x.config for x in failed.loops.values()])
    failure=failed.step(modes={k:'auto' for k in failed.loops});all_rows.append(observation(failed,failure))
    failure_ok=(failure['receipt']['trip_latched'] and 'measurement_unavailable:sg_level' in failure['receipt']['trip_reasons']
        and not any(failed.plant.controls[k] for k in ('feed_flow_control','condensate_flow_control','feed_inventory_control')))
    evidence.save(output/'measurement_failure.json',dict(result=failure,final=failed.checkpoint(),passed=bool(failure_ok)))
    tripped=feed_control_session(protection=ProtectionConfig(overspeed_fraction=.999));trip=[]
    for i in range(5):
        row=observation(tripped,tripped.step(modes={k:'auto' for k in tripped.loops} if i==0 else None));trip.append(row);all_rows.append(row)
    trip_ok=all(r['result']['receipt']['trip_latched'] and not any(r['plant']['controls'][k] for k in ('feed_flow_control','condensate_flow_control','feed_inventory_control')) for r in trip)
    evidence.save(output/'trip.json',dict(trace=trip,final=tripped.checkpoint(),passed=bool(trip_ok)))
    limited=PWRBypassSession(max_step_s=.1);q0=limited.snapshot()['feed_train']['feed_flow_kg_s']
    limit=limited.advance(.05,dict(feed_flow_control=True,feed_flow_fraction=1.))
    f=limit['feed_train'];servo_ok=(f['feed_flow_kg_s']>q0 and f['feed_flow_kg_s']<f['feed_controller']['requested_flow_kg_s']
        and f['feed_controller']['delivered_head_pa']<=limited.plant.maximum_feed_head_pa)
    derated=PWRBypassSession(max_step_s=.1)
    capped=derated.advance(.05,dict(feed_flow_control=True,feed_flow_fraction=1.,feed_pump_fraction=.72))
    cap=capped['feed_train']['feed_controller'];available=derated.plant.maximum_feed_head_pa*.72**2
    servo_ok=servo_ok and cap['saturated'] and cap['requested_head_pa']>available and cap['delivered_head_pa']==available
    servo_ok=servo_ok and capped['feed_train']['feed_flow_kg_s']<q0
    evidence.save(output/'finite_hydraulics.json',dict(initial_flow_kg_s=q0,final=limit,derated=capped,available_derated_head_pa=available,passed=bool(servo_ok)))
    invalid=feed_control_session();before=invalid.checkpoint();rollback=False
    try:invalid.step(dict(feed_inventory_fraction=True),modes={k:'auto' for k in invalid.loops})
    except ValueError:rollback=invalid.checkpoint()==before
    evidence.save(output/'rollback.json',dict(exact=rollback))
    physical=[r['plant'] for r in all_rows]+[limit,capped]
    stock=all(r['energy_relative']<=CRITERIA['energy_relative'] and r['mass_relative']<=CRITERIA['mass_relative'] and r['boron_relative']<=CRITERIA['boron_relative'] for r in physical)
    exclusive=all(r['feed_train']['makeup_opening']*r['feed_train']['drain_opening']==0. for r in physical)
    normal=all(not r['result']['receipt']['trip_latched'] for r in nominal)
    tail_ok=(errors['sg']<=CRITERIA['tail_sg_level_m'] and errors['hotwell']<=CRITERIA['tail_hotwell_level_m']
        and errors['feed']<=CRITERIA['tail_feed_level_m'] and errors['power']<=CRITERIA['tail_power_relative']
        and errors['speed']<=CRITERIA['tail_speed_relative'] and errors['pressure']<=CRITERIA['tail_pressure_relative'])
    unchanged=source==control_signature() and driver==sha(__file__) and serializer==sha(evidence.__file__)
    passed=bool(unchanged and stock and exclusive and normal and tail_ok and restart and rollback and servo_ok and trip_ok and failure_ok and error<=CRITERIA['refinement_relative'] and geometry_error<=CRITERIA['geometry_reference_relative'])
    report=dict(completed=True,passed=passed,source_unchanged=unchanged,closed_tasks=TASKS if passed else [],
        nominal_duration_s=600.,trip_duration_s=5.,restart_exact=bool(restart),tail_errors=errors,refinement_relative_error=error,
        geometry_reference_relative_error=geometry_error,split_range_exclusive=exclusive,finite_hydraulics_qualified=bool(servo_ok),measurement_failure_safe=bool(failure_ok),
        trip_fallback_qualified=bool(trip_ok),invalid_step_rollback_exact=rollback,
        maximum_energy_relative=max(r['energy_relative'] for r in physical),maximum_mass_relative=max(r['mass_relative'] for r in physical),
        maximum_boron_relative=max(r['boron_relative'] for r in physical),
        complete_npp_09=passed,full_r1_control_matrix=False,full_600s_checkpoint_restore=False,
        complete_package_c=False,r1_accepted=False,validated=False,plant_referenced=False,
        limitations=['Declared equivalent geometry and collapsed SG level, not swollen level or real differential-pressure gauge',
            'Research gain/threshold family and selected 600s reference; no manufacturer control-system qualification',
            'Finite flow servos have declared head feedback laws, not manufacturer pump maps',
            'Full R1 load/fault/horizon matrix and long-checkpoint continuation remain open'])
    evidence.save(output/'report.json',report)
    if reference is not None:
        reference=Path(reference);files={p.relative_to(output) for p in output.rglob('*.json')}
        if files!={p.relative_to(reference) for p in reference.rglob('*.json')}:raise ValueError('Feed regulation evidence sets differ')
        for name in sorted(files):
            if evidence.read(output/name)!=evidence.read(reference/name):raise ValueError('Installed feed regulation evidence differs:'+str(name))
        print('All installed six-loop regulation evidence matches exactly',flush=True)
    if not passed:raise ValueError('Six-loop regulation qualification failed')
    print(json.dumps(dict(passed=True,closed_tasks=TASKS)),flush=True)
    return report


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference',type=Path);args=parser.parse_args();run(args.output,args.reference)
