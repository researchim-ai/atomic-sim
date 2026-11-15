"""Research supervisory authority/events and three PI foundations; NPP-09.02 stays open."""
from argparse import ArgumentParser
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

from atomic_model_station import PWRBypassSession
from atomic_model_station_control import ControlSession,ProtectionConfig,Supervision,PI,PIConfig,control_signature
from atomic_model_station_signals import StationSignals,Sensor
from atomic_model_multiphysics.provenance import sha
from scripts import qualify_station_actuators as evidence

TASKS=['NPP-09.03','NPP-09.04']
CRITERIA=dict(energy_relative=1e-6,mass_relative=1e-10,boron_relative=1e-10,pi_reference_absolute=1e-12,
    stationary_error_not_a_gate=True,duplicate_trip_actions=0)


def frame(t,*,power=3e9,pressure=6.5e6,speed=157.):
    return dict(time_s=float(t),measurements={n:dict(value=v,unit=u,sampled_at_s=float(t),available_at_s=float(t),age_s=0.,quality=['good'])
        for n,v,u in [('power',power,'W'),('pressure',pressure,'Pa'),('speed',speed,'rad/s')]})


def normal_kwargs(t,reference_power):
    if t==0:return dict(modes={k:'auto' for k in ('rod_target','steam_bypass','steam_admission')})
    if t==4:return dict(targets=dict(rod_target=.995*reference_power,steam_admission=156.8))
    if t==8:return dict(modes=dict(rod_target='manual'))
    if t==10:return dict(modes=dict(rod_target='auto'))
    if t==12:return dict(targets=dict(steam_bypass=6.32e6))
    if t==18:return dict(modes=dict(steam_admission='manual'))
    if t==20:return dict(modes=dict(steam_admission='auto'))
    return {}


def trip_kwargs(t):
    if t==0:return dict(operator=dict(rod_target=0.,steam_admission=1.,steam_inlet_isolated=True,generator_connected=True),modes=dict(steam_admission='auto'))
    return dict(operator=dict(scram=False,turbine_trip=False,rod_target=0.,steam_admission=1.,generator_connected=True,steam_inlet_isolated=False),reset=t==1)


def run(output,reference=None):
    output=Path(output).absolute()
    if any(p.is_symlink() for p in (output,*output.parents)):raise ValueError('Symlink output refused')
    output.mkdir(parents=True,exist_ok=False)
    source=control_signature();driver=sha(__file__);serializer=sha(evidence.__file__)
    evidence.save(output/'protocol.json',dict(source=source,driver_sha256=driver,serializer_sha256=serializer,
        criteria=CRITERIA,candidate_tasks=TASKS,open_tasks=['NPP-09.02'],nominal_duration_s=24.,trip_duration_s=10.,
        policy_component_duration_s=120.,tick_s=1.,
        priority=['operator','controller','interlock','protection'],
        reset='Pressure interlock releases by hysteresis; physical reactor/turbine trip requires a new station session',
        scope='Sampled hot research supervision and three PI foundations; no qualified feed/level regulation or full R1 matrix'))
    pi=PI(PIConfig('x','steam_admission',1.,1.,kp=.2,ki=.1,bias=.4,tracking_tau_s=1.))
    reference_rows=[]
    for t in range(5):
        p=pi.propose(0.,1.,.4+.1*max(t-1,0),1.,entering=t==0)
        expected=.4+.1*t
        reference_rows.append(dict(tick=t,command=p['command'],expected=expected,error=abs(p['command']-expected)))
        pi.commit(p,p['command'],1.,automatic=True)
    saturation=PI(PIConfig('x','steam_admission',1.,1.,kp=1.,ki=.1,tracking_tau_s=1.))
    for _ in range(100):
        p=saturation.propose(0.,1.,0.,1.);saturation.commit(p,0.,1.,automatic=True)
    evidence.save(output/'pi_reference.json',dict(constant_error=reference_rows,overridden_integral=saturation.integral,
        scope='Independent constant-error discrete PI recurrence; ideal reference actuator, not station dynamics'))
    policy=Supervision();normal=dict(grid_available=True,scram=False,turbine_trip=False);policy_rows=[]
    for t in range(120):
        pressure=[6.5e6,6.8e6,6.7e6,6.6e6][t] if t<4 else 6.5e6
        f=frame(t,pressure=pressure);request=normal
        if t==20:f=frame(t,power=4e9,pressure=7.6e6,speed=170.);request=dict(grid_available=False,scram=True,turbine_trip=True)
        policy.advance(f,request,tick=t,nominal_speed=157.)
        if t==21:policy.request_reset()
        policy_rows.append(dict(tick=t,pressure_active=policy.pressure_active,trip_latched=policy.trip_latched,reasons=deepcopy(policy.trip_reasons)))
    before=deepcopy(policy.__dict__);duplicate_rejected=False
    try:policy.advance(frame(119),normal,tick=119,nominal_speed=157.)
    except ValueError:duplicate_rejected=policy.__dict__==before
    policy_ok=(policy_rows[1]['pressure_active'] and policy_rows[2]['pressure_active'] and not policy_rows[3]['pressure_active']
        and not any(r['trip_latched'] for r in policy_rows[:20]) and all(r['trip_latched'] for r in policy_rows[20:])
        and len(policy_rows[20]['reasons'])==6 and sum(e['kind']=='protection_latched' for e in policy.events)==1
        and duplicate_rejected and not Supervision().trip_latched)
    evidence.save(output/'policy_120s.json',dict(synthetic_input=True,physical_station_advanced=False,trace=policy_rows,
        events=policy.events,duplicate_rejected_atomically=duplicate_rejected,passed=bool(policy_ok)))
    plant=PWRBypassSession(max_step_s=.5);initial=plant.snapshot();p0=initial['core']['fission_total_w']
    bank=StationSignals([Sensor('power','fission_power_w',delay_ticks=1,lag_s=.5,sigma=1e4),
        Sensor('pressure','steam_pressure_pa',lag_s=.2,sigma=20.),Sensor('speed','turbine_speed_rad_s',lag_s=.2,sigma=.001)],seed=23)
    loops=[PIConfig('power','rod_target',p0,p0,kp=.01,ki=.002,direction=-1,bias=initial['controls']['rod_target']),
        PIConfig('pressure','steam_bypass',initial['steam_header']['pressure_pa'],1e6,kp=.1,ki=.02,direction=-1),
        PIConfig('speed','steam_admission',157.,10.,kp=.03,ki=.01,bias=1.)]
    s=ControlSession(plant,instruments=bank,loops=loops,protection=ProtectionConfig(startup_grace_ticks=1))
    normal_results=[];physical=[];saved=None
    for t in range(24):
        normal_results.append(s.step(**normal_kwargs(t,p0)));physical.append(s.plant.snapshot())
        if t==6:saved=json.loads(json.dumps(s.checkpoint(),sort_keys=True))
        if t%6==0:print('station control nominal',t+1,'/24',flush=True)
    normal_final=s.checkpoint();restored=ControlSession.restore(saved);normal_exact=True
    for t in range(7,24):normal_exact=restored.step(**normal_kwargs(t,p0))==normal_results[t] and normal_exact
    normal_exact=normal_exact and restored.checkpoint()==normal_final
    evidence.save(output/'normal.json',dict(trace=normal_results,trusted_plant_trace=physical,checkpoint=saved,
        final=normal_final,restart_exact=normal_exact))
    s=ControlSession(protection=ProtectionConfig(overspeed_fraction=.999));initial_rod=s.plant.snapshot()['actuator_status']['rod_position']
    tripped=[];trip_physical=[];saved_trip=None
    for t in range(10):
        tripped.append(s.step(**trip_kwargs(t)));trip_physical.append(s.plant.snapshot())
        if t==0:saved_trip=json.loads(json.dumps(s.checkpoint(),sort_keys=True))
    restored=ControlSession.restore(saved_trip);trip_exact=True
    for t in range(1,10):trip_exact=restored.step(**trip_kwargs(t))==tripped[t] and trip_exact
    trip_exact=trip_exact and restored.checkpoint()==s.checkpoint()
    events=s.plant.snapshot()['events'];decisions=tripped[0]['receipt']['decisions']['steam_admission']
    trip_ok=(decisions['authority']=='protection' and set(decisions['candidates'])>={'operator','controller','interlock','protection'}
        and sum(e['kind']=='scram_latched' for e in events)==1 and sum(e['kind']=='turbine_trip_latched' for e in events)==1
        and initial_rod<tripped[0]['receipt']['actual_positions']['rod']<1.
        and all(r['receipt']['trip_latched'] for r in tripped)
        and all(not r['receipt']['dispatched'] for r in tripped[1:])
        and any(e['kind']=='reset_denied' for e in s.events))
    evidence.save(output/'trip.json',dict(trace=tripped,trusted_plant_trace=trip_physical,checkpoint=saved_trip,
        final=s.checkpoint(),restart_exact=trip_exact,passed=bool(trip_ok)))
    s=ControlSession(protection=ProtectionConfig(pressure_open_pa=6.36e6,pressure_close_pa=6.35e6));release=[]
    for t in range(4):release.append(s.step());physical.append(s.plant.snapshot())
    release_ok=(release[0]['receipt']['pressure_interlock'] and any(e['kind']=='pressure_interlock_released' for e in s.events)
        and any(r['receipt']['dispatched'].get('steam_bypass')==0. for r in release[1:]) and s.manual['steam_bypass']==0.)
    evidence.save(output/'interlock.json',dict(trace=release,events=s.events,passed=bool(release_ok),final=s.checkpoint()))
    invalid=ControlSession();before=invalid.checkpoint();rollback=False
    try:invalid.step(dict(steam_admission=2.),modes=dict(rod_target='auto'))
    except ValueError:rollback=invalid.checkpoint()==before
    evidence.save(output/'rollback.json',dict(exact=rollback))
    all_physical=physical+trip_physical
    inventory_ok=all(r['energy_relative']<=CRITERIA['energy_relative'] and r['mass_relative']<=CRITERIA['mass_relative'] and r['boron_relative']<=CRITERIA['boron_relative'] for r in all_physical)
    normal_ok=not any(r['receipt']['trip_latched'] for r in normal_results)
    dispatch_ack=all(set(r['receipt']['dispatched'])<=set(r['receipt']['decisions']) for r in normal_results+tripped+release)
    unchanged=source==control_signature() and driver==sha(__file__) and serializer==sha(evidence.__file__)
    passed=bool(unchanged and policy_ok and trip_ok and release_ok and inventory_ok and normal_ok and dispatch_ack
        and normal_exact and trip_exact and rollback and max(r['error'] for r in reference_rows)<=CRITERIA['pi_reference_absolute']
        and abs(saturation.integral)<1.1)
    report=dict(completed=True,passed=passed,source_unchanged=unchanged,closed_tasks=TASKS if passed else [],
        nominal_duration_s=24.,trip_duration_s=10.,synthetic_policy_duration_s=120.,
        restart_exact=bool(normal_exact and trip_exact),operator_controller_interlock_protection_qualified=bool(trip_ok),
        pressure_interlock_return_qualified=bool(release_ok),invalid_step_rollback_exact=rollback,
        duplicate_policy_epoch_rejected=duplicate_rejected,pi_reference_error=max(r['error'] for r in reference_rows),
        maximum_energy_relative=max(r['energy_relative'] for r in all_physical),
        maximum_mass_relative=max(r['mass_relative'] for r in all_physical),
        feed_and_level_regulators_qualified=False,complete_npp_09_02=False,
        full_r1_event_matrix_qualified=False,complete_package_c=False,r1_accepted=False,validated=False,
        limitations=['Sampled research thresholds, not plant safety settings or licensed protection',
            'Three PI foundations only; feed and geometric level regulation NPP-09.02 remains open',
            'Actual physical trip is irreversible inside the current plant session; reset request is denied',
            '120s test is synthetic policy input, not a 120s physical station acceptance',
            'Short hot-station cases only; full R1 controller/fault/horizon qualification remains open'])
    evidence.save(output/'report.json',report)
    if reference is not None:
        reference=Path(reference);files={p.relative_to(output) for p in output.rglob('*.json')}
        if files!={p.relative_to(reference) for p in reference.rglob('*.json')}:raise ValueError('Control evidence sets differ')
        for name in sorted(files):
            if evidence.read(output/name)!=evidence.read(reference/name):raise ValueError('Installed control evidence differs:'+str(name))
        print('All installed control evidence matches exactly',flush=True)
    if not passed:raise ValueError('Station supervisory qualification failed')
    print(json.dumps(dict(passed=True,closed_tasks=TASKS)),flush=True)
    return report


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference',type=Path);args=parser.parse_args();run(args.output,args.reference)
