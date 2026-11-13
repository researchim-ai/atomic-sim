"""Primary coastdown and true zero-pump-head circulation in a research station."""
from argparse import ArgumentParser
import json
import math
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
from atomic_model_station import PWRPrimarySession,PrimaryConfig
from atomic_model_station.core import core_signature
from atomic_model_station.primary import frozen_gravity_head
from physics.hydraulics.network import G
from atomic_model_multiphysics.provenance import sha
from scripts import qualify_station_actuators as evidence

TASKS=['NPP-04.03']
SETTINGS=[(.5,1e-7),(.25,2e-8),(.1,2e-9)]
CRITERIA=dict(energy_relative=1e-6,mass_relative=1e-10,boron_relative=1e-10,instantaneous_relative=1e-8,
    momentum_relative=1e-8,gravity_reference_pa=1e-6,temporal_relative=1e-4,nominal_tail_relative=1e-5,
    natural_frozen_flow_relative=.03,natural_inertial_fraction=.1,natural_flow_spread_relative=.02,
    isolated_shaft_relative=1e-8,minimum_natural_flow_kg_s=1.)


def vector(r):return np.r_[r['pwr_state'],r['turbine_energy_j'],r['clad_temperature_k'],r['cooling_energy_j'],r['cooling']['flow_kg_s'],r['feed_inventory'],r['electrical']['backup_energy_j'],r['core']['owned_state'],r['primary']['log_speed'],r['primary']['bypass_fraction']]


def admit(r):
    p=r['primary'];scale=max(abs(p['pump_head_pa'])+abs(p['buoyancy_head_pa'])+abs(p['friction_head_pa'])+abs(p['inertial_head_pa']),1.)
    return bool(r['energy_relative']<=CRITERIA['energy_relative'] and r['mass_relative']<=CRITERIA['mass_relative'] and r['boron_relative']<=CRITERIA['boron_relative']
        and abs(r['instantaneous_energy_defect_w'])/max(abs(r['powers']['nuclear_heat_w']),1.)<=CRITERIA['instantaneous_relative']
        and abs(p['momentum_residual_pa'])/scale<=CRITERIA['momentum_relative']
        and min(p['cycle_flows_kg_s'].values())>=0. and p['minimum_sampled_spray_flow_kg_s']>=-p['spray_signed_roundoff_band_kg_s'])


def run(output,reference=None):
    output=Path(output).absolute()
    if any(p.is_symlink() for p in (output,*output.parents)):raise ValueError('Symlink output')
    output.mkdir(parents=True,exist_ok=False)
    source=core_signature();driver=sha(__file__);serializer=sha(evidence.__file__)
    s=PWRPrimarySession(max_step_s=1.);initial=s.snapshot()
    evidence.save(output/'protocol.json',dict(source=source,driver_sha256=driver,serializer_sha256=serializer,criteria=CRITERIA,settings=SETTINGS,
        parameters=s.parameter_registry(),initial=initial,candidate_tasks=TASKS,nominal_duration_s=3600.,coastdown_duration_s=600.,natural_study_duration_s=1800.,
        natural_zero_head_from_s=60.,bypass='Declared equivalent passive path, fixed existing resistance/inertance,60s linear motion; pump head/torque scaled equally, motor inhibited',
        shaft='Owned log speed, old omega derived; stored shaft energy unchanged; no powered near-stall start',
        steam='Closed turbine target solved by exact exponential, no signed negative-flow projection',
        spray='Existing pressurizer controller retained dynamically, signed roundoff band1e-9kg/s, no new manufacturer map',
        scope='Point core, declared10m geometry and limited rotor band0.5..1.1; research numerical admission only'))
    print('primary study protocol sealed',flush=True)
    rows=[]
    for i in range(60):
        rows.append(s.advance(60.))
        if i%5==0:print('primary nominal',i+1,'/60',flush=True)
    tail=float(np.max(abs(vector(rows[-1])-vector(rows[-3]))/np.maximum(abs(vector(rows[-3])),1.)))
    evidence.save(output/'nominal.json',dict(trace=rows,tail_relative_drift=tail,checkpoint=s.checkpoint()))
    observations=[initial,*rows];cases=[];natural_session=None
    for name,count,patch in [('coastdown',10,dict(scram=True,turbine_trip=True,primary_pump_on=False)),
        ('natural',30,dict(scram=True,turbine_trip=True,primary_pump_on=False,primary_bypass_open=True))]:
        t=PWRPrimarySession(max_step_s=1.);trace=[]
        for i in range(count):
            trace.append(t.advance(60.,patch if i==0 else None))
            if i%5==0:print('primary',name,i+1,'/',count,flush=True)
        evidence.save(output/(name+'.json'),dict(trace=trace,checkpoint=t.checkpoint()))
        cases.append(trace);observations.extend(trace)
        if name=='natural':natural_session=t
    final=cases[-1][-1];p=final['primary'];plant=natural_session.plant
    rho={name:plant.network.volumes[name].thermo.rho_kg_m3 for name in ('core','hot_leg','sg_primary','cold_leg')}
    independent_head=frozen_gravity_head(rho,plant.height_m)
    resistance=sum(plant.network.junctions[name].loss_coefficient/(2*plant.network.volumes[plant.network.junctions[name].upstream].thermo.rho_kg_m3*plant.network.junctions[name].area_m2**2)
        for name in ('pump','core_out','to_sg','to_cold'))
    independent_flow=brentq(lambda q:resistance*q*abs(q)-independent_head,0.,1e6,xtol=1e-10)
    natural_error=abs(p['mean_flow_kg_s']-independent_flow)/independent_flow
    inertial_fraction=abs(p['inertial_head_pa'])/max(abs(p['buoyancy_head_pa']),1.)
    q0=cases[-1][0]['primary']['speed_rad_s'];decay=plant.loss_coef/plant.inertia
    shaft_errors=[abs(r['primary']['speed_rad_s']-q0*math.exp(-decay*(r['time_s']-60.)))/max(q0*math.exp(-decay*(r['time_s']-60.)),1e-300) for r in cases[-1]]
    zero_head=all(r['primary']['pump_head_pa']==0. and not r['primary']['effective_motor_on'] and r['primary']['bypass_fraction']==1. for r in cases[-1])
    trials=[]
    for step,rtol in SETTINGS:
        t=PWRPrimarySession(max_step_s=step,rtol=rtol)
        r=t.advance(10.,dict(scram=True,turbine_trip=True,primary_pump_on=False,primary_bypass_open=True))
        trials.append(dict(max_step_s=step,rtol=rtol,observation=r));observations.append(r)
        print('primary transient refinement',step,flush=True)
    fine=vector(trials[-1]['observation']);errors=[float(np.max(abs(vector(t['observation'])-fine)/np.maximum(abs(fine),1.))) for t in trials]
    evidence.save(output/'references.json',dict(independent_gravity_head_pa=independent_head,independent_frozen_flow_kg_s=independent_flow,
        natural_frozen_flow_error=natural_error,natural_inertial_fraction=inertial_fraction,isolated_shaft_errors=shaft_errors,
        no_gravity_frozen_cycle_head_pa=-p['friction_head_pa'],trials=trials,refinement_errors=errors))
    restart=PWRPrimarySession(max_step_s=.2);restart.advance(.2,dict(scram=True,turbine_trip=True,primary_bypass_open=True))
    saved=json.loads(json.dumps(restart.checkpoint()));restored=PWRPrimarySession.restore(saved);r=restart.advance(.2);exact=restored.advance(.2)==r;observations.append(r)
    before=restart.checkpoint();rollback=False
    try:restart.advance(.1,{'primary_bypass_open':1})
    except ValueError:rollback=restart.checkpoint()==before
    zero=frozen_gravity_head(rho,0.)
    evidence.save(output/'restart.json',dict(checkpoint=saved,resumed=r,restart_exact=exact,invalid_command_rollback=rollback,zero_height_frozen_gravity_pa=zero))
    coast=cases[0][-1]['primary']
    natural_pass=bool(zero_head and independent_head>0 and p['mean_flow_kg_s']>=CRITERIA['minimum_natural_flow_kg_s']
        and natural_error<=CRITERIA['natural_frozen_flow_relative'] and inertial_fraction<=CRITERIA['natural_inertial_fraction']
        and p['flow_spread_relative']<=CRITERIA['natural_flow_spread_relative'] and abs(independent_head-p['buoyancy_head_pa'])<=CRITERIA['gravity_reference_pa']
        and zero==0. and max(shaft_errors)<=CRITERIA['isolated_shaft_relative'])
    unchanged=source==core_signature() and driver==sha(__file__) and serializer==sha(evidence.__file__)
    passed=bool(unchanged and all(admit(r) for r in observations) and natural_pass and coast['speed_rad_s']<initial['primary']['speed_rad_s']
        and coast['mean_flow_kg_s']<initial['primary']['mean_flow_kg_s'] and tail<=CRITERIA['nominal_tail_relative']
        and max(errors)<=CRITERIA['temporal_relative'] and exact and rollback)
    report=dict(completed=True,passed=passed,source_unchanged=unchanged,task_checks={'NPP-04.03':passed},closed_tasks=TASKS if passed else [],
        nominal_duration_s=rows[-1]['time_s'],nominal_tail_relative_drift=tail,coastdown_duration_s=600.,natural_duration_s=1800.,
        exact_zero_head=zero_head,natural_final_flow_kg_s=p['mean_flow_kg_s'],natural_frozen_reference_kg_s=independent_flow,
        natural_frozen_flow_error=natural_error,natural_inertial_fraction=inertial_fraction,natural_flow_spread_relative=p['flow_spread_relative'],
        isolated_shaft_max_error=max(shaft_errors),refinement_errors=errors,restart_exact=exact,rollback=rollback,
        max_energy_relative=max(r['energy_relative'] for r in observations),max_mass_relative=max(r['mass_relative'] for r in observations),
        max_boron_relative=max(r['boron_relative'] for r in observations),
        complete_package_b=False,complete_package_c=False,r1_accepted=False,validated=False,plant_referenced=False,full_long_checkpoint_replay=False,
        limitations=['Reduced point-core station and declared10m loop, no surveyed primary geometry or manufacturer pump/bypass maps',
            'Natural circulation admitted only on the declared equivalent passive path after60s motion, not inferred from positive coastdown flow',
            'Fluid reference-density kinetic closure and existing pressure-controller target-flow law retained',
            'No powered near-stall/cold start or reverse primary cycle; no new boiling/CHF or accident-safety admission',
            'Broad turbine rotor study band0.5..1.1 is illustrative, no hardware operating-speed qualification',
            'Accepted nodes and0.05s samples are checked, not a continuous mathematical safety proof'])
    evidence.save(output/'report.json',report)
    if reference is not None:
        reference=Path(reference);files={p.relative_to(output) for p in output.rglob('*.json')}
        if files!={p.relative_to(reference) for p in reference.rglob('*.json')}:raise ValueError('Evidence file sets differ')
        for path in sorted(files):
            if evidence.read(output/path)!=evidence.read(reference/path):raise ValueError('Installed primary evidence differs:'+str(path))
        print('All installed primary circulation evidence matches exactly',flush=True)
    if not passed:raise ValueError('Primary circulation qualification failed; no task accepted')
    print(json.dumps(dict(passed=passed,closed_tasks=report['closed_tasks'])),flush=True)
    return report


if __name__=='__main__':
    p=ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--reference',type=Path)
    args=p.parse_args();run(args.output,args.reference)
