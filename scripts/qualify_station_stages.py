"""Two equivalent stages with separator and finite-wall reheat, research scope."""
from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np
from atomic_model_station import PWRStagedSession,PWRSteamSession,StageConfig
from atomic_model_station.stages import stages_signature,staged_path
from atomic_model_multiphysics.provenance import sha
from physics.materials import iapws
from scripts import qualify_station_actuators as evidence
from scripts.qualify_station_steam import vector

TASKS=['NPP-05.03']
SETTINGS=[(1.,1e-7),(.5,2e-8),(.2,2e-9)]
CRITERIA=dict(energy_relative=1e-6,mass_relative=1e-10,boron_relative=1e-10,
    instantaneous_relative=1e-8,temporal_relative=1e-4,nominal_tail_relative=1e-5,
    routing_energy_w=.01,mass_stream_kg_s=1e-8,entropy_roundoff_w_k=.01,
    ideal_reference_relative=1e-10,wall_debit_w=.001,decay_reference_relative=1e-6)


def admit(r):
    p=r['staged_turbine'];cfg=r['stage_config'];audit=r['steam_port_audit']
    active_ok=(p['lp_exit_quality']>=cfg['minimum_lp_exit_quality'] and p['hp']['outlet_quality']>=cfg['minimum_hp_exit_quality']
        and p['reheater']['outlet_temperature_k']<=p['reheater']['wall_temperature_k']+1e-7) if p['active'] else (p['total_work_w']==0 and p['reheat_w']==0 and p['lp_exit_quality'] is None)
    return bool(active_ok and r['energy_relative']<=CRITERIA['energy_relative'] and r['mass_relative']<=CRITERIA['mass_relative']
        and r['boron_relative']<=CRITERIA['boron_relative']
        and abs(r['instantaneous_energy_defect_w'])/max(abs(r['powers']['nuclear_heat_w']),1.)<=CRITERIA['instantaneous_relative']
        and abs(p['energy_residual_w'])<=CRITERIA['routing_energy_w']
        and abs(p['routing_energy_residual_w'])<=CRITERIA['routing_energy_w']
        and abs(audit['energy_residual_w'])<=CRITERIA['routing_energy_w']
        and abs(p['mass_residual_kg_s'])<=CRITERIA['mass_stream_kg_s']
        and min(p['entropy_generation_w_k'].values())>=-CRITERIA['entropy_roundoff_w_k'])


def independent_reference():
    cfg=StageConfig(hp_efficiency=1.,lp_efficiency=1.,reheat_ua_w_k=0.,minimum_hp_exit_quality=.7,minimum_lp_exit_quality=.8)
    pin,pout,flow=6e6,8000.,1000.;up=iapws.saturation_properties(pin)[1]
    a,b=iapws.saturation_properties(cfg.intermediate_pressure_pa);c,d=iapws.saturation_properties(pout)
    hp_quality=(up.s_j_kg_k-a.s_j_kg_k)/(b.s_j_kg_k-a.s_j_kg_k)
    hp_enthalpy=a.h_j_kg+hp_quality*(b.h_j_kg-a.h_j_kg)
    lp_quality=(b.s_j_kg_k-c.s_j_kg_k)/(d.s_j_kg_k-c.s_j_kg_k)
    lp_enthalpy=c.h_j_kg+lp_quality*(d.h_j_kg-c.h_j_kg)
    expected_work=flow*(up.h_j_kg-hp_enthalpy)+flow*hp_quality*(b.h_j_kg-lp_enthalpy)
    expected_cond=flow*(1-hp_quality)*a.h_j_kg+flow*hp_quality*lp_enthalpy
    observed=staged_path(pin,up.h_j_kg,pout,flow,558.,cfg)
    error=max(abs(observed['total_work_w']-expected_work)/expected_work,abs(observed['condenser_energy_w']-expected_cond)/expected_cond)
    old=PWRSteamSession();new=PWRStagedSession()
    base=old._rhs(0.,old.state);changed=new._rhs(0.,new.state);i=new.base_size
    wall_delta=(changed[i-1]-base[i-1])*new.plant.sg_wall_capacity
    return dict(ideal=dict(expected_work_w=expected_work,expected_condenser_energy_w=expected_cond,observed=observed,relative_error=float(error)),
        finite_wall=dict(removed_heat_w=new.staged_terms['reheat_w'],wall_rate_power_delta_w=float(wall_delta),
            wall_debit_residual_w=float(wall_delta+new.staged_terms['reheat_w']),
            total_routing_residual_w=float(wall_delta+changed[i-2]-base[i-2]+changed[i]-base[i]),
            initial_stocks_unchanged=bool(np.array_equal(new.state,old.state)),initial_stored_energy_unchanged=new._stored()==old._stored()),
        scope='Independent entropy/enthalpy ideal cycle and difference of actual existing energy-owner rates; numerical, not experimental')


def late_decay_reference(rows):
    """Independent analytic convolution with linearly observed late forcing."""
    last=rows[-3:];initial=np.array(last[0]['pwr_state'][7:10]);predicted=initial.copy()
    # Existing declared illustrative groups; inspect live coefficients rather
    # than assign a fitted decay time from the failed stationarity check.
    model=PWRStagedSession().plant.decay
    lam=np.asarray(model.decay_constants);fraction=np.asarray(model.fractions)
    for a,b in zip(last[:-1],last[1:]):
        dt=b['time_s']-a['time_s'];p0=a['core']['fission_total_w'];p1=b['core']['fission_total_w']
        x=lam*dt;change=np.expm1(-x)
        predicted=predicted*np.exp(-x)+fraction*(p0*(-change)/lam+(p1-p0)/dt*(x+change)/lam**2)
    actual=np.array(last[-1]['pwr_state'][7:10])
    return dict(initial=initial.tolist(),predicted=predicted.tolist(),actual=actual.tolist(),
        decay_constants_per_s=lam.tolist(),fractions=fraction.tolist(),
        relative_error=float(max(abs(predicted-actual)/np.maximum(abs(actual),1.))),
        scope='Last120s, piecewise-linear fission between60s observations; independent analytic integration, no stock reset')


def run(output,reference=None):
    output=Path(output).absolute()
    if any(p.is_symlink() for p in (output,*output.parents)):raise ValueError('Symlink output')
    output.mkdir(parents=True,exist_ok=False)
    source=stages_signature();driver=sha(__file__);serializer=sha(evidence.__file__)
    s=PWRStagedSession(max_step_s=1.);initial=s.snapshot()
    evidence.save(output/'protocol.json',dict(source=source,driver_sha256=driver,serializer_sha256=serializer,criteria=CRITERIA,
        settings=SETTINGS,initial=initial,parameters=s.parameter_registry(),candidate_tasks=TASKS,
        nominal_duration_s=3600.,warmup_duration_s=3600.,total_operating_duration_s=7200.,trip_duration_s=60.,
        scope='Two equivalent thermodynamic stages, fixed intermediate pressure, quasi-steady separator/reheater; no blade-stage or hardware MSR maps',
        heat_source='Existing finite SG wall, heat subtracted once; no new thermal reservoir or fixed-temperature boundary',
        moisture='HP/LP limits are declared research admission parameters, not manufacturer erosion limits',
        reheat='Secant-cp effectiveness law; no distributed heat exchanger or separator hold-up'))
    print('stages protocol sealed',flush=True)
    warmup=[]
    for i in range(60):
        warmup.append(s.advance(60.))
        if i%5==0:print('stages warmup',i+1,'/60',flush=True)
    warmup_tail=float(max(abs(vector(warmup[-1])-vector(warmup[-3]))/np.maximum(abs(vector(warmup[-3])),1.)))
    evidence.save(output/'warmup.json',dict(trace=warmup,tail_relative_drift=warmup_tail,checkpoint=s.checkpoint()))
    nominal=[]
    for i in range(60):
        nominal.append(s.advance(60.))
        if i%5==0:print('stages nominal',i+1,'/60',flush=True)
    tail=float(max(abs(vector(nominal[-1])-vector(nominal[-3]))/np.maximum(abs(vector(nominal[-3])),1.)))
    evidence.save(output/'nominal.json',dict(trace=nominal,tail_relative_drift=tail,checkpoint=s.checkpoint()))
    s=PWRStagedSession(max_step_s=.5);regulated=[]
    for patch in [dict(steam_admission=.98),dict(steam_admission=1.)]:regulated.append(s.advance(10.,patch))
    evidence.save(output/'regulation.json',dict(trace=regulated,checkpoint=s.checkpoint()))
    s=PWRStagedSession(max_step_s=.5);trip=[]
    for i in range(6):
        trip.append(s.advance(10.,dict(scram=True,turbine_trip=True) if i==0 else None))
        print('stages trip',i+1,'/6',flush=True)
    off=s.advance(.1,dict(separator_available=False,reheater_available=False))
    evidence.save(output/'trip.json',dict(trace=trip,equipment_off=off,checkpoint=s.checkpoint()))
    imperfect=PWRStagedSession(stages=StageConfig(separator_efficiency=.95),max_step_s=.5).advance(10.)
    evidence.save(output/'imperfect_separator.json',imperfect)
    trials=[]
    for step,rtol in SETTINGS:
        s=PWRStagedSession(max_step_s=step,rtol=rtol);moving=s.advance(.1,dict(scram=True,turbine_trip=True))
        saved=json.loads(json.dumps(s.checkpoint()));t=PWRStagedSession.restore(saved)
        final=s.advance(4.9);exact=t.advance(4.9)==final
        trials.append(dict(step_s=step,rtol=rtol,moving=moving,checkpoint=saved,final=final,restart_exact=exact))
        print('stages refinement',step,flush=True)
    fine=vector(trials[-1]['final']);errors=[float(max(abs(vector(x['final'])-fine)/np.maximum(abs(fine),1.))) for x in trials]
    refs=independent_reference();decay_reference=late_decay_reference(nominal)
    evidence.save(output/'references.json',dict(trials=trials,refinement_errors=errors,independent=refs,decay_reference=decay_reference))
    faults=[]
    for patch in [dict(reheater_available=False),dict(separator_available=False)]:
        s=PWRStagedSession();before=s.checkpoint();rejected=False
        try:s.advance(.1,patch)
        except ValueError:rejected=s.checkpoint()==before
        faults.append(dict(request=patch,rejected_atomically=rejected))
    evidence.save(output/'faults.json',dict(cases=faults))
    observations=[initial,*warmup,*nominal,*regulated,*trip,off,imperfect,*[x[k] for x in trials for k in ('moving','final')]]
    unchanged=source==stages_signature() and driver==sha(__file__) and serializer==sha(evidence.__file__)
    debit=refs['finite_wall']
    passed=bool(unchanged and all(admit(r) for r in observations) and all(f['rejected_atomically'] for f in faults)
        and all(t['restart_exact'] for t in trials) and max(errors)<=CRITERIA['temporal_relative']
        and tail<=CRITERIA['nominal_tail_relative'] and decay_reference['relative_error']<=CRITERIA['decay_reference_relative'] and refs['ideal']['relative_error']<=CRITERIA['ideal_reference_relative']
        and abs(debit['wall_debit_residual_w'])<=CRITERIA['wall_debit_w']
        and abs(debit['total_routing_residual_w'])<=CRITERIA['routing_energy_w']
        and debit['initial_stocks_unchanged'] and debit['initial_stored_energy_unchanged'])
    active=[r['staged_turbine'] for r in observations if r['staged_turbine']['active']]
    report=dict(completed=True,passed=passed,source_unchanged=unchanged,closed_tasks=TASKS if passed else [],task_checks={'NPP-05.03':passed},
        nominal_duration_s=3600.,warmup_duration_s=3600.,total_operating_duration_s=7200.,trip_duration_s=60.,nominal_tail_relative_drift=tail,warmup_tail_relative_drift=warmup_tail,
        decay_reference_relative_error=decay_reference['relative_error'],refinement_errors=errors,
        restart_exact=all(t['restart_exact'] for t in trials),faults_rejected_atomically=all(f['rejected_atomically'] for f in faults),
        ideal_reference_relative_error=refs['ideal']['relative_error'],wall_debit_residual_w=debit['wall_debit_residual_w'],
        minimum_hp_exit_quality=min(p['hp']['outlet_quality'] for p in active),minimum_lp_exit_quality=min(p['lp_exit_quality'] for p in active),
        nominal_final_reheat_w=nominal[-1]['staged_turbine']['reheat_w'],
        max_energy_relative=max(r['energy_relative'] for r in observations),max_mass_relative=max(r['mass_relative'] for r in observations),
        max_boron_relative=max(r['boron_relative'] for r in observations),
        max_path_energy_residual_w=max(abs(r['staged_turbine']['energy_residual_w']) for r in observations),
        complete_package_b=False,complete_package_c=False,r1_accepted=False,validated=False,plant_referenced=False,full_long_checkpoint_replay=False,
        limitations=['Two equivalent stages, not the number of real blade rows; fixed interstage pressure, no finite interstage storage or nozzle map',
            'Zero-hold-up separator with declared efficiency and isenthalpic drain to condenser, no droplet residence/erosion qualification',
            'Secant-cp reheater supplied by the existing SG wall, not a manufacturer steam-bleed MSR topology',
            'Moisture thresholds and efficiencies are illustrative admission parameters, no assigned plant uncertainty',
            'Manual admission and hot start only; no automatic reheat controller, cold start or hardware turbine map',
            'NPP04.04/05.04, full packagesB/C andR1 remain open; no plant/experimental validation'])
    evidence.save(output/'report.json',report)
    if reference is not None:
        reference=Path(reference);files={f.relative_to(output) for f in output.rglob('*.json')}
        if files!={f.relative_to(reference) for f in reference.rglob('*.json')}:raise ValueError('Evidence sets differ')
        for name in sorted(files):
            if evidence.read(output/name)!=evidence.read(reference/name):raise ValueError('Installed stages evidence differs:'+str(name))
        print('All installed two-stage finite-wall evidence matches exactly',flush=True)
    if not passed:raise ValueError('Two-stage qualification failed; no task accepted')
    print(json.dumps(dict(passed=True,closed_tasks=TASKS)),flush=True)
    return report


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference',type=Path);args=parser.parse_args();run(args.output,args.reference)
