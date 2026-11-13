"""Finite station cooling: local nominal, causal disturbances and wheel replay."""
from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np

from atomic_model_station import PWRCooledSession,CoolingConfig
from atomic_model_station.pwr_train import physical_signature
from atomic_model_multiphysics.provenance import sha
from core.status import DomainError,SolverError
from scripts import qualify_station_actuators as evidence

TASKS=['NPP-08.01','NPP-08.02']
SETTINGS=[(.5,1e-7),(.2,2e-8),(.1,2e-9)]
CRITERIA=dict(energy_relative=1e-6,mass_relative=1e-10,boron_relative=1e-10,
    instantaneous_energy_relative=1e-8,refinement_relative=1e-4,pump_flow_analytic_relative=5e-7,
    cooling_energy_residual_w=1e-4,boundary_power_relative=1e-12,entropy_min_w_k=-1e-6,nominal_drift_relative=1e-5)


def vector(row):
    return np.r_[row['pwr_state'],row['turbine_energy_j'],row['clad_temperature_k'],row['cooling_energy_j'],row['cooling']['flow_kg_s']]


def admit(row):
    c=row['cooling'];p=row['powers']
    return bool(row['energy_relative']<=CRITERIA['energy_relative'] and row['mass_relative']<=CRITERIA['mass_relative']
        and row['boron_relative']<=CRITERIA['boron_relative']
        and abs(row['instantaneous_energy_defect_w'])/max(abs(p['nuclear_heat_w']),1.)<=CRITERIA['instantaneous_energy_relative']
        and abs(c['energy_residual_w'])<=CRITERIA['cooling_energy_residual_w']
        and abs(c['pump_energy_residual_w'])/max(abs(c['pump_electric_w']),abs(c['kinetic_rate_w']),1.)<=CRITERIA['boundary_power_relative']
        and abs(c['water_boundary_residual_w'])/max(abs(c['external_removed_w']),1.)<=CRITERIA['boundary_power_relative']
        and min(c['entropy_generation_w_k'].values())>=CRITERIA['entropy_min_w_k'])


def run(output,reference=None):
    output=Path(output).absolute()
    if any(p.is_symlink() for p in (output,*output.parents)):raise ValueError('Symlink output')
    output.mkdir(parents=True,exist_ok=False)
    source=physical_signature();driver=sha(__file__);serializer=sha(evidence.__file__)
    nominal=PWRCooledSession(max_step_s=1.);initial=nominal.snapshot()
    evidence.save(output/'protocol.json',dict(source=source,driver_sha256=driver,serializer_sha256=serializer,
        criteria=CRITERIA,settings=SETTINGS,nominal_duration_s=3600.,disturbance_duration_s=10.,candidate_tasks=TASKS,
        parameters=nominal.parameter_registry(),initial=initial,
        external_source='Prescribed inlet flow/temperature, finite heat-capacity rate, computed outlet water; no tower',
        energy_boundary='Condenser heat internal; external water removes heat; electric pump input counted once in auxiliaries and cooling energy',
        approximations=['Constant-density/cp incompressible cooling water, two fixed finite masses',
            'Head proportional to command squared, quadratic pipe resistance, fluid inertance; no motor rotor or cavitation map',
            'External water delivery infrastructure outside the plant boundary; auxiliary bus still ideally available']))
    print('finite cooling protocol sealed',flush=True)
    rows=[]
    for i in range(60):
        rows.append(nominal.advance(60.))
        if i%5==0:print('finite cooling nominal',i+1,'/60',flush=True)
    last=rows[-1];drift=float(np.max(abs(vector(last)-vector(initial))/np.maximum(abs(vector(initial)),1.)))
    evidence.save(output/'nominal.json',dict(trace=rows,relative_drift=drift,checkpoint=nominal.checkpoint()))
    cases=[];observations=[initial,*rows];analytic_errors=[]
    for name,patch in [('pump_off',{'cooling_pump_on':False}),('external_water_loss',{'external_water_fraction':0.})]:
        trials=[]
        for step,rtol in SETTINGS:
            s=PWRCooledSession(max_step_s=step,rtol=rtol);start=s.snapshot();row=s.advance(10.,patch)
            record=dict(max_step_s=step,rtol=rtol,initial=start,observation=row)
            if name=='pump_off':
                q=start['cooling']['flow_kg_s'];expected=q/(1+s.cooling_resistance*q*10./s.cooling_config.pipe_inertance_m_inv)
                error=abs(row['cooling']['flow_kg_s']-expected)/expected;record.update(analytic_flow_kg_s=expected,analytic_error=error);analytic_errors.append(error)
            observations.append(row);trials.append(record)
            print('finite cooling disturbance',name,step,flush=True)
        fine=vector(trials[-1]['observation'])
        errors=[float(np.max(abs(vector(t['observation'])-fine)/np.maximum(abs(fine),1.))) for t in trials]
        cases.append(dict(name=name,trials=trials,refinement_errors=errors))
    pump=cases[0]['trials'][0]['observation'];water=cases[1]['trials'][0]['observation']
    causal=bool(pump['cooling']['flow_kg_s']<initial['cooling']['flow_kg_s']
        and pump['cooling']['hot_temperature_k']>last['cooling']['hot_temperature_k']
        and pump['condenser_pressure_pa']>last['condenser_pressure_pa']
        and water['cooling']['external_removed_w']==0.
        and water['cooling']['cold_temperature_k']>last['cooling']['cold_temperature_k']
        and water['condenser_pressure_pa']>last['condenser_pressure_pa']
        and water['powers']['gross_electrical_w']<last['powers']['gross_electrical_w'])
    evidence.save(output/'disturbances.json',dict(cases=cases,causal=causal))
    restart=PWRCooledSession(max_step_s=.5);restart.advance(2.,{'cooling_pump_fraction':.8,'external_water_fraction':.7})
    saved=json.loads(json.dumps(restart.checkpoint()));restored=PWRCooledSession.restore(saved)
    resumed=restart.advance(2.);exact=restored.advance(2.)==resumed;observations.append(resumed)
    trip=PWRCooledSession(max_step_s=.2);trip_row=trip.advance(10.,{'scram':True,'turbine_trip':True});observations.append(trip_row)
    limited=PWRCooledSession(cooling=CoolingConfig(cold_mass_kg=1e4,hot_mass_kg=1e4),max_step_s=.2)
    before=limited.checkpoint();rollback=False
    try:limited.advance(30.,{'external_water_fraction':0.,'scram':True,'turbine_trip':True})
    except (ValueError,DomainError,SolverError):rollback=limited.checkpoint()==before
    evidence.save(output/'switches.json',dict(checkpoint=saved,resumed=resumed,restart_exact=exact,trip=trip_row,domain_rejection_rollback=rollback))
    all_admitted=all(admit(row) for row in observations)
    refinements=[c['refinement_errors'] for c in cases]
    boundary=initial['cooling'];pump_accounted=abs(boundary['external_removed_w']-boundary['condenser_received_w']-boundary['pump_electric_w'])<1e-4
    unchanged=source==physical_signature() and driver==sha(__file__) and serializer==sha(evidence.__file__)
    checks={'NPP-08.01':bool(all_admitted and causal and exact and rollback and pump_accounted and max(analytic_errors)<=CRITERIA['pump_flow_analytic_relative']),
            'NPP-08.02':bool(all_admitted and causal and boundary['external_flow_kg_s']>0 and boundary['external_effectiveness']<1)}
    passed=bool(unchanged and all(checks.values()) and drift<=CRITERIA['nominal_drift_relative']
        and all(max(err)<=CRITERIA['refinement_relative'] for err in refinements))
    report=dict(completed=True,passed=passed,source_unchanged=unchanged,task_checks=checks,closed_tasks=TASKS if passed else [],
        nominal_duration_s=last['time_s'],nominal_relative_drift=drift,refinement_errors=refinements,pump_analytic_errors=analytic_errors,
        causal_cooling_feedback=causal,restart_exact=exact,domain_rejection_rollback=rollback,
        max_energy_relative=max(r['energy_relative'] for r in observations),max_mass_relative=max(r['mass_relative'] for r in observations),
        max_boron_relative=max(r['boron_relative'] for r in observations),min_entropy_generation_w_k=min(min(r['cooling']['entropy_generation_w_k'].values()) for r in observations),
        nominal_cooling_pump_electric_w=boundary['pump_electric_w'],nominal_net_export_w=initial['powers']['net_export_w'],
        complete_package_b=False,complete_package_c=False,r1_accepted=False,validated=False,plant_referenced=False,
        full_nominal_checkpoint_replay=False,
        limitations=['Hot research configuration, illustrative constant cooling-water properties and component sizes',
            'Finite cooling loop only; no global station electrical blackout, reserve bus or automatic protection admission',
            'No cooling tower, external delivery pump or seasonal field calibration',
            'No primary pump coastdown/natural circulation or steam-line/MSR admission',
            'Original primary EOS, feed density, phase and inactive spray limits still apply',
            'Declared sampled domain checks are not continuous safety proofs'])
    evidence.save(output/'report.json',report)
    if reference is not None:
        reference=Path(reference);files={p.relative_to(output) for p in output.rglob('*.json')}
        if files!={p.relative_to(reference) for p in reference.rglob('*.json')}:raise ValueError('Evidence file sets differ')
        for path in sorted(files):
            if evidence.read(output/path)!=evidence.read(reference/path):raise ValueError('Installed cooling evidence differs:'+str(path))
        print('All installed finite cooling evidence matches exactly',flush=True)
    if not passed:raise ValueError('Finite cooling qualification failed; no tasks accepted')
    print(json.dumps(dict(passed=passed,closed_tasks=report['closed_tasks'])),flush=True)
    return report


if __name__=='__main__':
    p=ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--reference',type=Path)
    args=p.parse_args();run(args.output,args.reference)
