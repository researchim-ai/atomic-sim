"""Local bypass component qualification; does not accept NPP-05.04 as complete."""
from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np
from atomic_model_station import PWRBypassSession, BypassConfig
from atomic_model_station.bypass import bypass_signature, critical_nozzle
from atomic_model_multiphysics.provenance import sha
from physics.materials import iapws
from scripts import qualify_station_actuators as evidence

# Task closure requires the subsequent full-duration station study.
TASKS = []
CRITERIA = dict(energy_relative=1e-6, mass_relative=1e-10,
                condenser_energy_w=.01, nozzle_grid_relative=1e-4,
                entropy_roundoff_w_k=.01, temporal_relative=1e-4,
                pressure_event_time_s=.001)
SETTINGS = [(1.,1e-7),(.5,2e-8),(.2,2e-9)]


def run(output, reference=None):
    output = Path(output).absolute()
    if any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError('Symlink output refused')
    output.mkdir(parents=True, exist_ok=False)
    source = bypass_signature(); driver = sha(__file__)
    serializer = sha(evidence.__file__)
    s = PWRBypassSession(max_step_s=.5)
    evidence.save(output/'protocol.json', dict(source=source, driver_sha256=driver,
        serializer_sha256=serializer, criteria=CRITERIA, settings=SETTINGS, parameters=s.parameter_registry(),
        candidate_tasks=['NPP-05.04'], accepted_tasks=[], nominal_duration_s=0.,
        trip_duration_s=16., upper_cycle_duration_s=3., upper_cycle_restart_s=.1,
        upper_cycle_thresholds_pa=dict(open=6.4e6,close=6.38e6), scope='Component checks only; no full-duration station acceptance'))
    p, h = s.header.thermo.p_pa, s.header.thermo.h_j_kg
    entropy = iapws.properties_ph(p,h).s_j_kg_k
    grid = []
    # Independent equilibrium lever-rule reference, without the nozzle optimizer.
    for pressure in np.linspace(.2*p,.99*p,301):
        f,g = iapws.saturation_properties(pressure)
        x = (entropy-f.s_j_kg_k)/(g.s_j_kg_k-f.s_j_kg_k)
        if not 0 <= x <= 1: raise ValueError('Reference mixture domain')
        ht = (1-x)*f.h_j_kg+x*g.h_j_kg
        rho = 1/((1-x)/f.rho_kg_m3+x/g.rho_kg_m3)
        grid.append(float(rho*np.sqrt(2*(h-ht))))
    peak = critical_nozzle(float(p),float(h))
    error = abs(max(grid)/peak['mass_flux_kg_m2_s']-1.)
    evidence.save(output/'reference.json',dict(grid_mass_flux=grid, nozzle=peak, relative_error=error))
    observations = []
    for dt,patch in [(1.,dict(scram=True,turbine_trip=True)),(15.,None)]:
        observations.append(s.advance(dt,patch))
        print('bypass trip',s.time,flush=True)
    trip = observations[-1]
    evidence.save(output/'trip.json',dict(trace=observations,checkpoint=s.checkpoint()))
    s = PWRBypassSession(max_step_s=.5)
    moving = s.advance(.02,dict(steam_bypass=.2))
    saved = json.loads(json.dumps(s.checkpoint()))
    t = PWRBypassSession.restore(saved)
    final = s.advance(.2)
    exact = t.advance(.2)==final
    unavailable = s.advance(.2,dict(bypass_available=False))
    observations += [moving,final,unavailable]
    evidence.save(output/'motion.json',dict(moving=moving,checkpoint=saved,
        final=final,restart_exact=exact,unavailable=unavailable))
    s = PWRBypassSession(bypass=BypassConfig(relief_open_pa=6.36e6,relief_close_pa=6.33e6),max_step_s=.5)
    relief = s.advance(.1); observations.append(relief)
    evidence.save(output/'relief.json',relief)
    s = PWRBypassSession(); before = s.checkpoint(); rollback = False
    try: s.advance(.1,dict(scram=True,steam_bypass=1.1))
    except ValueError: rollback = s.checkpoint()==before
    evidence.save(output/'rollback.json',dict(exact=rollback))
    trials = []
    upper_trials = []
    for step,rtol in SETTINGS:
        session = PWRBypassSession(max_step_s=step,rtol=rtol)
        session.advance(9.,dict(scram=True,turbine_trip=True))
        checkpoint = json.loads(json.dumps(session.checkpoint()))
        restored = PWRBypassSession.restore(checkpoint)
        result = session.advance(7.)
        restart = restored.advance(7.)==result
        events = [e for e in result['events'] if e['kind']=='bypass_supply_close']
        if len(events)!=1: raise ValueError('Expected one lower-pressure crossing')
        observations.append(result)
        trials.append(dict(step_s=step,rtol=rtol,state=session.state.tolist(),
            pressure_event_time_s=events[0]['time_s'],restart_exact=restart))
        print('bypass pressure refinement',step,flush=True)
        upper = PWRBypassSession(bypass=BypassConfig(
            relief_open_pa=6.4e6,relief_close_pa=6.38e6),max_step_s=step,rtol=rtol)
        before = upper.advance(.1,dict(scram=True,turbine_trip=True))
        saved = json.loads(json.dumps(upper.checkpoint()))
        restored = PWRBypassSession.restore(saved)
        after = upper.advance(2.9)
        exact_upper = restored.advance(2.9)==after
        roots = [e for e in after['events'] if e['kind'] in ('bypass_relief_open','bypass_relief_close')]
        cycle_ok = (not before['header_bypass']['relief_latched']
            and [e['kind'] for e in roots]==['bypass_relief_open','bypass_relief_close']
            and .1<roots[0]['time_s']<roots[1]['time_s']<3.
            and all(e['inventories_projected'] is False for e in roots)
            and abs(roots[0]['pressure_pa']-6.4e6)<=.05
            and abs(roots[1]['pressure_pa']-6.38e6)<=.05
            and not after['header_bypass']['relief_latched']
            and after['bypass_position']==upper.bypass_config.trip_opening)
        observations.extend([before,after])
        upper_trials.append(dict(step_s=step,rtol=rtol,before=before,checkpoint=saved,
            final=after,state=upper.state.tolist(),roots=roots,
            restart_exact=exact_upper,cycle_ok=cycle_ok))
        print('bypass upper-cycle refinement',step,flush=True)

    fine = np.asarray(trials[-1]['state'])
    errors = [float(max(abs(np.asarray(t['state'])-fine)/np.maximum(abs(fine),1.))) for t in trials]
    event_errors = [abs(t['pressure_event_time_s']-trials[-1]['pressure_event_time_s']) for t in trials]
    evidence.save(output/'refinement.json',dict(trials=trials,state_relative_errors=errors,
        pressure_event_time_errors_s=event_errors))
    upper_fine = np.asarray(upper_trials[-1]['state'])
    upper_errors = [float(max(abs(np.asarray(t['state'])-upper_fine)/np.maximum(abs(upper_fine),1.))) for t in upper_trials]
    upper_event_errors = [max(abs(a['time_s']-b['time_s']) for a,b in zip(t['roots'],upper_trials[-1]['roots'])) for t in upper_trials]
    evidence.save(output/'upper_cycle.json',dict(trials=upper_trials,state_relative_errors=upper_errors,
        pressure_event_time_errors_s=upper_event_errors))
    unchanged = source==bypass_signature() and driver==sha(__file__) and serializer==sha(evidence.__file__)
    balances = all(r['energy_relative']<=CRITERIA['energy_relative']
        and r['mass_relative']<=CRITERIA['mass_relative']
        and abs(r['condenser_balance']['energy_residual_w'])<=CRITERIA['condenser_energy_w']
        and r['header_bypass']['entropy_generation_w_k']>=-CRITERIA['entropy_roundoff_w_k']
        for r in observations)
    passed = bool(unchanged and balances and exact and rollback
        and error<=CRITERIA['nozzle_grid_relative']
        and all(t['restart_exact'] and t['cycle_ok'] for t in upper_trials)
        and max(upper_errors)<=CRITERIA['temporal_relative']
        and max(upper_event_errors)<=CRITERIA['pressure_event_time_s']
        and all(t['restart_exact'] for t in trials)
        and max(errors)<=CRITERIA['temporal_relative']
        and max(event_errors)<=CRITERIA['pressure_event_time_s']
        and observations[0]['header_bypass']['flow_kg_s']>0
        and trip['header_bypass']['flow_kg_s']==0
        and not trip['header_bypass']['supply_permitted']
        and unavailable['bypass_position']==0
        and relief['header_bypass']['relief_latched'])
    report = dict(passed=passed,completed=True,source_unchanged=unchanged,
        closed_tasks=[],restart_exact=exact and all(t['restart_exact'] for t in trials+upper_trials),rollback_exact=rollback,
        upper_cycle_qualified=all(t['cycle_ok'] for t in upper_trials),
        upper_refinement_relative_errors=upper_errors,upper_pressure_event_time_errors_s=upper_event_errors,
        refinement_relative_errors=errors,pressure_event_time_errors_s=event_errors,
        nominal_duration_s=0.,trip_duration_s=16.,nozzle_grid_relative_error=error,
        max_energy_relative=max(r['energy_relative'] for r in observations),
        max_mass_relative=max(r['mass_relative'] for r in observations),
        full_duration_qualified=False,complete_package_b=False,complete_package_c=False,
        r1_accepted=False,validated=False,plant_referenced=False,
        limitations=['No task closure; full-duration station acceptance remains pending',
            'No manufacturer valve data or experimental validation; equilibrium no-slip nozzle'])
    evidence.save(output/'report.json',report)
    if reference is not None:
        reference = Path(reference)
        files = {p.relative_to(output) for p in output.rglob('*.json')}
        if files!={p.relative_to(reference) for p in reference.rglob('*.json')}:
            raise ValueError('Evidence sets differ')
        for name in sorted(files):
            if evidence.read(output/name)!=evidence.read(reference/name):
                raise ValueError('Bypass replay differs:'+str(name))
    if not passed: raise ValueError('Bypass component study failed')
    print(json.dumps(dict(passed=True,closed_tasks=[])),flush=True)
    return report


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference',type=Path)
    args=parser.parse_args();run(args.output,args.reference)
