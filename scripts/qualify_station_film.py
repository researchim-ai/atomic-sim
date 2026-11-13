"""Flow-dependent single-phase station qualification; NPP-04.04 remains open."""
from argparse import ArgumentParser
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from atomic_model_station import PWRFilmSession
from atomic_model_station.film import film_signature, film_transfer
from atomic_model_multiphysics.provenance import sha
from scripts import qualify_station_actuators as evidence
from scripts.qualify_station_primary import vector

TASKS = []  # Progress on04.04 is not full boiling/CHF integration.
SETTINGS = [(1.,1e-7),(.5,2e-8),(.2,2e-9)]
CRITERIA = dict(energy_relative=1e-6, mass_relative=1e-10, boron_relative=1e-10,
                instantaneous_relative=1e-8, refinement_relative=1e-4,
                nominal_tail_relative=1e-5, frozen_reference_relative=1e-9)


def frozen_reference(session):
    """Independent conservative three-stock matrix at frozen material properties."""
    p=session.plant;point=p.network.volumes['core'].thermo
    capacities=np.array([p.fuel_capacity,session.config.clad_capacity_j_k,
                        p.network.volumes['core'].mass_kg*point.cp_j_kg_k])
    kf=p.fuel_clad_ua;kc=session.snapshot()['film']['conductance_w_k']
    conductance=np.array([[-kf,kf,0.],[kf,-kf-kc,kc],[0.,kc,-kc]])
    matrix=conductance/capacities[:,None]
    initial=np.array([p.fuel_temperature,p.clad_temperature_k,point.T_k])
    def rhs(t,y):
        probe=replace(point,T_k=float(y[2]))
        q=film_transfer(probe,p.network.junctions['core_out'].mass_flow_kg_s,
                        session.film_geometry,float(y[1]))['heat_w']
        f=kf*(y[0]-y[1])
        return np.array([-f,f-q,q])/capacities
    solution=solve_ivp(rhs,(0.,.1),initial,method='DOP853',rtol=1e-12,atol=1e-11,max_step=.001)
    if not solution.success:raise ValueError('Frozen thermal reference integration failed')
    exact=expm(matrix*.1)@initial
    observed=solution.y[:,-1]
    return dict(exact=exact.tolist(),observed=observed.tolist(),matrix=matrix.tolist(),
                relative_error=float(max(abs(observed-exact)/np.maximum(abs(exact),1.))),
                stored_energy_relative=float(abs(capacities@(observed-initial))/abs(capacities@initial)),
                scope='Isolated finite fuel/clad/bulk capacities, constant properties, no fission or advection')


def admit(r):
    f=r['film'];p=r['phase_inventory'];cfg=r['film_config']
    return bool(r['energy_relative']<=CRITERIA['energy_relative']
        and r['mass_relative']<=CRITERIA['mass_relative'] and r['boron_relative']<=CRITERIA['boron_relative']
        and abs(r['instantaneous_energy_defect_w'])/max(abs(r['powers']['nuclear_heat_w']),1.)<=CRITERIA['instantaneous_relative']
        and f['reynolds']>=10000 and .7<=f['prandtl']<=160 and f['heat_w']>=0
        and f['chf_w_m2'] is None and f['dnbr'] is None and not f['boiling_supported']
        and all(p[n]['region']==1 and p[n]['subcooling_k']>=cfg['minimum_subcooling_k'] for n in ('core','hot_leg','sg_primary','cold_leg'))
        and p['sg_secondary']['liquid_mass_kg']>0 and p['sg_secondary']['vapor_mass_kg']>0)


def run(output,reference=None):
    output=Path(output).absolute()
    if any(p.is_symlink() for p in (output,*output.parents)):raise ValueError('Symlink output')
    output.mkdir(parents=True,exist_ok=False)
    source=film_signature();driver=sha(__file__);serializer=sha(evidence.__file__)
    session=PWRFilmSession(max_step_s=1.);initial=session.snapshot()
    evidence.save(output/'protocol.json',dict(source=source,driver_sha256=driver,serializer_sha256=serializer,
        criteria=CRITERIA,settings=SETTINGS,parameters=session.parameter_registry(),initial=initial,
        candidate_tasks=TASKS,in_progress_tasks=['NPP-04.04'],nominal_duration_s=3600.,coastdown_duration_s=60.,
        geometry='Equivalent round tubes, same actual core volume and nominal UA; no changed initial stocks or measured geometry',
        scope='Heating-only subcooled primary film and finite equilibrium SG/pressurizer phases; no CHF/boiling admission'))
    print('film protocol sealed',flush=True)
    rows=[]
    for i in range(60):
        rows.append(session.advance(60.))
        if i%5==0:print('film nominal',i+1,'/60',flush=True)
    tail=float(max(abs(vector(rows[-1])-vector(rows[-3]))/np.maximum(abs(vector(rows[-3])),1.)))
    evidence.save(output/'nominal.json',dict(trace=rows,tail_relative_drift=tail,checkpoint=session.checkpoint()))
    trip=PWRFilmSession(max_step_s=1.);coast=[]
    for i in range(6):
        coast.append(trip.advance(10.,dict(scram=True,turbine_trip=True,primary_pump_on=False) if i==0 else None))
        print('film coastdown',i+1,'/6',flush=True)
    evidence.save(output/'coastdown.json',dict(trace=coast,checkpoint=trip.checkpoint()))
    trials=[]
    for step,rtol in SETTINGS:
        s=PWRFilmSession(max_step_s=step,rtol=rtol)
        r=s.advance(10.,dict(scram=True,turbine_trip=True,primary_pump_on=False))
        trials.append(dict(step_s=step,rtol=rtol,observation=r))
        print('film refinement',step,flush=True)
    fine=vector(trials[-1]['observation'])
    errors=[float(max(abs(vector(x['observation'])-fine)/np.maximum(abs(fine),1.))) for x in trials]
    frozen=frozen_reference(PWRFilmSession())
    evidence.save(output/'references.json',dict(trials=trials,refinement_errors=errors,frozen_reference=frozen))
    s=PWRFilmSession(max_step_s=.2);s.advance(.2,dict(scram=True,turbine_trip=True))
    saved=json.loads(json.dumps(s.checkpoint()));t=PWRFilmSession.restore(saved)
    resumed=s.advance(.2);restart=t.advance(.2)==resumed;before=s.checkpoint();rollback=False
    try:s.advance(.1,dict(scram=True,load_fraction=2.))
    except ValueError:rollback=s.checkpoint()==before
    evidence.save(output/'restart.json',dict(checkpoint=saved,resumed=resumed,exact=restart,rollback=rollback))
    observations=[initial,*rows,*coast,*[t['observation'] for t in trials],resumed]
    unchanged=source==film_signature() and driver==sha(__file__) and serializer==sha(evidence.__file__)
    ua_ratio=coast[-1]['film']['conductance_w_k']/initial['film']['conductance_w_k']
    passed=bool(unchanged and all(admit(r) for r in observations) and restart and rollback
        and tail<=CRITERIA['nominal_tail_relative'] and max(errors)<=CRITERIA['refinement_relative']
        and frozen['relative_error']<=CRITERIA['frozen_reference_relative']
        and frozen['stored_energy_relative']<=CRITERIA['frozen_reference_relative'] and 0<ua_ratio<1)
    report=dict(completed=True,passed=passed,source_unchanged=unchanged,closed_tasks=[],in_progress_tasks=['NPP-04.04'],
        nominal_duration_s=3600.,coastdown_duration_s=60.,nominal_tail_relative_drift=tail,
        coastdown_final_ua_ratio=ua_ratio,refinement_errors=errors,frozen_reference_relative_error=frozen['relative_error'],
        frozen_energy_relative=frozen['stored_energy_relative'],restart_exact=restart,rollback=rollback,
        max_energy_relative=max(r['energy_relative'] for r in observations),max_mass_relative=max(r['mass_relative'] for r in observations),
        max_boron_relative=max(r['boron_relative'] for r in observations),
        complete_package_b=False,complete_package_c=False,r1_accepted=False,validated=False,plant_referenced=False,
        full_long_checkpoint_replay=False,
        limitations=['No boiling, CHF/DNBR, dryout, post-CHF or rod-bundle applicability claim',
            'Equivalent diameter fits previous nominal UA at fixed volume; not independent physical calibration',
            'Only turbulent heating admitted; low-Re transitional/natural circulation can leave this film scope',
            'Phase reserve checks are algebraic inventory admission, not resolved interfaces or phase-front tracking',
            'Long nominal uses point core; no experimental plant validation'])
    evidence.save(output/'report.json',report)
    if reference is not None:
        reference=Path(reference);files={f.relative_to(output) for f in output.rglob('*.json')}
        if files!={f.relative_to(reference) for f in reference.rglob('*.json')}:raise ValueError('Evidence sets differ')
        for name in sorted(files):
            if evidence.read(output/name)!=evidence.read(reference/name):raise ValueError('Installed film evidence differs:'+str(name))
        print('All installed station film evidence matches exactly',flush=True)
    if not passed:raise ValueError('Film qualification failed; NPP-04.04 not accepted')
    print(json.dumps(dict(passed=True,closed_tasks=[])),flush=True)
    return report


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference',type=Path);args=parser.parse_args();run(args.output,args.reference)
