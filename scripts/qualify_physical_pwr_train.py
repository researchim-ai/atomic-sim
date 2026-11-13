"""Physical reduced-PWR/cladding/train hot-domain qualification; partial B only."""
from argparse import ArgumentParser
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from atomic_model_station.pwr_train import PWRTrainSession,TrainConfig,physical_signature
from atomic_model_multiphysics.provenance import seal,sha
from atomic_model_training.spec import atomic_json

CANDIDATE_TASKS=['NPP-03.01','NPP-03.02','NPP-04.01','NPP-04.02','NPP-05.02','NPP-06.01','NPP-06.02','NPP-06.03']
CRITERIA=dict(energy_relative=1e-6,mass_relative=1e-10,boron_relative=1e-10,
              refinement_relative=1e-4,instantaneous_energy_relative=1e-8,nominal_relative_drift=1e-5)


def vector(row):return np.r_[row['pwr_state'],row['turbine_energy_j'],row['clad_temperature_k']]


def admit(row):
    return (row['energy_relative']<=CRITERIA['energy_relative'] and row['mass_relative']<=CRITERIA['mass_relative']
            and row['boron_relative']<=CRITERIA['boron_relative']
            and abs(row['instantaneous_energy_defect_w'])/max(abs(row['powers']['nuclear_heat_w']),1.)<=CRITERIA['instantaneous_energy_relative'])


def run(output):
    output=Path(output).absolute()
    if any(p.is_symlink() for p in (output,*output.parents)):raise ValueError('Symlink output')
    source=physical_signature();driver=sha(__file__);output.mkdir(parents=True,exist_ok=False)
    model=PWRTrainSession(max_step_s=1.);initial=model.snapshot()
    atomic_json(output/'protocol.json',seal(dict(source=source,driver_sha256=driver,config=asdict(model.config),parameters=model.parameter_registry(),
        charging=model.charging,criteria=CRITERIA,nominal_duration_s=3600.,load_step_duration_s=2.,
        integration='Coupled Radau with domain-admitted fixed-probe Jacobian and exact known-mode elimination; independent turbine kinetic energy and cladding temperature',
        refinement=[dict(max_step_s=.2,rtol=1e-7),dict(max_step_s=.1,rtol=2e-8),dict(max_step_s=.05,rtol=2e-9)],
        initial=initial,candidate_task_scope='Hot reduced-PWR connection; no full B or integrated R1 admission',
        auxiliary_bus='External grid remains available to ideal-efficiency auxiliary drives even when generator disconnected',
        turbine_scope='Single equivalent saturated-steam expansion; no MSR or manufacturer moisture qualification')))
    print('physical protocol sealed',flush=True)
    results=[]
    for step,tol in ((.2,1e-7),(.1,2e-8),(.05,2e-9)):
        s=PWRTrainSession(max_step_s=step,rtol=tol);row=s.advance(2.,{'load_fraction':.9});results.append((s,row))
        print('load refinement',step,flush=True)
    reference=vector(results[-1][1]);differences=[float(np.max(abs(vector(row)-reference)/np.maximum(abs(reference),1.))) for _,row in results]
    refined=all(x<=CRITERIA['refinement_relative'] for x in differences)
    open_loop=PWRTrainSession(max_step_s=.2);unregulated=open_loop.advance(2.,{'load_fraction':.9,'governor_enabled':False})
    regulated=results[0][1]
    causal=(regulated['effective_valve']<unregulated['effective_valve'] and regulated['steam_flow_kg_s']<unregulated['steam_flow_kg_s']
            and regulated['turbine_speed_rad_s']<unregulated['turbine_speed_rad_s'] and regulated['steam_generator_pressure_pa']>unregulated['steam_generator_pressure_pa'])
    atomic_json(output/'load.json',seal(dict(refinements=[dict(max_step_s=s.max_step_s,rtol=s.rtol,observation=row,error=err) for (s,row),err in zip(results,differences,strict=True)],
        counterfactual=unregulated,causal_governor_effect=causal,checkpoint=results[0][0].checkpoint())))
    off=PWRTrainSession(max_step_s=.1);disconnected=off.advance(.2,{'generator_connected':False})
    saved=json.loads(json.dumps(off.checkpoint()));restored=PWRTrainSession.restore(saved)
    exact=restored.advance(.1)==off.advance(.1)
    atomic_json(output/'switch.json',seal(dict(observation=disconnected,checkpoint=saved,restart_exact=exact)))
    limited=PWRTrainSession(TrainConfig(inertia_kg_m2=1e5));before=limited.checkpoint();rollback=False
    try:limited.advance(1.,{'generator_connected':False,'governor_enabled':False})
    except ValueError:rollback=limited.checkpoint()==before
    rows=[]
    for i in range(60):
        rows.append(model.advance(60.))
        if i%5==0:print('physical nominal',i+1,'/60',flush=True)
    last=rows[-1];drift=float(np.max(abs(vector(last)-vector(initial))/np.maximum(abs(vector(initial)),1.)))
    atomic_json(output/'nominal.json',seal(dict(trace=rows,relative_drift=drift,checkpoint=model.checkpoint())))
    sg=model.plant;split=sg.sg_wall.power_split_w(sg.network.volumes['sg_primary'].temperature_k,sg.network.volumes['sg_secondary'].temperature_k)
    all_admitted=all(admit(row) for row in [initial,unregulated,disconnected,*rows,*[r for _,r in results]])
    checks={
        'NPP-03.01':last['reactor_connected'] and len(sg.kinetics_state)==7 and len(sg.decay_state)==3 and last['powers']['nuclear_heat_w']>0,
        'NPP-03.02':last['clad_temperature_k']>300 and sg.train_config.clad_capacity_j_k>0 and all_admitted,
        'NPP-04.01':all_admitted and all(name in sg.network.volumes for name in ('core','cold_leg','hot_leg','sg_primary','pressurizer')),
        'NPP-04.02':abs(sum(split))/max(abs(split[0]),1.)<1e-12 and sg.sg_wall_capacity>0,
        'NPP-05.02':causal and regulated['expansion']['outlet_entropy_j_kg_k']>=regulated['expansion']['inlet_entropy_j_kg_k'],
        'NPP-06.01':model.config.inertia_kg_m2!=sg.inertia and regulated['turbine_energy_j']>0 and all_admitted,
        'NPP-06.02':disconnected['powers']['gross_electrical_w']==0 and disconnected['powers']['net_export_w']<0,
        'NPP-06.03':all_admitted and all(abs(row['powers']['gross_electrical_w']-row['powers']['own_needs_w']-row['powers']['net_export_w'])<1e-6 for row in rows),
    }
    checks={name:bool(value) for name,value in checks.items()}
    unchanged=source==physical_signature() and driver==sha(__file__)
    passed=unchanged and all_admitted and refined and causal and exact and rollback and drift<CRITERIA['nominal_relative_drift'] and all(checks.values())
    report=dict(completed=True,passed=passed,source_unchanged=unchanged,task_checks=checks,closed_tasks=CANDIDATE_TASKS if passed else [],
        nominal_duration_s=last['time_s'],nominal_relative_drift=drift,
        max_energy_relative=max(row['energy_relative'] for row in rows),max_mass_relative=max(row['mass_relative'] for row in rows),
        max_boron_relative=max(row['boron_relative'] for row in rows),load_refinement_errors=differences,
        governor_causal_effect=causal,restart_exact=exact,limit_rollback_exact=rollback,
        gross_nominal_w=initial['powers']['gross_electrical_w'],net_nominal_w=initial['powers']['net_export_w'],
        nominal_exhaust_quality=initial['expansion']['outlet_quality'],
        complete_package_b=False,r1_accepted=False,validated=False,plant_referenced=False,
        limitations=['Hot forward-flow reduced configuration only; no cold start or independent bank control',
                     'No steam-line inventory, nozzle/choking map or moisture separation/reheat',
                     'Wet thermodynamic exhaust is not turbine hardware qualification',
                     'Auxiliary drives remain supplied by an ideal external electrical bus',
                     'Activation of an eliminated spray source is refused; corresponding hybrid branch is not yet qualified',
                     'New cladding resistance split and capacity are illustrative; no CHF/subcooled boiling model',
                     'No new loss-of-feed, coastdown, natural-circulation or full SCRAM qualification'])
    atomic_json(output/'report.json',seal(report));return report


if __name__=='__main__':
    p=ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();run(args.output)
