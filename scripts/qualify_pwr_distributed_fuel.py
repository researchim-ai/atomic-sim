"""Short source-bound PWR coupling with local fuel and decay heat owners."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
from atomic_model_station import PWRCoreSession,PWRStagedSession,CoreConfig
from atomic_model_training.spec import atomic_json
from atomic_model_multiphysics.provenance import seal
from coupling.pwr_fuel import DistributedFuelSession,StagedDistributedFuelSession,FuelGrid,identity

PRESETS={'coarse':dict(max_step_s=.05,rtol=2e-8),'fine':dict(max_step_s=.025,rtol=5e-9)}
CRITERIA=dict(energy_relative=1e-9,energy_abs_j=.1,mass_relative=1e-10,boron_relative=1e-10,
    source_identity_relative=1e-12,instantaneous_relative=1e-10,precursor_inventory_relative=1e-10,
    decay_exchange_abs_j=.1,refinement_relative=1e-4,uniform_limit_relative=2e-6)
ACTIONS=({}, {'load_fraction':.99}, {'scram':True})


def specifications():
    return dict(uniform=dict(core=CoreConfig(kind='diffusion',cells=4),fuel_grid=FuelGrid()),
        biased_three=dict(core=CoreConfig(kind='diffusion',cells=4,axial_bias_reactivity=1e-4),fuel_grid=FuelGrid()),
        biased_five=dict(core=CoreConfig(kind='diffusion',cells=4,axial_bias_reactivity=1e-4),fuel_grid=FuelGrid((0.,.2,.9,2.1,3.,4.))),
        staged_uniform=dict(core=CoreConfig(kind='diffusion',cells=4),fuel_grid=FuelGrid()))


def physical_vector(snapshot):
    return np.r_[snapshot['pwr_state'],snapshot['turbine_energy_j'],snapshot['clad_temperature_k'],
        snapshot['cooling_energy_j'],snapshot['cooling']['flow_kg_s'],snapshot['feed_inventory'],
        snapshot['electrical']['backup_energy_j'],snapshot.get('header_inventory',[])]


def local_vector(snapshot):
    f=snapshot['distributed_fuel']
    return np.r_[physical_vector(snapshot),f['temperature_k'],np.ravel(f['decay_stored_j']),f['decay_storage_exchange_j']]


def relative(first,second):
    return float(np.max(abs(np.asarray(first)-second)/np.maximum(abs(second),1.)))


def checks(snapshot):
    f=snapshot['distributed_fuel'];p=max(f['fission_w'],f['thermal_w'],1.)
    values=dict(global_energy=snapshot['energy_relative']<=CRITERIA['energy_relative'] and abs(snapshot['energy_residual_j'])<=CRITERIA['energy_abs_j'],
        mass=snapshot['mass_relative']<=CRITERIA['mass_relative'],boron=snapshot['boron_relative']<=CRITERIA['boron_relative'],
        instantaneous=abs(snapshot['instantaneous_energy_defect_w'])/p<=CRITERIA['instantaneous_relative'],
        nuclear_storage=abs(f['nuclear_storage_identity_residual_w'])/p<=CRITERIA['source_identity_relative'],
        fuel_transfer=abs(f['fuel_transfer_residual_w'])/p<=CRITERIA['source_identity_relative'],
        decay_integral=abs(f['decay_exchange_residual_j'])<=CRITERIA['decay_exchange_abs_j'],
        neutron_inventory=snapshot['core']['neutron_precursor_inventory']['relative_defect']<=CRITERIA['precursor_inventory_relative'],
        exchange=max(f['exchange']['power_relative_defect'],f['exchange']['pairing_relative_defect'])<=CRITERIA['source_identity_relative'],
        local_positive=min(np.ravel(f['decay_stored_j']))>=0 and min(snapshot['core']['owned_state'])>=0,
        local_temperature=all(300<=v<=2800 for v in f['temperature_k']))
    return {name:bool(value) for name,value in values.items()}


def run(output):
    out=Path(output).absolute()
    if out.exists() or any(p.is_symlink() for p in (out,*out.parents)):raise ValueError('Fresh ordinary output required')
    out.mkdir(parents=True);source=identity();driver_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    outcomes=[];science={}
    for name,spec in specifications().items():
        session_type=StagedDistributedFuelSession if name.startswith('staged_') else DistributedFuelSession
        trajectories={}
        for preset,options in PRESETS.items():
            session=session_type(**spec,**options);rows=[];saved=None
            for i,action in enumerate(ACTIONS):
                rows.append(session.advance(.05,action or None))
                if i==1:saved=json.loads(json.dumps(session.checkpoint()))
            restored=session_type.restore(saved)
            replay=restored.advance(.05,{'scram':True})
            before=session.checkpoint();invalid_refused=False
            try:session.advance(.01,{'load_fraction':99})
            except ValueError:invalid_refused=session.checkpoint()==before
            trajectories[preset]=dict(snapshots=rows,checks=[checks(r) for r in rows],
                restart_exact=replay==rows[-1] and restored.checkpoint()==before,
                invalid_action_rollback=invalid_refused,checkpoint=session.checkpoint())
        errors=[relative(local_vector(a),local_vector(b)) for a,b in zip(trajectories['coarse']['snapshots'],trajectories['fine']['snapshots'])]
        limit=None
        if name in ('uniform','staged_uniform'):
            baseline_type=PWRStagedSession if name.startswith('staged_') else PWRCoreSession
            baseline=baseline_type(core=spec['core'],**PRESETS['fine']);old=[]
            for action in ACTIONS:old.append(baseline.advance(.05,action or None))
            limit=max(relative(physical_vector(a),physical_vector(b)) for a,b in zip(trajectories['fine']['snapshots'],old))
        final=trajectories['fine']['snapshots'][-1]['distributed_fuel']
        gates=dict(all_budgets=all(all(c.values()) for t in trajectories.values() for c in t['checks']),
            restarts=all(t['restart_exact'] for t in trajectories.values()),
            invalid_actions_rollback=all(t['invalid_action_rollback'] for t in trajectories.values()),
            refined=max(errors)<=CRITERIA['refinement_relative'],
            uniform_limit=limit is None or limit<=CRITERIA['uniform_limit_relative'],
            scram_releases_heat=final['thermal_w']>final['fission_w'] and final['nuclear_storage_rate_w']<0)
        row=dict(case=name,backend=session_type.backend_kind,config=dict(core=asdict(spec['core']),fuel_grid=dict(faces_m=list(spec['fuel_grid'].faces_m))),
            gates=gates,passed=all(gates.values()),refinement_errors=errors,uniform_limit_error=limit,
            final_fission_w=final['fission_w'],final_thermal_w=final['thermal_w'],
            final_decay_storage_rate_w=final['nuclear_storage_rate_w'])
        outcomes.append(row);science[name]=dict(summary=row,trajectories=trajectories)
        atomic_json(out/(name+'.json'),seal(science[name]))
    report=dict(source=source,driver_sha256=driver_sha,criteria=CRITERIA,cases=outcomes,
        passed=all(r['passed'] for r in outcomes) and source==identity() and driver_sha==hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        native_trajectories=8,restart_suffixes=8,legacy_uniform_trajectories=2,
        local_decay_heat_connected=True,pwr_inventory_connected=True,distributed_fluid_channels=False,
        station_env_default_changed=False,station_env_adapter_available=False,experimental_validation=False,closed_tasks=[])
    atomic_json(out/'report.json',seal(report))
    atomic_json(out/'scientific.json',seal(dict(report=report,cases=science)))
    if not report['passed']:raise ValueError('PWR distributed-fuel qualification failed; all attempts retained')
    return dict(passed=True,cases=4,maximum_refinement_error=max(max(r['refinement_errors']) for r in outcomes))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    print(run(parser.parse_args().output))
