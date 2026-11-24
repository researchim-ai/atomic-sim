"""Conditional fuel/cladding mesh refinement; fixed neutron and fluid meshes.

Constant length-weighted capacities and UA are preserved, not refitted.
No continuum-error estimate or experimental validation is supplied.
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np
from atomic_model_multiphysics.provenance import seal
from atomic_model_training.spec import atomic_json
from atomic_model_station import CoreConfig
from coupling import mesh_comparison, thermal_grid
from coupling.conservative import vector
from coupling.mesh_comparison import profile_error
from coupling.thermal_grid import aligned_fuel_grid
from coupling.pwr_moderator import LocalModeratorPWRSession, identity
from scripts import qualify_moderator_transients as transient_driver
from scripts import qualify_pwr_distributed_fuel as budget_driver

MESHES = (4, 8, 16)
FIELDS = ('fuel_temperature', 'cladding_temperature', 'decay0', 'decay1', 'decay2',
          'fission', 'thermal_source', 'fuel_to_cladding', 'cladding_to_fluid')
LIMITS = {k: 1e-6 for k in FIELDS}
CRITERIA = dict(temporal_relative=1e-4,
                total_power_relative=1e-6, coefficient_relative=1e-12,
                local_balance_relative=1e-10, map_relative=1e-12,
                minimum_temperature_span_k=1.)


def grid(cells):
    if type(cells) is not int or cells not in MESHES:
        raise ValueError('Explicit aligned thermal mesh 4,8,16 required')
    return aligned_fuel_grid((0.,1.,2.,3.,4.),(0.,1.,2.,4.),subdivisions=cells//4)


def source():
    return dict(moderator=identity(),
        helpers_sha256={m.__name__:hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
                        for m in (transient_driver,budget_driver,mesh_comparison,thermal_grid)},
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def measures(session):
    return dict(fuel_capacity_j_k=session.fuel_capacities.tolist(),
        cladding_capacity_j_k=session.cladding_capacities.tolist(),
        fuel_cladding_ua_w_k=session.fuel_conductances.tolist(),
        cladding_fluid_ua_w_k=session.coolant_conductances.tolist())


def fields(row):
    n=len(row['fuel_grid']['faces_m'])-1
    f=row['distributed_fuel'];c=row['distributed_cladding']
    decay=np.asarray(f['decay_stored_j'])
    if decay.shape!=(3,n):raise ValueError('Three complete spatial decay-heat reservoirs required')
    values=dict(fuel_temperature=f['temperature_k'],cladding_temperature=c['temperature_k'],
        **{f'decay{i}':decay[i] for i in range(3)},fission=f['fission_by_fuel_cell_w'],
        thermal_source=f['thermal_source_by_cell_w'],fuel_to_cladding=f['fuel_to_clad_by_cell_w'],
        cladding_to_fluid=c['clad_to_coolant_by_cell_w'])
    return {k:vector(v,n,positive=k.endswith('temperature')) for k,v in values.items()}


def compare_snapshots(left,right,scales):
    if set(scales)!=set(FIELDS):raise ValueError('All nine thermal profile scales required')
    if left['core_config']!=right['core_config'] or left['fluid_grid']!=right['fluid_grid']:
        raise ValueError('Neutron and fluid meshes and coefficients must remain fixed')
    a,b=fields(left),fields(right)
    return {k:profile_error(left['fuel_grid']['faces_m'],a[k],right['fuel_grid']['faces_m'],b[k],
        kind='cell_average' if k.endswith('temperature') else 'cell_integral',scale=scales[k]) for k in FIELDS}


def coefficient_defect(trajectories):
    """Check per-length laws, not just totals that could hide redistribution."""
    reference=trajectories['16_fine'];width=np.diff(reference['initial']['fuel_grid']['faces_m'])
    density={k:vector(v,len(width),positive=True)/width for k,v in reference['measures'].items()}
    if set(density)!={'fuel_capacity_j_k','cladding_capacity_j_k','fuel_cladding_ua_w_k','cladding_fluid_ua_w_k'}:
        raise ValueError('All four capacity/conductance measures required')
    maximum=0.
    for t in trajectories.values():
        w=np.diff(t['initial']['fuel_grid']['faces_m'])
        if set(t['measures'])!=set(density):raise ValueError('Matching capacity/conductance measures required')
        for k,v in t['measures'].items():
            per_length=vector(v,len(w),positive=True)/w
            maximum=max(maximum,float(max(abs(per_length/density[k][0]-1.))),
                float(abs(sum(v)/sum(reference['measures'][k])-1.)))
    return maximum


def thermal_audit(row,coefficients):
    """Recompute cell energy and pair heat identities from physical outputs."""
    data=fields(row);n=len(data['fuel_temperature']);f=row['distributed_fuel'];c=row['distributed_cladding'];fluid=row['distributed_fluid']
    cf=vector(coefficients['fuel_capacity_j_k'],n,positive=True)
    cc=vector(coefficients['cladding_capacity_j_k'],n,positive=True)
    fuel_rate=vector(f['fuel_temperature_rate_k_s'],n);clad_rate=vector(c['temperature_rate_k_s'],n)
    incoming=data['fuel_to_cladding'];outgoing=data['cladding_to_fluid'];thermal=data['thermal_source']
    pairs=np.asarray(fluid['pair_heat_w'],dtype=float)
    if pairs.shape!=(len(fluid['volume_m3']),n) or not np.isfinite(pairs).all():raise ValueError('Complete finite heat-pair matrix required')
    scale=max(float(sum(abs(thermal))),float(sum(abs(incoming))),float(sum(abs(outgoing))),1.)
    defects=np.r_[cf*fuel_rate-thermal+incoming,cc*clad_rate-incoming+outgoing,
        vector(c['fuel_to_clad_by_cell_w'],n)-incoming,pairs.sum(axis=0)-outgoing,
        pairs.sum(axis=1)-vector(fluid['coolant_heat_by_cell_w'],len(pairs))]
    entropy=float(incoming@(1/data['cladding_temperature']-1/data['fuel_temperature'])+fluid['contact_entropy_w_k'])
    stored=float(cc@data['cladding_temperature'])
    recorded_entropy=vector(c['conductive_entropy_generation_w_k'],n)
    result=dict(local_balance=float(max(abs(defects)))/scale<=CRITERIA['local_balance_relative'],
        cladding_storage=abs(stored-c['stored_energy_j'])/max(abs(stored),1.)<=CRITERIA['local_balance_relative'],
        nonnegative_entropy=bool(np.all(recorded_entropy>=0.) and fluid['contact_entropy_w_k']>=0. and entropy>=0.),
        entropy_identity=abs(entropy-sum(recorded_entropy))/max(abs(entropy),1.)<=CRITERIA['local_balance_relative'])
    return {k:bool(v) for k,v in result.items()}


def assess(trajectories):
    if type(trajectories) is not dict or set(trajectories)!={f'{n}_{p}' for n in MESHES for p in transient_driver.PRESETS}:
        raise ValueError('Complete 4/8/16 mesh by two temporal presets required')
    reference=trajectories['16_fine']['initial'];p0=float(reference['powers']['nuclear_fission_w']);length=4.
    scales={k:p0/length for k in FIELDS};scales.update(fuel_temperature=1000.,cladding_temperature=600.)
    scales.update({f'decay{i}':float(sum(row))/length for i,row in enumerate(reference['distributed_fuel']['decay_stored_j'])})
    gates=dict(budgets=True,thermal_balances=True,commands=True,timeline=True,fixed_meshes=True,
               nested_thermal_meshes=True,stationary=True,reference_frozen=True,map_budgets=True,rollback=True,arrivals=True)
    temporal=[];noise_rows=[];span=0.
    for key,t in trajectories.items():
        initial=t['initial'];rows=t['snapshots'];n=int(key.split('_')[0])
        steps=transient_driver.protocol('insertion',float(initial['actuator_status']['rod_position']))
        if len(rows)!=len(steps):raise ValueError('Complete observation timeline required')
        gates['commands'] &= t['steps']==steps
        gates['timeline'] &= all(abs(r['time_s']-v)<1e-12 for r,v in zip(rows,np.cumsum([s['duration_s'] for s in steps])))
        for r in [initial,*rows]:
            gates['fixed_meshes'] &= r['core_config']==reference['core_config'] and r['fluid_grid']==reference['fluid_grid']
            gates['nested_thermal_meshes'] &= r['fuel_grid']['faces_m']==list(grid(n).faces_m)
            gates['budgets'] &= all(budget_driver.checks(r).values())
            gates['thermal_balances'] &= all(thermal_audit(r,t['measures']).values())
            gates['reference_frozen'] &= r['local_moderator']['reference']==initial['local_moderator']['reference']
            gates['map_budgets'] &= max(r['local_moderator']['mass_relative_defect'],r['local_moderator']['boron_relative_defect'])<=CRITERIA['map_relative']
            span=max(span,float(np.ptp(r['distributed_fuel']['temperature_k'])))
        s=initial['local_moderator_stationarity']
        gates['stationary'] &= s['constrained_rank']==s['constrained_unknowns'] and s['max_selected_tolerance_ratio']<=1.
        gates['rollback'] &= t['invalid_command_rollback'] is True
        arrivals=[e for e in rows[-1]['events'] if e['kind']=='rod_arrived']
        gates['arrivals'] &= len(arrivals)==1 and abs(arrivals[0]['time_s']-1.)<1e-12
        if key.endswith('_coarse'):
            fine=trajectories[key.replace('_coarse','_fine')]
            for a,b in zip([initial,*rows],[fine['initial'],*fine['snapshots']]):
                temporal.append(budget_driver.relative(a['coupled_state'],np.asarray(b['coupled_state'])))
                noise_rows.append(compare_snapshots(a,b,scales))
    pairs={}
    for a,b in ((4,8),(8,16),(4,16)):
        left,right=(trajectories[f'{n}_fine'] for n in (a,b))
        observations=list(zip([left['initial'],*left['snapshots']],[right['initial'],*right['snapshots']]))
        profiles=[compare_snapshots(x,y,scales) for x,y in observations]
        pairs[f'{a}_{b}']=dict(profiles=profiles,maximum_l2={k:max(p[k]['normalized_l2'] for p in profiles) for k in FIELDS},
            maximum_total_power_relative=max(abs(x['powers']['nuclear_fission_w']-y['powers']['nuclear_fission_w'])/p0 for x,y in observations))
    noise={k:max(r[k]['normalized_l2'] for r in noise_rows) for k in FIELDS}
    defect=coefficient_defect(trajectories)
    gates.update(coefficients_preserved=defect<=CRITERIA['coefficient_relative'],
        matched_initial_power=all(abs(t['initial']['powers']['nuclear_fission_w']-p0)/p0<=1e-8 for t in trajectories.values()),
        temporal=max(temporal)<=CRITERIA['temporal_relative'],nonuniform=span>=CRITERIA['minimum_temperature_span_k'],
        aligned_replication=all(pair['maximum_l2'][k]<=LIMITS[k] for pair in pairs.values() for k in FIELDS),
        total_power=all(pair['maximum_total_power_relative']<=CRITERIA['total_power_relative'] for pair in pairs.values()),
        restart=trajectories['16_fine']['restart_exact'] is True)
    return dict(passed=all(gates.values()),gates={k:bool(v) for k,v in gates.items()},scales=scales,pairs=pairs,
        temporal_maximum_relative=max(temporal),temporal_profile_maximum_l2=noise,coefficient_relative_defect=defect,
        maximum_fuel_temperature_span_k=span,initial_rod_positions={k:t['initial']['actuator_status']['rod_position'] for k,t in trajectories.items()})


def run(output):
    out=Path(output).absolute()
    if out.exists() or any(p.is_symlink() for p in (out,*out.parents)):raise ValueError('Fresh ordinary output required')
    out.mkdir(parents=True);frozen=source();trajectories={}
    for cells in MESHES:
        for preset,options in transient_driver.PRESETS.items():
            key=f'{cells}_{preset}';print('trajectory',key,flush=True)
            session=LocalModeratorPWRSession(core=CoreConfig(kind='diffusion',cells=4),fuel_grid=grid(cells),**options)
            coefficients=measures(session);steps=transient_driver.protocol('insertion',float(session.state[12]))
            trajectories[key]=dict(transient_driver._run(session,steps,restart=key=='16_fine'),measures=coefficients)
            atomic_json(out/(key+'.json'),seal(trajectories[key]))
    assessment=assess(trajectories)
    report=dict(source=frozen,criteria=CRITERIA,profile_limits=LIMITS,assessment=assessment,
        passed=assessment['passed'] and frozen==source(),trajectories=6,restart_suffixes=1,horizon_s=2.,
        thermal_cells=list(MESHES),neutron_cells=4,fluid_faces_m=[0.,1.,2.,4.],
        qualification_scope='Aligned thermal replication limit under local constant-per-length laws; not generic mesh convergence',
        generic_thermal_mesh_convergence=False,
        initial_states='Separate stationary solves with the same requested power and initial inventory anchors; rod trims reported',
        command_scope='Same +0.01 inserted-fraction increment from each stationary rod trim',
        continuum_error_estimated=False,full_multiphysics_spatial_convergence=False,parameters_fit=False,
        experimental_validation=False,long_horizon_accepted=False,station_env_connected=False,closed_tasks=[])
    atomic_json(out/'report.json',seal(report));atomic_json(out/'scientific.json',seal(dict(report=report,trajectories=trajectories)))
    if not report['passed']:raise ValueError('Conditional thermal-mesh qualification failed; attempts retained')
    return dict(passed=True,temporal=assessment['temporal_maximum_relative'],spatial=pairs_summary(assessment))


def pairs_summary(assessment):
    return {k:v['maximum_l2'] for k,v in assessment['pairs'].items()}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    print(run(parser.parse_args().output))
