"""Joint neutron/fuel/cladding/liquid mesh study for a two-second PWR case.

Same code family, synthetic data and fixed-reference kinetic closure.
No continuum-error estimate, experimental or long-duration validation.
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np
from atomic_model_multiphysics.provenance import seal
from atomic_model_training.spec import atomic_json
from coupling import refinement,mesh_comparison
from coupling.conservative import vector
from coupling.mesh_comparison import profile_error
from coupling.refinement import coupled_pwr_mesh
from coupling.pwr_moderator import LocalModeratorPWRSession,identity
from scripts import qualify_moderator_transients as transient
from scripts import qualify_pwr_distributed_fuel as budget
from scripts import qualify_fluid_geometry as fluid
from scripts import qualify_thermal_mesh as thermal

MESHES=(2,4,8)
FIELDS=tuple('fluid.'+k for k in fluid.FIELDS)+tuple('thermal.'+k for k in thermal.FIELDS)+('neutron.fission',)+tuple(f'neutron.precursor{i}' for i in range(6))
LIMITS={k:(fluid.LIMITS[k.split('.')[1]] if k.startswith('fluid.') else .01 if k.endswith('temperature') else .05) for k in FIELDS}
CRITERIA=dict(temporal_relative=1e-4,contraction=.9,numerical_margin=20.,
    total_power_relative=1e-3,coefficient_relative=1e-12,map_relative=1e-12,
    minimum_flux_span=.01,minimum_fluid_temperature_span_k=1.)


def source():
    return dict(moderator=identity(),helpers_sha256={m.__name__:hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
        for m in (refinement,mesh_comparison,transient,budget,fluid,thermal)},
        thermal_grid_sha256=hashlib.sha256(Path(thermal.thermal_grid.__file__).read_bytes()).hexdigest(),
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def configurations(row):
    c=dict(row['core_config']);c.pop('cells')
    geometry=row['momentum_geometry']
    return dict(core=c,loss=row['fluid_grid']['internal_loss_coefficient'],
        momentum=row['fluid_grid']['momentum_geometry'],area=geometry['dual']['area_m2'],
        channel_reference_density=geometry['channel_reference_density_kg_m3'],
        legacy={name:{key:value for key,value in data.items() if key.startswith('legacy_')}
                for name,data in geometry['boundaries'].items()})


def profiles(row):
    f=fluid.fields(row);t=thermal.fields(row);c=row['core_config'];n=c['cells']
    result={}
    for key,value in f.items():result['fluid.'+key]=(row['fluid_grid']['faces_m'],value,'cell_integral' if key in ('mass','energy','boron') else 'cell_average')
    for key,value in t.items():result['thermal.'+key]=(row['fuel_grid']['faces_m'],value,'cell_average' if key.endswith('temperature') else 'cell_integral')
    faces=np.linspace(0.,c['length_m'],n+1)
    result['neutron.fission']=(faces,vector(row['core']['fission_by_cell_w'],n),'cell_integral')
    precursors=np.asarray(row['core']['local_precursors'])
    if precursors.shape!=(6,n):raise ValueError('All six local precursor families required')
    for i in range(6):result[f'neutron.precursor{i}']=(faces,vector(precursors[i],n),'cell_average')
    return result


def compare_snapshots(left,right,scales):
    if set(scales)!=set(FIELDS):raise ValueError('All22 coupled profile scales required')
    if configurations(left)!=configurations(right):raise ValueError('Only mesh resolution may change')
    if left['core_config']['kind']!='diffusion' or left['fluid_grid']['momentum_geometry']!='complete_dual':
        raise ValueError('Spatial neutron and complete dual fluid configuration required')
    a,b=profiles(left),profiles(right)
    return {key:profile_error(a[key][0],a[key][1],b[key][0],b[key][1],kind=a[key][2],scale=scales[key]) for key in FIELDS}


def scales_for(row):
    length=row['core_config']['length_m'];power=float(row['powers']['nuclear_fission_w'])
    result={k:power/length for k in FIELDS}
    result.update({'thermal.fuel_temperature':1000.,'thermal.cladding_temperature':600.,
        'fluid.temperature':600.,'fluid.pressure':15.5e6,'fluid.density':1000.})
    inventory=np.asarray(row['distributed_fluid']['inventory_by_cell'])
    for i,name in enumerate(('mass','energy','boron')):result['fluid.'+name]=float(inventory[:,i].sum()/length)
    for i,values in enumerate(row['distributed_fuel']['decay_stored_j']):result[f'thermal.decay{i}']=float(sum(values)/length)
    for i,values in enumerate(row['core']['local_precursors']):result[f'neutron.precursor{i}']=float(np.mean(values))
    return result


def face_coefficient_defect(row,actual):
    """Verify local partitions as well as the total invariants from v135."""
    m=row['momentum_geometry'];dual=m['dual'];a=dual['area_m2'];rho=m['channel_reference_density_kg_m3']
    i=np.asarray(dual['inertance_m_inv']);k=np.asarray(dual['loss_coefficients'])/a**2
    inlet=m['boundaries']['pump'];outlet=m['boundaries']['core_out']
    expected=dict(inertance_m_inv=np.r_[inlet['total_inertance_m_inv'],i[1:-1],outlet['total_inertance_m_inv']],
        kinetic_coefficients_m2_kg=np.r_[inlet['kinetic_coefficient_m2_kg'],i[1:-1]/rho,outlet['kinetic_coefficient_m2_kg']],
        loss_per_m4=np.r_[inlet['legacy_loss_per_m4']+k[0],k[1:-1],outlet['legacy_loss_per_m4']+k[-1]])
    return max(float(max(abs(vector(actual[key],len(value))-value)/np.maximum(abs(value),1e-30))) for key,value in expected.items())


def assess(trajectories):
    if type(trajectories) is not dict or set(trajectories)!={f'{n}_{p}' for n in MESHES for p in transient.PRESETS}:
        raise ValueError('Complete coupled2/4/8 by two temporal presets required')
    ref=trajectories['8_fine']['initial'];scales=scales_for(ref);p0=float(ref['powers']['nuclear_fission_w'])
    gates=dict(configuration=True,commands=True,timeline=True,budgets=True,thermal_balances=True,
        geometry=True,reference_frozen=True,inventory_maps=True,stationary=True,rollback=True,arrivals=True)
    temporal_errors=[];noise_profiles=[];geometries=[];partition=[];flux_span=temperature_span=0.
    coefficient_density={k:np.asarray(v)/np.diff(ref['fuel_grid']['faces_m']) for k,v in trajectories['8_fine']['thermal_measures'].items()}
    coefficient_defects=[]
    for key,t in trajectories.items():
        n=int(key.split('_')[0]);initial=t['initial'];rows=t['snapshots'];recipe=coupled_pwr_mesh(n)
        steps=transient.protocol('insertion',float(initial['actuator_status']['rod_position']))
        if len(rows)!=len(steps):raise ValueError('Complete observation timeline required')
        gates['commands'] &= t['steps']==steps
        gates['timeline'] &= all(abs(r['time_s']-time)<1e-12 for r,time in zip(rows,np.cumsum([s['duration_s'] for s in steps])))
        widths=np.diff(initial['fuel_grid']['faces_m'])
        if set(t['thermal_measures'])!=set(coefficient_density):raise ValueError('All four thermal measures required')
        for k,v in t['thermal_measures'].items():
            coefficient_defects.append(float(max(abs(vector(v,len(widths),positive=True)/widths/coefficient_density[k][0]-1.))))
        for r in [initial,*rows]:
            gates['configuration'] &= configurations(r)==configurations(ref) and r['core_config']['cells']==n and r['fluid_grid']['faces_m']==list(recipe['fluid_grid'].faces_m) and r['fuel_grid']['faces_m']==list(recipe['fuel_grid'].faces_m)
            gates['budgets'] &= all(budget.checks(r).values())
            gates['thermal_balances'] &= all(thermal.thermal_audit(r,t['thermal_measures']).values())
            g=fluid.audit_geometry(r,t['hydraulic_coefficients']);geometries.append(g);gates['geometry'] &= g['passed']
            partition.append(face_coefficient_defect(r,t['hydraulic_coefficients']))
            gates['reference_frozen'] &= r['local_moderator']['reference']==initial['local_moderator']['reference']
            gates['inventory_maps'] &= max(r['local_moderator']['mass_relative_defect'],r['local_moderator']['boron_relative_defect'])<=CRITERIA['map_relative']
            flux_span=max(flux_span,float(np.ptp(r['core']['relative_flux'])))
            temperature_span=max(temperature_span,float(np.ptp(r['distributed_fluid']['temperature_k'])))
        s=initial['local_moderator_stationarity'];gates['stationary'] &= s['constrained_rank']==s['constrained_unknowns'] and s['max_selected_tolerance_ratio']<=1.
        gates['rollback'] &= t['invalid_command_rollback'] is True
        events=[e for e in rows[-1]['events'] if e['kind']=='rod_arrived'];gates['arrivals'] &= len(events)==1 and abs(events[0]['time_s']-1.)<1e-12
        if key.endswith('_coarse'):
            fine=trajectories[key.replace('_coarse','_fine')]
            for a,b in zip([initial,*rows],[fine['initial'],*fine['snapshots']]):
                temporal_errors.append(budget.relative(a['coupled_state'],np.asarray(b['coupled_state'])))
                noise_profiles.append(compare_snapshots(a,b,scales))
    pairs={}
    for a,b in ((2,4),(4,8),(2,8)):
        left,right=(trajectories[f'{n}_fine'] for n in (a,b))
        observations=list(zip([left['initial'],*left['snapshots']],[right['initial'],*right['snapshots']]))
        compared=[compare_snapshots(x,y,scales) for x,y in observations]
        pairs[f'{a}_{b}']=dict(profiles=compared,maximum_l2={k:max(p[k]['normalized_l2'] for p in compared) for k in FIELDS},
            maximum_total_power_relative=max(abs(x['powers']['nuclear_fission_w']-y['powers']['nuclear_fission_w'])/p0 for x,y in observations))
    coarse=pairs['2_4']['maximum_l2'];fine=pairs['4_8']['maximum_l2'];noise={k:max(r[k]['normalized_l2'] for r in noise_profiles) for k in FIELDS}
    gates.update(thermal_laws_fixed=max(coefficient_defects)<=CRITERIA['coefficient_relative'],face_partitions=max(partition)<=CRITERIA['coefficient_relative'],
        matched_initial_power=all(abs(t['initial']['powers']['nuclear_fission_w']-p0)/p0<=1e-8 for t in trajectories.values()),
        temporal=max(temporal_errors)<=CRITERIA['temporal_relative'],
        nonuniform=flux_span>=CRITERIA['minimum_flux_span'] and temperature_span>=CRITERIA['minimum_fluid_temperature_span_k'],
        spatial_threshold=all(fine[k]<=LIMITS[k] for k in FIELDS),spatial_contraction=all(fine[k]<=CRITERIA['contraction']*coarse[k] for k in FIELDS),
        spatial_effect_resolved=all(fine[k]>=CRITERIA['numerical_margin']*noise[k] for k in FIELDS),
        total_power=pairs['4_8']['maximum_total_power_relative']<=CRITERIA['total_power_relative'],restart=trajectories['8_fine']['restart_exact'] is True)
    return dict(passed=all(gates.values()),gates={k:bool(v) for k,v in gates.items()},scales=scales,pairs=pairs,
        temporal_maximum_relative=max(temporal_errors),temporal_profile_maximum_l2=noise,
        thermal_coefficient_defect=max(coefficient_defects),face_partition_defect=max(partition),
        geometry_maximum_relative=max(g['geometry_relative_defect'] for g in geometries),
        momentum_maximum_abs_pa=max(g['momentum_abs_pa'] for g in geometries),
        initial_rod_positions={k:t['initial']['actuator_status']['rod_position'] for k,t in trajectories.items()})


def run(output):
    out=Path(output).absolute()
    if out.exists() or any(p.is_symlink() for p in (out,*out.parents)):raise ValueError('Fresh ordinary output required')
    out.mkdir(parents=True);frozen=source();trajectories={}
    for n in MESHES:
        for preset,options in transient.PRESETS.items():
            key=f'{n}_{preset}';print('trajectory',key,flush=True)
            s=LocalModeratorPWRSession(**coupled_pwr_mesh(n),**options)
            metadata=dict(thermal_measures=thermal.measures(s),hydraulic_coefficients=fluid.coefficients(s))
            steps=transient.protocol('insertion',float(s.state[12]))
            trajectories[key]=dict(transient._run(s,steps,restart=key=='8_fine'),**metadata)
            atomic_json(out/(key+'.json'),seal(trajectories[key]))
    assessment=assess(trajectories)
    report=dict(source=frozen,criteria=CRITERIA,profile_limits=LIMITS,assessment=assessment,
        passed=assessment['passed'] and frozen==source(),trajectories=6,restart_suffixes=1,horizon_s=2.,
        mesh_counts=[dict(neutron=n,fluid=n,thermal=2*n) for n in MESHES],profiles=22,
        initial_states='Separate stationary solves at the same requested power and initial inventory constraints; rod trims reported',
        command_scope='Same +0.01 inserted-fraction increment from each stationary trim',
        scope='Joint spatial refinement for this single-phase two-second rod case; not general all-scenario qualification',
        continuum_error_estimated=False,experimental_validation=False,long_horizon_accepted=False,station_env_connected=False,closed_tasks=[])
    atomic_json(out/'report.json',seal(report));atomic_json(out/'scientific.json',seal(dict(report=report,trajectories=trajectories)))
    if not report['passed']:raise ValueError('Coupled mesh qualification failed; results retained')
    return dict(passed=True,temporal=assessment['temporal_maximum_relative'],profiles=22)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    print(run(parser.parse_args().output))
