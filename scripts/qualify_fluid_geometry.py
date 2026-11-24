"""Complete dual momentum geometry and conditional liquid-mesh refinement.

Fixed eight-cell fuel mesh, four neutron cells; same declared channel span,
inertance, uniform-flow reference kinetic coefficient and loss under refinement.
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np
from atomic_model_multiphysics.provenance import seal
from atomic_model_training.spec import atomic_json
from atomic_model_station import CoreConfig
from coupling import hydraulic_geometry,mesh_comparison
from coupling.conservative import vector
from coupling.mesh_comparison import profile_error
from coupling.pwr_fuel import FuelGrid
from coupling.pwr_channel import FluidGrid
from coupling.pwr_moderator import LocalModeratorPWRSession,identity
from scripts import qualify_moderator_transients as transient_driver
from scripts import qualify_pwr_distributed_fuel as budget_driver

MESHES=(2,4,8)
FIELDS=('mass','energy','boron','temperature','pressure','density')
LIMITS=dict(mass=.02,energy=.02,boron=.02,temperature=.01,pressure=.001,density=.02)
CRITERIA=dict(temporal_relative=1e-4,contraction=.9,numerical_margin=20.,
              total_power_relative=1e-3,geometry_relative=1e-12,momentum_abs_pa=1e-5,
              kinetic_work_relative=1e-10,minimum_temperature_span_k=1.)


def source():
    return dict(moderator=identity(),helpers_sha256={m.__name__:hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
        for m in (hydraulic_geometry,mesh_comparison,transient_driver,budget_driver)},
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def coefficients(session):
    p=session.plant
    links=[p.network.junctions['pump'],*session.channel.junctions.values(),p.network.junctions['core_out']]
    rho=np.r_[p.link_density_ref['pump'],session.internal_density_ref,p.link_density_ref['core_out']]
    return dict(inertance_m_inv=[j.length_m/j.area_m2 for j in links],
        kinetic_coefficients_m2_kg=[j.length_m/j.area_m2/r for j,r in zip(links,rho)],
        loss_per_m4=[j.loss_coefficient/j.area_m2**2 for j in links])


def fields(row):
    f=row['distributed_fluid'];n=len(row['fluid_grid']['faces_m'])-1
    inventory=np.asarray(f['inventory_by_cell'])
    if inventory.shape!=(n,3):raise ValueError('Complete local mass/energy/boron required')
    values=dict(mass=inventory[:,0],energy=inventory[:,1],boron=inventory[:,2],
        temperature=f['temperature_k'],pressure=f['pressure_pa'],density=f['density_kg_m3'])
    return {k:vector(v,n,positive=True) for k,v in values.items()}


def compare_snapshots(left,right,scales):
    if set(scales)!=set(FIELDS):raise ValueError('All six fluid normalization scales required')
    if left['core_config']!=right['core_config'] or left['fuel_grid']!=right['fuel_grid']:
        raise ValueError('Fixed neutron and thermal meshes required')
    for r in (left,right):
        if r['fluid_grid']['momentum_geometry']!='complete_dual':raise ValueError('Complete dual geometry required')
    if left['fluid_grid']['internal_loss_coefficient']!=right['fluid_grid']['internal_loss_coefficient']:
        raise ValueError('Fixed channel loss coefficient required')
    a,b=fields(left),fields(right)
    return {k:profile_error(left['fluid_grid']['faces_m'],a[k],right['fluid_grid']['faces_m'],b[k],
        kind='cell_integral' if k in ('mass','energy','boron') else 'cell_average',scale=scales[k]) for k in FIELDS}


def audit_geometry(row,actual):
    m=row['momentum_geometry'];g=m['dual'];f=row['distributed_fluid'];n=len(row['fluid_grid']['faces_m'])
    inertance=vector(actual['inertance_m_inv'],n,positive=True)
    kinetic=vector(actual['kinetic_coefficients_m2_kg'],n,positive=True)
    losses=vector(actual['loss_per_m4'],n)
    legacy=m['boundaries'];i0=sum(b['legacy_inertance_m_inv'] for b in legacy.values())
    e0=sum(b['legacy_kinetic_coefficient_m2_kg'] for b in legacy.values())
    k0=sum(b['legacy_loss_per_m4'] for b in legacy.values())
    length=g['length_m'];area=g['area_m2'];rho=m['channel_reference_density_kg_m3'];k=g['loss_coefficient']
    expected=(length/area,length/area/rho,k/area**2)
    defects=[abs(x-y)/max(abs(y),1e-30) for x,y in zip((sum(inertance)-i0,sum(kinetic)-e0,sum(losses)-k0),expected)]
    q=vector(f['face_flow_kg_s'],n);acceleration=vector(f['face_acceleration_kg_s2'],n)
    force=vector(f['face_driving_pressure_pa'],n)
    work=np.r_[f['boundary_kinetic_rates_w'][0],f['internal_kinetic_rate_w'],f['boundary_kinetic_rates_w'][1]]
    calculated=kinetic*q*acceleration
    momentum=float(max(abs(inertance*acceleration-force)))
    kinetic_defect=float(max(abs(calculated-work)))/max(float(sum(abs(calculated))),1.)
    return dict(geometry_relative_defect=float(max(defects)),momentum_abs_pa=momentum,
                kinetic_work_relative=kinetic_defect,
                passed=bool(max(defects)<=CRITERIA['geometry_relative'] and momentum<=CRITERIA['momentum_abs_pa']
                            and kinetic_defect<=CRITERIA['kinetic_work_relative']))


def assess(trajectories):
    if type(trajectories) is not dict or set(trajectories)!={f'{n}_{p}' for n in MESHES for p in transient_driver.PRESETS}:
        raise ValueError('Complete 2/4/8 fluid by two temporal presets required')
    ref=trajectories['8_fine']['initial'];inventory=np.asarray(ref['distributed_fluid']['inventory_by_cell']);p0=float(ref['powers']['nuclear_fission_w'])
    scales=dict(mass=float(inventory[:,0].sum()/4),energy=float(inventory[:,1].sum()/4),boron=float(inventory[:,2].sum()/4),temperature=600.,pressure=15.5e6,density=1000.)
    gates=dict(budgets=True,geometry=True,commands=True,timeline=True,stationary=True,
        fixed_meshes=True,reference_frozen=True,rollback=True,arrivals=True,grid=True)
    temporal=[];noise_rows=[];geometry_rows=[];span=0.;legacy=[]
    for key,t in trajectories.items():
        initial=t['initial'];rows=t['snapshots'];cells=int(key.split('_')[0])
        steps=transient_driver.protocol('insertion',float(initial['actuator_status']['rod_position']))
        if len(rows)!=len(steps):raise ValueError('Complete observation timeline required')
        gates['commands'] &= t['steps']==steps
        gates['timeline'] &= all(abs(r['time_s']-time)<1e-12 for r,time in zip(rows,np.cumsum([s['duration_s'] for s in steps])))
        legacy.append(initial['momentum_geometry']['boundaries'])
        for r in [initial,*rows]:
            gates['grid'] &= r['fluid_grid']['faces_m']==np.linspace(0.,4.,cells+1).tolist() and r['fluid_grid']['internal_loss_coefficient']==1. and r['fluid_grid']['momentum_geometry']=='complete_dual'
            gates['fixed_meshes'] &= r['core_config']==ref['core_config'] and r['fuel_grid']==ref['fuel_grid']
            gates['budgets'] &= all(budget_driver.checks(r).values())
            result=audit_geometry(r,t['coefficients']);geometry_rows.append(result);gates['geometry'] &= result['passed']
            gates['reference_frozen'] &= r['local_moderator']['reference']==initial['local_moderator']['reference']
            span=max(span,float(np.ptp(r['distributed_fluid']['temperature_k'])))
        s=initial['local_moderator_stationarity'];gates['stationary'] &= s['constrained_rank']==s['constrained_unknowns'] and s['max_selected_tolerance_ratio']<=1.
        gates['rollback'] &= t['invalid_command_rollback'] is True
        events=[e for e in rows[-1]['events'] if e['kind']=='rod_arrived'];gates['arrivals'] &= len(events)==1 and abs(events[0]['time_s']-1.)<1e-12
        if key.endswith('_coarse'):
            fine=trajectories[key.replace('_coarse','_fine')]
            for a,b in zip([initial,*rows],[fine['initial'],*fine['snapshots']]):
                temporal.append(budget_driver.relative(a['coupled_state'],np.asarray(b['coupled_state'])))
                noise_rows.append(compare_snapshots(a,b,scales))
    pairs={}
    for a,b in ((2,4),(4,8),(2,8)):
        left,right=(trajectories[f'{n}_fine'] for n in (a,b))
        observations=list(zip([left['initial'],*left['snapshots']],[right['initial'],*right['snapshots']]))
        profiles=[compare_snapshots(x,y,scales) for x,y in observations]
        pairs[f'{a}_{b}']=dict(profiles=profiles,maximum_l2={k:max(p[k]['normalized_l2'] for p in profiles) for k in FIELDS},
            maximum_total_power_relative=max(abs(x['powers']['nuclear_fission_w']-y['powers']['nuclear_fission_w'])/p0 for x,y in observations))
    coarse=pairs['2_4']['maximum_l2'];fine=pairs['4_8']['maximum_l2'];noise={k:max(r[k]['normalized_l2'] for r in noise_rows) for k in FIELDS}
    legacy_properties=[[(b['legacy_inertance_m_inv'],b['legacy_kinetic_coefficient_m2_kg'],b['legacy_loss_per_m4']) for b in r.values()] for r in legacy]
    gates.update(legacy_geometry_fixed=all(x==legacy_properties[0] for x in legacy_properties),
        matched_initial_power=all(abs(t['initial']['powers']['nuclear_fission_w']-p0)/p0<=1e-8 for t in trajectories.values()),
        temporal=max(temporal)<=CRITERIA['temporal_relative'],nonuniform=span>=CRITERIA['minimum_temperature_span_k'],
        spatial_threshold=all(fine[k]<=LIMITS[k] for k in FIELDS),
        spatial_contraction=all(fine[k]<=CRITERIA['contraction']*coarse[k] for k in FIELDS),
        spatial_effect_resolved=all(fine[k]>=CRITERIA['numerical_margin']*noise[k] for k in FIELDS),
        total_power=pairs['4_8']['maximum_total_power_relative']<=CRITERIA['total_power_relative'],
        restart=trajectories['8_fine']['restart_exact'] is True)
    return dict(passed=all(gates.values()),gates={k:bool(v) for k,v in gates.items()},scales=scales,pairs=pairs,
        temporal_maximum_relative=max(temporal),temporal_profile_maximum_l2=noise,
        geometry_maximum_relative=max(r['geometry_relative_defect'] for r in geometry_rows),
        momentum_maximum_abs_pa=max(r['momentum_abs_pa'] for r in geometry_rows),
        kinetic_work_maximum_relative=max(r['kinetic_work_relative'] for r in geometry_rows),
        maximum_temperature_span_k=span,initial_rod_positions={k:t['initial']['actuator_status']['rod_position'] for k,t in trajectories.items()})


def run(output):
    out=Path(output).absolute()
    if out.exists() or any(p.is_symlink() for p in (out,*out.parents)):raise ValueError('Fresh ordinary output required')
    out.mkdir(parents=True);frozen=source();trajectories={}
    for cells in MESHES:
        for preset,options in transient_driver.PRESETS.items():
            key=f'{cells}_{preset}';print('trajectory',key,flush=True)
            session=LocalModeratorPWRSession(core=CoreConfig(kind='diffusion',cells=4),
                fuel_grid=FuelGrid(tuple(np.linspace(0.,4.,9))),
                fluid_grid=FluidGrid(tuple(np.linspace(0.,4.,cells+1)),1.,'complete_dual'),**options)
            actual=coefficients(session);steps=transient_driver.protocol('insertion',float(session.state[12]))
            trajectories[key]=dict(transient_driver._run(session,steps,restart=key=='8_fine'),coefficients=actual)
            atomic_json(out/(key+'.json'),seal(trajectories[key]))
    assessment=assess(trajectories)
    report=dict(source=frozen,criteria=CRITERIA,profile_limits=LIMITS,assessment=assessment,
        passed=assessment['passed'] and frozen==source(),trajectories=6,restart_suffixes=1,horizon_s=2.,fluid_cells=list(MESHES),
        neutron_cells=4,thermal_cells=8,channel_loss_coefficient=1.,
        initial_states='Separate stationary solves at the same requested power and initial inventory anchors; rod trims reported',
        command_scope='Same +0.01 inserted fraction from each stationary rod trim',
        continuum_error_estimated=False,full_multiphysics_spatial_convergence=False,experimental_validation=False,
        long_horizon_accepted=False,station_env_connected=False,closed_tasks=[])
    atomic_json(out/'report.json',seal(report));atomic_json(out/'scientific.json',seal(dict(report=report,trajectories=trajectories)))
    if not report['passed']:raise ValueError('Fluid geometry qualification failed; attempts retained')
    return dict(passed=True,temporal=assessment['temporal_maximum_relative'],geometry=assessment['geometry_maximum_relative'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    print(run(parser.parse_args().output))
