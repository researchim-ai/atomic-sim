"""Conditional 2/4/8 neutron-mesh refinement with fixed thermal/fluid grids.

The finest mesh is a numerical reference within the same synthetic model,
not an independent code, a continuum solution or experimental validation.
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np
from atomic_model_multiphysics.provenance import seal
from atomic_model_training.spec import atomic_json
from atomic_model_station import CoreConfig
from coupling import mesh_comparison
from coupling.mesh_comparison import profile_error
from coupling.pwr_moderator import LocalModeratorPWRSession, identity
from scripts import qualify_moderator_transients as transient_driver
from scripts import qualify_pwr_distributed_fuel as budget_driver
from scripts.qualify_moderator_transients import PRESETS, protocol, _run
from scripts.qualify_pwr_distributed_fuel import checks, relative

MESHES = (2, 4, 8)
CRITERIA = dict(temporal_relative=1e-4, spatial_l2=0.05,
                contraction=0.9, total_power_relative=1e-3,
                minimum_flux_span=0.01, map_relative=1e-12, numerical_margin=20.)


def source():
    return dict(moderator=identity(),
                helpers_sha256={m.__name__:hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
                                for m in (transient_driver,budget_driver)},
                comparison_sha256=hashlib.sha256(Path(mesh_comparison.__file__).read_bytes()).hexdigest(),
                driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def compare_snapshots(left, right, scales):
    """Keep cell-integrated fission and six cell-average precursor fields distinct."""
    if set(scales) != {'fission', *('precursor'+str(i) for i in range(6))}:
        raise ValueError('Fission and all six precursor normalization scales required')
    configs = [r['core_config'] for r in (left, right)]
    if any(c['kind'] != 'diffusion' for c in configs):
        raise ValueError('Diffusion meshes required')
    if {k:v for k,v in configs[0].items() if k!='cells'} != {k:v for k,v in configs[1].items() if k!='cells'}:
        raise ValueError('Only neutron cell count may differ')
    faces = [np.linspace(0., c['length_m'], c['cells']+1) for c in configs]
    precursors = [np.asarray(r['core']['local_precursors']) for r in (left,right)]
    if any(p.shape != (6,c['cells']) for p,c in zip(precursors,configs)):
        raise ValueError('Complete six-family spatial precursor fields required')
    result = dict(fission=profile_error(faces[0], left['core']['fission_by_cell_w'],
                  faces[1], right['core']['fission_by_cell_w'],
                  kind='cell_integral', scale=scales['fission']))
    for i in range(6):
        key='precursor'+str(i)
        result[key]=profile_error(faces[0],precursors[0][i],faces[1],precursors[1][i],
                                 kind='cell_average',scale=scales[key])
    return result


def assess(trajectories):
    expected={f'{n}_{preset}' for n in MESHES for preset in PRESETS}
    if type(trajectories) is not dict or set(trajectories)!=expected:
        raise ValueError('Complete 2/4/8 mesh by two temporal presets required')
    reference=trajectories['8_fine']['initial']
    p0=float(reference['powers']['nuclear_fission_w'])
    scales=dict(fission=p0/reference['core_config']['length_m'],
                **{'precursor'+str(i):float(np.mean(row))
                   for i,row in enumerate(reference['core']['local_precursors'])})
    gates=dict(budgets=True,commands=True,timeline=True,fixed_thermal_meshes=True,
               stationary=True,reference_frozen=True,map_budgets=True,rollback=True,arrivals=True)
    time_errors=[];temporal_profiles=[];span=0.
    for key,t in trajectories.items():
        initial=t['initial'];rows=t['snapshots'];steps=protocol('insertion',float(initial['actuator_status']['rod_position']))
        if len(rows)!=len(steps):raise ValueError('Complete common observation timeline required')
        gates['commands'] &= t['steps']==steps
        gates['timeline'] &= all(abs(r['time_s']-v)<1e-12 for r,v in zip(rows,np.cumsum([s['duration_s'] for s in steps])))
        gates['fixed_thermal_meshes'] &= initial['fluid_grid']==reference['fluid_grid'] and initial['fuel_grid']==reference['fuel_grid']
        for r in [initial,*rows]:
            gates['budgets'] &= all(checks(r).values())
            gates['reference_frozen'] &= r['local_moderator']['reference']==initial['local_moderator']['reference']
            gates['map_budgets'] &= max(r['local_moderator']['mass_relative_defect'],r['local_moderator']['boron_relative_defect'])<=CRITERIA['map_relative']
            span=max(span,float(np.ptp(r['core']['relative_flux'])))
        a=initial['local_moderator_stationarity']
        gates['stationary'] &= a['constrained_rank']==a['constrained_unknowns'] and a['max_selected_tolerance_ratio']<=1.
        gates['rollback'] &= t['invalid_command_rollback'] is True
        arrivals=[e for e in rows[-1]['events'] if e['kind']=='rod_arrived']
        gates['arrivals'] &= len(arrivals)==1 and abs(arrivals[0]['time_s']-1.)<1e-12
        if key.endswith('_coarse'):
            fine=trajectories[key.replace('_coarse','_fine')]
            time_errors.extend(relative(a['coupled_state'],np.asarray(b['coupled_state']))
                for a,b in zip([initial,*rows],[fine['initial'],*fine['snapshots']]))
            temporal_profiles.extend(compare_snapshots(a,b,scales)
                for a,b in zip([initial,*rows],[fine['initial'],*fine['snapshots']]))
    pairs={}
    for a,b in ((2,4),(4,8),(2,8)):
        left,right=(trajectories[f'{n}_fine'] for n in (a,b))
        profiles=[compare_snapshots(x,y,scales) for x,y in zip(
            [left['initial'],*left['snapshots']],[right['initial'],*right['snapshots']])]
        pairs[f'{a}_{b}']=dict(profiles=profiles,
            maximum_l2={k:max(p[k]['normalized_l2'] for p in profiles) for k in scales},
            maximum_total_power_relative=max(abs(x['powers']['nuclear_fission_w']-y['powers']['nuclear_fission_w'])/p0
                for x,y in zip([left['initial'],*left['snapshots']],[right['initial'],*right['snapshots']])))
    coarse=pairs['2_4']['maximum_l2'];fine=pairs['4_8']['maximum_l2']
    noise={k:max(p[k]['normalized_l2'] for p in temporal_profiles) for k in scales}
    gates.update(temporal=max(time_errors)<=CRITERIA['temporal_relative'],
        matched_initial_power=all(abs(t['initial']['powers']['nuclear_fission_w']-p0)/p0<=1e-8 for t in trajectories.values()),
        nonuniform=span>=CRITERIA['minimum_flux_span'],
        spatial_threshold=all(v<=CRITERIA['spatial_l2'] for v in fine.values()),
        spatial_contraction=all(fine[k]<=CRITERIA['contraction']*coarse[k] for k in scales),
        spatial_effect_resolved=all(fine[k]>=CRITERIA['numerical_margin']*noise[k] for k in scales),
        total_power=pairs['4_8']['maximum_total_power_relative']<=CRITERIA['total_power_relative'],
        restart=trajectories['8_fine']['restart_exact'] is True)
    return dict(passed=all(gates.values()),gates={k:bool(v) for k,v in gates.items()},
                temporal_maximum_relative=max(time_errors),maximum_flux_span=span,
                scales=scales,pairs=pairs,temporal_profile_maximum_l2=noise,
                initial_rod_positions={k:t['initial']['actuator_status']['rod_position'] for k,t in trajectories.items()})


def run(output):
    out=Path(output).absolute()
    if out.exists() or any(p.is_symlink() for p in (out,*out.parents)):raise ValueError('Fresh ordinary output required')
    out.mkdir(parents=True);frozen=source();trajectories={}
    for cells in MESHES:
        for preset,options in PRESETS.items():
            key=f'{cells}_{preset}';print('trajectory',key,flush=True)
            session=LocalModeratorPWRSession(core=CoreConfig(kind='diffusion',cells=cells),**options)
            steps=protocol('insertion',float(session.state[12]))
            trajectories[key]=_run(session,steps,restart=key=='8_fine')
            atomic_json(out/(key+'.json'),seal(trajectories[key]))
    assessment=assess(trajectories)
    report=dict(source=frozen,criteria=CRITERIA,assessment=assessment,
        passed=assessment['passed'] and source()==frozen,trajectories=6,restart_suffixes=1,
        horizon_s=2.,neutron_cells=list(MESHES),thermal_and_fluid_meshes_fixed=True,
        initial_states='Separate stationary solve at the same requested power and inventory anchors; rod trims reported, not identical spatial states',
        command_scope='Same +0.01 inserted-fraction increment from each stationary rod trim',
        continuum_error_estimated=False,full_multiphysics_spatial_convergence=False,
        experimental_validation=False,long_horizon_accepted=False,station_env_connected=False,closed_tasks=[])
    atomic_json(out/'report.json',seal(report));atomic_json(out/'scientific.json',seal(dict(report=report,trajectories=trajectories)))
    if not report['passed']:raise ValueError('Conditional neutron-mesh qualification failed; results retained')
    return dict(passed=True,temporal=assessment['temporal_maximum_relative'],spatial=pairs_summary(assessment))


def pairs_summary(assessment):
    return {key:value['maximum_l2'] for key,value in assessment['pairs'].items()}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    print(run(parser.parse_args().output))
