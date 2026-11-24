"""Joint mesh qualification of finite dilution and boration, including fronts."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
from atomic_model_multiphysics.provenance import seal
from atomic_model_training.spec import atomic_json
from coupling import pwr_boron
from coupling.pwr_boron import BoronExchangeConfig,BoronExchangePWRSession
from coupling.refinement import coupled_pwr_mesh
from coupling.mesh_comparison import profile_error
from scripts import qualify_boron_exchange as chemical
from scripts import qualify_coupled_mesh as coupled

MESHES=(2,4,8)
CASES=dict(dilution=0.,boration=2000.)
FRONTS=('fluid.boron_increment_ppm','neutron.boron_increment_ppm')
FIELDS=(*coupled.FIELDS,*FRONTS)
LIMITS=dict(coupled.LIMITS,**{k:.25 for k in FRONTS})
CRITERIA=dict(temporal_relative=1e-4,contraction=.9,numerical_margin=20.,
    total_power_relative=.005,coefficient_relative=1e-12,map_relative=1e-12,
    minimum_primary_boron_change_kg=.001)


def source():
    return dict(model=pwr_boron.identity(),helpers_sha256={m.__name__:hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
        for m in (chemical,coupled,coupled.refinement,coupled.mesh_comparison,coupled.fluid,coupled.thermal,coupled.budget,coupled.thermal.thermal_grid)},
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def compare_fronts(left,right,left_initial,right_initial):
    """Subtract each mesh's own reference to expose chemical transport errors."""
    result={}
    for row,initial in ((left,left_initial),(right,right_initial)):
        if row['core_config']!=initial['core_config'] or row['fluid_grid']!=initial['fluid_grid']:
            raise ValueError('Concentration increments require each mesh own initial reference')
    for domain in ('fluid','neutron'):
        if domain=='fluid':
            faces_a=left['fluid_grid']['faces_m'];faces_b=right['fluid_grid']['faces_m']
            na=len(faces_a)-1;nb=len(faces_b)-1
            a=(coupled.vector(left['distributed_fluid']['boron_fraction'],na)-coupled.vector(left_initial['distributed_fluid']['boron_fraction'],na))*1e6
            b=(coupled.vector(right['distributed_fluid']['boron_fraction'],nb)-coupled.vector(right_initial['distributed_fluid']['boron_fraction'],nb))*1e6
        else:
            na=left['core_config']['cells'];nb=right['core_config']['cells']
            a=coupled.vector(left['local_moderator']['boron_ppm'],na)-coupled.vector(left_initial['local_moderator']['boron_ppm'],na)
            b=coupled.vector(right['local_moderator']['boron_ppm'],nb)-coupled.vector(right_initial['local_moderator']['boron_ppm'],nb)
            faces_a=np.linspace(0.,left['core_config']['length_m'],left['core_config']['cells']+1)
            faces_b=np.linspace(0.,right['core_config']['length_m'],right['core_config']['cells']+1)
        result[domain+'.boron_increment_ppm']=profile_error(faces_a,a,faces_b,b,kind='cell_average',scale=1.)
    return result


def comparison(a,b,a0,b0,scales):
    return dict(coupled.compare_snapshots(a,b,scales),**compare_fronts(a,b,a0,b0))


def assess_case(trajectories,case):
    if case not in CASES:raise ValueError('Known chemical case required')
    if set(trajectories)!={f'{n}_{p}' for n in MESHES for p in chemical.PRESETS}:raise ValueError('Complete spatial/temporal chemical matrix required')
    ref=trajectories['8_fine']['initial'];scales=coupled.scales_for(ref);p0=float(ref['powers']['nuclear_fission_w'])
    gates=dict(configuration=True,commands=True,timeline=True,budgets=True,transfer=True,thermal=True,geometry=True,
               reference_frozen=True,inventory_maps=True,direction=True,rollback=True,stationary_base=True)
    times=[];noises=[];partitions=[];laws=[];changes={};geometries=[]
    reference_laws=trajectories['8_fine']['thermal_measures'];width_ref=np.diff(ref['fuel_grid']['faces_m'])
    for key,t in trajectories.items():
        n=int(key.split('_')[0]);first=t['initial'];rows=t['snapshots'];recipe=coupled_pwr_mesh(n)
        if len(rows)!=len(chemical.STEPS):raise ValueError('Complete chemical timeline required')
        gates['commands'] &= t['steps']==chemical.STEPS and t['checkpoint']['payload']['history']==chemical.STEPS
        gates['timeline'] &= [r['time_s'] for r in rows]==[.1,.5,1.,2.,2.5]
        cfg=t['checkpoint']['payload']['config'];expected=vars(BoronExchangeConfig(supply_boron_ppm=CASES[case]))
        gates['configuration'] &= cfg==expected
        width=np.diff(first['fuel_grid']['faces_m'])
        if set(t['thermal_measures'])!=set(reference_laws):raise ValueError('Complete thermal measures required')
        for k,v in t['thermal_measures'].items():
            actual=coupled.vector(v,len(width),positive=True)/width
            reference=np.asarray(reference_laws[k])/width_ref
            laws.append(float(max(abs(actual/reference[0]-1.))))
        for row in [first,*rows]:
            gates['configuration'] &= coupled.configurations(row)==coupled.configurations(ref) and row['core_config']['cells']==n and row['fuel_grid']['faces_m']==list(recipe['fuel_grid'].faces_m) and row['fluid_grid']['faces_m']==list(recipe['fluid_grid'].faces_m)
            gates['budgets'] &= all(coupled.budget.checks(row).values())
            gates['transfer'] &= all(chemical.audit(row,t['metadata']).values())
            gates['thermal'] &= all(coupled.thermal.thermal_audit(row,t['thermal_measures']).values())
            g=coupled.fluid.audit_geometry(row,t['hydraulic_coefficients']);geometries.append(g);gates['geometry'] &= g['passed']
            partitions.append(coupled.face_coefficient_defect(row,t['hydraulic_coefficients']))
            gates['reference_frozen'] &= row['local_moderator']['reference']==first['local_moderator']['reference']
            gates['inventory_maps'] &= max(row['local_moderator']['mass_relative_defect'],row['local_moderator']['boron_relative_defect'])<=CRITERIA['map_relative']
        change=chemical.primary_boron(rows[-1])-chemical.primary_boron(first);changes[key]=change
        gates['direction'] &= change<-CRITERIA['minimum_primary_boron_change_kg'] if case=='dilution' else change>CRITERIA['minimum_primary_boron_change_kg']
        gates['rollback'] &= t['rollback'] is True
        stationary=first['local_moderator_stationarity'];gates['stationary_base'] &= stationary['constrained_rank']==stationary['constrained_unknowns'] and stationary['max_selected_tolerance_ratio']<=1.
        if key.endswith('_coarse'):
            fine=trajectories[key.replace('_coarse','_fine')]
            for a,b in zip([first,*rows],[fine['initial'],*fine['snapshots']]):
                times.append(coupled.budget.relative(a['coupled_state'],np.asarray(b['coupled_state'])))
                noises.append(comparison(a,b,first,fine['initial'],scales))
    pairs={}
    for a,b in ((2,4),(4,8),(2,8)):
        left,right=(trajectories[f'{n}_fine'] for n in (a,b))
        observations=list(zip([left['initial'],*left['snapshots']],[right['initial'],*right['snapshots']]))
        profiles=[comparison(x,y,left['initial'],right['initial'],scales) for x,y in observations]
        pairs[f'{a}_{b}']=dict(profiles=profiles,maximum_l2={k:max(p[k]['normalized_l2'] for p in profiles) for k in FIELDS},
            maximum_total_power_relative=max(abs(x['powers']['nuclear_fission_w']-y['powers']['nuclear_fission_w'])/p0 for x,y in observations))
    coarse=pairs['2_4']['maximum_l2'];fine=pairs['4_8']['maximum_l2'];noise={k:max(r[k]['normalized_l2'] for r in noises) for k in FIELDS}
    gates.update(thermal_laws=max(laws)<=CRITERIA['coefficient_relative'],face_partitions=max(partitions)<=CRITERIA['coefficient_relative'],
        temporal=max(times)<=CRITERIA['temporal_relative'],spatial_threshold=all(fine[k]<=LIMITS[k] for k in FIELDS),
        spatial_contraction=all(fine[k]<=CRITERIA['contraction']*coarse[k] for k in FIELDS),
        spatial_effect_resolved=all(fine[k]>=CRITERIA['numerical_margin']*noise[k] for k in FIELDS),
        total_power=pairs['4_8']['maximum_total_power_relative']<=CRITERIA['total_power_relative'],
        finest_restart=trajectories['8_fine']['restart_exact'] is True)
    return dict(case=case,passed=all(gates.values()),gates={k:bool(v) for k,v in gates.items()},pairs=pairs,
        temporal_maximum_relative=max(times),temporal_profile_maximum_l2=noise,primary_boron_changes_kg=changes,
        thermal_coefficient_defect=max(laws),face_partition_defect=max(partitions),
        geometry_maximum_relative=max(g['geometry_relative_defect'] for g in geometries))


def matched_cases(cases):
    if set(cases)!=set(CASES):raise ValueError('Both chemical cases required')
    results={}
    for n in MESHES:
        left,right=(cases[c][f'{n}_fine'] for c in CASES);size=left['metadata']['base_size']
        same=left['initial']['coupled_state'][:size]==right['initial']['coupled_state'][:size]
        a=dict(left['checkpoint']['payload']['config']);b=dict(right['checkpoint']['payload']['config'])
        a.pop('supply_boron_ppm');b.pop('supply_boron_ppm');same &= a==b
        p0=left['initial']['powers']['nuclear_fission_w']
        effect=max(abs(x['powers']['nuclear_fission_w']-y['powers']['nuclear_fission_w'])/p0 for x,y in zip(left['snapshots'],right['snapshots']))
        noise=max(abs(a['powers']['nuclear_fission_w']-b['powers']['nuclear_fission_w'])/p0 for c in CASES
            for a,b in zip(cases[c][f'{n}_coarse']['snapshots'],cases[c][f'{n}_fine']['snapshots']))
        results[str(n)]=dict(matched=bool(same),power_effect_relative=float(effect),numerical_power_relative=float(noise),
            passed=bool(same and effect>=chemical.CRITERIA['minimum_power_effect_relative'] and effect>=CRITERIA['numerical_margin']*noise))
    return results


def run(output):
    out=Path(output).absolute()
    if out.exists() or any(p.is_symlink() for p in (out,*out.parents)):raise ValueError('Fresh ordinary output required')
    out.mkdir(parents=True);frozen=source();cases={}
    for case,ppm in CASES.items():
        trajectories={}
        for n in MESHES:
            for preset,options in chemical.PRESETS.items():
                key=f'{n}_{preset}';print(case,key,flush=True)
                s=BoronExchangePWRSession(**coupled_pwr_mesh(n),boron=BoronExchangeConfig(supply_boron_ppm=ppm),**options)
                first=s.snapshot();rows=[];saved=None
                meta=dict(cold_index=13+3*s.plant.network.volume_names().index('cold_leg'),rho=s.boron_rho,
                    supply_mass=s.boron_config.supply_mass_kg,receiver_mass=s.boron_config.receiver_mass_kg,supply_fraction=ppm*1e-6,base_size=s.boron_index)
                measures=coupled.thermal.measures(s);coefficients=coupled.fluid.coefficients(s)
                for i,step in enumerate(chemical.STEPS):
                    rows.append(s.advance(**step));print(case,key,s.time,flush=True)
                    if i==1:saved=json.loads(json.dumps(s.checkpoint()))
                restart=None
                if key=='8_fine':
                    restored=BoronExchangePWRSession.restore(saved)
                    restart=[restored.advance(**step) for step in chemical.STEPS[2:]]==rows[2:] and restored.checkpoint()==s.checkpoint()
                before=s.checkpoint()
                try:s.advance(.01,{'boron_flow_target_kg_s':1000.})
                except ValueError:rollback=s.checkpoint()==before
                else:rollback=False
                trajectories[key]=dict(initial=first,snapshots=rows,steps=deepcopy(chemical.STEPS),metadata=meta,
                    thermal_measures=measures,hydraulic_coefficients=coefficients,restart_exact=restart,rollback=rollback,checkpoint=before)
                atomic_json(out/f'{case}_{key}.json',seal(trajectories[key]))
        cases[case]=trajectories
    assessments={c:assess_case(t,c) for c,t in cases.items()};matched=matched_cases(cases)
    passed=all(a['passed'] for a in assessments.values()) and all(m['passed'] for m in matched.values()) and source()==frozen
    report=dict(source=frozen,criteria=CRITERIA,profile_limits=LIMITS,assessments=assessments,matched_cases=matched,passed=passed,
        trajectories=12,restart_suffixes=2,horizon_s=2.5,profiles=24,mesh_counts=[dict(neutron=n,fluid=n,thermal=2*n) for n in MESHES],
        scope='Joint spatial checks of these finite dilution/boration cases; no continuum-error estimate or general plant qualification',
        experimental_validation=False,long_horizon_accepted=False,station_env_connected=False,closed_tasks=[])
    atomic_json(out/'report.json',seal(report));atomic_json(out/'scientific.json',seal(dict(report=report,cases=cases)))
    if not passed:raise ValueError('Boron spatial qualification failed; attempts retained')
    return dict(passed=True,profiles=24,cases=2)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    print(run(parser.parse_args().output))
