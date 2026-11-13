"""Replaceable point/spatial neutron owner: local numerical qualification only."""
from argparse import ArgumentParser
import json
from pathlib import Path
import numpy as np
from scipy.linalg import expm
from scipy.integrate import solve_ivp
from atomic_model_spatial.reference import dense_operators
from atomic_model_station import PWRCoreSession,PWRPowerSession,CoreConfig
from atomic_model_station.core import core_signature
from atomic_model_multiphysics.provenance import sha
from core.status import DomainError,SolverError
from scripts import qualify_station_actuators as evidence

TASKS=['NPP-03.04']
SETTINGS=[(.2,1e-7),(.1,2e-8),(.05,2e-9)]
CRITERIA=dict(energy_relative=1e-6,mass_relative=1e-10,boron_relative=1e-10,projection_relative=1e-12,
    neutron_inventory_relative=1e-10,instantaneous_relative=1e-8,point_limit_relative=1e-5,frozen_reference_relative=1e-8,
    refinement_relative=1e-4,mesh_comparison_relative=1e-2,tail_drift_relative=1e-5)


def plant_vector(r):return np.r_[r['pwr_state'],r['turbine_energy_j'],r['clad_temperature_k'],r['cooling_energy_j'],r['cooling']['flow_kg_s'],r['feed_inventory'],r['electrical']['backup_energy_j']]


def vector(r):return np.r_[plant_vector(r),r['core']['owned_state']]


def admit(r):
    c=r['core']
    return bool(r['energy_relative']<=CRITERIA['energy_relative'] and r['mass_relative']<=CRITERIA['mass_relative'] and r['boron_relative']<=CRITERIA['boron_relative']
        and abs(c['fission_projection_residual_w'])/max(abs(c['fission_total_w']),1.)<=CRITERIA['projection_relative']
        and abs(r['instantaneous_energy_defect_w'])/max(abs(r['powers']['nuclear_heat_w']),1.)<=CRITERIA['instantaneous_relative']
        and min(c['owned_state'])>=0. and c.get('neutron_precursor_inventory',dict(relative_defect=0.))['relative_defect']<=CRITERIA['neutron_inventory_relative'])


def run(output,reference=None):
    output=Path(output).absolute()
    if any(p.is_symlink() for p in (output,*output.parents)):raise ValueError('Symlink output')
    output.mkdir(parents=True,exist_ok=False)
    source=core_signature();driver=sha(__file__);serializer=sha(evidence.__file__)
    cfg=CoreConfig(kind='diffusion',cells=4,axial_bias_reactivity=1e-4)
    s=PWRCoreSession(core=cfg,max_step_s=1.);initial=s.snapshot()
    evidence.save(output/'protocol.json',dict(source=source,driver_sha256=driver,serializer_sha256=serializer,criteria=CRITERIA,
        parameters=s.parameter_registry(),initial=initial,settings=SETTINGS,candidate_tasks=TASKS,nominal_duration_s=3600.,
        state_owner='Appended core variables only; old7 neutron coordinates are algebraic aliases excluded from integration',
        nuclear_data='Constructed critical one-group reflecting slab; six declared teaching precursor families, no evaluated plant cross sections',
        feedback='Existing bulk reactivity applied as absorption shift plus fixed linear synthetic bias at cell centres',
        thermal='Sum fission source through existing prompt/decay3 accounting into common lumped fuel; no cell-resolved thermal feedback',
        mesh_scope='2/4/8-cell comparison at0.5s for fixed bias field; no claimed formal order or plant spatial validation'))
    print('station core protocol sealed',flush=True)
    rows=[]
    for i in range(60):
        rows.append(s.advance(60.))
        if i%5==0:print('spatial station nominal',i+1,'/60',flush=True)
    tail=float(np.max(abs(vector(rows[-1])-vector(rows[-3]))/np.maximum(abs(vector(rows[-3])),1.)))
    evidence.save(output/'nominal.json',dict(trace=rows,tail_relative_drift=tail,checkpoint=s.checkpoint()))
    observations=[initial,*rows];trials=[]
    for step,rtol in SETTINGS:
        t=PWRCoreSession(core=cfg,max_step_s=step,rtol=rtol)
        r=t.advance(.5,{'load_fraction':.95,'rod_target':float(t.state[12])+.005})
        trials.append(dict(max_step_s=step,rtol=rtol,observation=r));observations.append(r)
        print('station spatial transient',step,flush=True)
    fine=vector(trials[-1]['observation']);errors=[float(np.max(abs(vector(t['observation'])-fine)/np.maximum(abs(fine),1.))) for t in trials]
    point=PWRCoreSession(max_step_s=.1);uniform=PWRCoreSession(core=CoreConfig(kind='diffusion',cells=4),max_step_s=.1);legacy=PWRPowerSession(max_step_s=.1)
    patch={'load_fraction':.95,'rod_target':float(point.state[12])+.005}
    p=point.advance(.5,patch);u=uniform.advance(.5,patch);l=legacy.advance(.5,patch);observations.extend([p,u])
    limit_errors=[float(np.max(abs(plant_vector(row)-plant_vector(l))/np.maximum(abs(plant_vector(l)),1.))) for row in [p,u]]
    adapter=s.neutron_core;y=adapter.lift(s.plant.kinetics.initial_state());rho=-.002
    _,_,matrix=dense_operators(adapter.model);n=adapter.cells
    matrix[np.arange(n),np.arange(n)]+=(rho+adapter.bias)/adapter.kinetics.generation_time_s
    exact=expm(matrix*.1)@y
    numerical=solve_ivp(lambda t,y:adapter.rhs(y,rho),(0.,.1),y,method='Radau',rtol=1e-10,atol=1e-12).y[:,-1]
    frozen_error=float(np.max(abs(numerical-exact)/np.maximum(abs(exact),1e-12)))
    meshes=[]
    for cells in (2,4,8):
        t=PWRCoreSession(core=CoreConfig(kind='diffusion',cells=cells,axial_bias_reactivity=1e-4),max_step_s=.1,rtol=2e-8)
        r=t.advance(.5,patch);meshes.append(r);observations.append(r)
    mesh_errors=[float(np.max(abs(plant_vector(r)-plant_vector(meshes[-1]))/np.maximum(abs(plant_vector(meshes[-1])),1.))) for r in meshes]
    evidence.save(output/'comparisons.json',dict(transients=trials,refinement_errors=errors,point=p,uniform_spatial=u,legacy=l,point_limit_errors=limit_errors,
        frozen_reference_error=frozen_error,frozen_exact=exact.tolist(),frozen_numerical=numerical.tolist(),mesh_observations=meshes,mesh_errors=mesh_errors))
    restart=PWRCoreSession(core=cfg,max_step_s=.2);restart.advance(.1,{'grid_available':False})
    saved=json.loads(json.dumps(restart.checkpoint()));restored=PWRCoreSession.restore(saved)
    resumed=restart.advance(.1);exact_restart=restored.advance(.1)==resumed;observations.append(resumed)
    before=restart.checkpoint();rollback=False
    try:restart.advance(.1,{'unknown_core_knob':True})
    except ValueError:rollback=restart.checkpoint()==before
    evidence.save(output/'restart.json',dict(checkpoint=saved,resumed=resumed,restart_exact=exact_restart,invalid_command_rollback=rollback))
    all_admitted=all(admit(r) for r in observations)
    shape=trials[-1]['observation']['core']['fission_by_cell_w'];shape_changed=shape[0]<shape[-1]
    unchanged=source==core_signature() and driver==sha(__file__) and serializer==sha(evidence.__file__)
    passed=bool(unchanged and all_admitted and max(errors)<=CRITERIA['refinement_relative'] and max(limit_errors)<=CRITERIA['point_limit_relative']
        and frozen_error<=CRITERIA['frozen_reference_relative'] and max(mesh_errors)<=CRITERIA['mesh_comparison_relative']
        and tail<=CRITERIA['tail_drift_relative'] and exact_restart and rollback and shape_changed)
    report=dict(completed=True,passed=passed,source_unchanged=unchanged,task_checks={'NPP-03.04':passed},closed_tasks=TASKS if passed else [],
        nominal_duration_s=rows[-1]['time_s'],nominal_tail_relative_drift=tail,point_limit_errors=limit_errors,refinement_errors=errors,
        mesh_comparison_errors=mesh_errors,frozen_reference_error=frozen_error,spatial_shape_changes=bool(shape_changed),restart_exact=exact_restart,rollback=rollback,
        max_energy_relative=max(r['energy_relative'] for r in observations),max_mass_relative=max(r['mass_relative'] for r in observations),
        max_boron_relative=max(r['boron_relative'] for r in observations),
        max_neutron_inventory_relative=max(r['core'].get('neutron_precursor_inventory',dict(relative_defect=0.))['relative_defect'] for r in observations),
        complete_package_b=False,complete_package_c=False,r1_accepted=False,validated=False,plant_referenced=False,full_nominal_checkpoint_replay=False,
        limitations=['Synthetic one-group diffusion and six local precursor families; not evaluated plant nuclear data',
            'Reflecting1D slab only; no actual assembly/bank geometry, burnup or spectral feedback',
            'Bulk thermal/density/boron feedback imposed uniformly; no local fuel/clad/coolant thermal resolution',
            'Existing3-group decay heat accounting remains lumped; cell fission power is not local deposited heat',
            'Only built-in point/diffusion adapters are replay-registered; arbitrary external providers need their own source/config binding',
            'Point limit and mesh comparison are numerical checks, not spatial plant validation',
            'Existing steam path, phase, power-resource and inactive spray restrictions remain'])
    evidence.save(output/'report.json',report)
    if reference is not None:
        reference=Path(reference);files={p.relative_to(output) for p in output.rglob('*.json')}
        if files!={p.relative_to(reference) for p in reference.rglob('*.json')}:raise ValueError('Evidence file sets differ')
        for path in sorted(files):
            if evidence.read(output/path)!=evidence.read(reference/path):raise ValueError('Installed core evidence differs:'+str(path))
        print('All installed replaceable-core evidence matches exactly',flush=True)
    if not passed:raise ValueError('Replaceable core qualification failed; no task accepted')
    print(json.dumps(dict(passed=passed,closed_tasks=report['closed_tasks'])),flush=True)
    return report


if __name__=='__main__':
    p=ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--reference',type=Path)
    args=p.parse_args();run(args.output,args.reference)
