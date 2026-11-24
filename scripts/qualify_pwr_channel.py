"""Short monolithic local-fluid PWR qualification; no industrial acceptance."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
from atomic_model_multiphysics.provenance import seal
from atomic_model_training.spec import atomic_json
from atomic_model_station import CoreConfig
from coupling.pwr_channel import ChannelPWRSession,FluidGrid,identity
from coupling.pwr_cladding import DistributedCladdingSession
from scripts.qualify_pwr_distributed_fuel import checks as fuel_checks,physical_vector,relative

PRESETS=dict(coarse=dict(max_step_s=.02,rtol=2e-8),fine=dict(max_step_s=.01,rtol=2e-9))
CRITERIA=dict(refinement_relative=2e-5,single_cell_limit_relative=2e-6,local_mass_kg_s=1e-8,
    local_energy_w=1e-4,local_boron_kg_s=1e-10)
ACTIONS=(None,{'load_fraction':.99},{'scram':True})


def gates(row):
    checks=fuel_checks(row);f=row['distributed_fluid'];res=f['local_boundary_balance_residual']
    checks.update(local_mass=abs(res[0])<=CRITERIA['local_mass_kg_s'],local_energy=abs(res[1])<=CRITERIA['local_energy_w'],
        local_boron=abs(res[2])<=CRITERIA['local_boron_kg_s'],positive_contact_entropy=f['contact_entropy_w_k']>=0,
        single_owner=max(abs(v) for v in f['alias_rate_residual'])==0.)
    return {k:bool(v) for k,v in checks.items()}


def run(output):
    out=Path(output).absolute()
    if out.exists() or any(p.is_symlink() for p in (out,*out.parents)):raise ValueError('Fresh ordinary output required')
    out.mkdir(parents=True);source=identity();driver=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    cases={};summaries=[]
    for name,faces,bias,k in [('single_cell',(0.,4.),0.,0.),('three_cells',(0.,1.,2.,4.),0.,0.),
            ('five_cells',(0.,.3,1.,2.3,3.,4.),0.,0.),('biased_lossy',(0.,1.,2.,4.),1e-4,.25)]:
        print('case',name,flush=True)
        spec=dict(core=CoreConfig(kind='diffusion',cells=4,axial_bias_reactivity=bias),fluid_grid=FluidGrid(faces,k))
        trajectories={}
        for preset,options in PRESETS.items():
            s=ChannelPWRSession(**spec,**options);rows=[];saved=None
            for i,action in enumerate(ACTIONS):
                rows.append(s.advance(.02,action))
                print(name,preset,'interval',i,'completed',flush=True)
                if i==1:saved=json.loads(json.dumps(s.checkpoint()))
            restored=ChannelPWRSession.restore(saved);replayed=restored.advance(.02,{'scram':True})
            before=s.checkpoint()
            try:s.advance(.001,{'load_fraction':99})
            except ValueError:rollback=s.checkpoint()==before
            else:rollback=False
            trajectories[preset]=dict(snapshots=rows,checks=[gates(r) for r in rows],
                restart_exact=replayed==rows[-1] and restored.checkpoint()==before,invalid_action_rollback=rollback,checkpoint=before)
        errors=[relative(a['coupled_state'],np.asarray(b['coupled_state'])) for a,b in zip(trajectories['coarse']['snapshots'],trajectories['fine']['snapshots'])]
        limit=None
        if name=='single_cell':
            baseline=DistributedCladdingSession(core=spec['core'],**PRESETS['fine'])
            old=[baseline.advance(.02,a) for a in ACTIONS]
            limit=max(relative(physical_vector(a),physical_vector(b)) for a,b in zip(trajectories['fine']['snapshots'],old))
        final=trajectories['fine']['snapshots'][-1]
        checks=dict(balances=all(all(g.values()) for t in trajectories.values() for g in t['checks']),
            restart=all(t['restart_exact'] for t in trajectories.values()),rollback=all(t['invalid_action_rollback'] for t in trajectories.values()),
            refined=max(errors)<=CRITERIA['refinement_relative'],single_cell_limit=limit is None or limit<=CRITERIA['single_cell_limit_relative'],
            decay_heat_release=final['distributed_fuel']['thermal_w']>final['distributed_fuel']['fission_w'],
            spatial_profile=len(faces)==2 or np.ptp(final['distributed_fluid']['temperature_k'])>.1)
        checks={key:bool(value) for key,value in checks.items()}
        summary=dict(case=name,passed=all(checks.values()),gates=checks,refinement_errors=errors,single_cell_limit_error=limit)
        summaries.append(summary);cases[name]=dict(summary=summary,core=asdict(spec['core']),fluid_faces_m=list(faces),internal_loss=k,trajectories=trajectories)
        atomic_json(out/(name+'.json'),seal(cases[name]))
        print(name,'passed',summary['passed'],flush=True)
    report=dict(source=source,driver_sha256=driver,criteria=CRITERIA,cases=summaries,
        passed=all(s['passed'] for s in summaries) and source==identity() and driver==hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        native_trajectories=8,restart_suffixes=8,single_cell_reference_trajectories=1,horizon_s=.06,
        local_fluid_pwr_connected=True,station_env_connected=False,local_density_boron_feedback=False,
        stationary_channel_initialization=False,experimental_validation=False,long_horizon_accepted=False,closed_tasks=[])
    atomic_json(out/'report.json',seal(report));atomic_json(out/'scientific.json',seal(dict(report=report,cases=cases)))
    if not report['passed']:raise ValueError('PWR channel qualification failed; attempts retained')
    return dict(passed=True,cases=len(cases),maximum_refinement_error=max(max(s['refinement_errors']) for s in summaries))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    print(run(parser.parse_args().output))
