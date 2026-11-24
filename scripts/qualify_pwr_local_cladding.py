"""Short reduced-PWR local-cladding qualification; no channel/plant validation."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from atomic_model_multiphysics.provenance import seal
from atomic_model_station import CoreConfig
from atomic_model_training.spec import atomic_json
from coupling.pwr_cladding import DistributedCladdingSession,identity
from coupling.pwr_fuel import DistributedFuelSession,FuelGrid
from scripts.qualify_pwr_distributed_fuel import checks,local_vector,physical_vector,relative,ACTIONS,PRESETS


CRITERIA=dict(refinement_relative=1e-4,uniform_limit_relative=2e-6,bdf_reference_relative=2e-6,
    local_balance_relative=1e-12,alias_abs_k=1e-9,entropy_min_w_k=-1e-8)


def vector(snapshot):
    return np.r_[local_vector(snapshot),snapshot['distributed_cladding']['temperature_k']]


def gates(snapshot):
    result=checks(snapshot);c=snapshot['distributed_cladding'];scale=max(snapshot['distributed_fuel']['thermal_w'],1.)
    result.update(local_cladding_balance=max(abs(v) for v in c['local_energy_residual_w'])/scale<=CRITERIA['local_balance_relative'],
        cladding_alias=abs(c['alias_residual_k'])<=CRITERIA['alias_abs_k'],
        conductive_entropy=min(c['conductive_entropy_generation_w_k'])>=CRITERIA['entropy_min_w_k'])
    return {k:bool(v) for k,v in result.items()}


def bdf_reference(spec):
    """Different integrator of the same physical RHS, without native reduction.

    Only the initial 0.02 s hold is compared. No switching or zero crossing is
    qualified by this reference. It is numerical evidence, not independent data.
    """
    reference=DistributedCladdingSession(**spec,**PRESETS['fine'])
    reference._prepare_interval();reference.plant._rod_rate_override=0.
    atol=np.full(len(reference.state),1e-6);atol[reference.base_size:]=.001
    atol[reference.base_size+1]=1e-8;atol=reference._absolute_tolerances(atol)
    solution=solve_ivp(reference._rhs,(0.,.02),reference.state.copy(),method='BDF',rtol=2e-10,atol=atol,max_step=.002)
    if not solution.success or solution.t[-1]!=.02:raise ValueError('Complete BDF reference required')
    reference.state=reference._reconstruct_state(solution.y[:,-1].copy());reference.time=.02
    reference._admit(reference.state);ref=reference.snapshot()
    native=DistributedCladdingSession(**spec,**PRESETS['fine']).advance(.02)
    return dict(error=relative(vector(native),vector(ref)),native=native,reference=ref,
        method='BDF full state versus native reduced Radau; same RHS',steps=len(solution.t)-1,nfev=solution.nfev)


def run(output):
    out=Path(output).absolute()
    if out.exists() or any(p.is_symlink() for p in (out,*out.parents)):raise ValueError('Fresh ordinary output required')
    out.mkdir(parents=True);source=identity();driver=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    cases={};summaries=[]
    for name,faces,bias in [('uniform',(0.,.7,2.2,4.),0.),('biased_three',(0.,.7,2.2,4.),1e-4),
            ('biased_five',(0.,.2,.9,2.1,3.,4.),1e-4)]:
        spec=dict(core=CoreConfig(kind='diffusion',cells=4,axial_bias_reactivity=bias),fuel_grid=FuelGrid(faces))
        trajectories={}
        for preset,options in PRESETS.items():
            s=DistributedCladdingSession(**spec,**options);rows=[];saved=None
            for i,patch in enumerate(ACTIONS):
                rows.append(s.advance(.05,patch or None))
                if i==1:saved=json.loads(json.dumps(s.checkpoint()))
            restored=DistributedCladdingSession.restore(saved)
            replay=restored.advance(.05,{'scram':True});before=s.checkpoint()
            try:s.advance(.01,{'load_fraction':99})
            except ValueError:rollback=s.checkpoint()==before
            else:rollback=False
            trajectories[preset]=dict(snapshots=rows,checks=[gates(r) for r in rows],checkpoint=before,
                restart_exact=replay==rows[-1] and restored.checkpoint()==before,invalid_action_rollback=rollback)
        errors=[relative(vector(a),vector(b)) for a,b in zip(trajectories['coarse']['snapshots'],trajectories['fine']['snapshots'])]
        uniform=None
        if name=='uniform':
            previous=DistributedFuelSession(**spec,**PRESETS['fine']);old=[previous.advance(.05,p or None) for p in ACTIONS]
            uniform=max(relative(physical_vector(a),physical_vector(b)) for a,b in zip(trajectories['fine']['snapshots'],old))
        bdf=bdf_reference(spec)
        final=trajectories['fine']['snapshots'][-1]
        accepted=dict(all_balances=all(all(c.values()) for t in trajectories.values() for c in t['checks']),
            restart=all(t['restart_exact'] for t in trajectories.values()),rollback=all(t['invalid_action_rollback'] for t in trajectories.values()),
            refined=max(errors)<=CRITERIA['refinement_relative'],uniform_limit=uniform is None or uniform<=CRITERIA['uniform_limit_relative'],
            bdf_reference=bdf['error']<=CRITERIA['bdf_reference_relative'],
            scram_heat_release=final['distributed_fuel']['thermal_w']>final['distributed_fuel']['fission_w'],
            local_profile=bias==0 or np.ptp(final['distributed_cladding']['temperature_k'])>1e-6)
        accepted={k:bool(v) for k,v in accepted.items()}
        summary=dict(case=name,passed=all(accepted.values()),gates=accepted,refinement_errors=errors,
            uniform_limit_error=uniform,bdf_reference_error=bdf['error'])
        cases[name]=dict(summary=summary,core=asdict(spec['core']),fuel_faces_m=list(faces),trajectories=trajectories,bdf=bdf)
        summaries.append(summary);atomic_json(out/(name+'.json'),seal(cases[name]))
    report=dict(source=source,driver_sha256=driver,criteria=CRITERIA,cases=summaries,
        passed=all(s['passed'] for s in summaries) and source==identity() and driver==hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        native_trajectories=6,restart_suffixes=6,bdf_hold_references=3,horizon_s=.15,bdf_horizon_s=.02,
        local_cladding_connected=True,distributed_fluid_channels=False,staged_film_supported=False,
        station_env_adapter_available=False,experimental_validation=False,closed_tasks=[])
    atomic_json(out/'report.json',seal(report));atomic_json(out/'scientific.json',seal(dict(report=report,cases=cases)))
    if not report['passed']:raise ValueError('Local cladding qualification failed; results retained')
    return dict(passed=True,cases=len(cases),maximum_refinement_error=max(max(s['refinement_errors']) for s in summaries),
        maximum_bdf_reference_error=max(s['bdf_reference_error'] for s in summaries))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    print(run(parser.parse_args().output))
