"""Short qualification of anchored local moderator feedback, not local XS data."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from atomic_model_multiphysics.provenance import seal
from atomic_model_training.spec import atomic_json
from atomic_model_station import CoreConfig
from coupling.pwr_channel import FluidGrid
from coupling.pwr_stationary import StationaryConfig
from coupling.pwr_moderator import LocalModeratorPWRSession,identity
from scripts.qualify_pwr_stationary import PRESETS,ACTIONS
from scripts.qualify_pwr_distributed_fuel import checks as fuel_checks,relative

CRITERIA=dict(hold_relative=1e-7,refinement_relative=2e-5,inventory_map_relative=1e-12,minimum_active_correction=1e-14)


def run(output):
    out=Path(output).absolute()
    if out.exists() or any(p.is_symlink() for p in (out,*out.parents)):raise ValueError('Fresh ordinary output required')
    out.mkdir(parents=True);source=identity();driver=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    cases={};summaries=[]
    for name,faces,power in [('one_cell',(0.,4.),1.),('three_cells',(0.,1.,2.,4.),1.),('reduced_power',(0.,1.,2.,4.),.95)]:
        print('case',name,flush=True);trajectories={}
        for preset,options in PRESETS.items():
            s=LocalModeratorPWRSession(core=CoreConfig(kind='diffusion',cells=2),fluid_grid=FluidGrid(faces),stationary=StationaryConfig(relative_power=power),**options)
            initial=s.snapshot();reference=initial['local_moderator']['reference'];rows=[];saved=None
            for i,(duration,patch) in enumerate(ACTIONS):
                rows.append(s.advance(duration,patch))
                if i==1:saved=json.loads(json.dumps(s.checkpoint()))
            active=initial['stationary_initialization']['physical_indices']
            hold=relative(np.asarray(rows[0]['coupled_state'])[active],np.asarray(initial['coupled_state'])[active])
            replay=None
            if preset=='fine':
                restored=LocalModeratorPWRSession.restore(saved);end=restored.advance(.02,{'scram':True})
                replay=end==rows[-1] and restored.checkpoint()==s.checkpoint()
            before=s.checkpoint()
            try:s.advance(.01,{'load_fraction':99})
            except ValueError:rollback=s.checkpoint()==before
            else:rollback=False
            init=initial['local_moderator_stationarity']
            correction=max(max(abs(np.r_[r['local_moderator']['density_reactivity_correction'],r['local_moderator']['boron_reactivity_correction']])) for r in rows)
            gates=dict(unchanged_initial_rhs=init['baseline_rhs_max_difference']==0.,
                local_full_rank=init['constrained_rank']==init['constrained_unknowns'],local_initial_residual=init['max_selected_tolerance_ratio']<=1.,
                hold=hold<=CRITERIA['hold_relative'],budgets=all(all(fuel_checks(r).values()) for r in rows),
                map_budgets=all(max(r['local_moderator']['mass_relative_defect'],r['local_moderator']['boron_relative_defect'])<=CRITERIA['inventory_map_relative'] for r in rows),
                reference_frozen=all(r['local_moderator']['reference']==reference for r in rows),
                active_coupling=correction>=CRITERIA['minimum_active_correction'],restart=replay is not False,invalid_action_rollback=rollback)
            trajectories[preset]=dict(initial=initial,snapshots=rows,checkpoint=s.checkpoint(),hold_relative=hold,
                maximum_correction=correction,restart_exact=replay,gates={k:bool(v) for k,v in gates.items()})
            print(name,preset,'completed',flush=True)
        errors=[relative(a['coupled_state'],np.asarray(b['coupled_state'])) for a,b in zip(trajectories['coarse']['snapshots'],trajectories['fine']['snapshots'])]
        gates=dict(all_checks=all(all(t['gates'].values()) for t in trajectories.values()),refined=max(errors)<=CRITERIA['refinement_relative'],
            exact_restart=trajectories['fine']['restart_exact'] is True)
        summary=dict(case=name,relative_power=power,fluid_cells=len(faces)-1,passed=all(gates.values()),gates=gates,
            refinement_errors=errors,hold_relative=max(t['hold_relative'] for t in trajectories.values()),
            maximum_correction=max(t['maximum_correction'] for t in trajectories.values()))
        summaries.append(summary);cases[name]=dict(summary=summary,trajectories=trajectories)
        atomic_json(out/(name+'.json'),seal(cases[name]));print(name,'passed',summary['passed'],flush=True)
    report=dict(source=source,driver_sha256=driver,criteria=CRITERIA,cases=summaries,
        passed=all(c['passed'] for c in summaries) and source==identity() and driver==hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        native_trajectories=6,restart_suffixes=3,baseline_initializations=9,fixed_reference_jacobian_audits=9,total_horizon_s=.14,
        local_density_boron_feedback=True,reference_measured=False,evaluated_local_cross_sections=False,experimental_validation=False,
        long_horizon_accepted=False,station_env_connected=False,closed_tasks=[])
    atomic_json(out/'report.json',seal(report));atomic_json(out/'scientific.json',seal(dict(report=report,cases=cases)))
    if not report['passed']:raise ValueError('Local moderator qualification failed; results retained')
    return dict(passed=True,cases=len(cases),max_refinement_error=max(max(c['refinement_errors']) for c in summaries))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    print(run(parser.parse_args().output))
