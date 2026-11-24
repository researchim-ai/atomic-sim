"""Normal-hot stationary PWR channel: full residuals, hold and restart checks."""
import argparse
from pathlib import Path
import hashlib
import json

import numpy as np
from atomic_model_multiphysics.provenance import seal
from atomic_model_training.spec import atomic_json
from atomic_model_station import CoreConfig
from coupling.pwr_channel import FluidGrid
from coupling.pwr_stationary import StationaryChannelPWRSession,StationaryConfig,identity
from scripts.qualify_pwr_distributed_fuel import checks as fuel_checks,relative

PRESETS=dict(coarse=dict(max_step_s=.1,rtol=2e-8),fine=dict(max_step_s=.02,rtol=2e-9))
CRITERIA=dict(physical_tolerance_ratio=1.,hold_relative=1e-7,refinement_relative=2e-5)
ACTIONS=((.1,None),(.02,{'load_fraction':.99}),(.02,{'scram':True}))


def run(output):
    out=Path(output).absolute()
    if out.exists() or any(p.is_symlink() for p in (out,*out.parents)):raise ValueError('Fresh ordinary output required')
    out.mkdir(parents=True);source=identity();driver=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    cases={};summaries=[]
    for name,faces,power in [('one_cell',(0.,4.),1.),('three_cells',(0.,1.,2.,4.),1.),('reduced_power',(0.,1.,2.,4.),.95)]:
        print('case',name,flush=True);trajectories={}
        for preset,options in PRESETS.items():
            s=StationaryChannelPWRSession(core=CoreConfig(kind='diffusion',cells=2),fluid_grid=FluidGrid(faces),stationary=StationaryConfig(relative_power=power),**options)
            initial=s.snapshot();rows=[];saved=None
            for i,(duration,patch) in enumerate(ACTIONS):
                rows.append(s.advance(duration,patch))
                if i==1:saved=json.loads(json.dumps(s.checkpoint()))
            report=initial['stationary_initialization'];active=report['physical_indices']
            hold=relative(np.asarray(rows[0]['coupled_state'])[active],np.asarray(initial['coupled_state'])[active])
            replay=None
            if preset=='fine':
                restored=StationaryChannelPWRSession.restore(saved);replayed=restored.advance(.02,{'scram':True})
                replay=replayed==rows[-1] and restored.checkpoint()==s.checkpoint()
            before=s.checkpoint()
            try:s.advance(.01,{'load_fraction':99})
            except ValueError:rollback=s.checkpoint()==before
            else:rollback=False
            gates=dict(full_initial_residual=report['max_physical_tolerance_ratio']<=CRITERIA['physical_tolerance_ratio'],
                full_rank=report['constrained_rank']==report['constrained_unknowns'],hold=hold<=CRITERIA['hold_relative'],
                transient_budgets=all(all(fuel_checks(r).values()) for r in rows),
                restart=replay is not False,invalid_command_rollback=rollback,
                no_parameter_fit=not report['parameter_changes'] and not report['pressure_control_settings_changed'])
            trajectories[preset]=dict(initial=initial,snapshots=rows,checkpoint=s.checkpoint(),hold_relative=hold,restart_exact=replay,
                gates={k:bool(v) for k,v in gates.items()})
            print(name,preset,'completed',flush=True)
        errors=[relative(a['coupled_state'],np.asarray(b['coupled_state'])) for a,b in zip(trajectories['coarse']['snapshots'],trajectories['fine']['snapshots'])]
        gates=dict(all_checks=all(all(t['gates'].values()) for t in trajectories.values()),
            refined=max(errors)<=CRITERIA['refinement_relative'],restart_accepted=trajectories['fine']['restart_exact'] is True)
        summary=dict(case=name,relative_power=power,fluid_cells=len(faces)-1,passed=all(gates.values()),gates=gates,
            refinement_errors=errors,hold_relative=max(t['hold_relative'] for t in trajectories.values()))
        summaries.append(summary);cases[name]=dict(summary=summary,trajectories=trajectories)
        atomic_json(out/(name+'.json'),seal(cases[name]));print(name,'passed',summary['passed'],flush=True)
    report=dict(source=source,driver_sha256=driver,criteria=CRITERIA,cases=summaries,
        passed=all(c['passed'] for c in summaries) and source==identity() and driver==hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        initializations=9,native_trajectories=6,restart_suffixes=3,hold_horizon_s=.1,total_horizon_s=.14,
        stationary_normal_hot_start=True,experimental_validation=False,long_horizon_accepted=False,station_env_connected=False,closed_tasks=[])
    atomic_json(out/'report.json',seal(report));atomic_json(out/'scientific.json',seal(dict(report=report,cases=cases)))
    if not report['passed']:raise ValueError('Stationary PWR qualification failed; results retained')
    return dict(passed=True,cases=len(cases),max_refinement_error=max(max(c['refinement_errors']) for c in summaries))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    print(run(parser.parse_args().output))
