"""Component-level liquid-channel qualification, not full PWR acceptance."""
import argparse
from dataclasses import asdict
import hashlib
from pathlib import Path
import json

import numpy as np
from scipy.linalg import expm
from atomic_model_multiphysics.provenance import seal
from atomic_model_training.spec import atomic_json
from coupling.coolant_channel import CoolantChannel,ChannelConfig,Port,identity

PRESETS=dict(coarse=dict(max_step_s=.002,rtol=1e-8),fine=dict(max_step_s=.001,rtol=2e-9))
CRITERIA=dict(balance_relative=1e-9,instantaneous_mass_kg_s=1e-10,instantaneous_energy_w=1e-6,
    instantaneous_boron_kg_s=1e-12,refinement_relative=2e-5,boron_reference_abs_kg=1e-9,rest_state_relative=1e-9)


def specifications():
    return dict(rest=(ChannelConfig(),Port(),Port(),[0.]*3),
        forward_boron=(ChannelConfig(initial_flow_kg_s=5.,loss_coefficient=0.),Port(boron_fraction=.0008),Port(),[0.]*3),
        reverse_boron=(ChannelConfig(initial_flow_kg_s=-5.,loss_coefficient=0.),Port(),Port(boron_fraction=.0002),[0.]*3),
        heated_three=(ChannelConfig(initial_wall_temperature_k=552.),Port(),Port(),[1e4,2e4,5e3]),
        heated_five=(ChannelConfig(faces_m=(0.,.4,1.3,2.1,3.,4.),initial_wall_temperature_k=552.),Port(),Port(),[1e4,2e4,5e3]),
        pressure_reversal=(ChannelConfig(),Port(),Port(15.502e6),[0.]*3))


def checks(row):
    residual=row['terms']['instantaneous_residual']
    return dict(cumulative=max(row['balance_relative'])<=CRITERIA['balance_relative'],
        instantaneous_mass=abs(residual[0])<=CRITERIA['instantaneous_mass_kg_s'],
        instantaneous_energy=abs(residual[1])<=CRITERIA['instantaneous_energy_w'],
        instantaneous_boron=abs(residual[2])<=CRITERIA['instantaneous_boron_kg_s'],
        entropy=row['terms']['conductive_entropy_w_k']>=0.)


def tracer_reference(session,time_s):
    q=session.config.initial_flow_kg_s;mass=session.state[:3*session.cells:3];order=list(range(session.cells))
    if q<0:order.reverse()
    inlet=session.left.boron_fraction if q>0 else session.right.boron_fraction
    matrix=np.zeros((session.cells+1,session.cells+1))
    for k,i in enumerate(order):
        matrix[i,i]=-abs(q)/mass[i]
        if k==0:matrix[i,-1]=abs(q)*inlet
        else:matrix[i,order[k-1]]=abs(q)/mass[order[k-1]]
    return (expm(matrix*time_s)@np.r_[session.state[2:3*session.cells:3],1.])[:-1]


def run(output):
    out=Path(output).absolute()
    if out.exists() or any(p.is_symlink() for p in (out,*out.parents)):raise ValueError('Fresh ordinary output required')
    out.mkdir(parents=True);source=identity();driver=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    cases={};summaries=[]
    for name,(config,left,right,power) in specifications().items():
        trajectories={}
        for preset,options in PRESETS.items():
            s=CoolantChannel(config,left=left,right=right,**options);initial=s.state.copy()
            exact=tracer_reference(s,.01) if name.endswith('_boron') else None
            first=s.advance(.005,power);saved=json.loads(json.dumps(s.checkpoint()))
            second=s.advance(.005,power);restored=CoolantChannel.restore(saved)
            restarted=restored.advance(.005,power)
            before=s.checkpoint()
            try:s.advance(.001,[-1.]*s.wall_cells,left=Port(15.501e6))
            except ValueError:rollback=s.checkpoint()==before
            else:rollback=False
            error=None if exact is None else float(max(abs(s.state[2:3*s.cells:3]-exact)))
            rest=None if name!='rest' else float(max(abs(s.state[:s.ledger_index]-initial[:s.ledger_index])/np.maximum(abs(initial[:s.ledger_index]),1.)))
            trajectories[preset]=dict(snapshots=[first,second],checks=[checks(first),checks(second)],
                restart_exact=restarted==second and restored.checkpoint()==before,invalid_action_rollback=rollback,
                tracer_reference_kg=None if exact is None else exact.tolist(),tracer_max_error_kg=error,rest_state_relative=rest,checkpoint=before)
        errors=[]
        for a,b in zip(trajectories['coarse']['snapshots'],trajectories['fine']['snapshots']):
            errors.append(float(max(abs(np.asarray(a['state'])-b['state'])/np.maximum(abs(np.asarray(b['state'])),1.))))
        gates=dict(balances=all(all(g.values()) for t in trajectories.values() for g in t['checks']),
            restart=all(t['restart_exact'] for t in trajectories.values()),rollback=all(t['invalid_action_rollback'] for t in trajectories.values()),
            refined=max(errors)<=CRITERIA['refinement_relative'],
            exact_tracer=all(t['tracer_max_error_kg'] is None or t['tracer_max_error_kg']<=CRITERIA['boron_reference_abs_kg'] for t in trajectories.values()),
            exact_rest=all(t['rest_state_relative'] is None or t['rest_state_relative']<=CRITERIA['rest_state_relative'] for t in trajectories.values()),
            reversed_pressure_drives_reverse_flow=name!='pressure_reversal' or max(trajectories['fine']['snapshots'][-1]['terms']['face_flow_kg_s'])<0.)
        gates={k:bool(v) for k,v in gates.items()}
        row=dict(case=name,passed=all(gates.values()),gates=gates,refinement_errors=errors,
            tracer_max_error_kg=trajectories['fine']['tracer_max_error_kg'],rest_state_relative=trajectories['fine']['rest_state_relative'])
        summaries.append(row);cases[name]=dict(summary=row,config=asdict(config),left=asdict(left),right=asdict(right),trajectories=trajectories)
        atomic_json(out/(name+'.json'),seal(cases[name]))
    report=dict(source=source,driver_sha256=driver,criteria=CRITERIA,cases=summaries,
        passed=all(s['passed'] for s in summaries) and source==identity() and driver==hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        trajectories=12,restart_suffixes=12,exact_tracer_references=4,horizon_s=.01,
        pwr_connected=False,station_env_connected=False,two_phase_supported=False,experimental_validation=False,closed_tasks=[])
    atomic_json(out/'report.json',seal(report));atomic_json(out/'scientific.json',seal(dict(report=report,cases=cases)))
    if not report['passed']:raise ValueError('Channel qualification failed; attempts retained')
    return dict(passed=True,cases=len(cases),max_refinement_error=max(max(r['refinement_errors']) for r in summaries))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    print(run(parser.parse_args().output))
