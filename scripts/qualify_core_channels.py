"""Finite numerical qualification of two-mesh synthetic transient coupling."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from coupling.core_channels import ChannelConfig,CoreChannels,ChannelSession,identity
from atomic_model_training.spec import atomic_json
from atomic_model_multiphysics.provenance import seal,unseal

STEPS=(.02,.01,.005)
CRITERIA=dict(power_relative=1e-12,pairing_relative=1e-12,energy_relative=1e-8,
    precursor_inventory_relative=1e-10,reference_agreement=2e-7,temporal_order_min=.8,
    finest_normalized_error=.005,point_limit_error=1e-8)


def cases():
    return dict(uniform_linear=ChannelConfig(alpha_per_k=0.),
        local_feedback=ChannelConfig(reactivity=(1e-4,0.,-5e-5,5e-5)),
        nonmatching_five=ChannelConfig(channel_faces=(0.,.2,.9,2.1,3.,4.),reactivity=(1e-4,0.,-5e-5,5e-5)))


def run(output):
    out=Path(output).absolute()
    if out.exists() or any(p.is_symlink() for p in (out,*out.parents)):raise ValueError('Fresh ordinary output required')
    out.mkdir(parents=True);source=identity();recipe_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    rows=[]
    for name,config in cases().items():
        model=CoreChannels(config);initial=model.initial();refs={}
        for method in ('Radau','BDF'):
            sol=solve_ivp(model.rhs,(0.,.2),initial,method=method,jac=model.jacobian,
                rtol=1e-10,atol=1e-11*model.scale,max_step=.001)
            if not sol.success or sol.t[-1]!=.2:raise ValueError('Independent IVP algorithm failed')
            refs[method]=sol.y[:,-1]
        disagreement=float(np.max(abs(refs['Radau']-refs['BDF'])/model.scale))
        errors=[];trajectories=[]
        for h in STEPS:
            session=ChannelSession(config);result=session.advance(.2,step_s=h)
            error=float(np.max(abs(session.state-refs['Radau'])/model.scale));errors.append(error)
            trajectories.append(dict(step_s=h,normalized_error=error,result=result))
        orders=[float(np.log(errors[i]/errors[i+1])/np.log(2.)) for i in range(2)]
        original=ChannelSession(config);original.advance(.1,step_s=.01)
        saved=json.loads(json.dumps(original.checkpoint()));restored=ChannelSession.restore(saved)
        same=original.advance(.1,step_s=.005)==restored.advance(.1,step_s=.005) and original.checkpoint()==restored.checkpoint()
        forged=unseal(saved);forged['state'][0]+=1e-6;corruption_refused=False
        try:ChannelSession.restore(seal(forged))
        except ValueError:corruption_refused=True
        point_error=None
        if name=='uniform_linear':
            point=expm(model.kinetics.matrix(config.reactivity[0])*.2)@model.kinetics.initial_state()
            point_error=float(np.max(abs(model.core.project(refs['Radau'][:model.k])-point)/np.maximum(abs(point),1.)))
        audit=[t['result']['audit'] for t in trajectories]
        checks=dict(references_agree=disagreement<=CRITERIA['reference_agreement'],
            temporal_order=min(orders)>=CRITERIA['temporal_order_min'],
            finest_error=errors[-1]<=CRITERIA['finest_normalized_error'],restart_exact=same,
            resealed_corruption_refused=corruption_refused,
            point_limit=point_error is None or point_error<=CRITERIA['point_limit_error'],
            power_transfer=max(a['power_relative_defect'] for a in audit)<=CRITERIA['power_relative'],
            temperature_pairing=max(a['pairing_relative_defect'] for a in audit)<=CRITERIA['pairing_relative'],
            heat_balance=max(a['energy_relative_defect'] for a in audit)<=CRITERIA['energy_relative'],
            neutron_inventory=max(a['precursor_inventory_relative_defect'] for a in audit)<=CRITERIA['precursor_inventory_relative'])
        row=dict(name=name,config=asdict(config),checks=checks,passed=all(checks.values()),
            reference_disagreement=disagreement,reference_states={m:v.tolist() for m,v in refs.items()},
            temporal_orders=orders,point_limit_error=point_error,trajectories=trajectories,
            restart_checkpoint_sha256=original.checkpoint()['sha256'])
        rows.append(row);atomic_json(out/(name+'.json'),seal(row))
    report=dict(source=source,recipe_sha256=recipe_sha,criteria=CRITERIA,cases=rows,
        passed=all(row['passed'] for row in rows) and source==identity() and recipe_sha==hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        native_trajectories=9,reference_trajectories=6,restart_pairs=3,
        independent_numerical_algorithms=True,independent_physical_data=False,
        local_neutron_precursors=True,all_fission_heat_prompt=True,decay_heat_supported=False,
        station_connected=False,experimental_validation=False,closed_tasks=[])
    atomic_json(out/'report.json',seal(report))
    if not report['passed']:raise ValueError('Synthetic coupled qualification failed; results retained')
    return dict(passed=True,native_trajectories=9,reference_trajectories=6,
        minimum_temporal_order=min(min(r['temporal_orders']) for r in rows))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    print(run(parser.parse_args().output))
