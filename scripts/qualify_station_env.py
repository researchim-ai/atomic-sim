"""Measured station API, Gymnasium, JSONL and opaque planning qualification."""
from argparse import ArgumentParser
from copy import deepcopy
from dataclasses import asdict
import io
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from atomic_model_multiphysics.provenance import sha
from atomic_model_training.spec import atomic_json
from atomic_model_station_faults import Fault
from atomic_model_station_env import EnvConfig,StationEnv,StationGymEnv,ActionError
from atomic_model_station_env.model import signature
from atomic_model_station_env.__main__ import serve
from scripts import qualify_station_actuators as evidence

TASKS=['NPP-14.01','NPP-14.02']


def env_signature():return signature()


def plain(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,(tuple,list)):return [plain(v) for v in value]
    if isinstance(value,dict):return {k:plain(v) for k,v in value.items()}
    return value


def run(output,reference=None):
    output=Path(output).absolute()
    if any(p.is_symlink() for p in (output,*output.parents)):raise ValueError('Symlink output refused')
    output.mkdir(parents=True,exist_ok=False)
    source=signature();driver=sha(__file__);serializer=sha(evidence.__file__)
    cfg=EnvConfig(max_steps=4,faults=(Fault('hidden_source','bias','pressure',2,value=150000.),),react_on_alarm=True)
    evidence.save(output/'protocol.json',dict(source=source,driver_sha256=driver,serializer_sha256=serializer,
        candidate_tasks=TASKS,config=asdict(cfg),nominal_duration_s=4.,trip_duration_s=4.,
        scope='Short measured station API and explicitly privileged model-assisted one-step planning; not full R1 or task/train-split acceptance'))
    direct=StationEnv(cfg);reset=direct.reset(seed=5);rows=[];saved=None
    for index in range(4):
        rows.append(plain(direct.step()))
        if index==0:saved=json.loads(json.dumps(direct.checkpoint(),sort_keys=True))
    restored=StationEnv.restore(saved)
    restart=True
    for row in rows[1:]:restart=plain(restored.step())==row and restart
    restart=restart and restored.checkpoint()==direct.checkpoint()
    evidence.save(output/'python.json',dict(reset=plain(reset),trace=rows,checkpoint=direct.checkpoint()))
    evidence.save(output/'restart.json',dict(saved=saved,exact=bool(restart),final=restored.checkpoint()))
    requests=[dict(method='reset',seed=5)]+[dict(method='step',action={}) for _ in range(4)]
    response=io.StringIO();transport=StationEnv(cfg)
    serve(io.StringIO(''.join(json.dumps(r)+'\n' for r in requests)),response,transport)
    wire=[json.loads(line) for line in response.getvalue().splitlines()]
    wire_exact=(all(r['ok'] for r in wire) and wire[0]['result']==dict(observation=reset[0],info=reset[1])
        and all(r['result']==dict(observation=t[0],reward=t[1],terminated=t[2],truncated=t[3],info=t[4]) for r,t in zip(wire[1:],rows))
        and transport.checkpoint()==direct.checkpoint())
    evidence.save(output/'jsonl.json',dict(trace=wire,exact=bool(wire_exact)))
    # A fresh interpreter executes the installed/source JSONL implementation.
    from atomic_model_station_env import model
    root=str(Path(model.__file__).resolve().parents[1])
    bootstrap='import sys;sys.path.insert(0,sys.argv[1]);from atomic_model_station_env.__main__ import serve;serve(sys.stdin,sys.stdout)'
    messages=[dict(method='reset',seed=8),dict(method='step',action={}),dict(method='checkpoint'),dict(method='observe')]
    completed=subprocess.run([sys.executable,'-I','-c',bootstrap,root],input=''.join(json.dumps(r)+'\n' for r in messages),
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,check=True,timeout=120)
    fresh=[json.loads(s) for s in completed.stdout.splitlines()]
    reference_env=StationEnv();r0=reference_env.reset(seed=8);r1=reference_env.step()
    process_ok=(fresh[0]['result']==dict(observation=r0[0],info=r0[1]) and fresh[1]['result']['observation']==r1[0]
        and not fresh[2]['ok'] and fresh[3]['result']['info']['steps']==1)
    evidence.save(output/'new_process.json',dict(trace=fresh,exact=bool(process_ok)))
    gym=StationGymEnv(cfg);python=StationEnv(cfg);g0=gym.reset(seed=4);python.reset(seed=4)
    gym_rows=[];gym_equal=True
    for a in ([0.,0.,0.],[.1,-.1,.2],[0.,0.,0.],[0.,0.,0.]):
        action=dict(setpoints=np.array(a),scram=0)
        actual=gym.step(action)
        command=dict(operator=dict(load_fraction=1.+.02*a[2]),targets=dict(rod_target=3.15e9*(1+.02*a[0]),steam_admission=157.*(1+.005*a[1])),modes=dict(rod_target='auto',steam_admission='auto'))
        expected=python.step(command)
        gym_equal=gym_equal and gym.observation_space.contains(actual[0]) and plain(actual[0])==plain(gym._vector(expected[0]))
        gym_equal=gym_equal and actual[1:4]==expected[1:4] and actual[4]['raw_observation']==expected[0]
        gym_rows.append(plain(actual))
    gym_equal=gym_equal and gym.station.checkpoint()==python.checkpoint()
    evidence.save(output/'gymnasium.json',dict(initial=plain(g0),trace=gym_rows,exact=bool(gym_equal)))
    planner=StationEnv(EnvConfig(max_steps=3,tick_s=.1,automatic=False,
        faults=(Fault('hidden_planning_case','pump_off','cooling',1),)))
    planner.reset(seed=2);before=planner.checkpoint()
    comparison=planner.preview([{},dict(operator=dict(load_fraction=.98)),dict(operator=dict(scram=True)),dict(operator=dict(rod_target=True))])
    branch_unchanged=planner.checkpoint()==before
    public=json.dumps(comparison,allow_nan=False)
    opaque=not any(s in public for s in ('hidden_planning_case','checkpoint','base_sha256','start_tick','fault_memory'))
    chosen=planner.apply(json.loads(public),1)
    selected=plain(chosen)==comparison['payload']['candidates'][1]['transition']
    after=planner.checkpoint();stale=False
    try:planner.apply(comparison,0)
    except ActionError as error:stale=error.code=='stale_branch' and planner.checkpoint()==after
    evidence.save(output/'branches.json',dict(comparison=comparison,source_unchanged=branch_unchanged,
        public_opaque=opaque,selected_exact=bool(selected),stale_rejected_atomically=stale,final=after))
    # Hidden future scenario identity must not leak through initial frames or plan tokens.
    a=StationEnv(EnvConfig(max_steps=2,tick_s=.01,automatic=False));b=StationEnv(EnvConfig(max_steps=2,tick_s=.01,automatic=False,
        faults=(Fault('hidden_future','bias','pressure',2,value=150000.),)))
    same_reset=a.reset(seed=1)==b.reset(seed=1)
    same_plan=a.preview([{}])==b.preview([{}])
    evidence.save(output/'privacy.json',dict(initial_frames_exact=same_reset,initial_plans_exact=same_plan,
        private_checkpoints_different=a.checkpoint()!=b.checkpoint()))
    invalid=StationEnv(EnvConfig(max_steps=2,tick_s=.01));invalid.reset();before=invalid.checkpoint();rollback=False
    try:invalid.step(dict(operator=dict(grid_available=True)))
    except ActionError as error:rollback=error.code=='invalid_action' and invalid.checkpoint()==before
    done=StationEnv(EnvConfig(max_steps=1,tick_s=.01));done.reset();last=done.step();end_before=done.checkpoint();terminal_guard=False
    try:done.step()
    except ActionError as error:terminal_guard=error.code=='episode_finished' and done.checkpoint()==end_before
    evidence.save(output/'rejections.json',dict(invalid_action_rollback=rollback,post_horizon_rollback=terminal_guard,
        horizon_is_truncation=bool(last[3] and not last[2])))
    exposed=json.dumps(plain([reset,rows,wire,fresh,g0,gym_rows]),allow_nan=False)
    hidden=not any(s in exposed for s in ('hidden_source','hidden_future','start_tick','end_tick','physical_state','fault_memory','stuck_positions'))
    unchanged=source==signature() and driver==sha(__file__) and serializer==sha(evidence.__file__)
    passed=bool(unchanged and restart and wire_exact and process_ok and gym_equal and branch_unchanged and opaque and selected
        and stale and same_reset and same_plan and rollback and terminal_guard and hidden)
    report=dict(completed=True,passed=passed,source_unchanged=unchanged,closed_tasks=TASKS if passed else [],
        nominal_duration_s=4.,trip_duration_s=4.,restart_exact=bool(restart),jsonl_exact=bool(wire_exact),
        new_process_exact=bool(process_ok),gymnasium_exact=bool(gym_equal),agent_payloads_hide_private_state=hidden,
        branch_outputs_opaque=opaque,branch_commit_exact=bool(selected),stale_branch_rollback=stale,
        initial_private_scenarios_publicly_indistinguishable=bool(same_reset and same_plan),
        full_r1_matrix=False,complete_npp_11=False,complete_npp_14=False,r1_accepted=False,validated=False,
        limitations=['Short API/replay scope, not full station physical qualification',
            'Planning is explicitly model-assisted and privileged; ordinary JSONL agents cannot invoke it',
            'Generic measured tracking reward, not accepted industrial task/utility definition',
            'Task suite and held-out train/dev/test split NPP-14.03/04 remain open'])
    evidence.save(output/'report.json',report)
    if reference is not None:
        reference=Path(reference);names={p.name for p in output.glob('*.json')}
        if names!={p.name for p in reference.glob('*.json')}:raise ValueError('Environment evidence sets differ')
        for name in sorted(names):
            if evidence.read(output/name)!=evidence.read(reference/name):raise ValueError('Installed environment evidence differs:'+name)
        print('All installed station environment evidence matches exactly',flush=True)
    if not passed:raise ValueError('Station environment qualification failed')
    print(json.dumps(dict(passed=True,closed_tasks=TASKS)),flush=True);return report


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference',type=Path);args=parser.parse_args();run(args.output,args.reference)
