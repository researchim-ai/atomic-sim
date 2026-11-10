"""Privileged train-only imitation, partial-observation controller evaluation.

Small public research corpus, not RL training or industrial transfer evidence.
"""
import argparse
from dataclasses import asdict,replace
from pathlib import Path
import numpy as np
from atomic_model_thermal_agent import ThermalEnv,ThermalConfig
from atomic_model_thermal_agent.environment import signature
from atomic_model_multiphysics.provenance import seal,sha
from atomic_model_training.spec import atomic_json


def cases(split):
    if split=='train':rows=[(100,4000,.2,0),(140,7000,.6,1),(180,10000,.3,0),(220,13000,.7,1)]
    elif split=='test':rows=[(120,5500,.3,1),(160,8500,.5,2),(200,11500,.4,1),(240,14000,.6,2)]
    else:raise ValueError('Declared train/test split')
    return [ThermalConfig(capacity_j_k=c,conductance_w_k=k,target_rise_k=t,delay_steps=d,steps=8,noise_std_k=.002 if split=='train' else .004) for c,k,t,d in rows]


def features(obs):
    return np.array([1.,obs[2]-obs[0],obs[0]-obs[1],(obs[0]-obs[3])/10.,obs[4]])


def teacher(env):
    snap=env.model.snapshot();w=snap['walls'][0];f=next(c for c in snap['cells'] if c['id']==w['cell']);cfg=env.config
    # Privileged physical state and parameters are used only for training labels.
    power=cfg.capacity_j_k*(env._target(env._index)-w['temperature_k'])/.0008+cfg.conductance_w_k*(w['temperature_k']-f['temperature_k'])+cfg.ambient_ua_w_k*(w['temperature_k']-cfg.ambient_temperature_k)
    return float(np.clip(power/cfg.max_power_w,0,1))


def fit():
    x=[];y=[]
    for i,cfg in enumerate(cases('train')):
        env=ThermalEnv(cfg);obs,_=env.reset(seed=100+i)
        for _ in range(cfg.steps):
            action=teacher(env);x.append(features(obs));y.append(action)
            obs,_,a,b,_=env.step([action])
            if a or b:break
    x=np.array(x);y=np.array(y);weights=np.linalg.solve(x.T@x+1e-3*np.eye(x.shape[1]),x.T@y)
    return dict(weights=weights.tolist(),samples=len(y),training_cases=[asdict(c) for c in cases('train')],
                fit='Ridge imitation of privileged analytic teacher; no test labels',regularization=.001)


def episode(cfg,kind,weights,seed):
    env=ThermalEnv(cfg);obs,_=env.reset(seed=seed);integral=0.;trace=[]
    for _ in range(cfg.steps):
        error=obs[2]-obs[0]
        if kind=='learned':action=float(np.clip(features(obs)@weights,0,1))
        elif kind=='pi':
            integral+=error*cfg.action_dt_s
            # Fixed nominal controller, explicitly not optimized or best-in-class.
            action=float(np.clip((8000*(obs[2]-obs[1])+50000*error+1e6*integral+20*(obs[2]-obs[3]))/100000,0,1))
        elif kind=='hold':action=.04
        else:raise ValueError('Declared policy')
        obs,reward,terminated,truncated,info=env.step([action]);trace.append(info)
        if terminated or truncated:break
    return dict(policy=kind,seed=seed,config=asdict(cfg),completed=len(trace)==cfg.steps and not terminated,
        iae_k_s=sum(s['iae_increment_k_s'] for s in trace),energy_j=sum(s['energy_j'] for s in trace),violations=sum(s['violations'] for s in trace),
        trace=trace,final_checkpoint=env.checkpoint())


def run(output):
    output=Path(output).absolute()
    if any(p.is_symlink() for p in (output,*output.parents)):raise ValueError('Symlink output')
    output.mkdir(parents=True,exist_ok=False);source=signature();driver=sha(__file__);training=fit()
    atomic_json(output/'training.json',seal(training));records=[];transfers=[]
    for index,cfg in enumerate(cases('test')):
        for kind in ['hold','pi','learned']:
            record=episode(cfg,kind,np.array(training['weights']),300+index)
            atomic_json(output/f'test-{index}-{kind}.json',seal(record));records.append(record)
            print(index,kind,record['completed'],record['iae_k_s'],flush=True)
        if cfg.delay_steps>1:continue  # Preserve actuator delay duration; unsupported finer delay is not a transfer test.
        fine=episode(replace(cfg,action_dt_s=cfg.action_dt_s/2,steps=cfg.steps*2,delay_steps=cfg.delay_steps*2),
                     'learned',np.array(training['weights']),300+index)
        # Delay2 cannot preserve physical duration under supported delay0..3;
        # do not mislabel that changed actuator as pure timestep transfer.
        atomic_json(output/f'transfer-{index}.json',seal(dict(record=fine,delay_duration_preserved=True)))
        transfers.append(dict(test_case=index,completed=fine['completed'],iae_k_s=fine['iae_k_s'],energy_j=fine['energy_j'],delay_s=cfg.delay_steps*cfg.action_dt_s,noise_scope='Independent per-sample noise, not the same continuous sensor path'))
    summaries={k:dict(episodes=sum(r['policy']==k for r in records),completed=sum(r['completed'] for r in records if r['policy']==k),
        mean_iae_k_s=float(np.mean([r['iae_k_s'] for r in records if r['policy']==k])),
        total_energy_j=sum(r['energy_j'] for r in records if r['policy']==k)) for k in ['hold','pi','learned']}
    report=dict(completed=True,source_unchanged=signature()==source and sha(__file__)==driver,source=source,driver=driver,training=training,
        summaries=summaries,transfers=transfers,test_episodes=len(records),test_cases=4,blind_test=False,
        limitations=['Two-millisecond public synthetic tasks','Fixed PI is not tuned; no claim of superiority to optimized MPC/PID','Imitation is not RL; privileged teacher only on train','Different noise discretization on transfer','No equipment validation'],validated=False,plant_referenced=False)
    atomic_json(output/'report.json',seal(report));return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);run(p.parse_args().output)
