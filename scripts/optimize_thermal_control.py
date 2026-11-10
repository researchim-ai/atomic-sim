"""Train-only energy/tracking calibration with a tuned PI and actuator drain tail."""
import argparse
from dataclasses import asdict,replace
import math
from pathlib import Path
import numpy as np
from atomic_model_thermal_agent import ThermalEnv
from atomic_model_thermal_agent.environment import signature
from atomic_model_multiphysics.provenance import seal,sha
from atomic_model_training.spec import atomic_json
from scripts import train_thermal_agent as original

TRAIN_SEEDS=(110,111)
TEST_SEEDS=(310,311,312)
ENERGY_WEIGHT=.25
SCALES=(.15,.3,.6,1.)
PI_GAINS=(20000.,40000.,60000.,100000.)


class TailEnv(ThermalEnv):
    """Keep original target switch, then drain queued actuator requests to zero."""
    def __init__(self,config):
        self.decision_steps=config.steps
        self.tail_steps=config.delay_steps+math.ceil(config.max_power_w/(config.slew_w_per_s*config.action_dt_s))+2
        super().__init__(replace(config,steps=config.steps+self.tail_steps))
    def _target(self,index):
        return self.config.fluid_temperature_k+self.config.target_rise_k*(.5 if index>=self.decision_steps//2 else 1.)


def score(record):
    if not record['completed']:return None
    values=[record['iae_k_s'],record['energy_j']]
    if any(type(v) not in (float,int) or not math.isfinite(v) or v<0 for v in values):raise ValueError('Finite nonnegative metrics required')
    cfg=record['config'];duration=cfg['steps']*cfg['action_dt_s']
    return record['iae_k_s']/(duration*cfg['target_rise_k'])+ENERGY_WEIGHT*record['energy_j']/(duration*cfg['max_power_w'])


def rollout(config,kind,parameter,weights,seed):
    if kind not in ('learned','pi','original','hold'):raise ValueError('Declared controller')
    env=TailEnv(config);obs,_=env.reset(seed=seed);trace=[];integral=0.
    for index in range(env.config.steps):
        if index>=env.decision_steps:action=0.
        elif kind in ('learned','original'):
            action=float(np.clip((parameter if kind=='learned' else 1.)*(original.features(obs)@weights),0,1))
        elif kind=='hold':action=.04
        else:
            error=obs[2]-obs[0];proposal=integral+error*config.action_dt_s
            feedforward=8000*(obs[2]-obs[1])+20*(obs[2]-obs[3])
            raw=feedforward+parameter*error+1e6*proposal
            if not ((raw>config.max_power_w and error>0) or (raw<0 and error<0)):integral=proposal
            action=float(np.clip((feedforward+parameter*error+1e6*integral)/config.max_power_w,0,1))
        obs,_,terminated,truncated,info=env.step([action]);info['tail']=index>=env.decision_steps;trace.append(info)
        if terminated or truncated:break
    drained=env._power==0. and all(v==0. for v in env._queue)
    record=dict(policy=kind,parameter=parameter,seed=seed,config=asdict(config),tail_steps=env.tail_steps,
        completed=len(trace)==env.config.steps and not terminated and drained,
        iae_k_s=math.fsum(s['iae_increment_k_s'] for s in trace),energy_j=math.fsum(s['energy_j'] for s in trace),
        active_energy_j=math.fsum(s['energy_j'] for s in trace if not s['tail']),tail_energy_j=math.fsum(s['energy_j'] for s in trace if s['tail']),
        actuator_drained=drained,trace=trace)
    record['score']=score(record);return record


def select(rows):
    if not rows or any(r['parameter'] is None for r in rows):raise ValueError('Candidate scores required')
    valid=[r for r in rows if r['all_completed']]
    if not valid:raise ValueError('No complete train candidate')
    for r in valid:
        if type(r['mean_score']) not in (float,int) or not math.isfinite(r['mean_score']) or r['mean_score']<0:raise ValueError('Invalid training score')
    return min(valid,key=lambda r:(r['mean_score'],r['parameter']))['parameter']


def run(output):
    output=Path(output).absolute()
    if any(p.is_symlink() for p in (output,*output.parents)):raise ValueError('Symlink output')
    output.mkdir(parents=True,exist_ok=False);source=signature();drivers={str(Path(p).absolute()):sha(p) for p in (__file__,original.__file__)}
    training=original.fit();weights=np.array(training['weights']);search={};selected={}
    atomic_json(output/'protocol.json',seal(dict(source=source,drivers=drivers,train_seeds=TRAIN_SEEDS,test_seeds=TEST_SEEDS,
        energy_weight=ENERGY_WEIGHT,scales=SCALES,pi_gains=PI_GAINS,train_configs=[asdict(c) for c in original.cases('train')],
        test_configs=[asdict(c) for c in original.cases('test')],blind_test=False,selection='Mean normalized tracking plus energy, complete train paths only; test never selects')))
    for kind,parameters in [('learned',SCALES),('pi',PI_GAINS)]:
        rows=[]
        for parameter in parameters:
            samples=[]
            for index,cfg in enumerate(original.cases('train')):
                for seed in TRAIN_SEEDS:
                    record=rollout(cfg,kind,parameter,weights,seed);samples.append(record)
                    atomic_json(output/f'train-{kind}-{parameter}-{index}-{seed}.json',seal(record))
            complete=all(r['completed'] for r in samples)
            rows.append(dict(parameter=parameter,all_completed=complete,episodes=len(samples),mean_score=float(np.mean([r['score'] for r in samples])) if complete else None))
            print('train',kind,parameter,rows[-1]['mean_score'],flush=True)
        search[kind]=rows;selected[kind]=select(rows)
    # Freeze train selection before touching test outcomes.
    atomic_json(output/'training.json',seal(dict(imitation=training,search=search,selected=selected,test_used=False)))
    results=[]
    for index,cfg in enumerate(original.cases('test')):
        for seed in TEST_SEEDS:
            for kind in ('original','learned','pi','hold'):
                record=rollout(cfg,kind,selected.get(kind,1.),weights,seed);record['case']=index
                results.append(record);atomic_json(output/f'test-{kind}-{index}-{seed}.json',seal(record))
        print('test case',index,'completed',flush=True)
    summary={kind:dict(episodes=sum(r['policy']==kind for r in results),completed=sum(r['completed'] for r in results if r['policy']==kind),
        mean_iae_k_s=float(np.mean([r['iae_k_s'] for r in results if r['policy']==kind])),
        total_energy_j=math.fsum(r['energy_j'] for r in results if r['policy']==kind),
        total_tail_energy_j=math.fsum(r['tail_energy_j'] for r in results if r['policy']==kind),
        mean_score=float(np.mean([r['score'] for r in results if r['policy']==kind])) if all(r['completed'] for r in results if r['policy']==kind) else None) for kind in ('original','learned','pi','hold')}
    unchanged=source==signature() and all(sha(p)==v for p,v in drivers.items())
    report=dict(completed=True,source_unchanged=unchanged,source=source,drivers=drivers,training=training,search=search,selected=selected,summary=summary,
        test_episodes=len(results),train_calibration_episodes=64,paired_noise_seeds=True,blind_test=False,
        passed=unchanged and all(r['completed'] for r in results),superiority_established=False,
        scope='Public two-millisecond decisions plus finite actuator drain; fixed objective, finite train grids, no industrial optimum',validated=False,plant_referenced=False)
    atomic_json(output/'report.json',seal(report));return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);run(p.parse_args().output)
