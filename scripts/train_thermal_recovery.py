"""Train-only robust fallback selection with unseen fault magnitude/onset tests."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import numpy as np
from atomic_model_thermal_agent.environment import signature
from atomic_model_thermal_observer import ThermalMonitor,MonitorConfig,signature as monitor_signature
from atomic_model_multiphysics.provenance import seal,unseal,sha
from atomic_model_training.spec import atomic_json
from scripts import benchmark_thermal_faults as faults
from scripts import optimize_thermal_control as objective

CANDIDATES=('continue','zero','four','ten','twenty')
FRACTIONS={'zero':0.,'four':.04,'ten':.1,'twenty':.2}
TRAIN_SCENARIOS=(('healthy',0.,2),('bias',-.12,2),('stuck',0.,1),('model_mismatch',0.,2))
TEST_SCENARIOS=(('healthy',0.,3),('bias_low',-.22,3),('bias_high',.18,3),('stuck',0.,3),('model_mismatch',0.,3))
TRAIN_SEED=610
TEST_SEEDS=(710,711)


class RecoveryEnv(objective.TailEnv):
    def __init__(self,config,scenario):
        kind,bias,onset=scenario
        if kind not in ('healthy','bias','bias_low','bias_high','stuck','model_mismatch') or type(onset) is not int or not 1<=onset<config.steps or not math.isfinite(bias) or abs(bias)>.5:
            raise ValueError('Recovery scenario domain')
        self.scenario=scenario;self._frozen=None;super().__init__(config)
    def reset(self,**kwargs):
        self._frozen=None;return super().reset(**kwargs)
    def _observation(self,snapshot):
        o=super()._observation(snapshot);kind,bias,onset=self.scenario
        if self._index>=onset:
            if kind.startswith('bias'):o[0]+=bias
            elif kind=='stuck':
                if self._frozen is None:self._frozen=float(o[0])
                o[0]=self._frozen
        return o


def monitor_config(cfg,scenario,split):
    base=dict(dt_s=cfg.action_dt_s,max_power_w=cfg.max_power_w)
    if scenario[0]=='model_mismatch':
        # Deliberately wrong prior, not a change to measured or true plant state.
        c,k=(100.,1000.) if split=='train' else (500.,15000.)
        base.update(capacity_min=c,capacity_max=c,conductance_min=k,conductance_max=k)
    return MonitorConfig(**base)


def recovery_action(candidate,normal,alarm):
    if candidate not in CANDIDATES:raise ValueError('Declared fallback candidate')
    if not alarm or candidate=='continue':return normal
    return FRACTIONS[candidate]


def rollout(cfg,scenario,candidate,training,seed,split):
    if candidate not in CANDIDATES or split not in ('train','test'):raise ValueError('Declared recovery protocol')
    env=RecoveryEnv(cfg,scenario);obs,_=env.reset(seed=seed)
    actor=faults.Actor('learned',training['selected'],training['imitation']['weights'],cfg.action_dt_s,cfg.max_power_w)
    monitor=ThermalMonitor(monitor_config(cfg,scenario,split));trace=[];first=None
    for index in range(env.config.steps):
        diagnostic=monitor.observe(obs)
        if diagnostic['alarm'] and first is None:first=index
        action=recovery_action(candidate,actor.act(obs),diagnostic['alarm']) if index<env.decision_steps else 0.
        before=obs.tolist();obs,_,terminated,truncated,info=env.step([action])
        trace.append(dict(**info,tail=index>=env.decision_steps,policy_observation=before,action=action,diagnostic=diagnostic))
        if terminated or truncated:break
    drained=env._power==0. and all(v==0. for v in env._queue)
    row=dict(candidate=candidate,split=split,scenario=list(scenario),seed=seed,config=asdict(cfg),trace=trace,alarm_index=first,
        completed=len(trace)==env.config.steps and not terminated and drained,actuator_drained=drained,
        iae_k_s=math.fsum(s['iae_increment_k_s'] for s in trace),energy_j=math.fsum(s['energy_j'] for s in trace))
    row['score']=objective.score(row);return row


def select(rows):
    rows=sorted(rows,key=lambda r:(r['candidate'],r['case'],r['scenario'][0]))
    expected={(candidate,case,scenario[0]) for candidate in CANDIDATES for case in range(2) for scenario in TRAIN_SCENARIOS}
    seen=set()
    for r in rows:
        key=(r['candidate'],r['case'],r['scenario'][0])
        if key in seen or r['split']!='train' or r['seed']!=TRAIN_SEED:raise ValueError('Training coverage/provenance')
        if tuple(r['scenario']) not in TRAIN_SCENARIOS:raise ValueError('Unknown train fault')
        seen.add(key)
        if r['completed'] and (type(r['score']) not in (float,int) or not math.isfinite(r['score']) or r['score']<0):raise ValueError('Finite training score required')
    if seen!=expected:raise ValueError('Incomplete train matrix')
    candidates=[]
    for candidate in CANDIDATES:
        subset=[r for r in rows if r['candidate']==candidate]
        if not all(r['completed'] for r in subset):continue
        group_means=[float(np.mean([r['score'] for r in subset if r['scenario'][0]==s[0]])) for s in TRAIN_SCENARIOS]
        candidates.append(dict(candidate=candidate,worst_group_score=max(group_means),mean_score=float(np.mean([r['score'] for r in subset]))))
    if not candidates:raise ValueError('No complete recovery candidate')
    chosen=min(candidates,key=lambda c:(c['worst_group_score'],c['mean_score'],c['candidate']))['candidate']
    return dict(selected=chosen,candidates=candidates,criterion='Minimize worst scenario-group mean J, then total mean J and id',test_used=False)


def summary(rows):
    rows=sorted(rows,key=lambda r:(r['policy'],r['case'],r['scenario'][0],r['seed']))
    expected={(policy,case,s[0],seed) for policy in ('continue','four','selected') for case in range(4) for s in TEST_SCENARIOS for seed in TEST_SEEDS}
    by={}
    for r in rows:
        key=(r['policy'],r['case'],r['scenario'][0],r['seed'])
        if key in by or r['split']!='test' or tuple(r['scenario']) not in TEST_SCENARIOS:raise ValueError('Test pairing/provenance')
        if any(type(r[k]) not in (float,int) or not math.isfinite(r[k]) or r[k]<0 for k in ('iae_k_s','energy_j')):raise ValueError('Invalid test metric')
        by[key]=r
    if set(by)!=expected:raise ValueError('Incomplete test matrix')
    result=[]
    for policy in ('continue','four','selected'):
        for scenario in TEST_SCENARIOS:
            group=[r for r in rows if r['policy']==policy and r['scenario'][0]==scenario[0]]
            healthy=[by[(policy,r['case'],'healthy',r['seed'])] for r in group]
            result.append(dict(policy=policy,scenario=scenario[0],episodes=len(group),completed=sum(r['completed'] for r in group),
                mean_iae_k_s=float(np.mean([r['iae_k_s'] for r in group])),total_energy_j=math.fsum(r['energy_j'] for r in group),
                mean_score=float(np.mean([r['score'] for r in group])) if all(r['completed'] for r in group) else None,
                alarms=sum(r['alarm_index'] is not None for r in group),
                tracking_degraded=sum(r['iae_k_s']>1.25*h['iae_k_s']+1e-6 for r,h in zip(group,healthy,strict=True))))
    return result


def run(output,training_path):
    output=Path(output).absolute();training_path=Path(training_path).absolute()
    if any(p.is_symlink() for p in (output,*output.parents)):raise ValueError('Symlink output')
    training=unseal(json.loads(training_path.read_text()));prior=unseal(json.loads(training_path.with_name('protocol.json').read_text()))
    if training['test_used'] is not False or prior['source']!=signature() or any(sha(p)!=v for p,v in prior['drivers'].items()):raise ValueError('Frozen policy source')
    output.mkdir(parents=True,exist_ok=False);source=signature();ms=monitor_signature()
    drivers={str(Path(p).absolute()):sha(p) for p in (__file__,faults.__file__,objective.__file__,faults.original.__file__)}
    atomic_json(output/'protocol.json',seal(dict(source=source,monitor_source=ms,drivers=drivers,training_sha256=sha(training_path),candidates=CANDIDATES,
        train_scenarios=TRAIN_SCENARIOS,test_scenarios=TEST_SCENARIOS,train_seed=TRAIN_SEED,test_seeds=TEST_SEEDS,
        scoring='Prior fixed normalized IAE +0.25 normalized energy; includes actuator drain tail',blind_test=False)))
    rows=[]
    train_cfg=[faults.original.cases('train')[i] for i in (0,3)]
    for candidate in CANDIDATES:
        for case,cfg in enumerate(train_cfg):
            for scenario in TRAIN_SCENARIOS:
                r=rollout(cfg,scenario,candidate,training,TRAIN_SEED,'train');r['case']=case;rows.append(r)
                atomic_json(output/f'train-{candidate}-{case}-{scenario[0]}.json',seal(r))
        print('train',candidate,'done',flush=True)
    selection=select(rows);atomic_json(output/'selection.json',seal(selection));print('selected',selection['selected'],flush=True)
    tests=[]
    for case,cfg in enumerate(faults.original.cases('test')):
        for scenario in TEST_SCENARIOS:
            for seed in TEST_SEEDS:
                for policy,candidate in [('continue','continue'),('four','four'),('selected',selection['selected'])]:
                    r=rollout(cfg,scenario,candidate,training,seed,'test');r.update(case=case,policy=policy);tests.append(r)
                    atomic_json(output/f'test-{case}-{scenario[0]}-{seed}-{policy}.json',seal(r))
        print('test case',case,'done',flush=True)
    unchanged=source==signature() and ms==monitor_signature() and all(sha(p)==v for p,v in drivers.items())
    report=dict(completed=True,passed=unchanged and all(r['completed'] for r in tests),source_unchanged=unchanged,
        selection=selection,summary=summary(tests),train_episodes=len(rows),test_episodes=len(tests),
        superiority_established=False,validated=False,plant_referenced=False,
        limitations=['Public short synthetic episodes','Finite constant fallback grid, not optimal recovery','Model mismatch is wrong monitor prior, not experimental plant degradation'])
    atomic_json(output/'report.json',seal(report));return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--training',type=Path,required=True)
    a=p.parse_args();run(a.output,a.training)
