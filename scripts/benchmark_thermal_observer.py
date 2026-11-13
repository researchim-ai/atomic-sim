"""Frozen-policy sensor consistency observer under broad parameter hypotheses."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import math
import numpy as np
from atomic_model_thermal_observer import ThermalMonitor,MonitorConfig,signature as monitor_signature
from atomic_model_thermal_agent.environment import signature
from atomic_model_multiphysics.provenance import seal,unseal,sha
from atomic_model_training.spec import atomic_json
from scripts import benchmark_thermal_faults as faults

POLICIES=('learned','repeat_guard','observer_guard')
SEEDS=(510,511)


def rollout(cfg,fault,policy,training,seed):
    if policy not in POLICIES:raise ValueError('Declared observer policy')
    env=faults.SensorEnv(cfg,fault);obs,_=env.reset(seed=seed)
    actor=faults.Actor('guarded' if policy=='repeat_guard' else 'learned',training['selected'],training['imitation']['weights'],cfg.action_dt_s,cfg.max_power_w)
    monitor=ThermalMonitor(MonitorConfig(dt_s=cfg.action_dt_s,max_power_w=cfg.max_power_w))
    trace=[];first=None
    for index in range(env.config.steps):
        observation=obs.tolist();diagnostic=monitor.observe(obs)
        if policy=='observer_guard':
            action=.04 if diagnostic['alarm'] else actor.act(obs)
            alarm=diagnostic['alarm']
        else:action=actor.act(obs);alarm=actor.alarm
        if index>=env.decision_steps:action=0.
        if alarm and first is None:first=index
        obs,_,terminated,truncated,info=env.step([action])
        trace.append(dict(**info,tail=index>=env.decision_steps,policy_observation=observation,action=action,
            alarm=alarm,observer=diagnostic))
        if terminated or truncated:break
    drained=env._power==0. and all(v==0. for v in env._queue)
    return dict(policy=policy,fault=fault,seed=seed,config=asdict(cfg),completed=len(trace)==env.config.steps and not terminated and drained,
        iae_k_s=math.fsum(s['iae_increment_k_s'] for s in trace),energy_j=math.fsum(s['energy_j'] for s in trace),trace=trace,
        alarm_index=first,false_alarm=fault=='healthy' and first is not None,monitor_checkpoint=monitor.checkpoint())


def summarize(rows):
    expected={(f,p,c,s) for f in faults.FAULTS for p in POLICIES for c in range(4) for s in SEEDS}
    by={}
    for r in rows:
        key=(r['fault'],r['policy'],r['case'],r['seed'])
        if key in by:raise ValueError('Duplicate observer episode')
        if any(type(r[k]) not in (float,int) or not math.isfinite(r[k]) or r[k]<0 for k in ['iae_k_s','energy_j']):raise ValueError('Invalid observer metric')
        by[key]=r
    if set(by)!=expected:raise ValueError('Incomplete observer matrix')
    result=[]
    for f in faults.FAULTS:
        for p in POLICIES:
            group=[by[(f,p,c,s)] for c in range(4) for s in SEEDS]
            healthy=[by[('healthy',p,r['case'],r['seed'])] for r in group]
            result.append(dict(fault=f,policy=p,episodes=len(group),completed=sum(r['completed'] for r in group),
                alarms=sum(r['alarm_index'] is not None for r in group),false_alarms=sum(r['false_alarm'] for r in group),
                tracking_degraded=sum(r['iae_k_s']>1.25*h['iae_k_s']+1e-6 for r,h in zip(group,healthy,strict=True)),
                mean_iae_k_s=float(np.mean([r['iae_k_s'] for r in group])),total_energy_j=math.fsum(r['energy_j'] for r in group)))
    return result


def run(output,training_path):
    output=Path(output).absolute();training_path=Path(training_path).absolute()
    if any(p.is_symlink() for p in (output,*output.parents)):raise ValueError('Symlink output')
    training=unseal(json.loads(training_path.read_text()));tp=unseal(json.loads(training_path.with_name('protocol.json').read_text()))
    if training['test_used'] is not False or tp['source']!=signature() or any(sha(p)!=v for p,v in tp['drivers'].items()):raise ValueError('Frozen training source')
    source=signature();ms=monitor_signature();drivers={str(Path(p).absolute()):sha(p) for p in (__file__,faults.__file__,faults.control.__file__,faults.original.__file__)}
    output.mkdir(parents=True,exist_ok=False)
    atomic_json(output/'protocol.json',seal(dict(source=source,monitor_source=ms,drivers=drivers,training_sha256=sha(training_path),
      monitor_config=asdict(MonitorConfig()),seeds=SEEDS,policies=POLICIES,faults=faults.FAULTS,
      scope='One-step corner hypothesis envelope, assumed noise/model margins; no formal enclosure or unique sensor diagnosis',blind_test=False)))
    rows=[]
    for case,cfg in enumerate(faults.original.cases('test')):
        for seed in SEEDS:
            for fault in faults.FAULTS:
                for policy in POLICIES:
                    row=rollout(cfg,fault,policy,training,seed);row['case']=case;rows.append(row)
                    atomic_json(output/f'{case}-{seed}-{fault}-{policy}.json',seal(row))
        print('case',case,'done',flush=True)
    unchanged=source==signature() and ms==monitor_signature() and all(sha(p)==v for p,v in drivers.items())
    report=dict(completed=True,passed=unchanged and all(r['completed'] for r in rows),source_unchanged=unchanged,episodes=len(rows),summary=summarize(rows),
      passed_means='Complete numerical experiment, not guaranteed robustness',superiority_established=False,validated=False,plant_referenced=False)
    atomic_json(output/'report.json',seal(report));return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--training',type=Path,required=True)
    a=p.parse_args();run(a.output,a.training)
