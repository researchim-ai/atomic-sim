"""Frozen-policy paired sensor stress tests; no retraining on fault evaluation."""
import argparse
from dataclasses import asdict
from pathlib import Path
import math
import numpy as np
from atomic_model_thermal_agent.environment import signature
from atomic_model_multiphysics.provenance import seal,unseal,sha
from atomic_model_training.spec import atomic_json
from scripts import optimize_thermal_control as control
from scripts import train_thermal_agent as original

FAULTS=('healthy','bias_low','stuck')
POLICIES=('learned','pi','guarded')
SEEDS=(410,411)
ONSET=2


class SensorEnv(control.TailEnv):
    """Alter measurements only, retaining identical RNG draw counts for pairing."""
    def __init__(self,config,fault):
        if fault not in FAULTS:raise ValueError('Declared sensor fault')
        self.fault=fault;self._frozen=None
        super().__init__(config)
    def reset(self,**kwargs):
        self._frozen=None
        return super().reset(**kwargs)
    def _observation(self,snapshot):
        observation=super()._observation(snapshot)
        if self._index>=ONSET:
            if self.fault=='bias_low':observation[0]-=.15
            elif self.fault=='stuck':
                if self._frozen is None:self._frozen=float(observation[0])
                observation[0]=self._frozen
        return observation


class Actor:
    """Measurement-only actor; monitor never receives truth or fault identity."""
    def __init__(self,kind,selected,weights,dt,max_power):
        if kind not in POLICIES:raise ValueError('Declared controller')
        self.kind=kind;self.selected=selected;self.weights=np.asarray(weights);self.dt=dt;self.max_power=max_power
        self.integral=0.;self.last=None;self.repeats=0;self.alarm=False
    def act(self,obs):
        if np.asarray(obs).shape!=(6,) or not np.isfinite(obs).all():raise ValueError('Finite observation6 required')
        self.repeats=self.repeats+1 if self.last==float(obs[0]) else 1;self.last=float(obs[0])
        if self.kind=='guarded' and self.repeats>=3:self.alarm=True
        if self.alarm:return .04
        if self.kind in ('learned','guarded'):
            return float(np.clip(self.selected['learned']*(original.features(obs)@self.weights),0,1))
        error=obs[2]-obs[0];proposal=self.integral+error*self.dt
        ff=8000*(obs[2]-obs[1])+20*(obs[2]-obs[3]);raw=ff+self.selected['pi']*error+1e6*proposal
        if not ((raw>self.max_power and error>0) or (raw<0 and error<0)):self.integral=proposal
        return float(np.clip((ff+self.selected['pi']*error+1e6*self.integral)/self.max_power,0,1))


def rollout(cfg,fault,kind,selected,weights,seed):
    env=SensorEnv(cfg,fault);obs,_=env.reset(seed=seed);actor=Actor(kind,selected,weights,cfg.action_dt_s,cfg.max_power_w)
    trace=[];alarm_index=None
    for index in range(env.config.steps):
        action=actor.act(obs) if index<env.decision_steps else 0.
        if actor.alarm and alarm_index is None:alarm_index=index
        before=obs.tolist();obs,_,terminated,truncated,info=env.step([action])
        trace.append(dict(**info,tail=index>=env.decision_steps,policy_observation=before,action=action,alarm=actor.alarm))
        if terminated or truncated:break
    drained=env._power==0. and all(v==0. for v in env._queue)
    return dict(fault=fault,policy=kind,seed=seed,config=asdict(cfg),trace=trace,
        completed=len(trace)==env.config.steps and not terminated and drained,actuator_drained=drained,
        iae_k_s=math.fsum(s['iae_increment_k_s'] for s in trace),energy_j=math.fsum(s['energy_j'] for s in trace),
        max_error_k=max(abs(s['wall_temperature_k']-s['target_k']) for s in trace),
        alarm_index=alarm_index,alarm_time_s=None if alarm_index is None else alarm_index*cfg.action_dt_s,
        false_alarm=fault=='healthy' and alarm_index is not None)


def paired(rows):
    expected={(fault,policy,c,seed) for fault in FAULTS for policy in POLICIES for c in range(4) for seed in SEEDS}
    lookup={}
    for row in rows:
        key=(row['fault'],row['policy'],row['case'],row['seed'])
        if key in lookup:raise ValueError('Duplicate fault episode')
        if any(not math.isfinite(row[k]) or row[k]<0 for k in ('iae_k_s','energy_j')):raise ValueError('Invalid fault metric')
        lookup[key]=row
    if set(lookup)!=expected:raise ValueError('Incomplete paired fault matrix')
    result=[]
    for fault in FAULTS:
        for policy in POLICIES:
            subset=[lookup[(fault,policy,c,seed)] for c in range(4) for seed in SEEDS]
            ref=[lookup[('healthy',policy,r['case'],r['seed'])] for r in subset]
            result.append(dict(fault=fault,policy=policy,episodes=len(subset),completed=sum(r['completed'] for r in subset),
                mean_iae_k_s=float(np.mean([r['iae_k_s'] for r in subset])),total_energy_j=math.fsum(r['energy_j'] for r in subset),
                mean_delta_iae_vs_healthy_k_s=float(np.mean([r['iae_k_s']-h['iae_k_s'] for r,h in zip(subset,ref,strict=True)])),
                tracking_degraded=sum(r['iae_k_s']>1.25*h['iae_k_s']+1e-6 for r,h in zip(subset,ref,strict=True)),
                alarms=sum(r['alarm_index'] is not None for r in subset),false_alarms=sum(r['false_alarm'] for r in subset)))
    return result


def run(output,training_path):
    output=Path(output).absolute();training_path=Path(training_path).absolute()
    if any(p.is_symlink() for p in (output,*output.parents)):raise ValueError('Symlink output')
    training=unseal(__import__('json').loads(training_path.read_text()))
    # Bind frozen selection to the environment and original calibration driver.
    protocol_path=training_path.with_name('protocol.json');protocol=unseal(__import__('json').loads(protocol_path.read_text()))
    if protocol['source']!=signature() or training['test_used'] is not False or any(sha(p)!=v for p,v in protocol['drivers'].items()):raise ValueError('Frozen training source mismatch')
    output.mkdir(parents=True,exist_ok=False);source=signature();drivers={str(Path(p).absolute()):sha(p) for p in (__file__,control.__file__,original.__file__)}
    atomic_json(output/'protocol.json',seal(dict(source=source,drivers=drivers,training_sha256=sha(training_path),training_protocol_sha256=sha(protocol_path),
        faults=FAULTS,policies=POLICIES,seeds=SEEDS,onset_observation=ONSET,bias_k=-.15,monitor='Three exactly equal wall readings; latched4% request',
        tracking_degradation_rule='IAE > 1.25 * own paired healthy IAE + 1e-6 K*s',blind_test=False)))
    rows=[]
    for case,cfg in enumerate(original.cases('test')):
        for seed in SEEDS:
            for fault in FAULTS:
                for kind in POLICIES:
                    record=rollout(cfg,fault,kind,training['selected'],training['imitation']['weights'],seed);record['case']=case;rows.append(record)
                    atomic_json(output/f'{case}-{seed}-{fault}-{kind}.json',seal(record))
        print('case',case,'done',flush=True)
    summary=paired(rows);unchanged=signature()==source and all(sha(p)==v for p,v in drivers.items()) and sha(training_path)==unseal(__import__('json').loads((output/'protocol.json').read_text()))['training_sha256']
    report=dict(completed=True,source_unchanged=unchanged,passed=unchanged and all(r['completed'] for r in rows),episodes=len(rows),summary=summary,
        passed_means='Complete numerical experiment, not fault robustness or industrial safety',superiority_established=False,validated=False,plant_referenced=False)
    atomic_json(output/'report.json',seal(report));return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--training',type=Path,required=True)
    a=p.parse_args();run(a.output,a.training)
