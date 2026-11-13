"""Transactional fixed-step index-1 R0 runtime with root events and full replay."""
from copy import deepcopy
from dataclasses import asdict
import math
import numpy as np
from atomic_model_multiphysics.provenance import seal,unseal
from .model import StationConfig,signature
from .nonlinear import build,guess,commands


class MixedStationSession:
    def __init__(self,config=None,*,max_step_s=.2,seed=0):
        if type(max_step_s) not in (float,int) or not .01<=max_step_s<=1 or type(seed) is not int or not 0<=seed<2**32:raise ValueError('Solver step/seed domain')
        self.config=config if config is not None else StationConfig();self.max_step_s=float(max_step_s);self.seed=seed
        self.system=build(self.config);self.u=dict(source_power_w=self.config.nominal_source_power_w,generator_connected=True,load_fraction=1.)
        self.y,self.initialization=self.system.steady(guess(self.config),self.u)
        self.y,self.rate,_=self.system.consistent(0.,self.y,self.u)
        self.time=0.;self.ledger=dict(source_j=0.,export_j=0.,cooling_j=0.,residual_j=0.)
        self.queue=[];self.guards=[];self.events=[];self.revision=0;self.history=[]
        self.rng=np.random.default_rng(seed);self.measurement_buffer=[];self.integral=0.
    def _record(self,op,**args):
        if len(self.history)>=256:raise ValueError('Operation budget256')
        self.history.append(dict(op=op,args=args))
    def _patch(self,patch):
        if type(patch) is not dict or not patch or not set(patch)<=set(self.u):raise ValueError('Command patch schema')
        new=commands(dict(self.u,**patch));old=self.y.copy()
        y,rate,receipt=self.system.consistent(self.time,self.y,new)
        if not np.array_equal(y[self.system.mask],old[self.system.mask]):raise ValueError('Topology initialization changed inventories')
        if new['generator_connected']!=self.u['generator_connected']:self.revision+=1
        self.y=y;self.rate=rate;self.u=new
        return receipt
    def set_command(self,patch):
        trial=deepcopy(self);trial._patch(patch);trial._record('set_command',patch=deepcopy(patch));self.__dict__.update(trial.__dict__)
    def schedule(self,at_s,patch):
        if type(at_s) not in (float,int) or not math.isfinite(at_s) or not self.time<at_s<=3600 or len(self.queue)>=32 or any(e['at_s']==at_s for e in self.queue):raise ValueError('Unique future queue time/budget')
        if type(patch) is not dict or not patch or not set(patch)<=set(self.u):raise ValueError('Queue command schema')
        commands(dict(self.u,**patch))
        trial=deepcopy(self);trial.queue.append(dict(at_s=float(at_s),patch=deepcopy(patch)));trial.queue.sort(key=lambda e:e['at_s']);trial._record('schedule',at_s=float(at_s),patch=deepcopy(patch));self.__dict__.update(trial.__dict__)
    def guard(self,name,variable,threshold,direction,patch):
        if type(name) is not str or not name or len(name)>80 or any(g['name']==name for g in self.guards) or len(self.guards)>=8:raise ValueError('Guard id/budget')
        if variable not in self.system.index or type(threshold) not in (float,int) or not math.isfinite(threshold) or type(direction) is not int or direction not in (-1,1):raise ValueError('Guard variable/threshold/direction')
        if type(patch) is not dict or not patch or not set(patch)<=set(self.u):raise ValueError('Guard patch schema')
        commands(dict(self.u,**patch))
        trial=deepcopy(self);g=dict(name=name,variable=variable,threshold=float(threshold),direction=direction,patch=deepcopy(patch),latched=False)
        if (trial.y[trial.system.index[variable]]-threshold)*direction>=0:
            g['latched']=True;receipt=trial._patch(patch)
            trial.events.append(dict(time_s=trial.time,kind='root',name=name,patch=deepcopy(patch),initialization=receipt,localization_tolerance_s=0.))
        trial.guards.append(g);trial._record('guard',name=name,variable=variable,threshold=float(threshold),direction=direction,patch=deepcopy(patch));self.__dict__.update(trial.__dict__)
    def _accept(self,y,rate,h):
        c=self.config;weights=np.array([c.primary_capacity_j_k,c.wall_capacity_j_k,c.secondary_capacity_j_k,c.cold_capacity_j_k,1.])
        delta=float(weights@(y[:5]-self.y[:5]));source=self.u['source_power_w']*h;export=y[9]*h;cooling=y[10]*h
        defect=math.fsum([delta,-source,export,cooling]);scale=max(abs(source)+abs(export)+abs(cooling),1.)
        if abs(defect)>1e-5+1e-6*scale:raise ValueError('Mixed global ledger failure')
        self.y=y;self.rate=rate;self.time+=h
        for key,v in [('source_j',source),('export_j',export),('cooling_j',cooling),('residual_j',defect)]:self.ledger[key]+=float(v)
    def advance(self,duration_s):
        if type(duration_s) not in (int,float) or not math.isfinite(duration_s) or not 0<duration_s<=60 or self.time+duration_s>3600:raise ValueError('Mixed duration/horizon domain')
        trial=deepcopy(self);target=trial.time+duration_s;steps=0
        while trial.time<target-1e-12:
            if steps>=10000:raise ValueError('Mixed step budget10000')
            while trial.queue and trial.queue[0]['at_s']<=trial.time+1e-10:
                event=trial.queue.pop(0);receipt=trial._patch(event['patch']);trial.events.append(dict(time_s=trial.time,kind='scheduled',patch=event['patch'],initialization=receipt))
            h=min(trial.max_step_s,target-trial.time,trial.queue[0]['at_s']-trial.time if trial.queue else 1e9)
            # Failed numerical trial steps may be retried; no inventories committed.
            while True:
                try:y,rate,_=trial.system.step(trial.time,trial.y,h,trial.u);break
                except ValueError:
                    h*=.5
                    if h<1e-5:raise ValueError('Mixed step cannot be admitted')
            roots=[]
            for index,g in enumerate(trial.guards):
                if g['latched']:continue
                j=trial.system.index[g['variable']];a=(trial.y[j]-g['threshold'])*g['direction'];b=(y[j]-g['threshold'])*g['direction']
                if a<0<=b:
                    lo=0.;hi=h;event_y=y;event_rate=rate
                    for _ in range(40):
                        if hi-lo<=1e-6:break
                        mid=(lo+hi)/2;my,mr,_=trial.system.step(trial.time,trial.y,mid,trial.u)
                        if (my[j]-g['threshold'])*g['direction']>=0:hi=mid;event_y=my;event_rate=mr
                        else:lo=mid
                    roots.append((hi,g['name'],index,event_y,event_rate))
            if roots:
                event_h,_,index,y,rate=min(roots,key=lambda item:(item[0],item[1]))
                trial._accept(y,rate,event_h);g=trial.guards[index];g['latched']=True
                receipt=trial._patch(g['patch']);trial.events.append(dict(time_s=trial.time,kind='root',name=g['name'],patch=deepcopy(g['patch']),initialization=receipt,localization_tolerance_s=1e-6))
            else:trial._accept(y,rate,h)
            steps+=1
        # Apply commands exactly at the endpoint before returning the state.
        while trial.queue and trial.queue[0]['at_s']<=trial.time+1e-10:
            event=trial.queue.pop(0);receipt=trial._patch(event['patch']);trial.events.append(dict(time_s=trial.time,kind='scheduled',patch=event['patch'],initialization=receipt))
        trial._record('advance',duration_s=float(duration_s));self.__dict__.update(trial.__dict__)
        return self.snapshot()
    def observe(self):
        trial=deepcopy(self);sample=dict(time_s=trial.time,wall_temperature_k=float(trial.y[1]+trial.rng.normal(0.,.002)))
        trial.measurement_buffer.append(sample)
        if len(trial.measurement_buffer)>2:trial.measurement_buffer.pop(0)
        result=deepcopy(trial.measurement_buffer[0]);trial._record('observe');self.__dict__.update(trial.__dict__);return result
    def control(self,target_k,measured_k,dt_s):
        if any(type(v) not in (int,float) or not math.isfinite(v) for v in (target_k,measured_k,dt_s)) or not 280<=target_k<=900 or not 280<=measured_k<=900 or not 0<dt_s<=60:raise ValueError('PI input domain')
        trial=deepcopy(self);error=target_k-measured_k;proposed=trial.integral+error*dt_s
        raw=trial.config.nominal_source_power_w+20000*error+1000*proposed;delivered=min(2e6,max(0.,raw))
        if delivered==raw or (raw>2e6 and error<0) or (raw<0 and error>0):trial.integral=proposed
        trial._patch({'source_power_w':delivered});trial._record('control',target_k=target_k,measured_k=measured_k,dt_s=dt_s)
        self.__dict__.update(trial.__dict__);return delivered
    def snapshot(self):
        return dict(time_s=self.time,state=self.y.tolist(),rate=self.rate.tolist(),commands=deepcopy(self.u),ledger=deepcopy(self.ledger),
            queue=deepcopy(self.queue),guards=deepcopy(self.guards),events=deepcopy(self.events),topology_revision=self.revision,
            regulator_integral=self.integral,measurement_buffer=deepcopy(self.measurement_buffer),rng_state=deepcopy(self.rng.bit_generator.state),
            model_id='npp-r0-mixed-index1',manifest=self.system.manifest(),validated=False,plant_referenced=False)
    def checkpoint(self):
        return seal(deepcopy(dict(config=asdict(self.config),max_step_s=self.max_step_s,seed=self.seed,source=signature(),history=self.history,snapshot=self.snapshot())))
    @classmethod
    def restore(cls,saved):
        p=unseal(deepcopy(saved))
        if p['source']!=signature() or type(p['history']) is not list or len(p['history'])>256:raise ValueError('Mixed restart source/budget')
        s=cls(StationConfig(**p['config']),max_step_s=p['max_step_s'],seed=p['seed'])
        allowed={'set_command','schedule','guard','advance','observe','control'}
        for op in p['history']:
            if type(op) is not dict or set(op)!={'op','args'} or op['op'] not in allowed or type(op['args']) is not dict:raise ValueError('Mixed restart operation schema')
            getattr(s,op['op'])(**op['args'])
        if s.checkpoint()!=saved:raise ValueError('Mixed replay differs')
        return s
