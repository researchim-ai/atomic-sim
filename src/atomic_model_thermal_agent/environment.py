"""Noisy partial observations, delayed rate-limited heater, exact episode replay."""
from dataclasses import dataclass,asdict
from copy import deepcopy
import hashlib
import math
from pathlib import Path
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from atomic_model_hem_heated import HeatedHEMNetwork
from atomic_model_hem_heated.model import canonical,signature as physics_signature
from atomic_model_hem_network.cases import baseline
from atomic_model_multiphysics.provenance import seal,unseal


def signature():
    return dict(physics=physics_signature(),environment={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(Path(__file__).parent.glob('*.py'))})


@dataclass(frozen=True)
class ThermalConfig:
    capacity_j_k:float=150.
    conductance_w_k:float=8000.
    fluid_temperature_k:float=440.
    target_rise_k:float=.4
    ambient_temperature_k:float=430.
    ambient_ua_w_k:float=20.
    noise_std_k:float=.002
    delay_steps:int=1
    action_dt_s:float=.00025
    steps:int=12
    max_step_s:float=5e-6
    max_power_w:float=100000.
    slew_w_per_s:float=2e8
    def __post_init__(self):
        bounds=dict(capacity_j_k=(100,500),conductance_w_k=(1000,15000),fluid_temperature_k=(435,445),
            target_rise_k=(.05,1),ambient_temperature_k=(420,440),ambient_ua_w_k=(0,100),noise_std_k=(0,.05),
            action_dt_s=(1e-5,.001),max_step_s=(1e-6,5e-6),max_power_w=(10000,200000),slew_w_per_s=(1e6,1e10))
        for k,(a,b) in bounds.items():
            v=getattr(self,k)
            if type(v) not in (int,float) or not math.isfinite(v) or not a<=v<=b:raise ValueError('Thermal config domain: '+k)
        if type(self.delay_steps) is not int or not 0<=self.delay_steps<=3:raise ValueError('Delay0..3')
        if type(self.steps) is not int or not 2<=self.steps<=128:raise ValueError('Steps2..128')
        if self.action_dt_s*self.steps>.03:raise ValueError('Research horizon at most30ms')


def request_for(config):
    spec=baseline('parallel');spec['wave_speed_bound_m_s']=2500.;spec['max_step_s']=config.max_step_s
    for node in spec['nodes']:
        node['initial'].update(pressure_pa=4e6,temperature_k=config.fluid_temperature_k)
    for edge in spec['edges']:
        for point in edge['initial']:point.update(pressure_pa=4e6,temperature_k=config.fluid_temperature_k)
    spec['walls'][0].update(capacity_j_k=config.capacity_j_k,conductance_w_k=config.conductance_w_k,
        initial_temperature_k=config.fluid_temperature_k,ambient_temperature_k=config.ambient_temperature_k,
        ambient_ua_w_k=config.ambient_ua_w_k)
    return dict(schema_version=1,model=spec,friction={})


class ThermalEnv(gym.Env):
    """Actions[0..1] request absolute heater power. No state or parameter randomization on reset.

    Different configs are separate plants; seed controls measurement noise only.
    Observation: wall/fluid/target/ambient temperatures minus440K, applied power
    fraction, elapsed decision fraction. Truth in info is for evaluation only.
    """
    metadata={'render_modes':[]}
    def __init__(self,config=None):
        self.config=config or ThermalConfig()
        if type(self.config) is not ThermalConfig:raise ValueError('ThermalConfig required')
        self.action_space=spaces.Box(0.,1.,shape=(1,),dtype=np.float64)
        self.observation_space=spaces.Box(-1000.,1000.,shape=(6,),dtype=np.float64)
        self._ready=False

    def _target(self,index):
        return self.config.fluid_temperature_k+self.config.target_rise_k*(.5 if index>=self.config.steps//2 else 1.)

    def _observation(self,snapshot):
        wall=snapshot['walls'][0];fluid=next(c for c in snapshot['cells'] if c['id']==wall['cell'])
        noise=self.np_random.normal(0,self.config.noise_std_k,size=2)
        return np.array([wall['temperature_k']-440.+noise[0],fluid['temperature_k']-440.+noise[1],
            self._target(self._index)-440.,self.config.ambient_temperature_k-440.,self._power/self.config.max_power_w,
            self._index/self.config.steps],dtype=np.float64)

    def reset(self,*,seed=None,options=None):
        if options:raise ValueError('Reset options unsupported; use separate config')
        if seed is not None and (type(seed) is not int or not 0<=seed<2**32):raise ValueError('Seed must be integer0..2**32-1')
        # Explicit deterministic default, unlike implicit physical domain randomization.
        self._seed=0 if seed is None else seed;super().reset(seed=self._seed)
        self.model=HeatedHEMNetwork(request_for(self.config));self._index=0;self._power=0.;self._queue=[0.]*self.config.delay_steps
        self._actions=[];self._done=False;self._ready=True
        self._obs=self._observation(self.model.snapshot())
        return self._obs.copy(),{'config':asdict(self.config),'noise_seed':self._seed,'physical_randomization':False}

    def step(self,action):
        if not self._ready or self._done:raise ValueError('Reset required or episode finished')
        values=np.asarray(action)
        if values.shape!=(1,) or values.dtype.kind not in 'if' or not np.isfinite(values).all() or not 0<=float(values[0])<=1:
            raise ValueError('Finite action shape(1) in[0,1] required')
        requested=float(values[0])*self.config.max_power_w
        queue=self._queue+[requested]
        delayed=queue.pop(0)
        limit=self.config.slew_w_per_s*self.config.action_dt_s
        applied=max(self._power-limit,min(self._power+limit,delayed))
        controls=deepcopy(self.model.snapshot()['controls']);controls['walls']['w0']=applied
        target=self._target(self._index)
        result=self.model.step(controls,self.config.action_dt_s,expected_revision=self._index,controller_state={'thermal_tick':self._index})
        # Environment state and RNG only advance after the physical transaction commits.
        self._queue,self._power=queue,applied;self._index+=1;self._actions.append(float(values[0]))
        self._obs=self._observation(result)
        wall=result['walls'][0]['temperature_k'];error=abs(wall-target)
        violations=wall>self.config.fluid_temperature_k+2 or any(c['temperature_k']>500 or c['pressure_pa']>8e6 for c in result['cells'])
        terminated=bool(violations);truncated=self._index>=self.config.steps;self._done=terminated or truncated
        info=dict(target_k=target,wall_temperature_k=wall,requested_power_w=requested,delayed_power_w=delayed,applied_power_w=applied,
            iae_increment_k_s=error*self.config.action_dt_s,energy_j=applied*self.config.action_dt_s,
            violations=terminated,global_ledger=result['global_ledger'],time_s=result['time_s'])
        reward=-info['iae_increment_k_s']-1e-6*info['energy_j']
        return self._obs.copy(),float(reward),terminated,bool(truncated),info

    def checkpoint(self):
        if not self._ready:raise ValueError('Reset required')
        return seal(dict(config=asdict(self.config),seed=self._seed,actions=deepcopy(self._actions),source=signature(),
            observation=self._obs.tolist(),queue=self._queue.copy(),power_w=self._power,core=self.model.checkpoint(),done=self._done))

    @classmethod
    def restore(cls,saved):
        p=unseal(deepcopy(saved))
        if p['source']!=signature() or not isinstance(p['actions'],list) or len(p['actions'])>128:raise ValueError('Thermal source/replay budget')
        env=cls(ThermalConfig(**p['config']));env.reset(seed=p['seed'])
        for action in p['actions']:env.step([action])
        if canonical(env.checkpoint())!=canonical(saved):raise ValueError('Thermal episode replay differs')
        return env
