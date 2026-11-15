"""Gymnasium adapter with finite normalized measured values and quality masks."""
from copy import deepcopy
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .model import StationEnv,EnvConfig,ActionError

CHANNELS=('condensate_flow','feed_flow','feed_level','hotwell_level','power','pressure',
          'pressure_reference','sg_level','sg_outflow','speed')
SCALES=np.array([1200.,1200.,2.,.6,3.15e9,6.3e6,6.3e6,2.5,1200.,157.])


class StationGymEnv(gym.Env):
    metadata={'render_modes':[]}

    def __init__(self,config=None):
        super().__init__();self.station=StationEnv(config)
        self.action_space=spaces.Dict(setpoints=spaces.Box(-1.,1.,shape=(3,),dtype=np.float64),scram=spaces.Discrete(2))
        self.observation_space=spaces.Dict(values=spaces.Box(-1e6,1e6,shape=(10,),dtype=np.float64),
            valid=spaces.MultiBinary(10),age_s=spaces.Box(0.,7200.,shape=(10,),dtype=np.float64),
            time_s=spaces.Box(0.,7200.,shape=(1,),dtype=np.float64))

    @staticmethod
    def _vector(frame):
        values=np.zeros(10);valid=np.zeros(10,dtype=np.int8);ages=np.zeros(10)
        for i,name in enumerate(CHANNELS):
            r=frame['measurements'][name]
            if r['quality']==['good'] and r['value'] is not None:
                values[i]=np.clip(r['value']/SCALES[i],-1e6,1e6);valid[i]=1
            ages[i]=0. if r['age_s'] is None else min(7200.,r['age_s'])
        return dict(values=values,valid=valid,age_s=ages,time_s=np.array([frame['time_s']],dtype=np.float64))

    def reset(self,*,seed=None,options=None):
        if options not in (None,{}):raise ValueError('Scenario options are configured outside the agent interface')
        super().reset(seed=seed)
        frame,info=self.station.reset(seed=seed)
        return self._vector(frame),dict(info,raw_observation=deepcopy(frame))

    def step(self,action):
        if not self.action_space.contains(action):raise ActionError('invalid_action','Action is outside the Gymnasium action space')
        a=action['setpoints']
        operator=dict(load_fraction=float(1.+.02*a[2]))
        if action['scram']==1:operator['scram']=True
        command=dict(operator=operator,targets=dict(rod_target=float(3.15e9*(1+.02*a[0])),steam_admission=float(157.*(1+.005*a[1]))),
                     modes=dict(rod_target='auto',steam_admission='auto'))
        frame,reward,terminated,truncated,info=self.station.step(command)
        return self._vector(frame),reward,terminated,truncated,dict(info,raw_observation=deepcopy(frame))

    def close(self):pass
