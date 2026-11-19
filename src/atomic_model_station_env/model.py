"""Measured-only station API. Checkpoints are privileged research material."""
from copy import deepcopy
from dataclasses import asdict,dataclass
import hashlib
from pathlib import Path
from core.status import DomainError,SolverError

from atomic_model_multiphysics.provenance import seal,unseal
from atomic_model_station.journal import Journal
from atomic_model_station_faults import Fault,FaultSession
from atomic_model_station_control.policy import valid_reading
from atomic_model_station_campaign import Episode
from atomic_model_station_campaign.engine import signature as campaign_signature

FLOAT_ACTIONS=dict(load_fraction=(.5,1.2),rod_target=(0.,1.),steam_bypass=(0.,1.),steam_admission=(0.,1.),
    feed_flow_fraction=(0.,1.),condensate_flow_fraction=(0.,1.),feed_inventory_fraction=(0.,1.),
    feed_pump_fraction=(0.,1.),condensate_pump_fraction=(0.,1.),cooling_pump_fraction=(0.,1.2),
    external_water_fraction=(0.,1.5))
BOOL_ACTIONS=('scram','turbine_trip','dhr_requested','primary_pump_on','feed_pump_on','condensate_pump_on','cooling_pump_on')


def signature():
    return dict(campaign=campaign_signature(),files={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(Path(__file__).parent.glob('*.py'))})


class ActionError(ValueError):
    def __init__(self,code,reason):
        self.code=code;self.reason=reason
        super().__init__(reason)


@dataclass(frozen=True)
class EnvConfig:
    max_steps:int=256
    tick_s:float=1.
    seed:int=0
    automatic:bool=True
    react_on_alarm:bool=False
    terminal_on_trip:bool=False
    faults:tuple=()
    max_step_s:float=1.
    rtol:float=1e-7
    equilibrium_start:bool=False
    inventory_deadband_m:float=0.
    film_model:str="dittus_boelter"

    def __post_init__(self):
        e=Episode('environment',self.max_steps,tick_s=self.tick_s,seed=self.seed,automatic=self.automatic,
            react_on_alarm=self.react_on_alarm,faults=self.faults,max_step_s=self.max_step_s,rtol=self.rtol,
            record_every=max(1,(self.max_steps+1023)//1024),equilibrium_start=self.equilibrium_start,
            inventory_deadband_m=self.inventory_deadband_m,film_model=self.film_model)
        object.__setattr__(self,'faults',e.faults)
        if self.tick_s>10.:raise ValueError('Environment uses a declared 10s PI tracking constant')
        if type(self.terminal_on_trip) is not bool:raise ValueError('Boolean trip termination')


class StationEnv:
    def __init__(self,config=None):
        self.config=config if config is not None else EnvConfig()
        if type(self.config) is not EnvConfig:raise ValueError('EnvConfig required')
        self._session=None;self._history=Journal();self._observation=None;self._last_info=None
        self._finished=False;self._seed=self.config.seed;self._plan=None;self._plan_sequence=0

    def _require(self):
        if self._session is None:raise ActionError('not_initialized','Call reset before using the environment')

    def reset(self,*,seed=None):
        seed=self.config.seed if seed is None else seed
        if type(seed) is not int or not 0<=seed<2**63:raise ActionError('invalid_seed','Integer seed in 0..2**63-1 required')
        c=self.config
        session=FaultSession(c.faults,tick_s=c.tick_s,seed=seed,react_on_alarm=c.react_on_alarm,
            max_step_s=c.max_step_s,rtol=c.rtol,history_budget_steps=max(256,c.max_steps),
            equilibrium_start=c.equilibrium_start,inventory_deadband_m=c.inventory_deadband_m,film_model=c.film_model)
        observation=session.control.instruments.operator_frame()
        info=dict(steps=0,termination_reason=None,truncation_reason=None,trip_latched=False,
                  detection=deepcopy(session.detection),receipt=None,reaction=None,reward_terms=None)
        self._session=session;self._seed=seed;self._history=Journal();self._observation=observation
        self._last_info=info;self._finished=False;self._plan=None;self._plan_sequence=0
        return deepcopy(observation),deepcopy(info)

    def available_actions(self):
        self._require()
        return dict(operator={**{k:dict(type='number',minimum=a,maximum=b) for k,(a,b) in FLOAT_ACTIONS.items()},
            **{k:dict(type='boolean') for k in BOOL_ACTIONS}},
            modes={k:['manual','auto'] for k in self._session.control.loops},
            targets=deepcopy(self._session.control.targets),reset_in_place_available=False,
            step_s=self.config.tick_s,remaining_steps=max(0,self.config.max_steps-len(self._history)),finished=self._finished)

    def _action(self,action):
        from atomic_model_station_faults.spec import finite
        action={} if action is None else action
        if type(action) is not dict or not set(action)<={'operator','modes','targets','reset'}:
            raise ActionError('invalid_action','Unknown action fields')
        result=dict(operator=None,modes=None,targets=None,reset=False)
        for field in ('operator','modes','targets'):
            values=action.get(field)
            if values is not None and (type(values) is not dict or len(values)>32):
                raise ActionError('invalid_action','Action patches must be bounded dictionaries')
            if values is not None:
                for key,value in values.items():
                    if field=='operator':
                        valid=(key in BOOL_ACTIONS and type(value) is bool) or (key in FLOAT_ACTIONS and finite(value) and FLOAT_ACTIONS[key][0]<=value<=FLOAT_ACTIONS[key][1])
                    elif field=='modes':valid=type(key) is str and key in self._session.control.loops and type(value) is str and value in ('manual','auto')
                    else:valid=type(key) is str and key in self._session.control.loops and self._session.control._target_ok(key,value)
                    if not valid:raise ActionError('invalid_action','Unsupported action value: '+str(key))
                result[field]=deepcopy(values) or None
        if 'reset' in action:
            if type(action['reset']) is not bool:raise ActionError('invalid_action','Boolean reset request required')
            result['reset']=action['reset']
        return result

    @staticmethod
    def _reward(frame,targets,trip):
        specs=[('power','fission_power_w','rod_target',3.15e9),
               ('pressure','steam_pressure_pa','steam_bypass',1e6),
               ('speed','turbine_speed_rad_s','steam_admission',157.)]
        errors={};missing=[]
        for sensor,quantity,target,scale in specs:
            value=valid_reading(frame,sensor,quantity)
            if value is None:missing.append(sensor)
            else:errors[sensor]=abs(value-targets[target])/scale
        penalty=sum(errors.values())+len(missing)+(1. if trip else 0.)
        return -float(penalty),dict(normalized_absolute_errors=errors,unavailable_channels=missing,trip_penalty=1. if trip else 0.)

    def step(self,action=None):
        self._require()
        if self._finished:raise ActionError('episode_finished','Episode finished; reset or restore an earlier checkpoint')
        normalized=self._action(action);trial=deepcopy(self);kwargs=deepcopy(normalized)
        if len(self._history)==0 and self.config.automatic:
            kwargs['modes']=dict({k:'auto' for k in trial._session.control.loops},**(kwargs['modes'] or {}))
        try:result=trial._session.step(**kwargs)
        except SolverError as error:raise ActionError('model_solver_rejected',str(error)) from None
        except (ValueError,DomainError) as error:raise ActionError('model_domain_rejected',str(error)) from None
        trial._history.append(normalized);trial._plan=None
        trip=result['receipt']['trip_latched']
        terminated=bool(trip and self.config.terminal_on_trip)
        truncated=len(trial._history)>=self.config.max_steps
        trial._finished=terminated or truncated;trial._observation=deepcopy(result['observation'])
        reward,terms=self._reward(result['observation'],result['receipt']['targets'],trip)
        info=dict(steps=len(trial._history),trip_latched=trip,
            termination_reason='protection_latched' if terminated else None,
            truncation_reason='declared_step_horizon' if truncated else None,
            receipt=deepcopy(result['receipt']),detection=deepcopy(result['detection']),
            reaction=deepcopy(result['reaction']),reward_terms=terms)
        trial._last_info=deepcopy(info);self.__dict__.update(trial.__dict__)
        return deepcopy(self._observation),reward,terminated,truncated,info

    def observation(self):
        self._require();return deepcopy(self._observation),deepcopy(self._last_info)

    def checkpoint(self):
        self._require()
        cfg=asdict(self.config);cfg['faults']=[asdict(f) for f in self.config.faults]
        return seal(dict(format='station-env/v1',source=signature(),config=cfg,seed=self._seed,
            history=list(self._history),session=self._session.checkpoint(),observation=deepcopy(self._observation),
            info=deepcopy(self._last_info),finished=self._finished))

    @classmethod
    def restore(cls,saved):
        p=unseal(deepcopy(saved))
        if p.get('format')!='station-env/v1' or p.get('source')!=signature():raise ValueError('Environment checkpoint source/format')
        if type(p.get('history')) is not list or len(p['history'])>8192:raise ValueError('Environment checkpoint budget')
        cfg=deepcopy(p['config']);cfg['faults']=tuple(Fault(**f) for f in cfg['faults'])
        obj=cls(EnvConfig(**cfg))
        if len(p['history'])>obj.config.max_steps:raise ValueError('Environment checkpoint horizon')
        obj.reset(seed=p['seed'])
        for action in p['history']:obj.step(action)
        if obj.checkpoint()!=saved:raise ValueError('Environment checkpoint replay differs')
        return obj

    def preview(self,actions):
        """Explicit model-assisted planner; no checkpoint or latent-state hash is public.

        Forecasts use the privileged model, so this method is absent from the
        ordinary agent transport. Cached plans are local and expire on advance.
        """
        self._require()
        if type(actions) not in (tuple,list) or not 1<=len(actions)<=8:raise ActionError('invalid_branches','One to eight branches required')
        base=self.checkpoint()['sha256'];candidates=[];expected=[]
        for action in actions:
            branch=deepcopy(self);branch._plan=None
            try:
                transition=branch.step(action)
                candidates.append(dict(action=self._action(action),accepted=True,transition=list(transition)))
                expected.append(branch.checkpoint()['sha256'])
            except ActionError as error:
                candidates.append(dict(action=None,accepted=False,error=dict(code=error.code,reason=error.reason)))
                expected.append(None)
        identifier=self._plan_sequence+1
        public=dict(format='station-branches/v1',source=signature(),plan_id=identifier,
                    planning_scope='privileged_model_assisted',candidates=candidates)
        self._plan=dict(base_sha256=base,public=deepcopy(public),expected=expected)
        self._plan_sequence=identifier
        return seal(public)

    def apply(self,comparison,index):
        self._require();p=unseal(deepcopy(comparison))
        if (type(p) is not dict or set(p)!={'format','source','plan_id','planning_scope','candidates'}
            or p.get('format')!='station-branches/v1' or p.get('source')!=signature()
            or self._plan is None or self._plan['base_sha256']!=self.checkpoint()['sha256']
            or p.get('plan_id')!=self._plan['public']['plan_id']):
            raise ActionError('stale_branch','Branch comparison does not match a live local plan')
        if p!=self._plan['public']:
            raise ActionError('branch_replay_mismatch','Branch proposal was modified')
        candidates=p['candidates']
        if type(candidates) is not list or not 1<=len(candidates)<=8 or type(index) is not int or not 0<=index<len(candidates):
            raise ActionError('invalid_branch','Branch index or budget')
        selected=candidates[index]
        if not selected['accepted']:raise ActionError('rejected_branch','Cannot apply a rejected branch')
        expected=self._plan['expected'][index]
        trial=deepcopy(self);trial._plan=None;transition=trial.step(selected['action'])
        if list(transition)!=selected['transition'] or trial.checkpoint()['sha256']!=expected:
            raise ActionError('branch_replay_mismatch','Predicted transition or checkpoint differs')
        self.__dict__.update(trial.__dict__);return transition
