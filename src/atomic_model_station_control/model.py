"""Hot research supervisory control. Not a licensed plant protection system."""
from copy import deepcopy
from dataclasses import asdict,dataclass
import hashlib
from pathlib import Path

from atomic_model_station import PWRBypassSession
from atomic_model_station.journal import Journal,history_budget
from atomic_model_station.faults import PWRFaultSession
from atomic_model_station.bypass import bypass_signature
from atomic_model_station_signals import Sensor,StationSignals,signature as signals_signature
from atomic_model_station_signals.model import CATALOG,finite
from atomic_model_multiphysics.provenance import seal,unseal
from .pi import PI,PIConfig


def control_signature():
    return dict(files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob('*.py'))},
                station=bypass_signature(),instruments=signals_signature())


from .policy import ProtectionConfig,Supervision,valid_reading


class ControlSession:
    def __init__(self,plant=None,*,instruments=None,loops=None,protection=None):
        self.plant=deepcopy(plant) if plant is not None else PWRBypassSession(max_step_s=.5)
        if type(self.plant) not in (PWRBypassSession,PWRFaultSession) or self.plant.time!=0:raise ValueError('Fresh PWRBypassSession required')
        self.instruments=deepcopy(instruments) if instruments is not None else StationSignals([
            Sensor('power','fission_power_w',lag_s=.5),Sensor('pressure','steam_pressure_pa',lag_s=.2),
            Sensor('speed','turbine_speed_rad_s',lag_s=.2)],seed=0)
        if type(self.instruments) is not StationSignals or self.instruments.tick!=-1:raise ValueError('Uninitialized instrument owner required')
        if type(self.plant) is PWRFaultSession and self.plant.fault_tick_s!=self.instruments.tick_s:raise ValueError('Physical fault/control clock mismatch')
        self.protection=protection if protection is not None else ProtectionConfig()
        if type(self.protection) is not ProtectionConfig:raise ValueError('ProtectionConfig required')
        quantities={s.name:s.quantity for s in self.instruments.sensors}
        if any(quantities.get(n)!=q for n,q in [('power','fission_power_w'),('pressure','steam_pressure_pa'),('speed','turbine_speed_rad_s')]):raise ValueError('Required named protection channels')
        self.initial_plant=self.plant.checkpoint();self.initial_instruments=self.instruments.checkpoint()
        snap=self.plant.snapshot();frame=self.instruments.ingest_station(snap)
        if loops is None:
            power=valid_reading(frame,'power','fission_power_w');pressure=valid_reading(frame,'pressure','steam_pressure_pa');speed=valid_reading(frame,'speed','turbine_speed_rad_s')
            if any(v is None for v in (power,pressure,speed)):raise ValueError('Delayed startup requires explicit PI references')
            loops=[PIConfig('power','rod_target',power,power,kp=.01,ki=.002,direction=-1,bias=snap['controls']['rod_target']),
                   PIConfig('pressure','steam_bypass',pressure,1e6,kp=.1,ki=.02,direction=-1),
                   PIConfig('speed','steam_admission',speed,10.,kp=.03,ki=.01,bias=1.)]
        if type(loops) not in (list,tuple) or not 0<len(loops)<=6 or any(type(c) is not PIConfig for c in loops):raise ValueError('PI loop list')
        if len({c.output for c in loops})!=len(loops) or any(c.sensor not in quantities for c in loops):raise ValueError('PI loop channel/output uniqueness')
        if any(not self._target_ok(c.output,c.target) for c in loops):raise ValueError('PI physical target domain')
        if any(c.tracking_tau_s<self.instruments.tick_s for c in loops):raise ValueError('PI tracking must cover one instrument tick')
        expected={'rod_target':'fission_power_w','steam_bypass':'steam_pressure_pa','steam_admission':'turbine_speed_rad_s','feed_flow_fraction':'sg_collapsed_level_m',
                  'condensate_flow_fraction':'hotwell_level_m','makeup_valve':'feed_tank_level_m','feed_inventory_fraction':'feed_tank_level_m'}
        if any(quantities[c.sensor]!=expected[c.output] for c in loops):raise ValueError('PI physical quantity/output contract')
        ff_expected={'feed_flow_fraction':'sg_outflow_kg_s','condensate_flow_fraction':'feed_flow_kg_s'}
        if any(c.feedforward_sensor is not None and quantities.get(c.feedforward_sensor)!=ff_expected.get(c.output) for c in loops):raise ValueError('PI feed-forward quantity/output contract')
        for output,name,quantity in [('feed_flow_fraction','feed_flow','feed_flow_kg_s'),('condensate_flow_fraction','condensate_flow','condensate_flow_kg_s')]:
            if any(c.output==output for c in loops) and quantities.get(name)!=quantity:raise ValueError('Measured inner-flow feedback channel required')
        self.loops={c.output:PI(c) for c in sorted(loops,key=lambda c:c.output)}
        self.targets={k:v.config.target for k,v in self.loops.items()};self.modes={k:'manual' for k in self.loops}
        self.manual=deepcopy(snap['controls']);self.policy=Supervision(self.protection)
        self.history=Journal();self.last_receipt=None

    def _target_ok(self,key,value):
        if not finite(value):return False
        if key=='rod_target':return 0<=value<=self.protection.power_trip_w
        if key=='steam_bypass':return self.plant.bypass_config.supply_open_pa<=value<=self.protection.pressure_trip_pa
        if key=='steam_admission':return self.plant.config.minimum_speed_fraction*self.plant.config.nominal_speed_rad_s<=value<=self.plant.config.maximum_speed_fraction*self.plant.config.nominal_speed_rad_s
        c=self.plant.feed_config
        if key=='feed_flow_fraction':return .05<=value<=.95*self.plant.plant.network.volumes['sg_secondary'].volume_m3/c.sg_area_m2
        if key=='condensate_flow_fraction':return .05<=value<=.95*c.hotwell_sump_height_m
        if key in ('makeup_valve','feed_inventory_fraction'):return .05<=value<=.9*self.plant.feed_tanks['feed']['total_volume_m3']/c.tank_area_m2
        return False

    def _feedback(self,snapshot,key):
        if key=='rod_target':value=snapshot['actuator_status']['rod_position']
        elif key=='steam_bypass':value=snapshot['bypass_position']
        elif key=='steam_admission':value=snapshot['steam_header']['outlet_position']/snapshot['controls']['valve_request']
        elif key in ('feed_flow_fraction','condensate_flow_fraction'):
            name='feed_flow' if key=='feed_flow_fraction' else 'condensate_flow'
            quantity='feed_flow_kg_s' if key=='feed_flow_fraction' else 'condensate_flow_kg_s'
            measured=valid_reading(self.instruments.operator_frame(),name,quantity)
            value=snapshot['controls'][key] if measured is None else measured/snapshot['feed_train']['flow_scale_kg_s']
        elif key=='feed_inventory_fraction':
            terms=snapshot['feed_train']
            value=.5+.5*terms['makeup_opening'] if terms['makeup_flow_kg_s']>0 else .5-.5*terms['drain_opening'] if terms['drain_flow_kg_s']>0 else .5
        elif key=='makeup_valve':value=snapshot['controls']['makeup_valve'] if snapshot['feed_train']['makeup_flow_kg_s']>0 else 0.
        else:raise ValueError('Unknown PI feedback')
        return float(min(1.,max(0.,value)))

    def _event(self,kind,**kw):self.events.append(dict(time_s=self.plant.time,kind=kind,**kw))

    @property
    def pressure_active(self):return self.policy.pressure_active
    @property
    def trip_latched(self):return self.policy.trip_latched
    @property
    def trip_reasons(self):return self.policy.trip_reasons
    @property
    def events(self):return self.policy.events

    def _protect(self,frame,requested):
        quantities={s.name:s.quantity for s in self.instruments.sensors};required={}
        for key,loop in self.loops.items():
            if key in ('feed_flow_fraction','condensate_flow_fraction','makeup_valve','feed_inventory_fraction') and self.modes[key]=='auto':
                required[loop.config.sensor]=quantities[loop.config.sensor]
                if loop.config.feedforward_sensor is not None:required[loop.config.feedforward_sensor]=quantities[loop.config.feedforward_sensor]
                if key in ('feed_flow_fraction','condensate_flow_fraction'):
                    name='feed_flow' if key=='feed_flow_fraction' else 'condensate_flow';required[name]=quantities[name]
        return self.policy.advance(frame,requested,tick=self.instruments.tick,
            nominal_speed=self.plant.config.nominal_speed_rad_s,backend_scram=self.plant.scram_latched,
            backend_turbine_trip=self.plant.turbine_tripped,additional_required=required)

    def step(self,operator=None,*,modes=None,targets=None,reset=False):
        if len(self.history)>=self.plant._maximum_history_steps():raise ValueError('Supervisory history budget')
        trial=deepcopy(self)
        result=trial._step(operator,modes,targets,reset)
        trial.history.append(dict(operator=deepcopy(operator),modes=deepcopy(modes),targets=deepcopy(targets),reset=reset))
        self.__dict__.update(trial.__dict__);return result

    def _step(self,operator,modes,targets,reset):
        if type(reset) is not bool:raise ValueError('Boolean reset request')
        if operator is not None:
            if type(operator) is not dict or not operator:raise ValueError('Operator patch')
            probe=deepcopy(self.plant);probe._set_controls(deepcopy(operator))
        for supplied,kind in [(modes,'mode'),(targets,'target')]:
            if supplied is not None and (type(supplied) is not dict or not set(supplied)<=set(self.loops)):raise ValueError('Unknown PI '+kind)
        if modes is not None and any(type(v) is not str or v not in ('manual','auto') for v in modes.values()):raise ValueError('PI manual/auto mode')
        if targets is not None and any(not self._target_ok(k,v) for k,v in targets.items()):raise ValueError('PI target domain')
        snap=self.plant.snapshot();frame=self.instruments.operator_frame();dt=self.instruments.tick_s
        if frame['time_s']!=self.plant.time and abs(frame['time_s']-self.plant.time)>1e-8:raise ValueError('Supervisor/instrument clock mismatch')
        entering=set();operator=operator or {}
        self.manual.update(deepcopy(operator))
        for key,value in sorted((modes or {}).items()):
            if value!=self.modes[key]:
                self._event('mode_changed',output=key,old=self.modes[key],new=value)
                if value=='auto':entering.add(key)
                elif key not in operator:self.manual[key]=self._feedback(snap,key)
                if value=='manual' and key in ('feed_flow_fraction','condensate_flow_fraction','feed_inventory_fraction'):
                    flag={'feed_flow_fraction':'feed_flow_control','condensate_flow_fraction':'condensate_flow_control','feed_inventory_fraction':'feed_inventory_control'}[key]
                    if flag not in operator:self.manual[flag]=True
                self.modes[key]=value
        for key,value in sorted((targets or {}).items()):
            if value!=self.targets[key]:self._event('target_changed',output=key,old=self.targets[key],new=value)
            self.targets[key]=value
        self._protect(frame,self.manual)
        if reset:self.policy.request_reset()
        proposals={};automatic={};controller={}
        quantities={s.name:s.quantity for s in self.instruments.sensors}
        for key,loop in self.loops.items():
            value=valid_reading(frame,loop.config.sensor,quantities[loop.config.sensor])
            proposal=None
            if key in ('feed_flow_fraction','condensate_flow_fraction'):
                name='feed_flow' if key=='feed_flow_fraction' else 'condensate_flow'
                if valid_reading(frame,name,quantities[name]) is None:value=None
            ff=0.
            if loop.config.feedforward_sensor is not None:
                name=loop.config.feedforward_sensor;measured=valid_reading(frame,name,quantities[name])
                if measured is None:value=None
                else:ff=(measured-loop.config.feedforward_reference)/loop.config.feedforward_scale
            if value is not None:
                proposal=loop.propose(value,self.targets[key],self._feedback(snap,key),dt,entering=key in entering,feedforward=ff)
            proposals[key]=proposal;requested_auto=self.modes[key]=='auto' and proposal is not None
            automatic[key]=requested_auto and not self.trip_latched
            if requested_auto and (not self.trip_latched or key not in ('feed_flow_fraction','condensate_flow_fraction','makeup_valve','feed_inventory_fraction')):
                controller[key]=proposal['command']
                if key in ('feed_flow_fraction','condensate_flow_fraction','feed_inventory_fraction'):
                    controller[{'feed_flow_fraction':'feed_flow_control','condensate_flow_fraction':'condensate_flow_control','feed_inventory_fraction':'feed_inventory_control'}[key]]=True
        interlock={};why={}
        if self.pressure_active:interlock['steam_bypass']=1.;why['steam_bypass']='measured_pressure_hysteresis'
        if self.manual['steam_inlet_isolated'] or not self.manual['generator_connected']:
            interlock['steam_admission']=0.;why['steam_admission']='steam_isolated_or_generator_disconnected'
        for on,fraction in [('feed_pump_on','feed_pump_fraction'),('condensate_pump_on','condensate_pump_fraction')]:
            if not self.manual[on]:
                interlock[fraction]=0.;why[fraction]='pump_not_requested'
                flag='feed_flow_control' if on=='feed_pump_on' else 'condensate_flow_control'
                interlock[flag]=False;why[flag]='pump_not_requested'
        protection=dict(scram=True,turbine_trip=True,generator_connected=False,rod_target=1.,steam_inlet_isolated=True,steam_admission=0.,feed_flow_control=False,condensate_flow_control=False,feed_inventory_control=False,makeup_valve=0.,drain_valve=0.) if self.trip_latched else {}
        selected=deepcopy(self.manual);authority={k:'operator' if k in operator else 'held_operator_intent' for k in selected}
        candidates={k:dict(operator=v) for k,v in operator.items()}
        for layer,patch in [('controller',controller),('interlock',interlock),('protection',protection)]:
            for key in sorted(patch):
                selected[key]=patch[key];authority[key]=layer;candidates.setdefault(key,{})[layer]=patch[key]
        dispatch={k:selected[k] for k in sorted(selected) if selected[k]!=self.plant.controls[k]}
        for key in dispatch:candidates.setdefault(key,{}).setdefault('held_operator_intent',self.manual[key])
        before=self.plant.time
        after=self.plant.advance(dt,dispatch or None)
        observed=self.instruments.ingest_station(after)
        for key,loop in self.loops.items():loop.commit(proposals[key],self._feedback(after,key),dt,automatic=automatic[key])
        receipt=dict(sequence=len(self.history),time_s=before,completed_at_s=self.plant.time,dispatched=deepcopy(dispatch),
            decisions={k:dict(candidates=candidates[k],selected=selected[k],authority=authority[k],reason=why.get(k) if authority[k]=='interlock' else 'latched_trip' if authority[k]=='protection' else 'automatic_loop' if authority[k]=='controller' else 'restored_operator_intent' if authority[k]=='held_operator_intent' else 'operator_request') for k in sorted(candidates)},
            actuator_feedback={k:self._feedback(after,k) for k in self.loops if k not in ('feed_flow_fraction','condensate_flow_fraction')},
            trip_latched=self.trip_latched,trip_reasons=deepcopy(self.trip_reasons),pressure_interlock=self.pressure_active,
            actual_positions=dict(rod=after['actuator_status']['rod_position'],admission=after['steam_header']['outlet_position'],bypass=after['bypass_position']),
            reset_in_place_available=False,modes=deepcopy(self.modes),targets=deepcopy(self.targets))
        self.last_receipt=receipt
        return dict(observation=observed,receipt=deepcopy(receipt))

    def checkpoint(self):
        return seal(dict(format='station-control/v1',source=control_signature(),initial_plant=deepcopy(self.initial_plant),
            initial_instruments=deepcopy(self.initial_instruments),loops=[asdict(v.config) for v in self.loops.values()],
            protection=asdict(self.protection),history=list(self.history),state=dict(
                plant=self.plant.checkpoint(),instruments=self.instruments.checkpoint(),
                pi={k:v.state() for k,v in self.loops.items()},modes=deepcopy(self.modes),targets=deepcopy(self.targets),
                manual=deepcopy(self.manual),pressure_active=self.pressure_active,trip_latched=self.trip_latched,
                trip_reasons=deepcopy(self.trip_reasons),events=deepcopy(self.events),
                policy_clock=dict(last_time_s=self.policy.last_time_s,last_tick=self.policy.last_tick),last_receipt=deepcopy(self.last_receipt))))

    @classmethod
    def restore(cls,saved):
        p=unseal(deepcopy(saved))
        budget=history_budget(unseal(p['initial_plant']).get('history_budget_steps',256))
        if p['format']!='station-control/v1' or p['source']!=control_signature() or type(p['history']) is not list or len(p['history'])>budget:raise ValueError('Control replay source/budget')
        plant_type=PWRFaultSession if 'equipment_faults' in unseal(p['initial_plant']) else PWRBypassSession
        obj=cls(plant_type.restore(p['initial_plant']),instruments=StationSignals.restore(p['initial_instruments']),
                loops=[PIConfig(**x) for x in p['loops']],protection=ProtectionConfig(**p['protection']))
        for row in p['history']:obj.step(**row)
        if obj.checkpoint()!=saved:raise ValueError('Control checkpoint replay differs')
        return obj
