"""Research feed train with finite gas-cushioned tanks and check-valve events."""
from copy import deepcopy
from dataclasses import asdict,dataclass
import math

import numpy as np
from atomic_model_multiphysics.provenance import seal,unseal
from .pwr_train import GovernedPWR,TrainConfig,physical_signature
from .cooling import PWRCooledSession,CoolingConfig


@dataclass(frozen=True)
class FeedConfig:
    feed_mass_kg:float=20000.
    makeup_mass_kg:float=5000.
    drain_mass_kg:float=100.
    feed_gas_volume_m3:float=20.
    makeup_gas_volume_m3:float=5.
    drain_gas_volume_m3:float=10.
    feed_pressure_pa:float=2e5
    makeup_pressure_pa:float=4e5
    drain_pressure_pa:float=1e5
    pump_efficiency:float=.85
    condensate_inertance_m_inv:float=100.
    condensate_friction_pa:float=5e4
    minimum_mass_kg:float=20.
    suction_static_head_m:float=6.
    required_suction_head_m:float=3.
    tank_area_m2:float=10.
    check_opening_band_pa:float=1.

    def __post_init__(self):
        for k,v in asdict(self).items():
            if type(v) not in (int,float) or not math.isfinite(v) or v<=0:raise ValueError('Positive finite feed parameter:'+k)
        ranges=dict(feed_mass_kg=(100.,1e6),makeup_mass_kg=(1.,1e6),drain_mass_kg=(1.,1e6),
            feed_gas_volume_m3=(.1,1e4),makeup_gas_volume_m3=(.1,1e4),drain_gas_volume_m3=(.1,1e4),
            feed_pressure_pa=(1e4,1e6),makeup_pressure_pa=(1e4,1e6),drain_pressure_pa=(1e4,1e6),
            condensate_friction_pa=(1e3,1e6),minimum_mass_kg=(1.,1000.),suction_static_head_m=(.1,100.),required_suction_head_m=(.1,50.),tank_area_m2=(1.,1000.),check_opening_band_pa=(1e-3,100.))
        if any(not low<=getattr(self,k)<=high for k,(low,high) in ranges.items()):raise ValueError('Feed parameter applicability')
        if not .5<=self.pump_efficiency<=1 or not 1<=self.condensate_inertance_m_inv<=1000:raise ValueError('Feed efficiency/inertance domain')
        if not self.minimum_mass_kg<min(self.feed_mass_kg,self.makeup_mass_kg,self.drain_mass_kg):raise ValueError('Tank initial reserve below minimum')
        if not self.drain_pressure_pa<self.feed_pressure_pa<self.makeup_pressure_pa<=1e6:raise ValueError('Declared tank pressure order')


class FeedPWR(GovernedPWR):
    def _sync_hotwell(self):
        super()._sync_hotwell()
        if getattr(self,'feed_train_ready',False):
            b=self.network.boundaries['condenser']
            self.true_hotwell=dict(h=float(b.enthalpy_j_kg),rho=float(b.density_kg_m3),p=float(b.pressure_pa))
            b.enthalpy_j_kg=self.feed_tank_h;b.pressure_pa=self.feed_tank_p;b.density_kg_m3=self.feed_density_ref

    def _apply_heat_and_actuators(self):
        super()._apply_heat_and_actuators()
        if not getattr(self,'feed_train_ready',False):return
        feed=self.network.junctions['feed'];source=self.network.boundaries['condenser'];c=self.feed_controls
        feed.loss_coefficient=self.feed_base_loss/c['feed_valve_fraction']**2
        _,friction,back=self.feed_terms()
        steam=self.network.junctions['steam'].mass_flow_kg_s+self.network.junctions['dump'].mass_flow_kg_s
        demand=max(0.,steam+.05*(self.feed_sg_mass_ref-self.network.volumes['sg_secondary'].mass_kg))
        requested=back+friction+self.feed_inertance/.2*(demand-feed.mass_flow_kg_s)
        cap=self.maximum_feed_head_pa*c['feed_pump_fraction']**2 if c['feed_pump_on'] else 0.
        delivered=max(0.,min(cap,requested));feed.pump_pressure_pa=delivered;feed.flow_target_kg_s=demand
        feed.delivery_enthalpy_j_kg=source.enthalpy_j_kg+(back+friction)/self.feed_density_ref-delivered/source.density_kg_m3
        self.feed_controller=dict(requested_flow_kg_s=demand,requested_head_pa=requested,delivered_head_pa=delivered,
            saturated=bool(requested!=delivered),available=c['feed_pump_on'])


class PWRFeedSession(PWRCooledSession):
    def _make_plant(self):return FeedPWR()

    def __init__(self,config=None,*,cooling=None,feed=None,max_step_s=.1,rtol=1e-7):
        self._hybrid_active=False;self.feed_ready=False;self.feed_config=feed if feed is not None else FeedConfig()
        if type(self.feed_config) is not FeedConfig:raise ValueError('FeedConfig required')
        super().__init__(config,cooling=cooling,max_step_s=max_step_s,rtol=rtol)
        p=self.plant;c=self.feed_config;b=p.network.boundaries['condenser'];self.feed_rho=float(p.feed_density_ref)
        self.feed_temperature_ref=float(p.hotwell_temperature_k);self.feed_u_ref=float(b.enthalpy_j_kg-b.pressure_pa/self.feed_rho)
        q=float(p.network.junctions['feed'].mass_flow_kg_s);self.condensate_resistance=c.condensate_friction_pa/q**2
        self.feed_index=len(self.state);self.condensate_open=True;self.hybrid_count=0
        self.feed_tanks={}
        for name in ('feed','makeup','drain'):
            mass=getattr(c,name+'_mass_kg');vg=getattr(c,name+'_gas_volume_m3');pressure=getattr(c,name+'_pressure_pa')
            self.feed_tanks[name]=dict(initial_mass_kg=mass,initial_gas_m3=vg,initial_pressure_pa=pressure,total_volume_m3=vg+mass/self.feed_rho)
        self.controls.update(feed_pump_on=True,feed_pump_fraction=1.,feed_valve_fraction=1.,condensate_pump_on=True,
            condensate_pump_fraction=1.,makeup_valve=0.,drain_valve=0.)
        self.state=np.r_[self.state,*[v for name in ('feed','makeup','drain') for v in (self.feed_tanks[name]['initial_mass_kg'],self.feed_tanks[name]['initial_mass_kg']*self.feed_u_ref)],q]
        p.feed_base_loss=p.network.junctions['feed'].loss_coefficient;p.feed_sg_mass_ref=p.network.volumes['sg_secondary'].mass_kg
        p.feed_train_ready=True;self.feed_ready=True
        self.feed_design=dict(density_kg_m3=self.feed_rho,reference_temperature_k=self.feed_temperature_ref,reference_internal_energy_j_kg=self.feed_u_ref,
            condensate_resistance_pa_s2_kg2=self.condensate_resistance,water_cp_j_kg_k=4180.,gas_gamma=1.4,
            gas_model='Sealed ideal adiabatic cushions; gas energy included',measured=False,initial_state='Declared hot start; added pump losses create a physical thermal settling transient')
        self.origin_feed_mass=sum(self.feed_tanks[n]['initial_mass_kg'] for n in self.feed_tanks)
        self._rhs(0.,self.state);self.origin_energy=self._stored();self._admit(self.state)

    def _tank(self,y,name):
        offset={'feed':0,'makeup':2,'drain':4}[name];i=self.feed_index+offset;m=float(y[i]);u=float(y[i+1]);s=self.feed_tanks[name]
        gas=s['total_volume_m3']-m/self.feed_rho
        if m<=0 or gas<=0:raise ValueError('Finite tank dry or gas cushion exhausted:'+name)
        pressure=s['initial_pressure_pa']*(s['initial_gas_m3']/gas)**1.4
        temp=self.feed_temperature_ref+(u/m-self.feed_u_ref)/4180.
        return dict(mass_kg=m,water_energy_j=u,gas_volume_m3=gas,pressure_pa=pressure,temperature_k=temp,
            h_j_kg=u/m+pressure/self.feed_rho,gas_energy_j=pressure*gas/.4,level_m=m/(self.feed_rho*self.feed_config.tank_area_m2))

    def _stored(self):
        stored=super()._stored()
        if not self.feed_ready:return stored
        tanks=[self._tank(self.state,n) for n in self.feed_tanks];q=self.state[self.feed_index+6]
        return float(stored+sum(t['water_energy_j']+t['gas_energy_j'] for t in tanks)+.5*self.feed_config.condensate_inertance_m_inv*q*q/self.feed_rho)

    def _rhs(self,t,y):
        if not self.feed_ready:return super()._rhs(t,y)
        c=self.feed_config;i=self.feed_index;p=self.plant;tanks={n:self._tank(y,n) for n in self.feed_tanks};tank=tanks['feed']
        p.feed_tank_p=tank['pressure_pa'];p.feed_tank_h=tank['h_j_kg'];p.feed_controls=self.controls
        rate=super()._rhs(t,y);l=self.ledger_index;rho=self.feed_rho
        qf=float(p.network.junctions['feed'].mass_flow_kg_s);qc=float(y[i+6]);actual=p.true_hotwell
        back=tank['pressure_pa']-actual['p'];friction=self.condensate_resistance*qc*abs(qc)
        demand=max(0.,qf+.05*(c.feed_mass_kg-tank['mass_kg'])+.02*(p.hotwell_mass_kg-p.condenser_mass_ref_kg))
        requested=back+friction+c.condensate_inertance_m_inv/.5*(demand-qc)
        cap=2e6*self.controls['condensate_pump_fraction']**2 if self.controls['condensate_pump_on'] else 0.
        head=max(0.,min(cap,requested));drive=head-back-friction
        dq=drive/c.condensate_inertance_m_inv if self.condensate_open else 0.
        cond_hyd=qc*head/rho;cond_elec=cond_hyd/c.pump_efficiency
        cond_loss=cond_elec-cond_hyd
        feed_hyd=p.boundary_terms_w()['feed_work_w'];feed_loss=feed_hyd*(1/c.pump_efficiency-1)
        makeup=self.controls['makeup_valve']*5.*math.sqrt(max(0.,(tanks['makeup']['pressure_pa']-tank['pressure_pa'])/2e5))
        drain=self.controls['drain_valve']*5.*math.sqrt(max(0.,(tank['pressure_pa']-tanks['drain']['pressure_pa'])/1e5))
        dm=[qc-qf+makeup-drain,-makeup,drain]
        incoming=actual['h']+(back+friction)/rho
        enthalpy=[qc*incoming-qf*tank['h_j_kg']+makeup*tanks['makeup']['h_j_kg']-drain*tank['h_j_kg']+cond_loss+feed_loss,
                  -makeup*tanks['makeup']['h_j_kg'],drain*tank['h_j_kg']]
        water_rates=[energy-tanks[name]['pressure_pa']*mass_rate/rho for name,energy,mass_rate in zip(self.feed_tanks,enthalpy,dm)]
        # Parent drained its feed source from the hotwell; replace that edge by
        # the real condensate edge. The SG still receives actual tank feed.
        rate[p.pack().size-3]+=qf-qc
        hotwell_correction=qf*tank['h_j_kg']-qc*actual['h'];rate[p.pack().size-2]+=hotwell_correction
        extra=cond_elec+feed_loss
        rate[l]+=extra;rate[l+2]+=extra
        self.powers['own_needs_w']+=extra;self.powers['net_export_w']-=extra;self.powers['external_net_w']=float(rate[l])
        self.powers.update(condensate_pump_electric_w=cond_elec,feed_pump_extra_loss_w=feed_loss,feed_pump_electric_w=feed_hyd+feed_loss)
        rate[l+3]=abs(self.powers['nuclear_heat_w'])+abs(self.powers['primary_pump_loss_out_w'])+abs(self.powers['external_cooling_heat_out_w'])+abs(self.powers['net_export_w'])
        kinetic=c.condensate_inertance_m_inv*qc*dq/rho
        added=hotwell_correction+sum(enthalpy)+kinetic
        self.instantaneous_energy_defect_w=float(self.instantaneous_energy_defect_w+added-extra)
        self.feed_terms=dict(tanks=tanks,feed_flow_kg_s=qf,condensate_flow_kg_s=qc,makeup_flow_kg_s=makeup,drain_flow_kg_s=drain,
            condensate_head_pa=head,condensate_requested_head_pa=requested,condensate_drive_pa=drive,condensate_saturated=bool(head!=requested),
            condensate_electric_w=cond_elec,feed_electric_w=feed_hyd+feed_loss,feed_controller=deepcopy(p.feed_controller),
            check_opening_band_pa=c.check_opening_band_pa,feed_check_open=p._feed_open,condensate_check_open=self.condensate_open,internal_energy_residual_w=added-extra,
            hotwell_mass_rate_kg_s=float(rate[self.base_size-3]),hotwell_energy_rate_w=float(rate[self.base_size-2]),hotwell_liquid_h_j_kg=actual['h'],
            total_mass_kg=float(p.closed_inventories()['closed_mass_kg']+sum(t['mass_kg'] for t in tanks.values())))
        return np.r_[rate,*[v for mass,energy in zip(dm,water_rates) for v in (mass,energy)],dq]

    def _admit(self,y):
        if self.feed_ready:
            p=self.plant;i=self.feed_index;tank=self._tank(y,'feed')
            p.feed_tank_p=tank['pressure_pa'];p.feed_tank_h=tank['h_j_kg'];p.feed_controls=self.controls
        super()._admit(y)
        if not self.feed_ready:return
        from physics.materials.iapws import saturation_pressure_mpa
        for name in self.feed_tanks:
            v=self._tank(y,name)
            if v['mass_kg']<self.feed_config.minimum_mass_kg or not 280<=v['temperature_k']<=350:
                raise ValueError('Tank reserve/temperature domain:'+name)
            if v['pressure_pa']<=saturation_pressure_mpa(v['temperature_k'])*1e6:raise ValueError('Tank liquid phase scope:'+name)
            if v['pressure_pa']<1e4 or v['pressure_pa']>2e6:raise ValueError('Gas cushion pressure domain:'+name)
        v=tank
        vapor=float(saturation_pressure_mpa(v['temperature_k'])*1e6)
        self.feed_suction_m=(v['pressure_pa']-vapor)/(self.feed_rho*9.80665)+v['level_m']
        state=p.condenser.state();liquid_mass=p.hotwell_mass_kg*(1-state.quality)
        self.condensate_suction_m=self.feed_config.suction_static_head_m+(liquid_mass-p.condenser_mass_ref_kg*.95)/(self.feed_rho*100.)
        if self.feed_suction_m<self.feed_config.required_suction_head_m or self.condensate_suction_m<self.feed_config.required_suction_head_m:
            raise ValueError('Suction head outside declared noncavitating scope')
        if y[i+6]<0:raise ValueError('Reverse condensate flow unsupported')
        if abs(p.true_hotwell['rho']/self.feed_rho-1)>.02:raise ValueError('Condensate reference-density scope exceeded')

    def _set_controls(self,patch):
        extra={'feed_pump_on','feed_pump_fraction','feed_valve_fraction','condensate_pump_on','condensate_pump_fraction','makeup_valve','drain_valve'}
        if type(patch) is not dict or not patch or not set(patch)<=set(self.controls):raise ValueError('Feed command patch')
        values=dict(self.controls,**patch)
        for key in ('feed_pump_on','condensate_pump_on'):
            if type(values[key]) is not bool:raise ValueError('Boolean pump availability')
        for key,low,high in [('feed_pump_fraction',0.,1.),('feed_valve_fraction',.1,1.),('condensate_pump_fraction',0.,1.),('makeup_valve',0.,1.),('drain_valve',0.,1.)]:
            value=values[key]
            if type(value) not in (int,float) or not math.isfinite(value) or not low<=value<=high:raise ValueError('Feed command domain:'+key)
        regular={k:v for k,v in patch.items() if k not in extra}
        if regular:super()._set_controls(regular)
        self.controls.update({k:values[k] for k in extra})
        if any(k in extra for k in patch):self.events.append(dict(time_s=self.time,kind='feed_command',requested={k:v for k,v in patch.items() if k in extra}))

    def advance(self,duration_s,patch=None):
        if self._hybrid_active:return super().advance(duration_s,patch)
        trial=deepcopy(self);trial._hybrid_active=True;trial.hybrid_count=0
        super(PWRFeedSession,trial).advance(duration_s,patch)
        trial._hybrid_active=False
        self.__dict__.update(trial.__dict__);return self.snapshot()

    def _stop_at_endpoint(self,solution):return any(len(t) for t in solution.t_events[2:])

    def _prepare_interval(self):
        if not self.feed_ready:return
        self._rhs(self.time,self.state)
        for name,index,opened,drive in [('feed',self.plant.feed_index,self.plant._feed_open,self.plant.feed_terms()[0]),
            ('condensate',self.feed_index+6,self.condensate_open,self.feed_terms['condensate_drive_pa'])]:
            if not opened and drive>self.feed_config.check_opening_band_pa:
                if self.state[index]!=0:raise ValueError('Closed valve has stored momentum')
                if name=='feed':self.plant._feed_open=True
                else:self.condensate_open=True
                self.events.append(dict(time_s=self.time,kind=name+'_check_opened',reason='positive_drive_above_declared_band'))

    def _extend_known(self,known,modes):
        if not self.plant._feed_open:known[self.plant.feed_index]=lambda t:0.;modes.append(dict(name='feed_check',kind='exact_closed'))
        if not self.condensate_open:known[self.feed_index+6]=lambda t:0.;modes.append(dict(name='condensate_check',kind='exact_closed'))

    def _flow_guards(self,feed):
        guards=[];self.guard_labels=[]
        for name,index,opened in [('feed',self.plant.feed_index,self.plant._feed_open),('condensate',self.feed_index+6,self.condensate_open)]:
            if opened:
                def event(t,y,j=index):return float(y[j])
                event.direction=-1;self.guard_labels.append(name+'_closed')
            else:
                def event(t,y,name=name):
                    self._rhs(t,y)
                    return (self.plant.feed_terms()[0] if name=='feed' else self.feed_terms['condensate_drive_pa'])-self.feed_config.check_opening_band_pa
                event.direction=1;self.guard_labels.append(name+'_opened')
            guards.append(event)
        return guards

    def _custom_root(self,label,endpoint,now):return False

    def _resume_stop(self,trial,solution,full,end,duration_s,patch):
        hits=[i for i,times in enumerate(solution.t_events) if len(times)]
        if not hits or any(i<2 for i in hits):raise ValueError('Turbine operating limit reached')
        now=float(solution.t[-1]);endpoint=full(now,solution.y[:,-1]);trial.state=endpoint;trial.time=now
        for hit in hits:
            label=trial.guard_labels[hit-2]
            if trial._custom_root(label,endpoint,now):continue
            name=label.split('_')[0];index=trial.plant.feed_index if name=='feed' else trial.feed_index+6
            if label.endswith('closed'):
                residual=float(endpoint[index])
                if abs(residual)>1e-6:raise ValueError('Check-valve zero-root residual too large')
                inertia=trial.plant.feed_inertance if name=='feed' else trial.feed_config.condensate_inertance_m_inv
                energy=.5*inertia*residual**2/trial.feed_rho
                endpoint[index]=0.;endpoint[trial.feed_index+1]+=energy
                if name=='feed':trial.plant._feed_open=False
                else:trial.condensate_open=False
                trial.events.append(dict(time_s=now,kind=name+'_check_closed',root_residual_kg_s=residual,root_roundoff_energy_to_tank_j=energy))
            else:
                if name=='feed':trial.plant._feed_open=True
                else:trial.condensate_open=True
                trial.events.append(dict(time_s=now,kind=name+'_check_opened',reason='drive_zero_crossing'))
        # Validate the accepted prefix as well as the constraint-consistent root.
        for t in sorted(set(solution.t[:-1].tolist()+np.linspace(self.time,now,max(2,math.ceil((now-self.time)/.05)+1))[:-1].tolist())):
            trial._admit(full(t,solution.sol(t)))
        _,motion=trial.rod_drive.trajectory(float(self.state[12]),float(trial.controls['rod_target']),trial.scram_latched,self.time)
        if motion['direction'] and now>=motion['arrival_s']:
            trial.events.append(dict(time_s=motion['arrival_s'],kind='rod_arrived',position=motion['target'],reason='scram_stop' if trial.scram_latched else 'requested_position'))
        trial.events.sort(key=lambda event:event['time_s'])
        trial._rhs(now,endpoint);trial._admit(endpoint)
        trial.hybrid_count=self.hybrid_count+1
        if trial.hybrid_count>8:raise ValueError('Feed hybrid event budget')
        if now<end:trial.advance(end-now)
        trial.history=deepcopy(self.history);trial.history.append(dict(duration_s=duration_s,patch=deepcopy(patch)))
        self.__dict__.update(trial.__dict__);return self.snapshot()

    def parameter_registry(self):
        result=super().parameter_registry()
        if not self.feed_ready:return result
        units=dict(feed_mass_kg='kg',makeup_mass_kg='kg',drain_mass_kg='kg',feed_gas_volume_m3='m^3',makeup_gas_volume_m3='m^3',drain_gas_volume_m3='m^3',
            feed_pressure_pa='Pa',makeup_pressure_pa='Pa',drain_pressure_pa='Pa',pump_efficiency='1',condensate_inertance_m_inv='1/m',
            condensate_friction_pa='Pa',minimum_mass_kg='kg',suction_static_head_m='m',required_suction_head_m='m',tank_area_m2='m^2',check_opening_band_pa='Pa')
        for key,value in asdict(self.feed_config).items():
            result['parameters']['feed.'+key]=dict(value=float(value),unit=units[key],source='Declared reduced feed train, not manufacturer data',family='illustrative',version='1',uncertainty=None)
        for key,value,unit in [('gamma',1.4,'1'),('cp',4180.,'J/(kg*K)'),('density',self.feed_rho,'kg/m^3'),('condensate_resistance',self.condensate_resistance,'Pa*s^2/kg^2')]:
            result['parameters']['feed.design.'+key]=dict(value=float(value),unit=unit,source='Declared constitutive constant or constructed nominal value',family='illustrative',version='1',uncertainty=None)
        return result

    def snapshot(self):
        result=super().snapshot()
        if self.feed_ready:
            inventory=self.plant.closed_inventories()['closed_mass_kg'];total=inventory+sum(self._tank(self.state,n)['mass_kg'] for n in self.feed_tanks)
            origin=self.origin_mass+self.origin_feed_mass
            terms=self.feed_terms;steam=self.plant.network.junctions['steam'];dump=self.plant.network.junctions['dump']
            expected_mass=steam.mass_flow_kg_s+dump.mass_flow_kg_s-terms['condensate_flow_kg_s']
            expected_energy=steam.mass_flow_kg_s*steam.delivery_enthalpy_j_kg+dump.mass_flow_kg_s*dump.delivery_enthalpy_j_kg-terms['condensate_flow_kg_s']*terms['hotwell_liquid_h_j_kg']-self.powers['condenser_heat_out_w']+self.powers['returned_losses_w']
            result['condenser_balance']=dict(mass_rate_kg_s=terms['hotwell_mass_rate_kg_s'],expected_mass_rate_kg_s=float(expected_mass),
                energy_rate_w=terms['hotwell_energy_rate_w'],expected_energy_rate_w=float(expected_energy),mass_residual_kg_s=float(terms['hotwell_mass_rate_kg_s']-expected_mass),
                energy_residual_w=float(terms['hotwell_energy_rate_w']-expected_energy),condensate_out_kg_s=terms['condensate_flow_kg_s'])
            result.update(model_id='npp-pwr-train-finite-feed',feed_train=deepcopy(self.feed_terms),feed_config=asdict(self.feed_config),feed_design=deepcopy(self.feed_design),
                feed_inventory=self.state[self.feed_index:self.feed_index+7].tolist(),feed_suction_m=self.feed_suction_m,condensate_suction_m=self.condensate_suction_m,
                mass_relative=float(abs(total-origin)/origin),finite_feed_connected=True)
        return result

    def checkpoint(self):
        p=unseal(super().checkpoint());p['feed_config']=asdict(self.feed_config);return seal(p)

    @classmethod
    def restore(cls,saved):
        p=unseal(deepcopy(saved))
        if p['source']!=physical_signature() or type(p['history']) is not list or len(p['history'])>256:raise ValueError('Feed replay source/budget')
        s=cls(TrainConfig(**p['config']),cooling=CoolingConfig(**p['cooling_config']),feed=FeedConfig(**p['feed_config']),max_step_s=p['max_step_s'],rtol=p['rtol'])
        for row in p['history']:s.advance(**row)
        if s.checkpoint()!=saved:raise ValueError('Feed replay differs')
        return s
