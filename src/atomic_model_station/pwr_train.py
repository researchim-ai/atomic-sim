"""Hot reduced-PWR / finite turbogenerator coupling; not a complete NPP R1."""
from copy import deepcopy
from dataclasses import asdict,dataclass
import hashlib
import math
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from core.status import DomainError,SolverError
from atomic_model_controlled_plant.model import ControlledPWR
from atomic_model_controlled_plant.session import identity as controlled_identity
from atomic_model_plant_extension.initialization import solve as initialize
from atomic_model_multiphysics.provenance import seal,unseal
from physics.power import steam_turbine_train as turbine
from .model import signature
from .contracts import FluidState,fluid_connection
from .actuators import RodDrive,boron_capability


def physical_signature():
    return dict(station=signature(),controlled_plant=controlled_identity(),
                turbine_sha256=hashlib.sha256(Path(turbine.__file__).read_bytes()).hexdigest())


@dataclass(frozen=True)
class TrainConfig:
    inertia_kg_m2:float=2e6
    nominal_speed_rad_s:float=157.
    generator_efficiency:float=.98
    nominal_bearing_fraction:float=.01
    governor_gain:float=5.
    minimum_speed_fraction:float=.9
    maximum_speed_fraction:float=1.1
    clad_capacity_j_k:float=1e7
    clad_resistance_fraction:float=.02
    def __post_init__(self):
        ranges={'inertia_kg_m2':(1e5,1e8),'nominal_speed_rad_s':(50.,400.),'generator_efficiency':(.5,1.),
                'nominal_bearing_fraction':(.001,.1),'governor_gain':(0.,20.),
                'minimum_speed_fraction':(.5,.99),'maximum_speed_fraction':(1.01,1.5),
                'clad_capacity_j_k':(1e5,1e9),'clad_resistance_fraction':(.001,.2)}
        for k,(low,high) in ranges.items():
            v=getattr(self,k)
            if type(v) not in (int,float) or not math.isfinite(v) or not low<=v<=high:raise ValueError('Train parameter domain:'+k)


class GovernedPWR(ControlledPWR):
    def turbine_outlet_enthalpy(self,sg_pressure_pa):
        result=turbine.expansion(dict(inlet_pressure_pa=sg_pressure_pa,
            outlet_pressure_pa=self._condenser_pressure_pa,isentropic_efficiency=self.eta_turbine))
        self.live_expansion=result
        return result['outlet_enthalpy_j_kg']
    def _apply_heat_and_actuators(self):
        requested=self.throttle
        if getattr(self,'train_ready',False):
            cfg=self.train_config;speed=math.sqrt(2*self.train_energy_j/cfg.inertia_kg_m2)
            correction=cfg.governor_gain*(1-speed/cfg.nominal_speed_rad_s) if self.train_controls['governor_enabled'] else 0.
            self.throttle=0. if self.train_turbine_tripped else min(1.,max(0.,requested+correction))
        try:super()._apply_heat_and_actuators()
        finally:self.throttle=requested
        if getattr(self,'train_ready',False):
            coolant=self.network.volumes['core'].temperature_k
            self.fuel_to_clad_w=self.fuel_clad_ua*(self.fuel_temperature-self.clad_temperature_k)
            self.clad_to_coolant_w=self.clad_coolant_ua*(self.clad_temperature_k-coolant)
            self.fuel_to_coolant_w=self.fuel_to_clad_w
            self.network.volumes['core'].heat_w=self.clad_to_coolant_w


class PWRTrainSession:
    def __init__(self,config=None,*,max_step_s=.1,rtol=1e-7):
        if type(max_step_s) not in (int,float) or not .01<=max_step_s<=1 or type(rtol) not in (float,int) or not 1e-10<=rtol<=1e-5:
            raise ValueError('Physical integration options')
        self.config=config if config is not None else TrainConfig()
        if type(self.config) is not TrainConfig:raise ValueError('TrainConfig required')
        self.max_step_s=float(max_step_s);self.rtol=float(rtol);self.time=0.;self.history=[];self.last_reduction=[]
        self.controls=dict(load_fraction=1.,generator_connected=True,valve_request=1.,governor_enabled=True,
            rod_target=0.,scram=False,turbine_trip=False)
        self.scram_latched=False;self.turbine_tripped=False;self.events=[];self.rod_drive=RodDrive();self.minimum_dump_sample_kg_s=0.
        self.plant=self._make_plant();self.initialization=initialize(self.plant)
        self.controls['rod_target']=float(self.plant.rod)
        p=self.plant;c=self.config;initial=p.pack().copy();p.derivatives(0.,initial)
        q=p.network.junctions['steam'];work=q.mass_flow_kg_s*(p.h_g_current-q.delivery_enthalpy_j_kg)
        if not math.isfinite(work) or work<=0:raise ValueError('Positive nominal turbine work required')
        self.nominal_work_w=work
        self.bearing_drag=c.nominal_bearing_fraction*work/c.nominal_speed_rad_s**2
        self.generator_drag=(1-c.nominal_bearing_fraction)*work/c.nominal_speed_rad_s**2
        energy=.5*c.inertia_kg_m2*c.nominal_speed_rad_s**2
        nominal_losses=work-(c.generator_efficiency*(1-c.nominal_bearing_fraction)*work)
        gap=p.hotwell_temperature_k-p.t_sink
        if gap<=0:raise ValueError('Nominal condenser sink temperature gap')
        old_ua=p.ua_condenser;p.ua_condenser+=nominal_losses/gap
        self.charging=dict(base_condenser_ua_w_k=old_ua,condenser_ua_w_k=p.ua_condenser,
            returned_nominal_losses_w=nominal_losses,reason='Explicit design sizing for added shaft/generator heat at the unchanged hot inventory',
            measured=False,inventories_projected=False)
        coolant=p.network.volumes['core'].temperature_k
        clad=coolant+c.clad_resistance_fraction*(p.fuel_temperature-coolant)
        p.fuel_clad_ua=p.ua_fuel/(1-c.clad_resistance_fraction)
        p.clad_coolant_ua=p.ua_fuel/c.clad_resistance_fraction
        self.charging['fuel_clad_ua_w_k']=p.fuel_clad_ua
        self.charging['clad_coolant_ua_w_k']=p.clad_coolant_ua
        self.charging['initial_clad_temperature_k']=clad
        self.charging['thermal_resistance_split']='Declared series split; same nominal effective UA, no measured fit'
        p.train_turbine_tripped=False;p.train_ready=True;p.train_config=c;p.train_energy_j=energy;p.train_controls=self.controls;p.clad_temperature_k=clad
        self.base_size=len(initial);self.ledger_index=self.base_size+2
        self.state=np.r_[initial,energy,clad,0.,0.,0.,0.]
        self.origin_mass=p.closed_inventories()['closed_mass_kg'];self.origin_boron=p.closed_inventories()['boron_kg']
        self._rhs(0.,self.state)
        self.origin_energy=self._stored()
        self._admit(self.state)
    def _make_plant(self):return GovernedPWR()
    def _reconstruct_state(self,y):return y
    def _absolute_tolerances(self,atol):return atol
    def _prepare_interval(self):pass
    def _maximum_time_s(self):return 3600.
    def _inactive_target_names(self):return ('spray',)
    def _extend_known(self,known,modes):pass
    def _flow_guards(self,feed):return [feed]
    def _stop_at_endpoint(self,solution):return False
    def _resume_stop(self,trial,solution,full,end,duration_s,patch):
        raise ValueError('Physical interval incomplete or operating limit reached')
    def _stored(self):return float(self.plant.closed_inventories()['stored_j']+self.state[self.base_size]+self.config.clad_capacity_j_k*self.state[self.base_size+1])
    def parameter_registry(self):
        params={}
        units=dict(inertia_kg_m2='kg*m^2',nominal_speed_rad_s='rad/s',generator_efficiency='1',nominal_bearing_fraction='1',
            governor_gain='1',minimum_speed_fraction='1',maximum_speed_fraction='1',clad_capacity_j_k='J/K',clad_resistance_fraction='1')
        def add(key,value,unit,source):params[key]=dict(value=float(value),unit=unit,source=source,family='illustrative',version='1',uncertainty=None)
        for k,v in asdict(self.rod_drive).items():add('rod.'+k,v,'1/s','Declared aggregate rod-drive law; not plant actuator data')
        for k,v in asdict(self.config).items():add('train.'+k,v,units[k],'Declared research train/cladding parameter, not manufacturer data')
        for k in ('base_condenser_ua_w_k','condenser_ua_w_k','fuel_clad_ua_w_k','clad_coolant_ua_w_k'):
            add('design.'+k,self.charging[k],'W/K','Constructed hot nominal sizing; not measured calibration')
        add('design.nominal_turbine_work_w',self.nominal_work_w,'W','Derived from live initialized steam ports')
        add('design.generator_drag',self.generator_drag,'N*m*s/rad','Constructed nominal shaft/load balance')
        add('design.bearing_drag',self.bearing_drag,'N*m*s/rad','Constructed nominal shaft loss allocation')
        return dict(schema_version=1,model_version='0.1.0',parameters=params,future_families=['illustrative','measured','calibrated'])
    def _rhs(self,t,y):
        p=self.plant;c=self.config;n=self.base_size;k=float(y[n])
        if not np.isfinite(y).all() or k<=0:raise ValueError('Positive finite turbine rotor state')
        p.train_energy_j=k;p.train_controls=self.controls;p.throttle=self.controls['valve_request'];p.clad_temperature_k=float(y[n+1])
        p.scram_latched=self.scram_latched;p.train_turbine_tripped=self.turbine_tripped
        rate=p.derivatives(t,y[:n]);omega2=2*k/c.inertia_kg_m2
        steam=p.network.junctions['steam'];work=steam.mass_flow_kg_s*(p.h_g_current-steam.delivery_enthalpy_j_kg)
        if work<0:raise ValueError('Negative turbine power outside current forward-flow model')
        generator=(self.generator_drag*self.controls['load_fraction'] if self.controls['generator_connected'] and not self.turbine_tripped else 0.)*omega2
        bearing=self.bearing_drag*omega2;gross=c.generator_efficiency*generator;losses=bearing+(1-c.generator_efficiency)*generator
        # Expansion removes shaft work once; this returns only dissipative losses.
        rate[-2]+=losses
        terms=p.boundary_terms_w()
        own=terms['motor_w']+terms['feed_work_w']+terms['pressurizer_heater_w']
        net=gross-own
        boundary=terms['nuclear_w']+terms['shaft_loss_w']+terms['condenser_w']-net
        self.powers=dict(steam_work_w=work,generator_mechanical_w=generator,turbine_bearing_w=bearing,
            returned_losses_w=losses,gross_electrical_w=gross,own_needs_w=own,net_export_w=net,
            nuclear_heat_w=terms['nuclear_w'],condenser_heat_out_w=-terms['condenser_w'],
            primary_pump_loss_out_w=-terms['shaft_loss_w'],external_net_w=boundary)
        clad_rate=(p.fuel_to_clad_w-p.clad_to_coolant_w)/c.clad_capacity_j_k
        self.instantaneous_energy_defect_w=float(sum(p.account_rates_w(rate).values())+(work-generator-bearing)+c.clad_capacity_j_k*clad_rate-boundary)
        return np.r_[rate,work-generator-bearing,clad_rate,boundary,gross,own,abs(terms['nuclear_w'])+abs(terms['condenser_w'])+abs(net)+abs(terms['shaft_loss_w'])]
    def _admit(self,y):
        p=self.plant;c=self.config;n=self.base_size
        p.train_energy_j=float(y[n]);p.clad_temperature_k=float(y[n+1]);p.refresh_algebraic(y[:n]);p._reject_if_out_of_domain()
        if any(y[13+3*i+2]<0 for i in range(len(p.network.volumes))):raise ValueError('Negative dissolved boron inventory')
        from physics.materials.iapws import saturation_temperature_k
        saturation=saturation_temperature_k(p.network.volumes['core'].pressure_pa*1e-6)
        if not 300<=y[n+1]<saturation:raise ValueError('Cladding surface saturation/temperature outside current single-phase film model')
        speed=math.sqrt(2*y[n]/c.inertia_kg_m2)
        if not c.minimum_speed_fraction*c.nominal_speed_rad_s<=speed<=c.maximum_speed_fraction*c.nominal_speed_rad_s:raise ValueError('Turbogenerator speed outside admitted band')
        self.minimum_dump_sample_kg_s=min(self.minimum_dump_sample_kg_s,float(p.network.junctions['dump'].mass_flow_kg_s))
        if p.network.junctions['dump'].mass_flow_kg_s < -1e-9:raise ValueError('Reverse steam bypass flow unsupported')
        if p.network.junctions['steam'].mass_flow_kg_s<0:raise ValueError('Reverse turbine steam flow unsupported')
    def _set_controls(self,patch):
        if type(patch) is not dict or not patch or not set(patch)<=set(self.controls):raise ValueError('Train command patch')
        v=dict(self.controls,**patch)
        if any(type(v[k]) is not bool for k in ('generator_connected','governor_enabled','scram','turbine_trip')):raise ValueError('Boolean train switches')
        for k,lo,hi in [('load_fraction',.5,1.2),('valve_request',.5,1.),('rod_target',0.,1.)]:
            x=v[k]
            if type(x) not in (float,int) or not math.isfinite(x) or not lo<=x<=hi:raise ValueError('Train command domain:'+k)
        self.controls=v
        self.events.append(dict(time_s=self.time,kind='command',requested=deepcopy(patch)))
        if v['scram'] and not self.scram_latched:
            self.scram_latched=True
            self.events.append(dict(time_s=self.time,kind='scram_latched',reason='explicit_reactor_trip',effective_rod_target=1.))
        if v['turbine_trip'] and not self.turbine_tripped:
            self.turbine_tripped=True
            self.events.append(dict(time_s=self.time,kind='turbine_trip_latched',reason='explicit_turbine_trip',effective_generator_connected=False))
        if self.scram_latched and ('rod_target' in patch or patch.get('scram') is False):
            self.events.append(dict(time_s=self.time,kind='command_inhibited',reason='latched_scram',effective_rod_target=1.))
        if self.turbine_tripped and any(k in patch for k in ('generator_connected','valve_request','turbine_trip')):
            self.events.append(dict(time_s=self.time,kind='command_inhibited',reason='latched_turbine_trip',effective_generator_connected=False,effective_valve=0.))
    def _passive_column(self,index):
        return self.ledger_index<=index<self.ledger_index+4
    def advance(self,duration_s,patch=None):
        if type(duration_s) not in (float,int) or not math.isfinite(duration_s) or not 0<duration_s<=60 or self.time+duration_s>self._maximum_time_s() or len(self.history)>=256:raise ValueError('Physical duration/history budget')
        trial=deepcopy(self)
        if patch is not None:trial._set_controls(patch)
        trial._prepare_interval()
        n=trial.base_size;cfg=trial.config
        def low(t,y):return y[n]-.5*cfg.inertia_kg_m2*(cfg.minimum_speed_fraction*cfg.nominal_speed_rad_s)**2
        def high(t,y):return .5*cfg.inertia_kg_m2*(cfg.maximum_speed_fraction*cfg.nominal_speed_rad_s)**2-y[n]
        def feed(t,y):return float(y[trial.plant.feed_index])
        for event in (low,high,feed):event.terminal=True;event.direction=-1
        end=trial.time+duration_s
        atol=np.full(len(trial.state),1e-6);atol[n:]=.01;atol[n+1]=1e-7
        atol=trial._absolute_tolerances(atol)
        # Exact known modes are eliminated, never clipped: secondary nonvolatile
        # boron is constant; inactive target flows solve q'=-q/tau analytically.
        trial._rhs(trial.time,trial.state)
        plant=trial.plant;volume_names=plant.network.volume_names();junction_names=plant.network.junction_names()
        secondary_b=13+3*volume_names.index('sg_secondary')+2
        if plant.network.junctions['steam'].boron_carry!=0 or plant.network.junctions['dump'].boron_carry!=0 or plant.network.boundaries['condenser'].boron_fraction!=0:
            raise ValueError('Secondary boron invariant not applicable')
        rod_at,motion=trial.rod_drive.trajectory(float(trial.state[12]),float(trial.controls['rod_target']),trial.scram_latched,trial.time)
        plant._rod_rate_override=0.  # Rod is supplied by the exact independent drive, never integrated/clipped.
        known={secondary_b:lambda t,v=float(trial.state[secondary_b]):v,12:rod_at}
        reduced_modes=[dict(name='secondary_boron',kind='exact_constant',value=float(trial.state[secondary_b]))]
        reduced_modes.append(dict(name='rod_position',kind='exact_bounded_drive',**motion))
        inactive={}
        offset=13+3*len(plant.network.volumes)
        for name in trial._inactive_target_names():
            j=plant.network.junctions[name]
            if j.flow_law=='target' and j.flow_target_kg_s==0.:
                index=offset+junction_names.index(name);q0=float(trial.state[index]);tau=j.flow_time_constant_s
                known[index]=lambda t,q=q0,tau=tau,origin=trial.time:q*math.exp(-(t-origin)/tau)
                inactive[index]=name;reduced_modes.append(dict(name=name,kind='exact_exponential',initial=q0,tau_s=tau))
        trial._extend_known(known,reduced_modes)
        retained=np.array([i for i in range(len(trial.state)) if i not in known]);origin_state=trial.state.copy()
        def full(t,small):
            y=origin_state.copy();y[retained]=small
            for i,fn in known.items():y[i]=fn(t)
            return trial._reconstruct_state(y)
        def fun(t,small):
            rate=trial._rhs(t,full(t,small))
            if rate[secondary_b]!=0. or any(plant.network.junctions[name].flow_target_kg_s!=0. for name in inactive.values()):
                raise ValueError('Eliminated mode activated; this hot-domain adapter requires a new hybrid branch')
            return rate[retained]
        boron_indices=[13+3*i+2 for i in range(len(plant.network.volumes))]
        def jac(t,small):
            base=fun(t,small);matrix=np.zeros((len(small),len(small)))
            def probe(candidate):
                if any(full(t,candidate)[i]<0 for i in boron_indices):raise ValueError('Jacobian probe outside boron domain')
                return fun(t,candidate)
            for column,index in enumerate(retained):
                if trial._passive_column(index):continue  # physical extension states remain coupled
                delta=1e-7*max(abs(float(small[column])),1.)
                for _ in range(10):
                    up=small.copy();down=small.copy();up[column]+=delta;down[column]-=delta
                    try:above=probe(up)
                    except (ValueError,DomainError,SolverError):above=None
                    try:below=probe(down)
                    except (ValueError,DomainError,SolverError):below=None
                    if above is not None or below is not None:break
                    delta*=.5
                else:raise ValueError('No admissible physical Jacobian probe')
                matrix[:,column]=(above-below)/(2*delta) if above is not None and below is not None else (above-base)/delta if above is not None else (base-below)/delta
            fun(t,small)
            return matrix
        events=[]
        for original in (low,high,*trial._flow_guards(feed)):
            def event(t,small,fn=original):return fn(t,full(t,small))
            event.terminal=True;event.direction=getattr(original,'direction',-1);events.append(event)
        solution=solve_ivp(fun,(trial.time,end),trial.state[retained],method='Radau',rtol=trial.rtol,atol=atol[retained],
            max_step=trial.max_step_s,jac=jac,events=events,dense_output=True)
        if not solution.success:raise ValueError('Physical solver failed')
        if solution.t[-1]!=end or trial._stop_at_endpoint(solution):return self._resume_stop(trial,solution,full,end,duration_s,patch)
        # Accepted solver nodes and a separate dense-output sampling grid.
        for t in sorted(set(solution.t.tolist()+np.linspace(trial.time,end,max(2,math.ceil(duration_s/.05)+1)).tolist())):
            trial._admit(full(t,solution.sol(t)))
        trial.state=full(end,solution.y[:,-1]);trial.time=end;trial.last_reduction=reduced_modes;trial._rhs(end,trial.state)
        trial._admit(trial.state);observation=trial.snapshot()
        if observation['energy_relative']>1e-6 or observation['mass_relative']>1e-10 or observation['boron_relative']>1e-10:raise ValueError('Physical global inventory ledger failure')
        if motion['direction'] and trial.time>=motion['arrival_s']:
            trial.events.append(dict(time_s=motion['arrival_s'],kind='rod_arrived',position=motion['target'],reason='scram_stop' if trial.scram_latched else 'requested_position'))
        trial.history.append(dict(duration_s=duration_s,patch=deepcopy(patch)))
        self.__dict__.update(trial.__dict__);return self.snapshot()
    def snapshot(self):
        p=self.plant;n=self.base_size;c=self.config;l=self.ledger_index
        # _rhs restores all live thermodynamic ports to the committed state.
        rate=self._rhs(self.time,self.state);inventory=p.closed_inventories();stored=self._stored()
        defect=stored-self.origin_energy-self.state[l]
        scale=max(abs(self.origin_energy),float(self.state[l+3]),1.)
        q=float(p.network.junctions['steam'].mass_flow_kg_s)
        pin=float(p.network.volumes['sg_secondary'].pressure_pa);pout=float(p._condenser_pressure_pa)
        hin=float(p.h_g_current);hout=float(p.network.junctions['steam'].delivery_enthalpy_j_kg)
        bulk_h=float(p.hotwell_energy_j/p.hotwell_mass_kg+pout*p.condenser.volume_m3/p.hotwell_mass_kg)
        feed=p.network.junctions['feed'];dump=p.network.junctions['dump']
        condenser_mass_rate=q+dump.mass_flow_kg_s-feed.mass_flow_kg_s
        condenser_energy_rate=q*hout+dump.mass_flow_kg_s*dump.delivery_enthalpy_j_kg-feed.mass_flow_kg_s*p.network.boundaries['condenser'].enthalpy_j_kg-self.powers['condenser_heat_out_w']+self.powers['returned_losses_w']
        inlet=fluid_connection(FluidState(pin,hin,q,phase='vapor'),FluidState(pin,hin,-q,phase='vapor'))
        outlet=fluid_connection(FluidState(pout,hout,q,phase='mixture'),FluidState(pout,bulk_h,-q,phase='mixture'))
        return dict(model_id='npp-pwr-train-hot',time_s=self.time,pwr_state=self.state[:n].tolist(),
            turbine_energy_j=float(self.state[n]),turbine_speed_rad_s=math.sqrt(2*self.state[n]/c.inertia_kg_m2),
            clad_temperature_k=float(self.state[n+1]),fuel_to_clad_w=p.fuel_to_clad_w,clad_to_coolant_w=p.clad_to_coolant_w,
            primary_pump_speed_rad_s=p.omega,controls=deepcopy(self.controls),powers=deepcopy(self.powers),
            actuator_status=dict(rod_requested=self.controls['rod_target'],rod_position=float(self.state[12]),
                scram_latched=self.scram_latched,turbine_trip_latched=self.turbine_tripped,
                effective_generator_connected=self.controls['generator_connected'] and not self.turbine_tripped,
                reset_available=False,boron=boron_capability(),rod_drive=asdict(self.rod_drive)),
            events=deepcopy(self.events),effective_valve=p.effective_steam_throttle,steam_flow_kg_s=p.network.junctions['steam'].mass_flow_kg_s,
            condenser_balance=dict(mass_rate_kg_s=float(rate[n-3]),expected_mass_rate_kg_s=float(condenser_mass_rate),
                energy_rate_w=float(rate[n-2]),expected_energy_rate_w=float(condenser_energy_rate),
                mass_residual_kg_s=float(rate[n-3]-condenser_mass_rate),energy_residual_w=float(rate[n-2]-condenser_energy_rate),
                feed_out_kg_s=float(feed.mass_flow_kg_s),turbine_in_kg_s=q,bypass_in_kg_s=float(dump.mass_flow_kg_s)),
            minimum_sampled_dump_flow_kg_s=self.minimum_dump_sample_kg_s,
            dump_signed_roundoff_band_kg_s=1e-9,
            dump_flow_kg_s=p.network.junctions['dump'].mass_flow_kg_s,
            dump_target_kg_s=p.network.junctions['dump'].flow_target_kg_s,
            steam_generator_pressure_pa=p.network.volumes['sg_secondary'].pressure_pa,
            condenser_pressure_pa=p._condenser_pressure_pa,hotwell_mass_kg=p.hotwell_mass_kg,
            gross_generated_j=float(self.state[l+1]),own_needs_j=float(self.state[l+2]),net_exported_j=float(self.state[l+1]-self.state[l+2]),
            energy_residual_j=float(defect),energy_relative=float(abs(defect)/scale),
            energy_scale='max(abs(initial stored energy),gross external throughput,1 J)',
            mass_relative=abs(inventory['closed_mass_kg']-self.origin_mass)/max(abs(self.origin_mass),1.),
            boron_relative=abs(inventory['boron_kg']-self.origin_boron)/max(abs(self.origin_boron),1e-9),
            instantaneous_energy_defect_w=self.instantaneous_energy_defect_w,
            expansion=deepcopy(p.live_expansion),charging=deepcopy(self.charging),
            exact_mode_reduction=deepcopy(self.last_reduction),jacobian='Fixed relative probes with domain-admitted one-sided fallback',
            steam_port_audit=dict(inlet=inlet,outlet=outlet,enthalpy_drop_work_w=inlet['enthalpy_a_to_b_w']-outlet['enthalpy_a_to_b_w']),
            reactor_connected=True,steam_cycle_connected=True,complete_package_b=False,
            validated=False,plant_referenced=False)
    def checkpoint(self):
        return seal(deepcopy(dict(config=asdict(self.config),max_step_s=self.max_step_s,rtol=self.rtol,
            source=physical_signature(),history=self.history,snapshot=self.snapshot())))
    @classmethod
    def restore(cls,saved):
        p=unseal(deepcopy(saved))
        if p['source']!=physical_signature() or type(p['history']) is not list or len(p['history'])>256:raise ValueError('Physical restart source/budget')
        s=cls(TrainConfig(**p['config']),max_step_s=p['max_step_s'],rtol=p['rtol'])
        for row in p['history']:s.advance(**row)
        if s.checkpoint()!=saved:raise ValueError('Physical replay differs')
        return s
