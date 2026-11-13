"""Finite mixed steam header with explicit rerouting of legacy fluid edges.

Manual aggregate turbine admission, bounded small-drop inlet valve, IF97
M/U inversion and actual inlet enthalpy. No pipe waves or manufacturer maps.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
import math

import numpy as np
from physics.materials import iapws
from atomic_model_plant_extension.volume import StrictVolume
from atomic_model_multiphysics.provenance import seal, unseal, sha
from .film import FilmPWR, FilmConfig, PWRFilmSession, film_signature
from .primary import PrimaryConfig
from .core import CoreConfig
from .power import BackupConfig
from .feed import FeedConfig
from .cooling import CoolingConfig
from .pwr_train import TrainConfig
from .contracts import FluidState, fluid_connection


def steam_signature():
    return dict(film=film_signature(), material_sha256=sha(iapws.__file__))


@dataclass(frozen=True)
class SteamConfig:
    volume_m3: float = 50.
    nominal_pressure_loss_fraction: float = .005
    maximum_pressure_loss_fraction: float = .05
    valve_stroke_time_s: float = .5
    heat_loss_ua_w_k: float = 0.
    minimum_quality: float = .99
    minimum_mass_kg: float = 1.

    def __post_init__(self):
        for key, low, high in [('volume_m3',10.,200.),('nominal_pressure_loss_fraction',.001,.01),
                              ('maximum_pressure_loss_fraction',.02,.1),('valve_stroke_time_s',.1,5.),
                              ('heat_loss_ua_w_k',0.,1e4),
                              ('minimum_quality',.95,1.),('minimum_mass_kg',.1,10.)]:
            value=getattr(self,key)
            if type(value) not in (int,float) or not math.isfinite(value) or not low<=value<=high:
                raise ValueError('Steam parameter domain: '+key)


def actual_expansion(pin, hin, pout, efficiency):
    """Actual near-dry inlet, saturated-dome isentrope, constant aggregate eta."""
    if any(not math.isfinite(x) for x in (pin,hin,pout,efficiency)) or not 0<pout<pin or not 0<efficiency<=1:
        raise ValueError('Forward finite steam expansion domain')
    inlet=iapws.properties_ph(pin,hin)
    if inlet.region not in (2,4) or inlet.quality<.95:
        raise ValueError('Near-dry steam turbine inlet required')
    liquid,vapor=iapws.saturation_properties(pout)
    x=(inlet.s_j_kg_k-liquid.s_j_kg_k)/(vapor.s_j_kg_k-liquid.s_j_kg_k)
    if not 0<=x<=1:raise ValueError('Isentropic reference outside supported saturation dome')
    his=liquid.h_j_kg+x*(vapor.h_j_kg-liquid.h_j_kg)
    work=efficiency*(hin-his)
    outlet=iapws.properties_ph(pout,hin-work)
    if work<=0 or outlet.s_j_kg_k<inlet.s_j_kg_k-1e-7:
        raise ValueError('Expansion work/entropy domain')
    return dict(inlet_enthalpy_j_kg=float(hin),outlet_enthalpy_j_kg=float(outlet.h_j_kg),
        isentropic_outlet_enthalpy_j_kg=float(his),isentropic_quality=float(x),
        inlet_entropy_j_kg_k=float(inlet.s_j_kg_k),outlet_entropy_j_kg_k=float(outlet.s_j_kg_k),
        inlet_temperature_k=float(inlet.T_k),outlet_temperature_k=float(outlet.T_k),
        inlet_quality=float(inlet.quality),outlet_quality=float(outlet.quality),work_j_kg=float(work))


class HeaderPWR(FilmPWR):
    def _apply_heat_and_actuators(self):
        super()._apply_heat_and_actuators()
        if not hasattr(self,'steam_header_pressure_pa'):return
        upstream=self.network.volumes['sg_secondary'].pressure_pa
        density=iapws.saturation_properties(upstream)[1].rho_kg_m3
        # One-way equilibrium valve law. A reverse/large pressure contrast is
        # refused on accepted states, not promoted to an unqualified gas map.
        difference=max(upstream-self.steam_header_pressure_pa,0.)
        self.network.junctions['steam'].flow_target_kg_s=(self.steam_inlet_coefficient
            *self.steam_inlet_position*math.sqrt(density*difference))


class PWRSteamSession(PWRFilmSession):
    def _make_plant(self):return HeaderPWR(height_m=self.primary_config.height_m)
    def _closed_steam_mode(self):return False  # Header owns both closure modes.

    def __init__(self,config=None,*,cooling=None,feed=None,backup=None,core=None,
                 primary=None,film=None,steam=None,max_step_s=.1,rtol=1e-7):
        self.steam_ready=False
        self.steam_config=steam if steam is not None else SteamConfig()
        if type(self.steam_config) is not SteamConfig:raise ValueError('SteamConfig required')
        super().__init__(config,cooling=cooling,feed=feed,backup=backup,core=core,
                         primary=primary,film=film,max_step_s=max_step_s,rtol=rtol)
        p=self.plant;c=self.steam_config
        source=p.network.volumes['sg_secondary'].pressure_pa
        pressure=source*(1-c.nominal_pressure_loss_fraction)
        point=iapws.properties_ph(pressure,p.h_g_current)
        self.header=StrictVolume('steam_header',c.volume_m3,0.)
        self.header.pressure_pa=float(pressure);self.header.temperature_k=float(point.T_k)
        self.header.mass_kg=float(c.volume_m3*point.rho_kg_m3)
        self.header.internal_energy_j=float(self.header.mass_kg*point.u_j_kg)
        self.header.sync()
        q=float(p.network.junctions['steam'].mass_flow_kg_s)
        density=iapws.saturation_properties(source)[1].rho_kg_m3
        self.steam_inlet_coefficient=q/math.sqrt(density*(source-pressure))
        self.steam_outlet_coefficient=q/pressure
        self.steam_design=dict(inlet_coefficient=self.steam_inlet_coefficient,outlet_coefficient=self.steam_outlet_coefficient,
            initial_source_pressure_pa=float(source),initial_header_pressure_pa=float(pressure),initial_flow_kg_s=q,
            source='Constructed single nominal point, not measured valve/turbine map',
            inherited_sizing='Rotor drag and condenser/cooling sizing retain pre-header SG reference; new header settles dynamically')
        self.steam_index=len(self.state)
        self.state=np.r_[self.state,self.header.mass_kg,self.header.internal_energy_j,q,1.,1.]
        self.origin_header_mass=self.header.mass_kg
        self.controls.update(steam_inlet_isolated=False,steam_admission=1.,governor_enabled=False)
        self.steam_ready=True
        self.state=self._reconstruct_state(self.state)
        self._rhs(0.,self.state);self.origin_energy=self._stored();self._admit(self.state)

    def _sync_header(self,y):
        i=self.steam_index
        self.header.mass_kg=float(y[i]);self.header.internal_energy_j=float(y[i+1])
        point=self.header.sync()
        self.plant.steam_header_pressure_pa=float(point.p_pa)
        self.plant.steam_inlet_position=float(y[i+3])
        self.plant.steam_inlet_coefficient=self.steam_inlet_coefficient
        return point

    def _reconstruct_state(self,y):
        y=super()._reconstruct_state(y)
        if not self.steam_ready:return y
        point=self._sync_header(y)
        node=self.plant.network.volumes['sg_secondary']
        j=13+3*self.plant.network.volume_names().index('sg_secondary')
        node.mass_kg=float(y[j]);node.internal_energy_j=float(y[j+1]);node.sync()
        density=iapws.saturation_properties(node.pressure_pa)[1].rho_kg_m3
        offset=13+3*len(self.plant.network.volumes)+self.plant.network.junction_names().index('steam')
        y[offset]=self.steam_inlet_coefficient*float(y[self.steam_index+3])*math.sqrt(density*max(node.pressure_pa-point.p_pa,0.))
        y[self.steam_index+2]=self.steam_outlet_coefficient*point.p_pa*float(y[self.steam_index+4])
        return y

    def _stored(self):
        total=super()._stored()
        return float(total+self.state[self.steam_index+1]) if self.steam_ready else total

    def _absolute_tolerances(self,atol):
        atol=super()._absolute_tolerances(atol)
        if self.steam_ready:atol[self.steam_index:self.steam_index+5]=[1e-7,.01,1e-8,1e-10,1e-10]
        return atol

    def _valve_targets(self):
        trip=self.scram_latched or self.turbine_tripped
        return (0. if trip or self.controls['steam_inlet_isolated'] else 1.,
                0. if trip else self.controls['steam_admission']*self.controls['valve_request'])

    def _rhs(self,t,y):
        if not self.steam_ready:return super()._rhs(t,y)
        y=self._reconstruct_state(np.array(y,dtype=float,copy=True))
        point=self._sync_header(y)
        rate=super()._rhs(t,y);p=self.plant;c=self.steam_config;i=self.steam_index
        qs=float(p.network.junctions['steam'].mass_flow_kg_s);qt=float(y[i+2])
        if qs<0 or qt<0:raise ValueError('Reverse steam flow outside header scope')
        inlet_h=float(p.h_g_current);old_out=float(p.network.junctions['steam'].delivery_enthalpy_j_kg)
        expansion=actual_expansion(float(point.p_pa),float(point.h_j_kg),float(p._condenser_pressure_pa),p.eta_turbine)
        out_h=expansion['outlet_enthalpy_j_kg'];work=qt*expansion['work_j_kg'];old_work=qs*(inlet_h-old_out)
        cold=float(y[self.cooling_index]/(self.cooling_config.cold_mass_kg*self.cooling_config.heat_capacity_j_kg_k))
        loss=c.heat_loss_ua_w_k*(point.T_k-cold)
        dm=qs-qt;du=qs*inlet_h-qt*point.h_j_kg-loss
        mass_correction=qt-qs;energy_correction=qt*out_h-qs*old_out;work_correction=work-old_work
        rate[self.base_size-3]+=mass_correction
        rate[self.base_size-2]+=energy_correction
        rate[self.base_size]+=work_correction
        rate[self.cooling_index]+=loss
        self.feed_terms['hotwell_mass_rate_kg_s']+=mass_correction
        self.feed_terms['hotwell_energy_rate_w']+=energy_correction
        self.cooling_terms['cold_energy_rate_w']+=loss
        self.cooling_terms['header_received_w']=float(loss)
        residual=du+energy_correction+work_correction+loss
        self.instantaneous_energy_defect_w+=residual
        self.powers.update(steam_work_w=float(work),steam_header_heat_to_cooling_w=float(loss))
        p.live_expansion=expansion
        target=self.steam_outlet_coefficient*point.p_pa*float(y[i+4])
        dq=0.  # Derived outlet-flow alias, no fictitious fluid momentum stock.
        targets=self._valve_targets()
        valve_rates=[(1. if target>y[i+3+j] else -1. if target<y[i+3+j] else 0.)/c.valve_stroke_time_s
                     for j,target in enumerate(targets)]
        upstream=p.network.volumes['sg_secondary'].pressure_pa
        upstream_s=iapws.saturation_properties(upstream)[1].s_j_kg_k
        after_valve=iapws.properties_ph(point.p_pa,inlet_h)
        valve_entropy=qs*(after_valve.s_j_kg_k-upstream_s)
        mix_entropy=qs*((inlet_h-point.h_j_kg)/point.T_k+point.s_j_kg_k-after_valve.s_j_kg_k)
        self.steam_terms=dict(pressure_pa=float(point.p_pa),temperature_k=float(point.T_k),quality=float(point.quality),region=int(point.region),
            thermo_energy_residual_j_kg=float(self.header.energy_mismatch_j_kg),thermo_volume_relative_residual=float(self.header.volume_mismatch/point.v_m3_kg),
            mass_kg=float(y[i]),energy_j=float(y[i+1]),liquid_mass_kg=float(y[i]*(1-point.quality)),vapor_mass_kg=float(y[i]*point.quality),
            inlet_flow_kg_s=qs,turbine_flow_kg_s=qt,turbine_flow_target_kg_s=float(target),mass_rate_kg_s=float(dm),energy_rate_w=float(du),
            inlet_enthalpy_j_kg=inlet_h,turbine_inlet_enthalpy_j_kg=float(point.h_j_kg),outlet_enthalpy_j_kg=out_h,
            inlet_position=float(y[i+3]),outlet_position=float(y[i+4]),inlet_target=targets[0],outlet_target=targets[1],
            outlet_pressure_ratio=float(p._condenser_pressure_pa/point.p_pa),
            source_pressure_pa=float(upstream),pressure_loss_fraction=float((upstream-point.p_pa)/upstream),
            heat_to_cooling_w=float(loss),heat_entropy_w_k=float(loss*(1/cold-1/point.T_k)),
            inlet_valve_entropy_w_k=float(valve_entropy),mixing_entropy_w_k=float(mix_entropy),
            routing_energy_residual_w=float(residual),routing_mass_residual_kg_s=float(dm+mass_correction),
            expansion_entropy_w_k=float(qt*(expansion['outlet_entropy_j_kg_k']-expansion['inlet_entropy_j_kg_k'])),
            governing='Manual valve positions; no automatic governor or hardware steam-flow map',
            flow_model='Algebraic one-way small-drop inlet and linear aggregate outlet; finite valve positions and header M/U, no pipe inertia',
            state_ownership='Header M/U and valve positions owned; SG and turbine flows are derived aliases')
        return np.r_[rate,dm,du,dq,valve_rates]

    def _admit(self,y):
        if self.steam_ready:
            canonical=self._reconstruct_state(np.array(y,dtype=float,copy=True))
            offset=13+3*len(self.plant.network.volumes)+self.plant.network.junction_names().index('steam')
            for index in (offset,self.steam_index+2):
                if abs(float(y[index]-canonical[index]))>1e-10*max(abs(float(canonical[index])),1.):
                    raise ValueError('Steam flow alias differs from owned inventories/positions')
            self._sync_header(y)
        super()._admit(y)
        if not self.steam_ready:return
        p=self.header.thermo;c=self.steam_config;i=self.steam_index
        if abs(self.header.energy_mismatch_j_kg)>.01 or abs(self.header.volume_mismatch)/p.v_m3_kg>1e-8:
            raise ValueError('Header conserved EOS residual')
        if y[i]<c.minimum_mass_kg or p.region not in (2,4) or p.quality<c.minimum_quality:
            raise ValueError('Header mass/near-dry phase limit')
        if not 5e6<=p.p_pa<=8e6 or not 0<=y[i+3]<=1 or not 0<=y[i+4]<=1 or y[i+2]<0:
            raise ValueError('Header pressure/valve/flow operating limit: '+repr((p.p_pa,float(y[i+3]),float(y[i+4]),float(y[i+2]))))
        if self.plant._condenser_pressure_pa/p.p_pa>.01:
            raise ValueError('Linear steam admission requires low-backpressure envelope')
        source=self.plant.network.volumes['sg_secondary'].pressure_pa
        if self.plant.network.junctions['steam'].mass_flow_kg_s>0 and not 0<=(source-p.p_pa)/source<=c.maximum_pressure_loss_fraction:
            raise ValueError('Inlet valve small forward pressure-drop scope')

    def _extend_known(self,known,modes):
        super()._extend_known(known,modes)
        if not self.steam_ready:return
        self.steam_arrivals=[];origin=self.time;c=self.steam_config;i=self.steam_index
        targets=self._valve_targets()
        for j,target in enumerate(targets):
            index=i+3+j;initial=float(self.state[index]);sign=1. if target>initial else -1. if target<initial else 0.
            arrival=origin+abs(target-initial)*c.valve_stroke_time_s
            known[index]=lambda t,a=arrival,v=target,x=initial,d=sign: v if t>=a else x+d*(t-origin)/c.valve_stroke_time_s
            modes.append(dict(name='steam_valve_'+str(j),kind='exact_bounded_motion',initial=initial,target=target,arrival_s=arrival))
            if sign:self.steam_arrivals.append((arrival,index,target))
        offset=13+3*len(self.plant.network.volumes)+self.plant.network.junction_names().index('steam')
        known[offset]=lambda t:0.
        known[i+2]=lambda t:0.
        modes.extend([dict(name='header_inlet_flow',kind='derived_equilibrium_flux'),
                      dict(name='header_outlet_flow',kind='derived_equilibrium_flux')])

    def _flow_guards(self,feed):
        guards=super()._flow_guards(feed)
        if self.steam_ready and self.steam_arrivals:
            arrival=min(row[0] for row in self.steam_arrivals)
            def event(t,y):return arrival-t
            event.direction=-1;guards.append(event);self.guard_labels.append('steam_valve_arrival')
        return guards

    def _custom_root(self,label,endpoint,now):
        if label!='steam_valve_arrival':return super()._custom_root(label,endpoint,now)
        for arrival,index,target in self.steam_arrivals:
            if abs(now-arrival)<=1e-10:
                residual=float(endpoint[index]-target)
                if abs(residual)>1e-10:raise ValueError('Steam valve analytical arrival residual')
                endpoint[index]=target
                self.events.append(dict(time_s=now,kind='steam_valve_arrived',index=index,target=target,position_roundoff=residual))
        return True

    def _set_controls(self,patch):
        if type(patch) is not dict or not patch or not set(patch)<=set(self.controls):raise ValueError('Steam command patch')
        if patch.get('governor_enabled',False) is not False:raise ValueError('Header session uses manual admission; automatic governor unavailable')
        if 'steam_inlet_isolated' in patch and type(patch['steam_inlet_isolated']) is not bool:raise ValueError('Boolean steam isolation')
        if 'steam_admission' in patch:
            value=patch['steam_admission']
            if type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<=1:raise ValueError('Steam admission0..1')
        regular={k:v for k,v in patch.items() if k not in ('steam_inlet_isolated','steam_admission')}
        if regular:super()._set_controls(regular)
        self.controls.update({k:v for k,v in patch.items() if k in ('steam_inlet_isolated','steam_admission')})
        self.events.append(dict(time_s=self.time,kind='steam_command',requested=deepcopy(patch)))

    def parameter_registry(self):
        result=super().parameter_registry()
        if self.steam_ready:
            result['parameters']['design.nominal_turbine_work_w']['source']='Inherited pre-header SG reference for rotor/condenser sizing; actual work uses header ports'
            for key,value in asdict(self.steam_config).items():
                unit='m^3' if key=='volume_m3' else 's' if key.endswith('_s') else 'kg' if key.endswith('_kg') else 'W/K' if key=='heat_loss_ua_w_k' else '1'
                result['parameters']['steam.'+key]=dict(value=float(value),unit=unit,source='Declared lumped header/valve research parameters; no hardware map',family='illustrative',version='1',uncertainty=None)
            for key,unit in [('inlet_coefficient','kg/(s*sqrt(kg/m^3*Pa))'),('outlet_coefficient','kg/(s*Pa)')]:
                result['parameters']['steam.design.'+key]=dict(value=float(self.steam_design[key]),unit=unit,source=self.steam_design['source'],family='illustrative',version='1',uncertainty=None)
        return result

    def snapshot(self):
        result=super().snapshot()
        if not self.steam_ready:return result
        rate=self._rhs(self.time,self.state);i=self.steam_index;p=self.plant;terms=self.steam_terms
        total=p.closed_inventories()['closed_mass_kg']+sum(self._tank(self.state,n)['mass_kg'] for n in self.feed_tanks)+self.state[i]
        origin=self.origin_mass+self.origin_feed_mass+self.origin_header_mass
        qt=terms['turbine_flow_kg_s'];dump=p.network.junctions['dump'];feed=self.feed_terms
        expected_mass=qt+dump.mass_flow_kg_s-feed['condensate_flow_kg_s']
        expected_energy=qt*terms['outlet_enthalpy_j_kg']+dump.mass_flow_kg_s*dump.delivery_enthalpy_j_kg-feed['condensate_flow_kg_s']*feed['hotwell_liquid_h_j_kg']-self.powers['condenser_heat_out_w']+self.powers['returned_losses_w']
        result.update(model_id='npp-pwr-finite-steam-header',steam_header=deepcopy(terms),steam_config=asdict(self.steam_config),
            header_inventory=self.state[i:i+5].tolist(),steam_design=deepcopy(self.steam_design),steam_flow_kg_s=qt,sg_steam_flow_kg_s=terms['inlet_flow_kg_s'],
            powers=deepcopy(self.powers),expansion=deepcopy(p.live_expansion),mass_relative=float(abs(total-origin)/origin),
            finite_steam_header_connected=True,effective_valve=terms['outlet_position'],
            condenser_balance=dict(mass_rate_kg_s=float(rate[self.base_size-3]),expected_mass_rate_kg_s=float(expected_mass),
                energy_rate_w=float(rate[self.base_size-2]),expected_energy_rate_w=float(expected_energy),
                mass_residual_kg_s=float(rate[self.base_size-3]-expected_mass),energy_residual_w=float(rate[self.base_size-2]-expected_energy),
                condensate_out_kg_s=feed['condensate_flow_kg_s']))
        result['actuator_status'].update(steam_inlet_position=terms['inlet_position'],steam_outlet_position=terms['outlet_position'],governor_available=False)
        inlet=fluid_connection(FluidState(terms['pressure_pa'],terms['turbine_inlet_enthalpy_j_kg'],qt,phase='mixture'),
                              FluidState(terms['pressure_pa'],terms['turbine_inlet_enthalpy_j_kg'],-qt,phase='mixture'))
        # Both audit enthalpies describe actual turbine ports, not the SG tap.
        outlet=fluid_connection(FluidState(float(p._condenser_pressure_pa),terms['outlet_enthalpy_j_kg'],qt,phase='mixture'),
                               FluidState(float(p._condenser_pressure_pa),terms['outlet_enthalpy_j_kg'],-qt,phase='mixture'))
        result['steam_port_audit']=dict(inlet=inlet,outlet=outlet,enthalpy_drop_work_w=inlet['enthalpy_a_to_b_w']-outlet['enthalpy_a_to_b_w'])
        return result

    def checkpoint(self):
        payload=unseal(super().checkpoint());payload.update(steam_config=asdict(self.steam_config),steam_source=steam_signature());return seal(payload)

    @classmethod
    def restore(cls,saved):
        p=unseal(deepcopy(saved))
        if p['steam_source']!=steam_signature() or type(p['history']) is not list or len(p['history'])>256:raise ValueError('Steam replay source/budget')
        s=cls(TrainConfig(**p['config']),cooling=CoolingConfig(**p['cooling_config']),feed=FeedConfig(**p['feed_config']),backup=BackupConfig(**p['backup_config']),
            core=CoreConfig(**p['core_config']),primary=PrimaryConfig(**p['primary_config']),film=FilmConfig(**p['film_config']),steam=SteamConfig(**p['steam_config']),max_step_s=p['max_step_s'],rtol=p['rtol'])
        for row in p['history']:s.advance(**row)
        if s.checkpoint()!=saved:raise ValueError('Steam replay differs')
        return s
