"""Finite balanced primary chemical exchange for the reduced liquid PWR.

Separate gas-cushioned supply/receiver, linear reference-density liquid,
adiabatic gas, first-order flow actuator and explicit electrical pump heat.
Not a plant CVCS reconstruction or a validated chemical/solubility model.
"""
from copy import deepcopy
from dataclasses import asdict,dataclass
import hashlib
import math
from pathlib import Path

import numpy as np
from atomic_model_multiphysics.provenance import seal,unseal
from physics.materials.iapws import properties_pt,saturation_pressure_mpa
from .pwr_moderator import LocalModeratorPWRSession,identity as moderator_identity
from . import regenerator,boron_ports


def identity():
    return dict(moderator=moderator_identity(),adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        regenerator_sha256=hashlib.sha256(Path(regenerator.__file__).read_bytes()).hexdigest(),
        ports_sha256=hashlib.sha256(Path(boron_ports.__file__).read_bytes()).hexdigest())


@dataclass(frozen=True)
class BoronExchangeConfig:
    supply_mass_kg:float=5000.
    receiver_mass_kg:float=5000.
    supply_boron_ppm:float=0.
    receiver_boron_ppm:float=500.
    temperature_k:float=350.
    pressure_pa:float=2e5
    gas_volume_m3:float=5.
    minimum_mass_kg:float=20.
    maximum_flow_kg_s:float=50.
    actuator_tau_s:float=.2
    pump_efficiency:float=.85
    maximum_pump_head_pa:float=20e6
    regenerator_ua_w_k:float=0.
    require_liquid_ports:bool=False

    def __post_init__(self):
        if type(self.require_liquid_ports) is not bool:raise ValueError('Liquid-port policy must be boolean')
        bounds=dict(supply_mass_kg=(100.,1e5),receiver_mass_kg=(100.,1e5),supply_boron_ppm=(0.,10000.),
            receiver_boron_ppm=(0.,10000.),temperature_k=(300.,370.),pressure_pa=(1e5,1e6),gas_volume_m3=(.1,100.),
            minimum_mass_kg=(1.,1000.),maximum_flow_kg_s=(.1,100.),actuator_tau_s=(.05,5.),
            pump_efficiency=(.5,1.),maximum_pump_head_pa=(1e6,25e6),regenerator_ua_w_k=(0.,1e7))
        for key,(lo,hi) in bounds.items():
            v=getattr(self,key)
            if type(v) not in (int,float) or not math.isfinite(v) or not lo<=v<=hi:raise ValueError('Boron exchange parameter domain: '+key)
        if self.minimum_mass_kg>=min(self.supply_mass_kg,self.receiver_mass_kg):raise ValueError('Finite exchange tank reserve required')
        if self.pressure_pa<=saturation_pressure_mpa(self.temperature_k)*1e6+1000.:raise ValueError('Subcooled exchange tanks required')


def tank_state(inventory,*,initial_mass,gas_volume,pressure,rho,u_ref,temperature):
    m,u,b=map(float,inventory);gas=gas_volume+(initial_mass-m)/rho
    if not all(math.isfinite(v) for v in (m,u,b,gas)) or m<=0 or gas<=0 or not 0<=b<=.01*m:
        raise ValueError('Finite liquid/gas/boron tank inventory required')
    p=pressure*(gas_volume/gas)**1.4;t=temperature+(u/m-u_ref)/4180.
    if not math.isfinite(p) or not math.isfinite(t) or not 1e4<=p<=2e6 or not 300<=t<=400:
        raise ValueError('Linear liquid and gas-cushion tank domain')
    if p<=saturation_pressure_mpa(t)*1e6+1000.:raise ValueError('Exchange tank flashing/cavitation unsupported')
    return dict(mass_kg=m,water_energy_j=u,boron_kg=b,boron_fraction=b/m,temperature_k=t,pressure_pa=p,
                gas_volume_m3=gas,gas_energy_j=p*gas/.4,h_j_kg=u/m+p/rho)


class BoronExchangePWRSession(LocalModeratorPWRSession):
    """Equal supply/letdown flows; primary mass unchanged, all boron retained."""
    def __init__(self,*args,boron=None,**kwargs):
        self.boron_ready=False;self.boron_config=BoronExchangeConfig() if boron is None else boron
        if type(self.boron_config) is not BoronExchangeConfig:raise ValueError('BoronExchangeConfig required')
        super().__init__(*args,**kwargs)
        self._initialize_exchange()

    def _initialize_exchange(self):
        """Attach new finite tanks to an already validated, unadvanced base."""
        self.initial_moderator=super().checkpoint();self.boron_index=len(self.state);c=self.boron_config
        water=properties_pt(c.pressure_pa,c.temperature_k)
        self.boron_rho=float(water.rho_kg_m3);self.boron_u=float(water.u_j_kg)
        self.state=np.r_[self.state,c.supply_mass_kg,c.supply_mass_kg*self.boron_u,c.supply_mass_kg*c.supply_boron_ppm*1e-6,
            c.receiver_mass_kg,c.receiver_mass_kg*self.boron_u,c.receiver_mass_kg*c.receiver_boron_ppm*1e-6,0.,0.]
        self.controls['boron_flow_target_kg_s']=0.
        self.boron_ready=True
        self.boron_origin_mass=c.supply_mass_kg+c.receiver_mass_kg
        self.boron_origin_inventory=c.supply_mass_kg*c.supply_boron_ppm*1e-6+c.receiver_mass_kg*c.receiver_boron_ppm*1e-6
        self._rhs(0.,self.state);self.origin_energy=self._stored();self._admit(self.state)

    def _sync_fluid(self,y):
        return super()._sync_fluid(y[:self.boron_index] if self.boron_ready else y)

    def _tanks(self,y):
        c=self.boron_config
        return {name:tank_state(y[self.boron_index+i:self.boron_index+i+3],initial_mass=mass,
            gas_volume=c.gas_volume_m3,pressure=c.pressure_pa,rho=self.boron_rho,u_ref=self.boron_u,temperature=c.temperature_k)
            for name,i,mass in (('supply',0,c.supply_mass_kg),('receiver',3,c.receiver_mass_kg))}

    def _stored(self):
        value=super()._stored()
        if self.boron_ready:value+=sum(t['water_energy_j']+t['gas_energy_j'] for t in self._tanks(self.state).values())
        return float(value)

    def _rhs(self,t,y):
        if self.boron_ready and (len(y)!=self.boron_index+8 or not np.isfinite(y).all()):raise ValueError('Boron exchange state shape/finite domain')
        rate=super()._rhs(t,y)
        if not self.boron_ready:return rate
        tanks=self._tanks(y);s=tanks['supply'];r=tanks['receiver'];c=self.boron_config;p=self.plant
        cold=p.network.volumes['cold_leg'];q=float(y[self.boron_index+6])
        if not 0<=q<=c.maximum_flow_kg_s:raise ValueError('Nonnegative bounded exchange flow required')
        if q>0 and not self.controls['grid_available']:raise ValueError('Active boron pump requires available external grid in this configuration')
        head=cold.pressure_pa-s['pressure_pa']
        if not 0<head<=c.maximum_pump_head_pa or cold.pressure_pa<=r['pressure_pa']:
            raise ValueError('Exchange pump/letdown pressure domain')
        hydraulic=q*head/self.boron_rho;electric=hydraulic/c.pump_efficiency
        pump_specific=head/(self.boron_rho*c.pump_efficiency)
        # In the reference-density liquid closure, ideal pressure work raises
        # p/rho in h. Only pump inefficiency raises inlet sensible temperature.
        pump_temperature_rise=(pump_specific-head/self.boron_rho)/4180.
        regen=regenerator.counterflow(flow_kg_s=q,ua_w_k=c.regenerator_ua_w_k,hot_k=cold.temperature_k,
            cold_k=s['temperature_k']+pump_temperature_rise,hot_cp_j_kg_k=float(cold.thermo.cp_j_kg_k),cold_cp_j_kg_k=4180.)
        recovered=regen['heat_w']
        ports_eos=dict(enabled=False,active=q>0,checked=False,passed=False,states={},gates={})
        if c.require_liquid_ports:
            ports_eos=boron_ports.assess_liquid_ports(flow_kg_s=q,primary_pressure_pa=float(cold.pressure_pa),
                supply_pressure_pa=s['pressure_pa'],receiver_pressure_pa=r['pressure_pa'],
                supply_enthalpy_j_kg=s['h_j_kg'],pump_work_j_kg=pump_specific,
                donor_enthalpy_j_kg=float(cold.thermo.h_j_kg),recovered_heat_w=recovered)
            if not ports_eos['passed']:raise ValueError('Unsupported liquid exchange ports: '+','.join(k for k,v in ports_eos['gates'].items() if not v))
        energy=q*(s['h_j_kg']-cold.thermo.h_j_kg)+electric+recovered
        boron=q*(s['boron_fraction']-cold.boron_fraction)
        index=13+3*p.network.volume_names().index('cold_leg')
        rate[index+1]+=energy;rate[index+2]+=boron
        dm=np.array([-q,q]);enthalpy=np.array([-q*s['h_j_kg'],q*cold.thermo.h_j_kg-recovered])
        water=enthalpy-np.array([s['pressure_pa'],r['pressure_pa']])*dm/self.boron_rho
        db=np.array([-q*s['boron_fraction'],q*cold.boron_fraction])
        dq=(self.controls['boron_flow_target_kg_s']-q)/c.actuator_tau_s
        before=self.powers['net_export_w'];rate[self.ledger_index]+=electric;rate[self.ledger_index+2]+=electric
        self.powers['own_needs_w']+=electric;self.powers['net_export_w']-=electric
        self.powers['external_net_w']=float(rate[self.ledger_index]);self.powers['boron_pump_electric_w']=electric
        rate[self.ledger_index+3]+=abs(self.powers['net_export_w'])-abs(before)
        residual=energy+float(enthalpy.sum())-electric
        self.instantaneous_energy_defect_w+=residual
        self.boron_terms=dict(tanks=deepcopy(tanks),flow_kg_s=q,target_kg_s=self.controls['boron_flow_target_kg_s'],
            hydraulic_power_w=hydraulic,electrical_power_w=electric,pump_heat_loss_w=electric-hydraulic,
            primary_energy_rate_w=energy,primary_boron_rate_kg_s=boron,
            tank_mass_rates_kg_s=dm.tolist(),tank_water_energy_rates_w=water.tolist(),tank_boron_rates_kg_s=db.tolist(),
            instantaneous_energy_residual_w=residual,instantaneous_boron_residual_kg_s=float(boron+db.sum()),
            integrated_pump_electric_j=float(y[self.boron_index+7]),
            regenerator=regen,
            liquid_ports=ports_eos,
            exchange_ports=dict(primary_donor_enthalpy_j_kg=float(cold.thermo.h_j_kg),
                supply_enthalpy_j_kg=s['h_j_kg'],pump_specific_work_j_kg=pump_specific,
                pump_head_pa=head,pump_inlet_density_kg_m3=self.boron_rho,
                pump_efficiency=c.pump_efficiency,pump_sensible_temperature_rise_k=pump_temperature_rise),
            liquid_closure='Constant reference density, cp4180; adiabatic gamma1.4 gas cushions',
            pressure_scope='Flow-controlled pump and letdown; no line inertia or valve sizing model',validated=False)
        return np.r_[rate,-q,water[0],db[0],q,water[1],db[1],dq,electric]

    def _extend_known(self,known,modes):
        super()._extend_known(known,modes)
        if self.boron_ready:
            q=float(self.state[self.boron_index+6]);target=self.controls['boron_flow_target_kg_s'];origin=self.time;tau=self.boron_config.actuator_tau_s
            known[self.boron_index+6]=lambda t:q if t==origin else target+(q-target)*math.exp(-(t-origin)/tau)
            modes.append(dict(name='boron_flow_actuator',kind='exact_first_order',initial=q,target=target,tau_s=tau))

    def _absolute_tolerances(self,atol):
        atol=super()._absolute_tolerances(atol)
        if self.boron_ready:atol[self.boron_index:]=[1e-7,.001,1e-10,1e-7,.001,1e-10,1e-7,.001]
        return atol

    def _set_controls(self,patch):
        if type(patch) is not dict or not patch:raise ValueError('Nonempty exchange command dictionary required')
        if self.boron_ready and not patch.get('grid_available',self.controls['grid_available']) and (self.state[self.boron_index+6]>0 or patch.get('boron_flow_target_kg_s',self.controls['boron_flow_target_kg_s'])!=0):
            raise ValueError('Active boron pump requires available external grid in this configuration')
        if 'boron_flow_target_kg_s' not in patch:return super()._set_controls(patch)
        value=patch['boron_flow_target_kg_s']
        if type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<=self.boron_config.maximum_flow_kg_s:raise ValueError('Boron flow command domain')
        other={k:v for k,v in patch.items() if k!='boron_flow_target_kg_s'}
        if other:super()._set_controls(other)
        self.controls['boron_flow_target_kg_s']=float(value)

    def _passive_column(self,index):
        return super()._passive_column(index) or (self.boron_ready and index==self.boron_index+7)

    def parameter_registry(self):
        result=super().parameter_registry()
        if self.boron_ready:
            units=dict(supply_mass_kg='kg',receiver_mass_kg='kg',supply_boron_ppm='ppm',receiver_boron_ppm='ppm',
                temperature_k='K',pressure_pa='Pa',gas_volume_m3='m3',minimum_mass_kg='kg',maximum_flow_kg_s='kg/s',
                actuator_tau_s='s',pump_efficiency='1',maximum_pump_head_pa='Pa',regenerator_ua_w_k='W/K')
            for key,value in asdict(self.boron_config).items():
                if key=='require_liquid_ports':continue
                result['parameters']['boron_exchange.'+key]=dict(value=float(value),unit=units[key],family='illustrative',
                    source='Declared finite balanced exchange component; not plant CVCS data',version='1',uncertainty=None)
            for key,value,unit in (('reference_density',self.boron_rho,'kg/m3'),('liquid_cp',4180.,'J/kg/K'),('gas_gamma',1.4,'1')):
                result['parameters']['boron_exchange.'+key]=dict(value=value,unit=unit,family='illustrative',
                    source='Reference-liquid/adiabatic-gas closure; no experimental calibration',version='1',uncertainty=None)
        return result

    def exchange_policy(self):
        return dict(require_liquid_ports=self.boron_config.require_liquid_ports,minimum_subcooling_k=1.)

    def _admit(self,y):
        super()._admit(y)
        if self.boron_ready:
            tanks=self._tanks(y)
            if any(t['mass_kg']<self.boron_config.minimum_mass_kg for t in tanks.values()):raise ValueError('Finite exchange tank reserve exhausted')
            if not 0<=y[self.boron_index+6]<=self.boron_config.maximum_flow_kg_s or y[self.boron_index+7]<0:raise ValueError('Exchange actuator/energy domain')

    def snapshot(self):
        result=super().snapshot()
        if self.boron_ready:
            tanks=self._tanks(self.state);terms=deepcopy(self.boron_terms)
            total_boron=self.plant.closed_inventories()['boron_kg']+sum(t['boron_kg'] for t in tanks.values())
            total_mass=self.feed_terms['total_mass_kg']+sum(t['mass_kg'] for t in tanks.values())
            terms.update(total_boron_kg=total_boron,total_mass_kg=total_mass)
            result.update(model_id='pwr-finite-boron-exchange/v1',finite_boron_exchange=terms,boron_exchange_source=identity(),
                boron_exchange_policy=self.exchange_policy(),
                boron_relative=abs(total_boron-self.origin_boron-self.boron_origin_inventory)/max(abs(self.origin_boron+self.boron_origin_inventory),1e-9),
                mass_relative=abs(total_mass-self.origin_mass-self.origin_feed_mass-self.boron_origin_mass)/max(abs(total_mass),1.))
            result['actuator_status']['boron']=dict(available=True,command='boron_flow_target_kg_s',
                direct_inventory_assignment_allowed=False,finite_supply=True,balanced_primary_mass=True)
        return result

    def checkpoint(self):
        if not self.boron_ready:return super().checkpoint()
        return seal(dict(format='pwr-finite-boron/v1',source=identity(),initial_moderator=deepcopy(self.initial_moderator),
            config=asdict(self.boron_config),history=list(deepcopy(self.history)),snapshot=self.snapshot()))

    @classmethod
    def restore(cls,saved):
        p=unseal(deepcopy(saved))
        if type(p) is not dict or set(p)!={'format','source','initial_moderator','config','history','snapshot'} or p['format']!='pwr-finite-boron/v1' or p['source']!=identity():raise ValueError('Finite boron checkpoint source/format')
        if type(p['history']) is not list or len(p['history'])>256:raise ValueError('Boron replay history budget')
        seed=LocalModeratorPWRSession.restore(p['initial_moderator'])
        if seed.time!=0 or seed.history:raise ValueError('Unadvanced boron initialization required')
        # The base restore above has already recomputed and compared its full
        # stationary state. Copy that verified object; do not solve it twice.
        obj=cls.__new__(cls);obj.__dict__.update(deepcopy(seed.__dict__))
        obj.boron_ready=False;obj.boron_config=BoronExchangeConfig(**p['config'])
        obj._initialize_exchange()
        for step in p['history']:
            if type(step) is not dict or set(step)!={'duration_s','patch'}:raise ValueError('Boron replay step')
            obj.advance(**step)
        if obj.checkpoint()!=saved:raise ValueError('Finite boron replay differs')
        return obj
