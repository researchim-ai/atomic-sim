"""Finite-header bypass with an IHEM nozzle and hybrid valve interlocks.

No-slip equilibrium critical flow, not a manufacturer safety-valve map.
Recovered jet stagnation enthalpy enters the existing finite condenser.
"""
from copy import deepcopy
from dataclasses import asdict,dataclass
from functools import lru_cache
import math

import numpy as np
from scipy.optimize import brentq,minimize_scalar
from physics.materials import iapws
from atomic_model_multiphysics.provenance import seal,unseal
from .stages import PWRStagedSession,StageConfig,stages_signature
from .steam import SteamConfig
from .film import FilmConfig
from .primary import PrimaryConfig
from .core import CoreConfig
from .power import BackupConfig
from .feed import FeedConfig
from .cooling import CoolingConfig
from .pwr_train import TrainConfig
from .contracts import FluidState,fluid_connection

NOZZLE_SOURCE='https://lwrs.inl.gov/content/uploads/11/2024/03/Development_of_RELAP5-3D-Modeling_RCIC_System.pdf'
NOZZLE_VERSION='iheM-equilibrium-enthalpy-maximum/v1'


def bypass_signature():return dict(stages=stages_signature())


def isentropic_point(pressure,entropy):
    """Equilibrium mixture or superheated branch, bounded IF97 entropy inverse."""
    liquid,vapor=iapws.saturation_properties(pressure)
    if liquid.s_j_kg_k<=entropy<=vapor.s_j_kg_k:
        x=(entropy-liquid.s_j_kg_k)/(vapor.s_j_kg_k-liquid.s_j_kg_k)
        h=(1-x)*liquid.h_j_kg+x*vapor.h_j_kg
        rho=1/((1-x)/liquid.rho_kg_m3+x/vapor.rho_kg_m3)
        return float(h),float(rho),float(x)
    if entropy<vapor.s_j_kg_k:raise ValueError('Nozzle liquid isentrope outside near-dry inlet scope')
    maximum=iapws.properties_pt(pressure,900.)
    if entropy>maximum.s_j_kg_k:raise ValueError('Nozzle entropy inverse outside900K')
    h=brentq(lambda value:iapws.properties_ph(pressure,value).s_j_kg_k-entropy,
             vapor.h_j_kg,maximum.h_j_kg,xtol=1e-6)
    point=iapws.properties_ph(pressure,h)
    return float(h),float(point.rho_kg_m3),float(point.quality)


def isentropic_flux(pressure,enthalpy,entropy,throat_pressure):
    if throat_pressure==pressure:return dict(mass_flux_kg_m2_s=0.,velocity_m_s=0.,enthalpy_j_kg=float(enthalpy),quality=None,density_kg_m3=None)
    if not .2*pressure<=throat_pressure<pressure:raise ValueError('Nozzle pressure-search domain')
    h,rho,x=isentropic_point(throat_pressure,entropy)
    drop=enthalpy-h
    if drop<=0:raise ValueError('Nonpositive nozzle kinetic enthalpy')
    speed=math.sqrt(2*drop)
    return dict(mass_flux_kg_m2_s=float(rho*speed),velocity_m_s=float(speed),enthalpy_j_kg=h,quality=x,density_kg_m3=rho)


@lru_cache(maxsize=2048)
def critical_nozzle(pressure,enthalpy):
    point=iapws.properties_ph(pressure,enthalpy)
    if point.region not in (2,4) or point.quality<.99:raise ValueError('Near-dry equilibrium nozzle stagnation state')
    entropy=float(point.s_j_kg_k)
    def flux(ratio):return isentropic_flux(pressure,enthalpy,entropy,pressure*ratio)['mass_flux_kg_m2_s']
    optimum=minimize_scalar(lambda ratio:-flux(ratio),bounds=(.2,.9999999),method='bounded',options={'xatol':1e-9})
    if not optimum.success or not .201<optimum.x<.998:raise ValueError('Nozzle maximum not interior to declared search box')
    throat=isentropic_flux(pressure,enthalpy,entropy,pressure*float(optimum.x))
    if throat['mass_flux_kg_m2_s']<max(flux(.2),flux(.9999)):raise ValueError('Nozzle maximum below search endpoints')
    return dict(throat,critical_pressure_pa=float(pressure*optimum.x),stagnation_entropy_j_kg_k=entropy,
                version=NOZZLE_VERSION,source=NOZZLE_SOURCE,validated=False)


def nozzle_flow(pressure,enthalpy,back_pressure,position,area,discharge_coefficient):
    values=(pressure,enthalpy,back_pressure,position,area,discharge_coefficient)
    if any(not math.isfinite(v) for v in values) or not 5e6<=pressure<=8e6 or back_pressure<=0 or not 0<=position<=1 or not .0001<=area<=.03 or not .5<=discharge_coefficient<=1:
        raise ValueError('Bounded nozzle pressure/geometry/position')
    if position==0 or back_pressure>=pressure:
        return dict(flow_kg_s=0.,regime='closed' if position==0 else 'blocked',critical_pressure_pa=None,
                    mass_flux_kg_m2_s=0.,velocity_m_s=0.,enthalpy_j_kg=None,quality=None,density_kg_m3=None)
    result=dict(critical_nozzle(float(pressure),float(enthalpy)))
    critical=result['critical_pressure_pa']
    if back_pressure>critical:
        result.update(isentropic_flux(pressure,enthalpy,result['stagnation_entropy_j_kg_k'],back_pressure))
    result.update(flow_kg_s=float(area*discharge_coefficient*position*result['mass_flux_kg_m2_s']),
                  regime='choked' if back_pressure<=critical else 'unchoked')
    return result


@dataclass(frozen=True)
class BypassConfig:
    area_m2:float=.01
    discharge_coefficient:float=.8
    stroke_time_s:float=.5
    relief_open_pa:float=7e6
    relief_close_pa:float=6.8e6
    supply_open_pa:float=6.3e6
    supply_close_pa:float=6.2e6
    trip_opening:float=.1

    def __post_init__(self):
        for key,low,high in [('area_m2',.0001,.03),('discharge_coefficient',.5,1.),('stroke_time_s',.1,5.),
            ('relief_open_pa',6.35e6,7.6e6),('relief_close_pa',6.31e6,7.5e6),
            ('supply_open_pa',6.2e6,6.34e6),('supply_close_pa',6.1e6,6.3e6),('trip_opening',0.,1.)]:
            value=getattr(self,key)
            if type(value) not in (int,float) or not math.isfinite(value) or not low<=value<=high:raise ValueError('Bypass parameter domain:'+key)
        if not self.supply_close_pa<self.supply_open_pa<self.relief_close_pa<self.relief_open_pa:
            raise ValueError('Bypass supply/relief hysteresis pressure order')


class PWRBypassSession(PWRStagedSession):
    def __init__(self,config=None,*,cooling=None,feed=None,backup=None,core=None,primary=None,
                 film=None,steam=None,stages=None,bypass=None,max_step_s=.1,rtol=1e-7):
        self.bypass_ready=False;self.relief_latched=False;self.bypass_supply_permitted=True
        self.bypass_config=bypass if bypass is not None else BypassConfig()
        if type(self.bypass_config) is not BypassConfig:raise ValueError('BypassConfig required')
        super().__init__(config,cooling=cooling,feed=feed,backup=backup,core=core,primary=primary,
                         film=film,steam=steam,stages=stages,max_step_s=max_step_s,rtol=rtol)
        self.bypass_index=len(self.state);self.state=np.r_[self.state,0.]
        self.controls.update(steam_bypass=0.,bypass_available=True)
        self.bypass_ready=True;self._rhs(0.,self.state);self._admit(self.state)

    def _absolute_tolerances(self,atol):
        atol=super()._absolute_tolerances(atol)
        if self.bypass_ready:atol[self.bypass_index]=1e-10
        return atol

    def _bypass_target(self):
        if not self.controls['bypass_available'] or not self.bypass_supply_permitted:return 0.
        trip=self.bypass_config.trip_opening if self.scram_latched or self.turbine_tripped else 0.
        return max(self.controls['steam_bypass'],trip,1. if self.relief_latched else 0.)

    def _rhs(self,t,y):
        rate=super()._rhs(t,y)
        if not self.bypass_ready:return rate
        c=self.bypass_config;h=self.header.thermo;p=self.plant;position=float(y[self.bypass_index])
        flow=nozzle_flow(float(h.p_pa),float(h.h_j_kg),float(p._condenser_pressure_pa),position,c.area_m2,c.discharge_coefficient)
        q=flow['flow_kg_s'];energy=q*h.h_j_kg;i=self.steam_index
        rate[i]-=q;rate[i+1]-=energy
        rate[self.base_size-3]+=q;rate[self.base_size-2]+=energy
        self.feed_terms['hotwell_mass_rate_kg_s']+=q;self.feed_terms['hotwell_energy_rate_w']+=energy
        self.steam_terms['mass_rate_kg_s']-=q;self.steam_terms['energy_rate_w']-=energy
        recovered=iapws.properties_ph(p._condenser_pressure_pa,h.h_j_kg) if q>0 else None
        entropy=q*(recovered.s_j_kg_k-h.s_j_kg_k) if q>0 else 0.
        target=self._bypass_target()
        self.bypass_terms=dict(flow,position=position,target=target,flow_kg_s=q,enthalpy_flow_w=float(energy),
            header_pressure_pa=float(h.p_pa),condenser_pressure_pa=float(p._condenser_pressure_pa),
            stagnation_enthalpy_j_kg=float(h.h_j_kg),recovered_enthalpy_j_kg=float(h.h_j_kg),
            entropy_generation_w_k=float(entropy),shaft_work_w=0.,mass_routing_residual_kg_s=0.,energy_routing_residual_w=0.,
            relief_latched=self.relief_latched,supply_permitted=self.bypass_supply_permitted,
            legacy_sg_dump_flow_kg_s=float(p.network.junctions['dump'].mass_flow_kg_s),
            flow_model='IHEM isentropic maximum flux; jet kinetic energy recovered into condenser, no external shaft work')
        self.powers['header_bypass_enthalpy_w']=float(energy)
        derivative=(1. if target>position else -1. if target<position else 0.)/c.stroke_time_s
        return np.r_[rate,derivative]

    def _admit(self,y):
        super()._admit(y)
        if self.bypass_ready and not 0<=y[self.bypass_index]<=1:raise ValueError('Bypass position domain')

    def _prepare_interval(self):
        super()._prepare_interval()
        if not self.bypass_ready:return
        pressure=self.header.thermo.p_pa;c=self.bypass_config
        if not self.relief_latched and pressure>c.relief_open_pa:self._switch('bypass_relief_open',self.time,pressure)
        elif self.relief_latched and pressure<c.relief_close_pa:self._switch('bypass_relief_close',self.time,pressure)
        if self.bypass_supply_permitted and pressure<c.supply_close_pa:self._switch('bypass_supply_close',self.time,pressure)
        elif not self.bypass_supply_permitted and pressure>c.supply_open_pa:self._switch('bypass_supply_open',self.time,pressure)

    def _switch(self,label,now,pressure):
        if label=='bypass_relief_open':self.relief_latched=True
        elif label=='bypass_relief_close':self.relief_latched=False
        elif label=='bypass_supply_close':self.bypass_supply_permitted=False
        elif label=='bypass_supply_open':self.bypass_supply_permitted=True
        else:raise ValueError('Unknown bypass pressure mode')
        self.events.append(dict(time_s=now,kind=label,pressure_pa=float(pressure),inventories_projected=False))

    def _extend_known(self,known,modes):
        super()._extend_known(known,modes)
        if not self.bypass_ready:return
        x=float(self.state[self.bypass_index]);target=self._bypass_target();origin=self.time;c=self.bypass_config
        sign=1. if target>x else -1. if target<x else 0.;arrival=origin+abs(target-x)*c.stroke_time_s
        known[self.bypass_index]=lambda t:target if t>=arrival else x+sign*(t-origin)/c.stroke_time_s
        self.bypass_arrival=(arrival,target) if sign else None
        modes.append(dict(name='header_bypass_position',kind='exact_bounded_motion',initial=x,target=target,arrival_s=arrival))

    def _flow_guards(self,feed):
        guards=super()._flow_guards(feed)
        if not self.bypass_ready:return guards
        if self.bypass_arrival:
            arrival=self.bypass_arrival[0]
            def event(t,y):return arrival-t
            event.direction=-1;guards.append(event);self.guard_labels.append('bypass_position_arrival')
        c=self.bypass_config
        for label,pressure,direction in [('bypass_relief_close' if self.relief_latched else 'bypass_relief_open',c.relief_close_pa if self.relief_latched else c.relief_open_pa,-1 if self.relief_latched else 1),
            ('bypass_supply_close' if self.bypass_supply_permitted else 'bypass_supply_open',c.supply_close_pa if self.bypass_supply_permitted else c.supply_open_pa,-1 if self.bypass_supply_permitted else 1)]:
            def event(t,y,value=pressure):return self._sync_header(y).p_pa-value
            event.direction=direction;guards.append(event);self.guard_labels.append(label)
        return guards

    def _custom_root(self,label,endpoint,now):
        if label=='bypass_position_arrival':
            arrival,target=self.bypass_arrival;residual=float(endpoint[self.bypass_index]-target)
            if abs(now-arrival)>1e-10 or abs(residual)>1e-10:raise ValueError('Bypass analytical arrival residual')
            endpoint[self.bypass_index]=target
            self.events.append(dict(time_s=now,kind=label,target=target,position_roundoff=residual,inventories_projected=False))
            return True
        thresholds=dict(bypass_relief_open=self.bypass_config.relief_open_pa,bypass_relief_close=self.bypass_config.relief_close_pa,
                        bypass_supply_close=self.bypass_config.supply_close_pa,bypass_supply_open=self.bypass_config.supply_open_pa)
        if label in thresholds:
            pressure=self._sync_header(endpoint).p_pa
            if abs(pressure-thresholds[label])>.05:raise ValueError('Bypass pressure-root residual')
            self._switch(label,now,pressure);return True
        return super()._custom_root(label,endpoint,now)

    def _set_controls(self,patch):
        if type(patch) is not dict or not patch or not set(patch)<=set(self.controls):raise ValueError('Bypass command patch')
        if 'bypass_available' in patch and type(patch['bypass_available']) is not bool:raise ValueError('Boolean bypass availability')
        if 'steam_bypass' in patch:
            x=patch['steam_bypass']
            if type(x) not in (float,int) or not math.isfinite(x) or not 0<=x<=1:raise ValueError('Bypass request0..1')
        regular={k:v for k,v in patch.items() if k not in ('steam_bypass','bypass_available')}
        if regular:super()._set_controls(regular)
        self.controls.update({k:v for k,v in patch.items() if k in ('steam_bypass','bypass_available')})
        self.events.append(dict(time_s=self.time,kind='bypass_command',requested=deepcopy(patch)))

    def parameter_registry(self):
        result=super().parameter_registry()
        if self.bypass_ready:
            for key,value in asdict(self.bypass_config).items():
                unit='m^2' if key.endswith('_m2') else 's' if key.endswith('_s') else 'Pa' if key.endswith('_pa') else '1'
                result['parameters']['header_bypass.'+key]=dict(value=float(value),unit=unit,
                    source='Declared valve geometry/discharge and hybrid thresholds; no manufacturer safety-valve qualification',family='illustrative',version='1',uncertainty=None)
        return result

    def snapshot(self):
        result=super().snapshot()
        if not self.bypass_ready:return result
        b=deepcopy(self.bypass_terms);q=b['flow_kg_s'];balance=result['condenser_balance']
        balance['expected_mass_rate_kg_s']+=q;balance['expected_energy_rate_w']+=b['enthalpy_flow_w']
        balance['mass_residual_kg_s']=balance['mass_rate_kg_s']-balance['expected_mass_rate_kg_s']
        balance['energy_residual_w']=balance['energy_rate_w']-balance['expected_energy_rate_w']
        inlet=fluid_connection(FluidState(b['header_pressure_pa'],b['stagnation_enthalpy_j_kg'],q,phase='mixture'),
                              FluidState(b['header_pressure_pa'],b['stagnation_enthalpy_j_kg'],-q,phase='mixture'))
        outlet=fluid_connection(FluidState(b['condenser_pressure_pa'],b['recovered_enthalpy_j_kg'],q,phase='mixture'),
                               FluidState(b['condenser_pressure_pa'],b['recovered_enthalpy_j_kg'],-q,phase='mixture'))
        result.update(model_id='npp-pwr-finite-header-bypass',header_bypass=b,bypass_config=asdict(self.bypass_config),
            bypass_position=float(self.state[self.bypass_index]),bypass_port_audit=dict(inlet=inlet,outlet=outlet,
                energy_residual_w=inlet['enthalpy_a_to_b_w']-outlet['enthalpy_a_to_b_w'],shaft_work_w=0.))
        return result

    def checkpoint(self):
        p=unseal(super().checkpoint());p.update(bypass_config=asdict(self.bypass_config),bypass_source=bypass_signature());return seal(p)

    @classmethod
    def restore(cls,saved):
        p=unseal(deepcopy(saved))
        if p['bypass_source']!=bypass_signature() or type(p['history']) is not list or len(p['history'])>256:raise ValueError('Bypass replay source/budget')
        s=cls(TrainConfig(**p['config']),cooling=CoolingConfig(**p['cooling_config']),feed=FeedConfig(**p['feed_config']),backup=BackupConfig(**p['backup_config']),core=CoreConfig(**p['core_config']),
            primary=PrimaryConfig(**p['primary_config']),film=FilmConfig(**p['film_config']),steam=SteamConfig(**p['steam_config']),stages=StageConfig(**p['stage_config']),
            bypass=BypassConfig(**p['bypass_config']),max_step_s=p['max_step_s'],rtol=p['rtol'])
        for row in p['history']:s.advance(**row)
        if s.checkpoint()!=saved:raise ValueError('Bypass replay differs')
        return s
