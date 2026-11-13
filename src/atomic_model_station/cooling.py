"""Finite incompressible cooling loop coupled to the hot PWR/train backend.

The external source is prescribed-flow water, not an unmodelled cooling tower.
All sizing is illustrative. No electrical blackout admission is implied.
"""
from copy import deepcopy
from dataclasses import asdict,dataclass
import math

import numpy as np

from atomic_model_multiphysics.provenance import seal,unseal
from .pwr_train import PWRTrainSession,TrainConfig,physical_signature


@dataclass(frozen=True)
class CoolingConfig:
    cold_mass_kg:float=1e6
    hot_mass_kg:float=1e6
    heat_capacity_j_kg_k:float=4180.
    density_kg_m3:float=1000.
    nominal_temperature_rise_k:float=5.
    external_inlet_temperature_k:float=283.15
    nominal_external_effectiveness:float=.8
    nominal_pump_head_pa:float=2e5
    pump_efficiency:float=.85
    pipe_inertance_m_inv:float=10.

    def __post_init__(self):
        domains=dict(cold_mass_kg=(1e4,1e8),hot_mass_kg=(1e4,1e8),heat_capacity_j_kg_k=(3800.,4500.),
            density_kg_m3=(950.,1050.),nominal_temperature_rise_k=(1.,10.),external_inlet_temperature_k=(273.15,303.15),
            nominal_external_effectiveness=(.1,.95),nominal_pump_head_pa=(1e4,1e6),pump_efficiency=(.5,1.),pipe_inertance_m_inv=(1.,100.))
        for key,(low,high) in domains.items():
            value=getattr(self,key)
            if type(value) not in (int,float) or not math.isfinite(value) or not low<=value<=high:
                raise ValueError('Cooling parameter domain:'+key)


class PWRCooledSession(PWRTrainSession):
    def __init__(self,config=None,*,cooling=None,max_step_s=.1,rtol=1e-7):
        self.cooling_ready=False
        self.cooling_config=cooling if cooling is not None else CoolingConfig()
        if type(self.cooling_config) is not CoolingConfig:raise ValueError('CoolingConfig required')
        super().__init__(config,max_step_s=max_step_s,rtol=rtol)
        c=self.cooling_config;hot=self.plant.t_sink;cold=hot-c.nominal_temperature_rise_k
        if not 273.16<=cold<hot<353.15 or not c.external_inlet_temperature_k<cold:
            raise ValueError('Constructed cooling start requires liquid cold/hot water and a colder external inlet')
        heat=float(self.powers['condenser_heat_out_w'])
        if heat<=0:raise ValueError('Positive nominal condenser heat required')
        flow=heat/(c.heat_capacity_j_kg_k*c.nominal_temperature_rise_k)
        self.cooling_resistance=c.nominal_pump_head_pa/flow**2
        pump=flow*c.nominal_pump_head_pa/(c.density_kg_m3*c.pump_efficiency)
        external_capacity=(heat+pump)/(c.nominal_external_effectiveness*(cold-c.external_inlet_temperature_k))
        self.external_nominal_flow_kg_s=external_capacity/c.heat_capacity_j_kg_k
        self.cooler_ua_w_k=-external_capacity*math.log1p(-c.nominal_external_effectiveness)
        self.cooling_design=dict(initial_cold_temperature_k=cold,initial_hot_temperature_k=hot,nominal_loop_flow_kg_s=flow,
            nominal_pump_electric_w=pump,nominal_condenser_heat_w=heat,external_nominal_flow_kg_s=self.external_nominal_flow_kg_s,
            cooler_ua_w_k=self.cooler_ua_w_k,flow_resistance_pa_s2_kg2=self.cooling_resistance,
            source='Constructed nominal sizing, no measured fit',measured=False,inventories_projected=False,
            external_source='Prescribed-flow liquid water; external delivery infrastructure outside the plant boundary')
        self.controls.update(cooling_pump_on=True,cooling_pump_fraction=1.,external_water_fraction=1.,
            external_water_temperature_k=c.external_inlet_temperature_k,cooler_available=True)
        self.cooling_index=len(self.state)
        self.state=np.r_[self.state,c.cold_mass_kg*c.heat_capacity_j_kg_k*cold,c.hot_mass_kg*c.heat_capacity_j_kg_k*hot,flow]
        self.cooling_ready=True
        self._rhs(0.,self.state);self.origin_energy=self._stored();self._admit(self.state)

    def _stored(self):
        stored=super()._stored()
        if not self.cooling_ready:return stored
        i=self.cooling_index;c=self.cooling_config
        return float(stored+self.state[i]+self.state[i+1]+.5*c.pipe_inertance_m_inv*self.state[i+2]**2/c.density_kg_m3)

    def _terms(self,y):
        c=self.cooling_config;i=self.cooling_index
        cold=float(y[i]/(c.cold_mass_kg*c.heat_capacity_j_kg_k))
        hot=float(y[i+1]/(c.hot_mass_kg*c.heat_capacity_j_kg_k));flow=float(y[i+2])
        head=c.nominal_pump_head_pa*self.controls['cooling_pump_fraction']**2 if self.controls['cooling_pump_on'] else 0.
        friction=self.cooling_resistance*flow*abs(flow)
        flow_rate=(head-friction)/c.pipe_inertance_m_inv
        hydraulic=flow*head/c.density_kg_m3;electric=hydraulic/c.pump_efficiency
        friction_heat=flow*friction/c.density_kg_m3;motor_heat=electric-hydraulic
        kinetic_rate=c.pipe_inertance_m_inv*flow*flow_rate/c.density_kg_m3
        outside_flow=self.external_nominal_flow_kg_s*self.controls['external_water_fraction']
        capacity=outside_flow*c.heat_capacity_j_kg_k
        ua=self.cooler_ua_w_k if self.controls['cooler_available'] else 0.
        effectiveness=-math.expm1(-ua/capacity) if capacity>0 else 0.
        inlet=self.controls['external_water_temperature_k']
        removed=capacity*effectiveness*(cold-inlet)
        outlet=inlet+removed/capacity if capacity>0 else inlet
        exchanged=flow*c.heat_capacity_j_kg_k*(hot-cold)
        return dict(cold_temperature_k=cold,hot_temperature_k=hot,flow_kg_s=flow,flow_rate_kg_s2=flow_rate,
            pump_head_pa=head,friction_head_pa=friction,pump_electric_w=electric,pump_hydraulic_w=hydraulic,
            friction_heat_w=friction_heat,motor_heat_w=motor_heat,kinetic_rate_w=kinetic_rate,loop_exchange_w=exchanged,
            external_flow_kg_s=outside_flow,external_inlet_temperature_k=inlet,external_outlet_temperature_k=outlet,
            external_removed_w=removed,external_effectiveness=effectiveness,
            pump_energy_residual_w=electric-friction_heat-motor_heat-kinetic_rate,
            water_boundary_residual_w=removed-capacity*(outlet-inlet))

    def _rhs(self,t,y):
        if not self.cooling_ready:return super()._rhs(t,y)
        terms=self._terms(y);self.plant.t_sink=terms['hot_temperature_k']
        rate=super()._rhs(t,y);l=self.ledger_index
        received=self.powers['condenser_heat_out_w'];pump=terms['pump_electric_w'];removed=terms['external_removed_w']
        hot_rate=received-terms['loop_exchange_w']
        cold_rate=terms['loop_exchange_w']-removed+terms['friction_heat_w']+terms['motor_heat_w']
        cooling_rate=hot_rate+cold_rate+terms['kinetic_rate_w']
        base_boundary=rate[l]
        rate[l]=base_boundary+received-removed+pump
        rate[l+2]+=pump
        own=self.powers['own_needs_w']+pump;net=self.powers['gross_electrical_w']-own
        rate[l+3]=abs(self.powers['nuclear_heat_w'])+abs(self.powers['primary_pump_loss_out_w'])+abs(removed)+abs(net)
        self.powers.update(own_needs_w=own,net_export_w=net,external_net_w=float(rate[l]),
            cooling_pump_electric_w=pump,external_cooling_heat_out_w=removed)
        self.instantaneous_energy_defect_w=float(self.instantaneous_energy_defect_w+cooling_rate-(received-removed+pump))
        outside_capacity=terms['external_flow_kg_s']*self.cooling_config.heat_capacity_j_kg_k
        external_entropy=outside_capacity*math.log(terms['external_outlet_temperature_k']/terms['external_inlet_temperature_k'])-removed/terms['cold_temperature_k']
        terms.update(entropy_generation_w_k=dict(
                circulation_mixing=terms['loop_exchange_w']*(1/terms['cold_temperature_k']-1/terms['hot_temperature_k']),
                condenser_transfer=received*(1/terms['hot_temperature_k']-1/self.plant.hotwell_temperature_k),
                external_exchanger=external_entropy,
                dissipation=(terms['friction_heat_w']+terms['motor_heat_w'])/terms['cold_temperature_k']),
            condenser_received_w=received,cold_energy_rate_w=cold_rate,hot_energy_rate_w=hot_rate,
            energy_residual_w=cooling_rate-received+removed-pump,
            total_mass_kg=self.cooling_config.cold_mass_kg+self.cooling_config.hot_mass_kg)
        self.cooling_terms=terms
        return np.r_[rate,cold_rate,hot_rate,terms['flow_rate_kg_s2']]

    def _admit(self,y):
        if self.cooling_ready:
            terms=self._terms(y)
            if not np.isfinite(y).all() or not 273.16<=terms['cold_temperature_k']<=353.15 or not 273.16<=terms['hot_temperature_k']<=353.15:
                raise ValueError('Cooling liquid-temperature domain')
            if not 0<=terms['flow_kg_s']<=2*self.cooling_design['nominal_loop_flow_kg_s']:
                raise ValueError('Cooling forward-flow domain')
            if not 273.15<=terms['external_outlet_temperature_k']<=353.15:
                raise ValueError('External water liquid-temperature domain')
            self.plant.t_sink=terms['hot_temperature_k']
        super()._admit(y)
        if self.cooling_ready and terms['hot_temperature_k']>self.plant.hotwell_temperature_k:
            raise ValueError('Reverse condenser heat transfer outside this hot configuration')

    def _set_controls(self,patch):
        extra={'cooling_pump_on','cooling_pump_fraction','external_water_fraction','external_water_temperature_k','cooler_available'}
        if type(patch) is not dict or not patch or not set(patch)<=set(self.controls):raise ValueError('Cooled station command patch')
        values=dict(self.controls,**patch)
        if any(type(values[k]) is not bool for k in ('cooling_pump_on','cooler_available')):raise ValueError('Boolean cooling availability')
        for key,low,high in [('cooling_pump_fraction',0.,1.2),('external_water_fraction',0.,1.5),('external_water_temperature_k',273.15,303.15)]:
            value=values[key]
            if type(value) not in (int,float) or not math.isfinite(value) or not low<=value<=high:raise ValueError('Cooling command domain:'+key)
        regular={k:v for k,v in patch.items() if k not in extra}
        if regular:super()._set_controls(regular)
        else:self.events.append(dict(time_s=self.time,kind='command',requested=deepcopy(patch)))
        for key in extra:self.controls[key]=values[key]
        if regular and any(k in extra for k in patch):
            self.events.append(dict(time_s=self.time,kind='cooling_command',requested={k:v for k,v in patch.items() if k in extra}))

    def parameter_registry(self):
        result=super().parameter_registry()
        if not self.cooling_ready:return result
        units=dict(cold_mass_kg='kg',hot_mass_kg='kg',heat_capacity_j_kg_k='J/(kg*K)',density_kg_m3='kg/m^3',
            nominal_temperature_rise_k='K',external_inlet_temperature_k='K',nominal_external_effectiveness='1',
            nominal_pump_head_pa='Pa',pump_efficiency='1',pipe_inertance_m_inv='1/m')
        for key,value in asdict(self.cooling_config).items():
            result['parameters']['cooling.'+key]=dict(value=float(value),unit=units[key],source='Declared incompressible research cooling loop',
                family='illustrative',version='1',uncertainty=None)
        for key,value,unit in [('cooler_ua_w_k',self.cooler_ua_w_k,'W/K'),('external_nominal_flow_kg_s',self.external_nominal_flow_kg_s,'kg/s'),
            ('flow_resistance',self.cooling_resistance,'Pa*s^2/kg^2')]:
            result['parameters']['cooling.design.'+key]=dict(value=float(value),unit=unit,source='Constructed nominal sizing; no measured fit',
                family='illustrative',version='1',uncertainty=None)
        return result

    def snapshot(self):
        result=super().snapshot()
        if self.cooling_ready:
            result.update(model_id='npp-pwr-train-finite-cooling',cooling=deepcopy(self.cooling_terms),
                cooling_design=deepcopy(self.cooling_design),cooling_config=asdict(self.cooling_config),
                cooling_energy_j=self.state[self.cooling_index:self.cooling_index+2].tolist(),
                external_source='Prescribed flow and inlet temperature of external water; no cooling-tower model',
                finite_cooling_connected=True)
        return result

    def checkpoint(self):
        payload=unseal(super().checkpoint());payload['cooling_config']=asdict(self.cooling_config)
        return seal(payload)

    @classmethod
    def restore(cls,saved):
        p=unseal(deepcopy(saved))
        if p['source']!=physical_signature() or type(p['history']) is not list or len(p['history'])>256:
            raise ValueError('Cooled station restart source/budget')
        s=cls(TrainConfig(**p['config']),cooling=CoolingConfig(**p['cooling_config']),max_step_s=p['max_step_s'],rtol=p['rtol'])
        for row in p['history']:s.advance(**row)
        if s.checkpoint()!=saved:raise ValueError('Cooled station replay differs')
        return s
