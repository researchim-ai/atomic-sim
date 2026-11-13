"""Positive primary-shaft coordinate and explicit zero-pump-head bypass study."""
from copy import deepcopy
from dataclasses import asdict,dataclass
import math
import numpy as np
from atomic_model_multiphysics.provenance import seal,unseal
from physics.hydraulics.network import G
from .core import PWRCoreSession,CoreConfig,core_signature
from .power import AuxiliaryPWR,BackupConfig
from .feed import FeedConfig
from .cooling import CoolingConfig
from .pwr_train import TrainConfig


def frozen_gravity_head(density_by_node,height_m):
    """Independent four-column head at fixed densities, no fill/state change."""
    names=('core','hot_leg','sg_primary','cold_leg')
    if type(height_m) not in (int,float) or set(density_by_node)!=set(names) or not math.isfinite(height_m) or not 0<=height_m<=20:raise ValueError('Frozen primary geometry')
    if any(not math.isfinite(density_by_node[n]) or density_by_node[n]<=0 for n in names):raise ValueError('Positive frozen liquid densities')
    r=density_by_node
    return G*height_m*(-.25*r['core']-.5*r['hot_leg']+.25*r['sg_primary']+.5*r['cold_leg'])


@dataclass(frozen=True)
class PrimaryConfig:
    height_m:float=10.
    maximum_speed_fraction:float=1.2
    powered_restart_minimum_fraction:float=1e-6
    bypass_transition_time_s:float=60.
    def __post_init__(self):
        for key,low,high in [('height_m',0.,20.),('maximum_speed_fraction',1.,1.5),('powered_restart_minimum_fraction',1e-8,.01),('bypass_transition_time_s',1.,120.)]:
            value=getattr(self,key)
            if type(value) not in (int,float) or not math.isfinite(value) or not low<=value<=high:raise ValueError('Primary study parameter:'+key)


class PrimaryPWR(AuxiliaryPWR):
    def _hydraulic_torque(self):
        fraction=getattr(self,'primary_bypass_fraction',0.)
        if fraction>=1.:
            self.network.junctions['pump'].pump_pressure_pa=0.
            return 0.
        torque=super()._hydraulic_torque()
        self.network.junctions['pump'].pump_pressure_pa*=1-fraction
        return torque*(1-fraction)


class PWRPrimarySession(PWRCoreSession):
    def _make_plant(self):return PrimaryPWR(height_m=self.primary_config.height_m)

    def __init__(self,config=None,*,cooling=None,feed=None,backup=None,core=None,primary=None,max_step_s=.1,rtol=1e-7):
        if config is None:config=TrainConfig(minimum_speed_fraction=.5)
        self.minimum_spray_sample_kg_s=0.;self.primary_ready=False;self.primary_config=primary if primary is not None else PrimaryConfig()
        if type(self.primary_config) is not PrimaryConfig:raise ValueError('PrimaryConfig required')
        super().__init__(config,cooling=cooling,feed=feed,backup=backup,core=core,max_step_s=max_step_s,rtol=rtol)
        self.primary_index=len(self.state)
        self.controls['primary_bypass_open']=False
        self.state=np.r_[self.state,math.log(self.plant.omega/self.plant.omega_nom),0.]
        self.primary_ready=True;self.state=self._reconstruct_state(self.state)
        self._rhs(0.,self.state);self.origin_energy=self._stored();self._admit(self.state)

    def _inactive_target_names(self):return ()

    def _closed_steam_mode(self):return self.scram_latched or self.turbine_tripped

    def _reconstruct_state(self,y):
        y=super()._reconstruct_state(y)
        if self.primary_ready:
            z=float(y[self.primary_index])
            if not math.isfinite(z) or z>math.log(self.primary_config.maximum_speed_fraction)+.1:raise ValueError('Primary shaft logarithm domain')
            y[11]=self.plant.omega_nom*math.exp(z)
        return y

    def _absolute_tolerances(self,atol):
        atol=super()._absolute_tolerances(atol)
        if self.primary_ready:atol[self.primary_index:self.primary_index+2]=1e-10
        return atol

    def _extend_known(self,known,modes):
        super()._extend_known(known,modes)
        if not self.primary_ready:return
        index=self.primary_index+1;initial=float(self.state[index]);target=1. if self.controls['primary_bypass_open'] else 0.
        direction=1. if target>initial else -1. if target<initial else 0.
        origin=self.time;speed=1/self.primary_config.bypass_transition_time_s;arrival=origin+abs(target-initial)/speed
        known[index]=lambda t:target if t>=arrival else initial+direction*speed*(t-origin)
        modes.append(dict(name='primary_bypass_position',kind='exact_bounded_motion',initial=initial,target=target,arrival_s=arrival))
        known[11]=lambda t:0.
        modes.append(dict(name='primary_shaft_speed',kind='derived_from_log_speed',index=11))
        if self._closed_steam_mode():
            p=self.plant;j=p.network.junctions['steam']
            if j.flow_law!='target' or j.flow_target_kg_s!=0.:raise ValueError('Closed steam mode not applicable')
            index=13+3*len(p.network.volumes)+p.network.junction_names().index('steam')
            q=float(self.state[index]);origin=self.time;tau=j.flow_time_constant_s
            known[index]=lambda t:q*math.exp(-(t-origin)/tau)
            modes.append(dict(name='closed_turbine_steam',kind='exact_exponential',initial=q,tau_s=tau))

    def _rhs(self,t,y):
        if not self.primary_ready:return super()._rhs(t,y)
        full=self._reconstruct_state(np.array(y,dtype=float,copy=True));p=self.plant
        bypass=self.controls['primary_bypass_open'];fraction=float(full[self.primary_index+1]);p.primary_bypass_fraction=fraction
        requested=self.controls;effective=dict(requested)
        if bypass or fraction>0.:effective['primary_pump_on']=False
        self.controls=effective
        try:rate=super()._rhs(t,full)
        finally:self.controls=requested
        q=p.network.junctions['pump'].mass_flow_kg_s;rho=p.network.volumes['cold_leg'].thermo.rho_kg_m3
        hydraulic_coefficient=(1-fraction)*q*p.dp_pump_nom/(rho*p.omega_nom**2)
        decay=(hydraulic_coefficient+p.loss_coef)/p.inertia
        if p.motor_on:
            if p.omega<self.primary_config.powered_restart_minimum_fraction*p.omega_nom:raise ValueError('Powered near-stall restart outside this shaft study')
            log_rate=p.drive_torque/(p.inertia*p.omega)-decay
        else:log_rate=-decay
        previous=float(rate[11]);rate[11]=p.omega*log_rate
        self.instantaneous_energy_defect_w+=p.inertia*p.omega*(rate[11]-previous)
        if self._closed_steam_mode() and p.network.junctions['steam'].flow_target_kg_s!=0.:
            raise ValueError('Closed steam source activated')
        cycle=['pump','core_out','to_sg','to_cold'];offset=13+3*len(p.network.volumes);names=p.network.junction_names()
        pressures=[];gravity=[];friction=[];inertia=[];flows=[];resistance=[]
        for name in cycle:
            link=p.network.junctions[name];up=p.network.volumes[link.upstream];down=p.network.volumes[link.downstream]
            donor=up if link.mass_flow_kg_s>=0 else down;density=donor.thermo.rho_kg_m3
            pressures.append(up.pressure_pa-down.pressure_pa)
            gravity.append(-.5*(up.thermo.rho_kg_m3+down.thermo.rho_kg_m3)*G*(down.elevation_m-up.elevation_m))
            friction.append(link.friction_dp(density));flows.append(link.mass_flow_kg_s)
            inertia.append(link.length_m/link.area_m2*rate[offset+names.index(name)])
            resistance.append(link.loss_coefficient/(2*density*link.area_m2**2))
        buoyancy=sum(gravity);head=p.network.junctions['pump'].pump_pressure_pa
        balance=sum(pressures)+head+buoyancy-sum(friction)-sum(inertia)
        reference=math.sqrt(buoyancy/sum(resistance)) if buoyancy>0 else None
        self.primary_terms=dict(log_speed=float(full[self.primary_index]),speed_rad_s=p.omega,log_speed_rate_per_s=log_rate,
            bypass_selected=bypass,bypass_fraction=fraction,effective_motor_on=p.motor_on,pump_head_pa=head,buoyancy_head_pa=buoyancy,
            cycle_pressure_sum_pa=sum(pressures),friction_head_pa=sum(friction),inertial_head_pa=sum(inertia),momentum_residual_pa=balance,
            cycle_flows_kg_s=dict(zip(cycle,flows)),mean_flow_kg_s=float(np.mean(flows)),flow_spread_relative=float(np.ptp(flows)/max(abs(np.mean(flows)),1.)),
            frozen_zero_head_reference_kg_s=reference,zero_head_reference_scope='Frozen instantaneous densities and common steady flow; independent scalar friction solve',
            shaft_drag_energy_out_w=p.loss_coef*p.omega**2,geometry_height_m=p.height_m,
            spray_flow_kg_s=p.network.junctions['spray'].mass_flow_kg_s,spray_target_kg_s=p.network.junctions['spray'].flow_target_kg_s,
            minimum_sampled_spray_flow_kg_s=self.minimum_spray_sample_kg_s,spray_signed_roundoff_band_kg_s=1e-9,
            flow_direction='qualified forward cycle only',shaft_coordinate='log(omega/omega_nom), old speed slot is a derived alias')
        target=1. if bypass else 0.
        valve_rate=(1. if target>fraction else -1. if target<fraction else 0.)/self.primary_config.bypass_transition_time_s
        return np.r_[rate,log_rate,valve_rate]

    def _admit(self,y):
        if self.primary_ready:
            expected=self.plant.omega_nom*math.exp(float(y[self.primary_index]))
            if abs(float(y[11])-expected)>1e-12*max(expected,1.):raise ValueError('Primary speed alias differs from its owner')
        super()._admit(y)
        if self.primary_ready:
            if not 0<=self.plant.omega<=self.primary_config.maximum_speed_fraction*self.plant.omega_nom:raise ValueError('Primary shaft speed band')
            if not 0<=y[self.primary_index+1]<=1:raise ValueError('Primary bypass position domain')
            spray=self.plant.network.junctions['spray'].mass_flow_kg_s
            self.minimum_spray_sample_kg_s=min(self.minimum_spray_sample_kg_s,float(spray))
            if spray < -1e-9:raise ValueError('Reverse spray flow outside numerical band')
            for name in ('pump','core_out','to_sg','to_cold'):
                if self.plant.network.junctions[name].mass_flow_kg_s<0:raise ValueError('Reverse primary cycle outside qualified study')

    def _set_controls(self,patch):
        if type(patch) is not dict or not patch or not set(patch)<=set(self.controls):raise ValueError('Primary study command patch')
        if 'primary_bypass_open' in patch and type(patch['primary_bypass_open']) is not bool:raise ValueError('Boolean bypass selection')
        regular={k:v for k,v in patch.items() if k!='primary_bypass_open'}
        if regular:super()._set_controls(regular)
        if 'primary_bypass_open' in patch:
            self.controls['primary_bypass_open']=patch['primary_bypass_open']
            self.events.append(dict(time_s=self.time,kind='primary_path_selected',bypass=patch['primary_bypass_open'],
                reason='Equivalent passive path; pump head and hydraulic torque removed, motor inhibited' if patch['primary_bypass_open'] else 'Shaft-coupled pump path'))

    def parameter_registry(self):
        result=super().parameter_registry()
        if self.primary_ready:
            for key,value in asdict(self.primary_config).items():
                result['parameters']['primary_study.'+key]=dict(value=float(value),unit='m' if key=='height_m' else 's' if key=='bypass_transition_time_s' else '1',
                    source='Declared reduced primary geometry and shaft limits, not surveyed plant data',family='illustrative',version='1',uncertainty=None)
        return result

    def snapshot(self):
        r=super().snapshot()
        if self.primary_ready:r.update(model_id='npp-pwr-primary-circulation',primary=deepcopy(self.primary_terms),primary_config=asdict(self.primary_config),primary_circulation_connected=True)
        return r

    def checkpoint(self):
        p=unseal(super().checkpoint());p['primary_config']=asdict(self.primary_config);return seal(p)

    @classmethod
    def restore(cls,saved):
        p=unseal(deepcopy(saved))
        if p['core_source']!=core_signature() or type(p['history']) is not list or len(p['history'])>256:raise ValueError('Primary study replay source/budget')
        s=cls(TrainConfig(**p['config']),cooling=CoolingConfig(**p['cooling_config']),feed=FeedConfig(**p['feed_config']),backup=BackupConfig(**p['backup_config']),
            core=CoreConfig(**p['core_config']),primary=PrimaryConfig(**p['primary_config']),max_step_s=p['max_step_s'],rtol=p['rtol'])
        for row in p['history']:s.advance(**row)
        if s.checkpoint()!=saved:raise ValueError('Primary study replay differs')
        return s
