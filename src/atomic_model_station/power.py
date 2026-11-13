"""Declared hot-station loss-of-power and finite emergency cooling research model."""
from copy import deepcopy
from dataclasses import asdict,dataclass
import math
import numpy as np
from atomic_model_multiphysics.provenance import seal,unseal
from .feed import FeedPWR,FeedConfig,PWRFeedSession
from .cooling import CoolingConfig
from .pwr_train import TrainConfig,physical_signature


@dataclass(frozen=True)
class BackupConfig:
    initial_energy_j:float=5e6
    dhr_electric_w:float=1e5
    dhr_ua_w_k:float=5e5
    dhr_maximum_heat_w:float=1e8

    def __post_init__(self):
        ranges=dict(initial_energy_j=(1e4,1e9),dhr_electric_w=(1e4,1e6),dhr_ua_w_k=(1e3,1e6),dhr_maximum_heat_w=(1e6,3e8))
        for key,(low,high) in ranges.items():
            value=getattr(self,key)
            if type(value) not in (int,float) or not math.isfinite(value) or not low<=value<=high:raise ValueError('Backup parameter domain:'+key)


class AuxiliaryPWR(FeedPWR):
    def _pressurizer_controls(self,pressure_pa):
        if not getattr(self,'aux_power_available',True):return 0.,0.
        return super()._pressurizer_controls(pressure_pa)


class PWRPowerSession(PWRFeedSession):
    def _make_plant(self):return AuxiliaryPWR()

    def __init__(self,config=None,*,cooling=None,feed=None,backup=None,max_step_s=.1,rtol=1e-7):
        self.power_ready=False;self.backup_config=backup if backup is not None else BackupConfig()
        if type(self.backup_config) is not BackupConfig:raise ValueError('BackupConfig required')
        self.emergency_latched=False;self.backup_exhausted=False
        super().__init__(config,cooling=cooling,feed=feed,max_step_s=max_step_s,rtol=rtol)
        self.controls.update(grid_available=True,primary_pump_on=True,backup_enabled=True,dhr_requested=False)
        self.power_index=len(self.state)
        # Stored backup energy, actual grid exchange, DHR heat and backup use.
        self.state=np.r_[self.state,self.backup_config.initial_energy_j,0.,0.,0.]
        self.power_ready=True;self._rhs(0.,self.state);self.origin_energy=self._stored();self._admit(self.state)

    def _stored(self):
        energy=super()._stored()
        return float(energy+self.state[self.power_index]) if self.power_ready else energy

    def _passive_column(self,index):
        return super()._passive_column(index) or (self.power_ready and self.power_index+1<=index<self.power_index+4)

    def _supply(self):
        grid=self.controls['grid_available'];wanted=self.emergency_latched or self.controls['dhr_requested']
        reserve=self.controls['backup_enabled'] and not self.backup_exhausted
        active=bool(wanted and (grid or reserve))
        return grid,active,bool(active and not grid)

    def _rhs(self,t,y):
        if not self.power_ready:return super()._rhs(t,y)
        grid,active,drawing=self._supply();requested=self.controls;effective=dict(requested)
        for key in ('feed_pump_on','condensate_pump_on','cooling_pump_on'):effective[key]=bool(requested[key] and grid)
        self.plant.aux_power_available=grid;self.plant.motor_on=bool(grid and requested['primary_pump_on'])
        self.controls=effective
        try:rate=super()._rhs(t,y)
        finally:self.controls=requested
        c=self.backup_config;l=self.ledger_index;i=self.power_index
        primary=self.plant.network.volumes['cold_leg'];cold=float(y[self.cooling_index]/(self.cooling_config.cold_mass_kg*self.cooling_config.heat_capacity_j_kg_k))
        heat=min(c.dhr_maximum_heat_w,c.dhr_ua_w_k*max(primary.temperature_k-cold,0.)) if active else 0.
        electric=c.dhr_electric_w if active else 0.;discharge=electric if drawing else 0.
        node=13+3*self.plant.network.volume_names().index('cold_leg')+1
        rate[node]-=heat;rate[self.cooling_index]+=heat+electric
        self.cooling_terms['cold_energy_rate_w']+=heat+electric
        self.cooling_terms.update(dhr_received_w=heat,dhr_motor_heat_w=electric)
        rate[l]+=electric if grid else 0.;rate[l+2]+=electric
        own=self.powers['own_needs_w']+electric;gross=self.powers['gross_electrical_w']
        export=gross-own if grid else 0.
        rate[l+3]=abs(self.powers['nuclear_heat_w'])+abs(self.powers['primary_pump_loss_out_w'])+abs(self.powers['external_cooling_heat_out_w'])+abs(export)
        self.powers.update(own_needs_w=own,net_export_w=export,net_generation_w=gross-own,external_net_w=float(rate[l]),
            grid_import_w=max(-export,0.),dhr_electric_w=electric,backup_discharge_w=discharge,dhr_heat_w=heat)
        self.power_terms=dict(grid_available=grid,main_bus_available=grid,emergency_latched=self.emergency_latched,
            backup_exhausted=self.backup_exhausted,backup_energy_j=float(y[i]),dhr_active=active,dhr_from_backup=drawing,
            dhr_heat_w=heat,dhr_electric_w=electric,backup_discharge_w=discharge,
            dhr_receiving_temperature_k=cold,dhr_donor_temperature_k=primary.temperature_k,
            dhr_entropy_generation_w_k=heat*(1/cold-1/primary.temperature_k)+electric/cold,
            effective_drives=dict(primary_pump=bool(grid and requested['primary_pump_on']),feed_pump=effective['feed_pump_on'],
                condensate_pump=effective['condensate_pump_on'],cooling_pump=effective['cooling_pump_on'],pressurizer_heater=grid),
            energy_transfer_residual_w=(-heat)+(heat+electric)-discharge-(electric if grid else 0.))
        return np.r_[rate,-discharge,export,heat,discharge]

    def _set_controls(self,patch):
        extra={'grid_available','primary_pump_on','backup_enabled','dhr_requested'}
        if type(patch) is not dict or not patch or not set(patch)<=set(self.controls):raise ValueError('Power command patch')
        if any(type(value) is not bool for key,value in patch.items() if key in extra):raise ValueError('Boolean supply/request command')
        old=self.controls['grid_available'];regular={key:value for key,value in patch.items() if key not in extra}
        if regular:super()._set_controls(regular)
        self.controls.update({key:value for key,value in patch.items() if key in extra})
        if any(key in extra for key in patch):self.events.append(dict(time_s=self.time,kind='power_command',requested={k:v for k,v in patch.items() if k in extra}))
        grid=self.controls['grid_available']
        if old and not grid:
            self.events.append(dict(time_s=self.time,kind='grid_lost',reason='external_supply_unavailable'))
            if not self.emergency_latched:
                self.emergency_latched=True;self.scram_latched=True;self.turbine_tripped=True
                self.events.append(dict(time_s=self.time,kind='power_loss_trip',reason='declared_fail_safe_interlock',scram=True,turbine_trip=True))
            self.events.append(dict(time_s=self.time,kind='main_bus_deenergized',reason='grid_lost',backup_scope='dedicated_DHR_only'))
        if not old and grid:self.events.append(dict(time_s=self.time,kind='grid_restored',trip_reset=False,backup_recharged=False))

    def _admit(self,y):
        super()._admit(y)
        if self.power_ready and not 0<=y[self.power_index]<=self.backup_config.initial_energy_j:
            raise ValueError('Finite backup energy domain')

    def _extend_known(self,known,modes):
        super()._extend_known(known,modes)
        if self.power_ready and self.backup_exhausted:
            known[self.power_index]=lambda t:0.;modes.append(dict(name='backup_energy',kind='exact_exhausted'))

    def _flow_guards(self,feed):
        guards=super()._flow_guards(feed)
        if self.power_ready and self._supply()[2]:
            def empty(t,y):return float(y[self.power_index])
            empty.direction=-1;guards.append(empty);self.guard_labels.append('backup_empty')
        return guards

    def _custom_root(self,label,endpoint,now):
        if label!='backup_empty':return super()._custom_root(label,endpoint,now)
        residual=float(endpoint[self.power_index])
        if abs(residual)>1e-3:raise ValueError('Backup exhaustion root residual')
        endpoint[self.power_index]=0.;endpoint[self.cooling_index]+=residual
        self.backup_exhausted=True
        self.events.append(dict(time_s=now,kind='backup_exhausted',root_residual_j=residual,root_roundoff_to_cooling_j=residual,reason='finite_energy_empty'))
        return True

    def parameter_registry(self):
        result=super().parameter_registry()
        if not self.power_ready:return result
        units=dict(initial_energy_j='J',dhr_electric_w='W',dhr_ua_w_k='W/K',dhr_maximum_heat_w='W')
        for key,value in asdict(self.backup_config).items():
            result['parameters']['backup.'+key]=dict(value=float(value),unit=units[key],source='Declared hypothetical research backup/DHR sizing, not licensed plant equipment',
                family='illustrative',version='1',uncertainty=None)
        return result

    def snapshot(self):
        result=super().snapshot()
        if self.power_ready:
            result.update(model_id='npp-pwr-train-finite-power',electrical=deepcopy(self.power_terms),backup_config=asdict(self.backup_config),
                net_generated_j=result['net_exported_j'],net_exported_j=float(self.state[self.power_index+1]),
                dhr_transferred_j=float(self.state[self.power_index+2]),backup_used_j=float(self.state[self.power_index+3]),
                finite_power_connected=True)
        return result

    def checkpoint(self):
        payload=unseal(super().checkpoint());payload['backup_config']=asdict(self.backup_config);return seal(payload)

    @classmethod
    def restore(cls,saved):
        p=unseal(deepcopy(saved))
        if p['source']!=physical_signature() or type(p['history']) is not list or len(p['history'])>256:raise ValueError('Power replay source/budget')
        s=cls(TrainConfig(**p['config']),cooling=CoolingConfig(**p['cooling_config']),feed=FeedConfig(**p['feed_config']),backup=BackupConfig(**p['backup_config']),
            max_step_s=p['max_step_s'],rtol=p['rtol'])
        for row in p['history']:s.advance(**row)
        if s.checkpoint()!=saved:raise ValueError('Power replay differs')
        return s
