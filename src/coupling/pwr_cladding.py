"""Local fuel/cladding storage on the reduced PWR, with one common coolant.

Uses the inherited constant conductances, not a boiling/CHF correlation.
Axial conduction, local fluid inventories and the staged film backend are absent.
"""
from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np
from atomic_model_multiphysics.provenance import seal,unseal
from atomic_model_station import TrainConfig,CoreConfig
from atomic_model_station.cooling import CoolingConfig
from atomic_model_station.feed import FeedConfig
from atomic_model_station.power import BackupConfig
from physics.materials.iapws import saturation_temperature_k
from .pwr_fuel import DistributedFuelSession,FuelGrid,identity as fuel_identity


def identity():
    return dict(fuel=fuel_identity(),cladding_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


class DistributedCladdingSession(DistributedFuelSession):
    """One cladding-temperature owner per fuel cell; inherited scalar is an alias."""
    def __init__(self,*args,**kwargs):
        self.cladding_ready=False
        super().__init__(*args,**kwargs)
        self.cladding_index=len(self.state)
        self.cladding_capacities=self.config.clad_capacity_j_k*self.fuel_weights
        self.coolant_conductances=self.plant.clad_coolant_ua*self.fuel_weights
        self.state=np.r_[self.state,np.full(self.fuel_cells,self.state[self.base_size+1])]
        self.cladding_ready=True;self.state=self._reconstruct_state(self.state)
        self._rhs(self.time,self.state);self.origin_energy=self._stored();self._admit(self.state)

    def _reconstruct_state(self,y):
        y=super()._reconstruct_state(y)
        if self.cladding_ready:y[self.base_size+1]=self.fuel_weights@y[self.cladding_index:self.cladding_index+self.fuel_cells]
        return y

    def _absolute_tolerances(self,atol):
        atol=super()._absolute_tolerances(atol)
        if self.cladding_ready:atol[self.cladding_index:self.cladding_index+self.fuel_cells]=1e-7
        return atol

    def _extend_known(self,known,modes):
        super()._extend_known(known,modes)
        if self.cladding_ready:
            known[self.base_size+1]=lambda t:0.
            modes.append(dict(name='lumped_cladding_alias',kind='derived_from_local_owners',indices=[self.base_size+1]))

    def _check_cladding(self,temperature):
        saturation=saturation_temperature_k(self.plant.network.volumes['core'].pressure_pa*1e-6)
        if (temperature.shape!=(self.fuel_cells,) or not np.isfinite(temperature).all()
                or np.any(temperature<300) or np.any(temperature>=saturation)):
            raise ValueError('Local cladding single-phase temperature domain')

    def _rhs(self,t,y):
        if not self.cladding_ready:return super()._rhs(t,y)
        full=self._reconstruct_state(np.array(y,dtype=float,copy=True))
        clad=full[self.cladding_index:self.cladding_index+self.fuel_cells]
        # Reject malformed local values even when their mean is admissible.
        if not np.isfinite(clad).all() or np.any(clad<300):raise ValueError('Local cladding temperature domain')
        rate=super()._rhs(t,full)
        self._check_cladding(clad)
        p=self.plant;temperature,_=self._local(full)
        coolant=float(p.network.volumes['core'].temperature_k)
        incoming=self.fuel_conductances*(temperature-clad)
        outgoing=self.coolant_conductances*(clad-coolant)
        fuel_rate=(np.asarray(self.fuel_terms['thermal_source_by_cell_w'])-incoming)/self.fuel_capacities
        clad_rate=(incoming-outgoing)/self.cladding_capacities
        old_fuel=rate[self.fuel_index:self.decay_index].copy()
        old_clad=float(rate[self.base_size+1])
        coolant_delta=float(outgoing.sum()-p.clad_to_coolant_w)
        coolant_energy=13+3*p.network.volume_names().index('core')+1
        rate[coolant_energy]+=coolant_delta
        rate[self.fuel_index:self.decay_index]=fuel_rate
        rate[10]=self.fuel_weights@fuel_rate
        rate[self.base_size+1]=self.fuel_weights@clad_rate
        correction=float(self.fuel_capacities@(fuel_rate-old_fuel)
            +self.cladding_capacities@clad_rate-self.config.clad_capacity_j_k*old_clad+coolant_delta)
        self.instantaneous_energy_defect_w=float(self.instantaneous_energy_defect_w+correction)
        p.fuel_to_clad_w=float(incoming.sum());p.fuel_to_coolant_w=p.fuel_to_clad_w
        p.clad_to_coolant_w=float(outgoing.sum());p.network.volumes['core'].heat_w=p.clad_to_coolant_w
        self.fuel_terms.update(fuel_to_clad_by_cell_w=incoming.tolist(),fuel_temperature_rate_k_s=fuel_rate.tolist(),
            fuel_transfer_residual_w=float(incoming.sum()-p.fuel_to_clad_w),
            coolant_scope='Local cladding storage; inherited common core coolant, no fluid channels')
        # Conductive entropy generation is nonnegative even for reversed heat flow.
        entropy=incoming*(1/clad-1/temperature)+outgoing*(1/coolant-1/clad)
        residual=self.cladding_capacities*clad_rate-incoming+outgoing
        self.cladding_terms=dict(temperature_k=clad.tolist(),temperature_rate_k_s=clad_rate.tolist(),
            capacity_j_k=self.cladding_capacities.tolist(),stored_energy_j=float(self.cladding_capacities@clad),
            fuel_to_clad_by_cell_w=incoming.tolist(),clad_to_coolant_by_cell_w=outgoing.tolist(),
            coolant_temperature_k=coolant,local_energy_residual_w=residual.tolist(),
            conductive_entropy_generation_w_k=entropy.tolist(),
            coolant_transfer_correction_w=coolant_delta,
            alias_residual_k=float(full[self.base_size+1]-self.fuel_weights@clad),
            state_ownership='Appended local cladding temperatures only; scalar cladding is a capacity-weighted alias',
            closure='Inherited constant UA, length-weighted; no local film/boiling/CHF or axial conduction',
            data_family='illustrative',validated=False)
        return np.r_[rate,clad_rate]

    def _admit(self,y):
        super()._admit(y)
        if self.cladding_ready:
            self._check_cladding(y[self.cladding_index:self.cladding_index+self.fuel_cells])
            if abs(y[self.base_size+1]-self.fuel_weights@y[self.cladding_index:self.cladding_index+self.fuel_cells])>1e-9:
                raise ValueError('Cladding alias differs from local owners')

    def snapshot(self):
        result=super().snapshot()
        if self.cladding_ready:
            result.update(model_id='pwr-local-cladding-research/v1',distributed_cladding=deepcopy(self.cladding_terms),
                distributed_cladding_source=identity(),local_cladding_connected=True)
        return result

    def checkpoint(self):
        return seal(dict(format='pwr-local-cladding/v1',source=identity(),fuel_checkpoint=super().checkpoint()))

    @classmethod
    def restore(cls,saved):
        envelope=unseal(deepcopy(saved))
        if (type(envelope) is not dict or set(envelope)!={'format','source','fuel_checkpoint'}
                or envelope['format']!='pwr-local-cladding/v1' or envelope['source']!=identity()):
            raise ValueError('Local cladding checkpoint source/format')
        data=unseal(envelope['fuel_checkpoint'])
        keys={'config','max_step_s','rtol','source','history','snapshot','cooling_config','feed_config',
            'backup_config','core_config','core_source','format','distributed_fuel_source','fuel_grid','backend_kind'}
        if type(data) is not dict or set(data)!=keys:raise ValueError('Local cladding checkpoint fields')
        source=fuel_identity()
        if (data['format']!='pwr-distributed-fuel/v1' or data['backend_kind']!='core'
                or data['distributed_fuel_source']!=source or data['core_source']!=source['backend']
                or data['source']!=source['backend']['station']):
            raise ValueError('Local cladding nested source/format')
        if type(data['history']) is not list or len(data['history'])>256:raise ValueError('Local cladding history budget')
        for name,kind in [('config',TrainConfig),('cooling_config',CoolingConfig),('feed_config',FeedConfig),('backup_config',BackupConfig),('core_config',CoreConfig)]:
            if type(data[name]) is not dict or set(data[name])!=set(kind.__dataclass_fields__):raise ValueError('Local cladding configuration fields')
        if (type(data['fuel_grid']) is not dict or set(data['fuel_grid'])!={'faces_m'}
                or type(data['fuel_grid']['faces_m']) not in (list,tuple)):
            raise ValueError('Local cladding grid fields')
        obj=cls(TrainConfig(**data['config']),cooling=CoolingConfig(**data['cooling_config']),feed=FeedConfig(**data['feed_config']),
            backup=BackupConfig(**data['backup_config']),core=CoreConfig(**data['core_config']),
            fuel_grid=FuelGrid(tuple(data['fuel_grid']['faces_m'])),max_step_s=data['max_step_s'],rtol=data['rtol'])
        for row in data['history']:
            if type(row) is not dict or set(row)!={'duration_s','patch'}:raise ValueError('Local cladding history record')
            obj.advance(**row)
        if obj.checkpoint()!=saved:raise ValueError('Local cladding replay differs')
        return obj

    def parameter_registry(self):
        result=super().parameter_registry()
        if self.cladding_ready:
            for j in range(self.fuel_cells):
                for name,value,unit in [('capacity',self.cladding_capacities[j],'J/K'),('coolant_ua',self.coolant_conductances[j],'W/K')]:
                    result['parameters'][f'distributed_cladding.{name}.{j}']=dict(value=float(value),unit=unit,
                        source='Length-weighted inherited constant-UA reduced model; no measured fit',family='illustrative',version='1',uncertainty=None)
        return result
