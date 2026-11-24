"""Opt-in PWR distributed fuel and local decay-heat inventories.

The existing coolant/core volume and cladding are still lumped. Synthetic
one-group neutron data and the inherited teaching Doppler law remain in use.
This adapter is not a plant reconstruction or a distributed fluid-channel model.
"""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
from atomic_model_station import PWRCoreSession,PWRStagedSession,CoreConfig,TrainConfig
from atomic_model_station.cooling import CoolingConfig
from atomic_model_station.feed import FeedConfig
from atomic_model_station.power import BackupConfig
from atomic_model_station.core import core_signature
from atomic_model_station.primary import PrimaryConfig
from atomic_model_station.film import FilmConfig
from atomic_model_station.steam import SteamConfig
from atomic_model_station.stages import StageConfig
from atomic_model_multiphysics.provenance import seal,unseal
from physics.depletion import chains
from plants.vver import doppler_reactivity
from .conservative import ConservativeMap,vector


def identity():
    from . import conservative
    return dict(backend=core_signature(),adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        exchange_sha256=hashlib.sha256(Path(conservative.__file__).read_bytes()).hexdigest(),
        decay_sha256=hashlib.sha256(Path(chains.__file__).read_bytes()).hexdigest())


@dataclass(frozen=True)
class FuelGrid:
    faces_m:tuple=(0.,.7,2.2,4.)

    def __post_init__(self):
        if type(self.faces_m) is not tuple:raise ValueError('Immutable fuel mesh required')
        if not 2<=len(self.faces_m)<=17:raise ValueError('One to sixteen fuel cells required')
        ConservativeMap([self.faces_m[0],self.faces_m[-1]],self.faces_m)


def local_heat(decay,fission_w,stored_j):
    """Return local thermal power and the three owned reservoir derivatives."""
    q=vector(fission_w);raw=np.asarray(stored_j)
    if raw.dtype.kind not in 'fiu':raise ValueError('Real local decay inventories required')
    h=np.array(raw,dtype=float,copy=True)
    if np.any(q<0) or h.shape!=(3,len(q)) or not np.isfinite(h).all() or np.any(h<0):
        raise ValueError('Nonnegative local fission and 3xM decay reservoirs required')
    if len(decay.fractions)!=3:raise ValueError('Inherited three-family heat model required')
    release=decay.decay_constants[:,None]*h
    heat=(1-decay.decay_fraction)*q+release.sum(axis=0)
    rate=decay.fractions[:,None]*q[None,:]-release
    return heat,rate


class _DistributedFuel:
    """Integrate local fuel/decay states; old scalar slots are derived aliases."""
    def __init__(self,config=None,*,cooling=None,feed=None,backup=None,core=None,fuel_grid=None,max_step_s=.1,rtol=1e-7,**backend_options):
        self.fuel_ready=False
        self.fuel_grid=FuelGrid() if fuel_grid is None else fuel_grid
        if type(self.fuel_grid) is not FuelGrid:raise ValueError('FuelGrid required')
        core=CoreConfig(kind='diffusion',cells=4) if core is None else core
        if type(core) is not CoreConfig or core.kind!='diffusion':raise ValueError('Spatial diffusion core required')
        super().__init__(config,cooling=cooling,feed=feed,backup=backup,core=core,max_step_s=max_step_s,rtol=rtol,**backend_options)
        if self.backend_kind=='staged' and self.core_config.length_m!=self.film_config.heated_length_m:
            raise ValueError('Neutron/fuel span and inherited heated length must match')
        self.exchange=ConservativeMap(self.neutron_core.model.faces,self.fuel_grid.faces_m)
        self.fuel_index=len(self.state);self.fuel_cells=len(self.fuel_grid.faces_m)-1
        self.decay_index=self.fuel_index+self.fuel_cells;self.exchange_index=self.decay_index+3*self.fuel_cells
        self.fuel_weights=self.exchange.target_widths/self.core_config.length_m
        self.fuel_capacities=self.plant.fuel_capacity*self.fuel_weights
        self.fuel_conductances=self.plant.fuel_clad_ua*self.fuel_weights
        initial_heat=np.outer(self.state[7:10],self.fuel_weights)
        self.state=np.r_[self.state,np.full(self.fuel_cells,self.state[10]),initial_heat.ravel(),0.]
        self.initial_decay_j=float(initial_heat.sum())
        self.fuel_ready=True;self.state=self._reconstruct_state(self.state)
        self._rhs(self.time,self.state);self.origin_energy=self._stored();self._admit(self.state)

    def _local(self,y):
        temperature=y[self.fuel_index:self.decay_index]
        heat=y[self.decay_index:self.exchange_index].reshape(3,self.fuel_cells)
        return temperature,heat

    def _reconstruct_state(self,y):
        y=super()._reconstruct_state(y)
        if self.fuel_ready:
            temperature,heat=self._local(y)
            y[10]=self.fuel_weights@temperature;y[7:10]=heat.sum(axis=1)
        return y

    def _stored(self):
        total=super()._stored()
        return float(total+self._local(self.state)[1].sum()) if self.fuel_ready else total

    def _absolute_tolerances(self,atol):
        atol=super()._absolute_tolerances(atol)
        if self.fuel_ready:
            atol[self.fuel_index:self.decay_index]=1e-7
            atol[self.decay_index:self.exchange_index+1]=.01
        return atol

    def _extend_known(self,known,modes):
        super()._extend_known(known,modes)
        if self.fuel_ready:
            for index in range(7,11):known[index]=lambda t:0.
            modes.append(dict(name='lumped_fuel_and_decay_aliases',kind='derived_from_local_owners',indices=list(range(7,11))))

    def _passive_column(self,index):
        return super()._passive_column(index) or (self.fuel_ready and index==self.exchange_index)

    def _spatial_reactivity(self,projected_temperature):
        """Default local Doppler with inherited bulk moderator contributions."""
        p=self.plant
        common=p.reactivity()-doppler_reactivity(p.fuel_temperature,p.t_fuel_ref,p.doppler_coef)
        return np.array([common+doppler_reactivity(v,p.t_fuel_ref,p.doppler_coef) for v in projected_temperature])+self.neutron_core.bias

    def _rhs(self,t,y):
        if not self.fuel_ready:return super()._rhs(t,y)
        full=self._reconstruct_state(np.array(y,dtype=float,copy=True))
        temperature,stored=self._local(full)
        if not np.isfinite(temperature).all() or np.any(temperature<300) or np.any(temperature>2800) or not np.isfinite(stored).all() or np.any(stored<0):
            raise ValueError('Local thermal state outside domain')
        rate=super()._rhs(t,full)
        p=self.plant;n=self.neutron_core.cells;owned=full[self.core_index:self.core_index+self.neutron_core.size]
        projected_temperature=self.exchange.feedback_temperature(temperature)
        rho=self._spatial_reactivity(projected_temperature)
        if np.any(1-rho<=0):raise ValueError('Positive local neutron absorption required')
        neutron=self.neutron_core.operator@owned
        neutron[:n]+=rho*owned[:n]/p.kinetics.generation_time_s
        rate[:7]=self.neutron_core.project(neutron)
        rate[self.core_index:self.core_index+self.neutron_core.size]=neutron
        fission=self.neutron_core.fission_by_cell(owned);mapped=self.exchange.powers(fission)
        heat,heat_rate=local_heat(p.decay,mapped,stored)
        clad=float(full[self.base_size+1]);transfer=self.fuel_conductances*(temperature-clad)
        temperature_rate=(heat-transfer)/self.fuel_capacities
        old_fuel_rate=float(rate[10]);rate[10]=self.fuel_weights@temperature_rate;rate[7:10]=heat_rate.sum(axis=1)
        # The inherited global ledger uses heat entering the thermal plant.
        # Add owned nuclear reservoirs and shift that external boundary to fission.
        storage_rate=float(fission.sum()-heat.sum())
        rate[self.ledger_index]+=storage_rate
        rate[self.ledger_index+3]+=abs(float(fission.sum()))-abs(float(heat.sum()))
        pwr_before=self.powers['external_net_w'];self.powers['external_net_w']=float(rate[self.ledger_index])
        self.powers['nuclear_fission_w']=float(fission.sum())
        self.instantaneous_energy_defect_w=float(self.instantaneous_energy_defect_w+p.fuel_capacity*(rate[10]-old_fuel_rate)+float(heat_rate.sum())-storage_rate)
        weights=np.r_[self.neutron_core.model.width/self.neutron_core.speed,np.tile(self.neutron_core.model.width,6)]
        actual=float(weights@neutron)
        expected=float(np.sum(self.neutron_core.config.nu_fission_per_m*rho*owned[:n]*self.neutron_core.model.width))
        components=abs(self.neutron_core.operator)@abs(owned)
        components[:n]+=abs(rho*owned[:n]/p.kinetics.generation_time_s)
        scale=max(float(abs(weights)@components),abs(expected),1e-30)
        self.core_terms.update(thermal_projection='Distributed fuel and local decay heat; common existing cladding/coolant',
            neutron_precursor_inventory=dict(rate=actual,expected_rate=expected,absolute_defect=abs(actual-expected),
                balance_scale=scale,relative_defect=abs(actual-expected)/scale,
                normalization='Absolute operator and local-reactivity contributions; not the near-zero net stationary rate'),
            local_reactivity=rho.tolist(),local_fuel_feedback_k=projected_temperature.tolist())
        self.fuel_terms=dict(temperature_k=temperature.tolist(),decay_stored_j=stored.tolist(),
            fission_by_fuel_cell_w=mapped.tolist(),thermal_source_by_cell_w=heat.tolist(),fuel_to_clad_by_cell_w=transfer.tolist(),
            local_heat_storage_rate_w=heat_rate.tolist(),fuel_temperature_rate_k_s=temperature_rate.tolist(),
            fission_w=float(fission.sum()),thermal_w=float(heat.sum()),decay_power_w=float((p.decay.decay_constants[:,None]*stored).sum()),
            decay_stored_total_j=float(stored.sum()),nuclear_storage_rate_w=float(heat_rate.sum()),
            nuclear_storage_identity_residual_w=float(fission.sum()-heat.sum()-heat_rate.sum()),
            fuel_alias_rate_residual_k_s=float(rate[10]-self.fuel_weights@temperature_rate),
            fuel_transfer_residual_w=float(transfer.sum()-p.fuel_to_clad_w),
            extended_boundary_shift_w=float(rate[self.ledger_index]-pwr_before),
            decay_exchange_residual_j=float(stored.sum()-self.initial_decay_j-full[self.exchange_index]),
            decay_storage_exchange_j=float(full[self.exchange_index]),
            exchange=self.exchange.audit(fission,temperature),
            state_ownership='Local fuel temperatures and 3xM decay reservoirs integrated once; legacy slots7..10 derived only',
            initial_decay_distribution='Constructed length-proportional lift of inherited reservoirs; not measured spatial history',
            global_boundary='Fission input; local nuclear heat reservoirs included once in stored energy',
            coolant_scope='Inherited common core coolant volume and common cladding; no distributed fluid channels',
            data_family='illustrative',validated=False,plant_referenced=False)
        return np.r_[rate,temperature_rate,heat_rate.ravel(),storage_rate]

    def _admit(self,y):
        if self.fuel_ready:
            temperature,stored=self._local(y)
            if not np.isfinite(temperature).all() or np.any(temperature<300) or np.any(temperature>2800) or np.any(stored<0) or not np.isfinite(stored).all():
                raise ValueError('Local fuel temperature/decay reservoir domain')
            if np.max(abs(y[7:10]-stored.sum(axis=1))/np.maximum(abs(y[7:10]),1.))>1e-12 or abs(y[10]-self.fuel_weights@temperature)>1e-9:
                raise ValueError('Lumped aliases differ from local owners')
        super()._admit(y)

    def snapshot(self):
        result=super().snapshot()
        if self.fuel_ready:
            result.update(model_id='pwr-distributed-fuel-research/v1',distributed_fuel=deepcopy(self.fuel_terms),
                distributed_fuel_source=identity(),fuel_grid=dict(faces_m=list(self.fuel_grid.faces_m)),
                distributed_fuel_backend=self.backend_kind,local_decay_heat_connected=True,distributed_fluid_channels=False)
        return result

    def checkpoint(self):
        data=unseal(super().checkpoint())
        if self.fuel_ready:
            data.update(format='pwr-distributed-fuel/v1',distributed_fuel_source=identity(),fuel_grid=dict(faces_m=list(self.fuel_grid.faces_m)),backend_kind=self.backend_kind)
        return seal(data)

    @classmethod
    def restore(cls,saved):
        data=unseal(deepcopy(saved))
        keys={'config','max_step_s','rtol','source','history','snapshot','cooling_config','feed_config',
            'backup_config','core_config','core_source','format','distributed_fuel_source','fuel_grid','backend_kind'}
        if cls.backend_kind=='staged':keys|={'primary_config','film_config','film_source','steam_config','steam_source','stage_config','stages_source'}
        if type(data) is not dict or set(data)!=keys:raise ValueError('Distributed fuel checkpoint fields')
        source=identity()
        if (data['format']!='pwr-distributed-fuel/v1' or data['backend_kind']!=cls.backend_kind or data['distributed_fuel_source']!=source or
                data['source']!=source['backend']['station'] or data['core_source']!=source['backend']):
            raise ValueError('Distributed fuel checkpoint source/format')
        if type(data.get('history')) is not list or len(data['history'])>256:raise ValueError('Distributed fuel checkpoint history budget')
        config_types=[('config',TrainConfig),('cooling_config',CoolingConfig),('feed_config',FeedConfig),('backup_config',BackupConfig),('core_config',CoreConfig)]
        if cls.backend_kind=='staged':config_types += [('primary_config',PrimaryConfig),('film_config',FilmConfig),('steam_config',SteamConfig),('stage_config',StageConfig)]
        for name,kind in config_types:
            if type(data[name]) is not dict or set(data[name])!=set(kind.__dataclass_fields__):raise ValueError('Distributed fuel configuration fields')
        if (type(data['fuel_grid']) is not dict or set(data['fuel_grid'])!={'faces_m'} or
                type(data['fuel_grid']['faces_m']) not in (tuple,list)):
            raise ValueError('Distributed fuel checkpoint geometry')
        options={}
        if cls.backend_kind=='staged':options=dict(primary=PrimaryConfig(**data['primary_config']),film=FilmConfig(**data['film_config']),steam=SteamConfig(**data['steam_config']),stages=StageConfig(**data['stage_config']))
        obj=cls(TrainConfig(**data['config']),cooling=CoolingConfig(**data['cooling_config']),feed=FeedConfig(**data['feed_config']),
            backup=BackupConfig(**data['backup_config']),core=CoreConfig(**data['core_config']),
            fuel_grid=FuelGrid(tuple(data['fuel_grid']['faces_m'])),max_step_s=data['max_step_s'],rtol=data['rtol'],**options)
        for row in data['history']:
            if type(row) is not dict or set(row)!={'duration_s','patch'}:raise ValueError('Distributed fuel checkpoint history record')
            obj.advance(**row)
        if obj.checkpoint()!=saved:raise ValueError('Distributed fuel checkpoint replay differs')
        return obj

    def parameter_registry(self):
        result=super().parameter_registry()
        if self.fuel_ready:
            for j,(capacity,conductance,width) in enumerate(zip(self.fuel_capacities,self.fuel_conductances,self.exchange.target_widths)):
                for name,value,unit in [('capacity',capacity,'J/K'),('fuel_clad_ua',conductance,'W/K'),('width',width,'m')]:
                    result['parameters'][f'distributed_fuel.{name}.{j}']=dict(value=float(value),unit=unit,
                        source='Length-weighted inherited fuel sizing; constructed research discretization',
                        family='illustrative',version='1',uncertainty=None)
        return result


class DistributedFuelSession(_DistributedFuel,PWRCoreSession):
    """Distributed fuel on the reduced PWRCoreSession backend."""
    backend_kind='core'


class StagedDistributedFuelSession(_DistributedFuel,PWRStagedSession):
    """Distributed fuel with inherited film, header, separator and reheat."""
    backend_kind='staged'
