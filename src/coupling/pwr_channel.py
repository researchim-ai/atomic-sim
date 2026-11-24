"""Monolithic reduced PWR with local fluid inventory and internal face momentum.

The old core M/U/B are derived aliases, not additional inventories. Uses
constructed geometry and constant UA; not a staged-film or StationEnv backend.
"""
from copy import deepcopy
from dataclasses import dataclass,asdict
import hashlib
from pathlib import Path

import numpy as np
from core.status import DomainError
from atomic_model_multiphysics.provenance import seal,unseal
from physics.hydraulics.network import FlowNetwork,Junction,G
from physics.materials.iapws import saturation_temperature_k
from .conservative import ConservativeMap
from .coolant_channel import _ChannelVolume,contact,identity as channel_identity
from .pwr_cladding import DistributedCladdingSession,identity as cladding_identity
from .hydraulic_geometry import dual_geometry,combine_boundary
from . import hydraulic_geometry


def identity():
    return dict(cladding=cladding_identity(),channel=channel_identity(),
        geometry_sha256=hashlib.sha256(Path(hydraulic_geometry.__file__).read_bytes()).hexdigest(),
        adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


@dataclass(frozen=True)
class FluidGrid:
    faces_m:tuple=(0.,1.,2.,4.)
    internal_loss_coefficient:float=0.
    momentum_geometry:str='legacy'

    def __post_init__(self):
        if type(self.faces_m) is not tuple or not 2<=len(self.faces_m)<=9:raise ValueError('Immutable one to eight fluid cells required')
        ConservativeMap(self.faces_m,self.faces_m)
        k=self.internal_loss_coefficient
        if type(k) not in (int,float) or not np.isfinite(k) or not 0<=k<=100:raise ValueError('Internal loss coefficient0..100 required')
        if type(self.momentum_geometry) is not str or self.momentum_geometry not in ('legacy','complete_dual'):
            raise ValueError('Explicit legacy or complete_dual momentum geometry required')
        if len(self.faces_m)==2 and k!=0 and self.momentum_geometry=='legacy':raise ValueError('One cell has no internal resistance')


class ChannelPWRSession(DistributedCladdingSession):
    """Local fluid and fuel/decay/cladding with the existing external PWR loop."""
    def __init__(self,*args,fluid_grid=None,**kwargs):
        self.fluid_ready=False;self.fluid_grid=FluidGrid() if fluid_grid is None else fluid_grid
        if type(self.fluid_grid) is not FluidGrid:raise ValueError('FluidGrid required')
        super().__init__(*args,**kwargs)
        self.initial_pwr=super().checkpoint()
        self.fluid_exchange=ConservativeMap(self.fuel_grid.faces_m,self.fluid_grid.faces_m)
        self.fluid_cells=len(self.fluid_grid.faces_m)-1
        self.fluid_index=len(self.state);self.internal_flow_index=self.fluid_index+3*self.fluid_cells
        p=self.plant;names=p.network.volume_names();self.core_alias_index=13+3*names.index('core')
        self.hot_index=13+3*names.index('hot_leg');self.junction_offset=13+3*len(names)
        self.pump_index=self.junction_offset+p.network.junction_names().index('pump')
        self.out_index=self.junction_offset+p.network.junction_names().index('core_out')
        original=p.network.volumes['core'];self.channel_area=original.volume_m3/self.core_config.length_m
        for name in ('core','cold_leg'):
            if p.network.volumes[name].elevation_m!=0:raise ValueError('Horizontal reduced-PWR port geometry required')
        w=self.fluid_exchange.target_widths/self.core_config.length_m
        original_inventory=self.state[self.core_alias_index:self.core_alias_index+3].copy()
        inventory=np.outer(w,original_inventory)
        volumes={}
        for i,width in enumerate(self.fluid_exchange.target_widths):
            name=f'cell{i}';v=_ChannelVolume(name,self.channel_area*width,0.)
            v.seed_pressure_pa=original.pressure_pa;v.seed_temperature_k=original.temperature_k
            v.mass_kg,v.internal_energy_j,v.boron_kg=inventory[i]
            volumes[name]=v
        centres=(np.asarray(self.fluid_grid.faces_m[:-1])+self.fluid_grid.faces_m[1:])/2
        lengths=np.diff(centres);links={}
        complete=self.fluid_grid.momentum_geometry=='complete_dual'
        self.momentum_geometry=None
        if complete:
            geometry=dual_geometry(self.fluid_grid.faces_m,self.channel_area,self.fluid_grid.internal_loss_coefficient)
            channel_density=float(original.thermo.rho_kg_m3)
            boundaries={}
            for name,i in (('pump',0),('core_out',-1)):
                link=p.network.junctions[name]
                if link.flow_law!='momentum':raise ValueError('Complete dual geometry requires momentum boundary links')
                combined=combine_boundary(area_m2=link.area_m2,length_m=link.length_m,
                    loss_coefficient=link.loss_coefficient,reference_density=p.link_density_ref[name],
                    channel_area_m2=self.channel_area,channel_length_m=geometry['lengths_m'][i],
                    channel_loss=geometry['loss_coefficients'][i],channel_reference_density=channel_density)
                boundaries[name]=combined
                link.length_m=combined['length_m'];link.loss_coefficient=combined['loss_coefficient']
                p.link_density_ref[name]=combined['reference_density_kg_m3']
            self.momentum_geometry=dict(mode='complete_dual',dual=geometry,boundaries=boundaries,
                channel_reference_density_kg_m3=channel_density,
                closure='Fixed reference-density kinetic storage; combined boundary ownership, no additional flow states')
        for i,length in enumerate(lengths):
            k=geometry['loss_coefficients'][i+1] if complete else self.fluid_grid.internal_loss_coefficient*length/lengths.sum()
            links[f'internal{i}']=Junction(f'internal{i}',f'cell{i}',f'cell{i+1}',self.channel_area,float(length),float(k))
        self.channel=FlowNetwork(volumes,links)
        self.channel.refresh_properties()
        self.internal_density_ref=np.array([volumes[f'cell{i}'].thermo.rho_kg_m3 for i in range(self.fluid_cells-1)])
        if complete:self.internal_density_ref[:]=channel_density
        qin=self.state[self.pump_index];qout=self.state[self.out_index]
        flows=qin+(qout-qin)*np.asarray(self.fluid_grid.faces_m[1:-1])/self.core_config.length_m
        self.state=np.r_[self.state,inventory.ravel(),flows]
        self.fluid_ready=True;self.state=self._reconstruct_state(self.state)
        self._rhs(self.time,self.state);self.origin_energy=self._stored()
        closed=p.closed_inventories();self.origin_mass=closed['closed_mass_kg'];self.origin_boron=closed['boron_kg']
        self._admit(self.state)
        self.channel_initialization=dict(method='Length-proportional conservative lift of solved lumped PWR core; internal flows interpolated between actual port flows',
            stationary_channel_solve=False,volume_m3=float(original.volume_m3),flow_area_m2=float(self.channel_area),
            alias_lift_defect=(inventory.sum(axis=0)-original_inventory).tolist())

    def _fluid(self,y):return y[self.fluid_index:self.internal_flow_index].reshape(self.fluid_cells,3)

    def _stored(self):
        result=super()._stored()
        if self.fluid_ready:result+=self._internal_kinetic(self.state)
        return float(result)

    def _internal_kinetic(self,y):
        return float(sum(.5*link.length_m/link.area_m2*y[self.internal_flow_index+i]**2/self.internal_density_ref[i]
            for i,link in enumerate(self.channel.junctions.values())))

    def _reconstruct_state(self,y):
        y=super()._reconstruct_state(y)
        if self.fluid_ready:y[self.core_alias_index:self.core_alias_index+3]=self._fluid(y).sum(axis=0)
        return y

    def _sync_fluid(self,y):
        if len(y)!=self.internal_flow_index+self.fluid_cells-1 or not np.isfinite(y).all():raise DomainError('Channel PWR state shape/finite domain')
        fluid=self._fluid(y)
        if np.any(fluid[:,0]<=0) or np.any(fluid[:,2]<0) or np.any(fluid[:,2]>.01*fluid[:,0]):raise DomainError('Local fluid mass/boron domain')
        if min(y[self.pump_index],y[self.out_index],*y[self.internal_flow_index:])<0:raise DomainError('Channel PWR forward flow domain; no reverse pump model')
        self.channel.unpack(y[self.fluid_index:]);self.channel.refresh_properties()
        for v in self.channel.volumes.values():
            if v.thermo.region!=1 or not 1e6<=v.pressure_pa<=20e6 or not 300<=v.temperature_k<=600:
                raise DomainError('Local PWR coolant requires liquid region1,1..20 MPa,300..600 K')

    def _check_cladding(self,temperature):
        if not self.fluid_ready:return super()._check_cladding(temperature)
        saturation=np.array([saturation_temperature_k(v.pressure_pa*1e-6) for v in self.channel.volumes.values()])
        if np.any(temperature<300) or not np.isfinite(temperature).all() or np.any((self.fluid_exchange.overlap>0)&(temperature[None,:]>=saturation[:,None])):
            raise DomainError('Local PWR cladding saturation domain')

    def _absolute_tolerances(self,atol):
        atol=super()._absolute_tolerances(atol)
        if self.fluid_ready:
            atol[self.fluid_index:self.internal_flow_index:3]=1e-7
            atol[self.fluid_index+1:self.internal_flow_index:3]=.001
            atol[self.fluid_index+2:self.internal_flow_index:3]=1e-10
            atol[self.internal_flow_index:]=1e-6
        return atol

    def _extend_known(self,known,modes):
        super()._extend_known(known,modes)
        if self.fluid_ready:
            for i in range(self.core_alias_index,self.core_alias_index+3):known[i]=lambda t:0.
            modes.append(dict(name='lumped_core_fluid_aliases',kind='derived_from_local_M_U_B',indices=list(range(self.core_alias_index,self.core_alias_index+3))))

    def _rhs(self,t,y):
        if not self.fluid_ready:return super()._rhs(t,y)
        full=self._reconstruct_state(np.array(y,dtype=float,copy=True));self._sync_fluid(full)
        rate=super()._rhs(t,full)
        p=self.plant;bulk=p.network.volumes['core'];cold=p.network.volumes['cold_leg']
        local=list(self.channel.volumes.values());first,last=local[0],local[-1]
        qin=float(full[self.pump_index]);qout=float(full[self.out_index])
        clad=full[self.cladding_index:self.cladding_index+self.fuel_cells]
        thermal=contact(self.fluid_exchange,clad,[v.temperature_k for v in local],self.coolant_conductances)
        for v,q in zip(local,thermal['coolant_heat_in_w']):v.heat_w=float(q)
        local_rate=self.channel.derivatives(full[self.fluid_index:])
        cells=local_rate[:3*self.fluid_cells].reshape(self.fluid_cells,3)
        pump=p.network.junctions['pump'];out=p.network.junctions['core_out']
        pump_heat=qin*pump.pump_pressure_pa/cold.thermo.rho_kg_m3
        inlet=np.array([qin,qin*pump.delivery_enthalpy_j_kg+pump_heat,qin*cold.boron_fraction])
        outlet=np.array([qout,qout*last.thermo.h_j_kg,qout*last.boron_fraction])
        cells[0]+=inlet;cells[-1]-=outlet
        old_core=rate[self.core_alias_index:self.core_alias_index+3].copy()
        old_clad=rate[self.cladding_index:self.cladding_index+self.fuel_cells].copy()
        incoming=np.asarray(self.cladding_terms['fuel_to_clad_by_cell_w'])
        clad_rate=(incoming-thermal['wall_heat_out_w'])/self.cladding_capacities
        hot_correction=np.array([0.,qout*(last.thermo.h_j_kg-bulk.thermo.h_j_kg),qout*(last.boron_fraction-bulk.boron_fraction)])
        # Existing boundary momentum owners see actual endpoint pressures and
        # outlet donor density; no duplicated boundary flow or inertia state.
        old_pump=float(rate[self.pump_index]);old_out=float(rate[self.out_index])
        hot=p.network.volumes['hot_leg'];dz=hot.elevation_m-bulk.elevation_m
        rate[self.pump_index]+=pump.area_m2/pump.length_m*(bulk.pressure_pa-first.pressure_pa)
        rate[self.out_index]+=out.area_m2/out.length_m*(last.pressure_pa-bulk.pressure_pa
            -.5*(last.thermo.rho_kg_m3-bulk.thermo.rho_kg_m3)*G*dz
            -out.friction_dp(last.thermo.rho_kg_m3)+out.friction_dp(bulk.thermo.rho_kg_m3))
        pump_factor=pump.length_m/pump.area_m2*qin/p.link_density_ref['pump']
        out_factor=out.length_m/out.area_m2*qout/p.link_density_ref['core_out']
        pump_work=pump_factor*rate[self.pump_index];out_work=out_factor*rate[self.out_index]
        delta_pump=pump_factor*(rate[self.pump_index]-old_pump)
        delta_out=out_factor*(rate[self.out_index]-old_out)
        cells[0,1]-=.5*pump_work;cells[-1,1]-=.5*out_work
        internal_work=[]
        for i,link in enumerate(self.channel.junctions.values()):
            work=link.length_m/link.area_m2*full[self.internal_flow_index+i]*local_rate[3*self.fluid_cells+i]/self.internal_density_ref[i]
            internal_work.append(float(work));cells[i,1]-=.5*work;cells[i+1,1]-=.5*work
        # Match the parent's reference-density hydraulic kinetic closure:
        # changed boundary acceleration changes both kinetic work and its
        # compensating thermal exchange. Internal faces get their own storage.
        cold_index=13+3*p.network.volume_names().index('cold_leg')
        rate[cold_index+1]-=.5*delta_pump
        hot_correction[1]-=.5*delta_out
        rate[self.hot_index:self.hot_index+3]+=hot_correction
        rate[self.core_alias_index:self.core_alias_index+3]=cells.sum(axis=0)
        rate[self.cladding_index:self.cladding_index+self.fuel_cells]=clad_rate
        rate[self.base_size+1]=self.fuel_weights@clad_rate
        correction=float(cells[:,1].sum()-old_core[1]+hot_correction[1]-.5*delta_pump
            +self.cladding_capacities@(clad_rate-old_clad)+delta_pump+delta_out+sum(internal_work))
        self.instantaneous_energy_defect_w=float(self.instantaneous_energy_defect_w+correction)
        p.clad_to_coolant_w=float(thermal['coolant_heat_in_w'].sum())
        bulk.heat_w=p.clad_to_coolant_w
        out.delivery_enthalpy_j_kg=last.thermo.h_j_kg-G*dz
        fuel_temperature=self._local(full)[0]
        self.cladding_terms.update(temperature_rate_k_s=clad_rate.tolist(),clad_to_coolant_by_cell_w=thermal['wall_heat_out_w'].tolist(),
            coolant_temperature_k=None,coolant_temperature_by_cell_k=[v.temperature_k for v in local],
            conductive_entropy_generation_w_k=(incoming*(1/clad-1/fuel_temperature)+thermal['entropy_w_k'].sum(axis=0)).tolist(),
            local_energy_residual_w=(self.cladding_capacities*clad_rate-incoming+thermal['wall_heat_out_w']).tolist())
        self.fuel_terms['coolant_scope']='Owned local fluid M/U/B and internal face momentum; common legacy core is an alias'
        kinetic_work=.5*(pump_work+out_work)+sum(internal_work)
        expected=inlet-outlet+np.array([0.,thermal['coolant_heat_in_w'].sum()-kinetic_work,0.])
        acceleration=np.r_[rate[self.pump_index],local_rate[3*self.fluid_cells:],rate[self.out_index]]
        force=[cold.pressure_pa-first.pressure_pa+pump.pump_pressure_pa-pump.friction_dp(cold.thermo.rho_kg_m3)]
        force.extend(local[i].pressure_pa-local[i+1].pressure_pa-link.friction_dp(local[i].thermo.rho_kg_m3)
                     for i,link in enumerate(self.channel.junctions.values()))
        force.append(last.pressure_pa-hot.pressure_pa-.5*(last.thermo.rho_kg_m3+hot.thermo.rho_kg_m3)*G*dz-out.friction_dp(last.thermo.rho_kg_m3))
        self.fluid_terms=dict(inventory_by_cell=self._fluid(full).tolist(),volume_m3=[v.volume_m3 for v in local],
            pressure_pa=[v.pressure_pa for v in local],temperature_k=[v.temperature_k for v in local],
            density_kg_m3=[v.thermo.rho_kg_m3 for v in local],boron_fraction=[v.boron_fraction for v in local],
            face_flow_kg_s=np.r_[qin,full[self.internal_flow_index:self.internal_flow_index+self.fluid_cells-1],qout].tolist(),
            face_acceleration_kg_s2=acceleration.tolist(),face_driving_pressure_pa=force,
            inventory_rate_by_cell=cells.tolist(),port_inlet_rates=inlet.tolist(),port_outlet_rates=outlet.tolist(),
            local_boundary_balance_residual=(cells.sum(axis=0)-expected).tolist(),
            alias_rate_residual=(rate[self.core_alias_index:self.core_alias_index+3]-cells.sum(axis=0)).tolist(),
            pair_heat_w=thermal['pair_heat_w'].tolist(),coolant_heat_by_cell_w=thermal['coolant_heat_in_w'].tolist(),
            contact_entropy_w_k=float(thermal['entropy_w_k'].sum()),hot_leg_correction=hot_correction.tolist(),
            internal_kinetic_rate_w=internal_work,boundary_kinetic_rates_w=[float(pump_work),float(out_work)],
            internal_loss_coefficient=self.fluid_grid.internal_loss_coefficient,
            feedback_scope='Existing local fuel Doppler; inherited cold-leg density and aggregate core boron feedback',
            state_ownership='Local M/U/B only; legacy core aliases not integrated; original pump/outlet flows and appended internal flows each owned once',
            geometry_scope='Equivalent area from inherited core volume/span; original boundary links retained, optional additional internal loss',
            data_family='illustrative',validated=False)
        return np.r_[rate,local_rate]

    def _admit(self,y):
        if self.fluid_ready:
            self._sync_fluid(y)
            aliases=y[self.core_alias_index:self.core_alias_index+3];owned=self._fluid(y).sum(axis=0)
            if np.max(abs(aliases-owned)/np.maximum(abs(owned),1.))>1e-12:raise DomainError('Core fluid aliases differ from local owners')
        super()._admit(y)

    def snapshot(self):
        result=super().snapshot()
        if self.fluid_ready:
            result.update(model_id='pwr-local-fluid-research/v1',distributed_fluid=deepcopy(self.fluid_terms),
                distributed_fluid_source=identity(),distributed_fluid_channels=True,
                coupled_state=self.state.tolist(),internal_kinetic_energy_j=self._internal_kinetic(self.state),
                channel_initialization=deepcopy(getattr(self,'channel_initialization',None)),
                fluid_grid=dict(faces_m=list(self.fluid_grid.faces_m),internal_loss_coefficient=self.fluid_grid.internal_loss_coefficient,
                                momentum_geometry=self.fluid_grid.momentum_geometry),
                momentum_geometry=deepcopy(self.momentum_geometry))
        return result

    def checkpoint(self):
        if not self.fluid_ready:return super().checkpoint()
        grid=asdict(self.fluid_grid);grid['faces_m']=list(grid['faces_m'])
        return seal(dict(format='pwr-local-fluid/v1',source=identity(),fluid_grid=grid,initial_pwr=deepcopy(self.initial_pwr),
            history=list(deepcopy(self.history)),snapshot=self.snapshot()))

    @classmethod
    def restore(cls,saved):
        p=unseal(deepcopy(saved))
        if type(p) is not dict or set(p)!={'format','source','fluid_grid','initial_pwr','history','snapshot'} or p['format']!='pwr-local-fluid/v1' or p['source']!=identity():raise ValueError('PWR channel checkpoint source/format')
        if type(p['history']) is not list or len(p['history'])>256:raise ValueError('PWR channel history budget')
        if type(p['fluid_grid']) is not dict or set(p['fluid_grid'])!=set(FluidGrid.__dataclass_fields__) or type(p['fluid_grid']['faces_m']) not in (tuple,list):raise ValueError('PWR channel grid fields')
        initial=DistributedCladdingSession.restore(p['initial_pwr'])
        if initial.time!=0 or initial.history:raise ValueError('PWR channel initial checkpoint must be unadvanced')
        grid=FluidGrid(**dict(p['fluid_grid'],faces_m=tuple(p['fluid_grid']['faces_m'])))
        obj=cls(initial.config,cooling=initial.cooling_config,feed=initial.feed_config,backup=initial.backup_config,
            core=initial.core_config,fuel_grid=initial.fuel_grid,max_step_s=initial.max_step_s,rtol=initial.rtol,fluid_grid=grid)
        for row in p['history']:
            if type(row) is not dict or set(row)!={'duration_s','patch'}:raise ValueError('PWR channel history fields')
            obj.advance(**row)
        if obj.checkpoint()!=saved:raise ValueError('PWR channel replay differs')
        return obj

    def parameter_registry(self):
        result=super().parameter_registry()
        if self.fluid_ready:
            for name,value,unit in [('flow_area',self.channel_area,'m2'),('internal_loss',self.fluid_grid.internal_loss_coefficient,'1')]:
                result['parameters']['distributed_fluid.'+name]=dict(value=float(value),unit=unit,
                    source='Constructed horizontal equivalent channel; not measured geometry or calibrated loss',family='illustrative',version='1',uncertainty=None)
        return result
