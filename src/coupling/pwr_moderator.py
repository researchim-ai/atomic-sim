"""Anchored local density/boron feedback around a constructed hot reference.

Uses inherited illustrative incremental coefficients, not evaluated local
cross sections. The reference is frozen at the unperturbed stationary start.
"""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
from core.status import DomainError,SolverError
from atomic_model_multiphysics.provenance import seal,unseal
from .conservative import ConservativeMap,vector
from .pwr_channel import ChannelPWRSession
from .pwr_stationary import StationaryChannelPWRSession,StationaryConfig,_Problem,identity as stationary_identity


def identity():
    return dict(stationary=stationary_identity(),moderator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def moderator_fields(exchange,mass_kg,boron_kg,volume_m3):
    """Volume-average density and water-mass-weighted ppm on neutron cells."""
    n=len(exchange.target_widths)
    mass=vector(mass_kg,n,positive=True);boron=vector(boron_kg,n);volume=vector(volume_m3,n,positive=True)
    if np.any(boron<0) or np.any(boron>.01*mass):raise DomainError('Local dilute boron mass domain')
    area=volume/exchange.target_widths
    if not np.isfinite(area).all() or not np.allclose(area,area[0],rtol=1e-12,atol=0.):raise DomainError('Uniform flow area required for moderator exchange')
    try:
        with np.errstate(over='raise',invalid='raise',divide='raise'):
            density=exchange.feedback_matrix@(mass/volume)
            boron_density=exchange.feedback_matrix@(boron/volume)
            ppm=1e6*boron_density/density
            mapped_mass=area[0]*exchange.source_widths*density
            mapped_boron=area[0]*exchange.source_widths*boron_density
    except FloatingPointError as exc:raise DomainError('Moderator exchange exceeds finite range') from exc
    if not all(np.isfinite(v).all() for v in (density,ppm,mapped_mass,mapped_boron)):raise DomainError('Nonfinite moderator field')
    return dict(density_kg_m3=density,boron_ppm=ppm,mass_by_neutron_cell_kg=mapped_mass,boron_by_neutron_cell_kg=mapped_boron,
        mass_relative_defect=float(abs(mapped_mass.sum()-mass.sum())/max(float(mass.sum()),1.)),
        boron_relative_defect=float(abs(mapped_boron.sum()-boron.sum())/max(float(boron.sum()),1e-12)))


@dataclass(frozen=True)
class _Reference:
    density_kg_m3:tuple
    boron_ppm:tuple
    cold_density_kg_m3:float
    bulk_boron_ppm:float

    def json(self):
        return dict(density_kg_m3=list(self.density_kg_m3),boron_ppm=list(self.boron_ppm),
            cold_density_kg_m3=self.cold_density_kg_m3,bulk_boron_ppm=self.bulk_boron_ppm,
            source='Constructed stationary reference; frozen before any commands',measured=False)


class LocalModeratorPWRSession(StationaryChannelPWRSession):
    """Replace bulk moderator changes with local changes from a fixed hot state."""
    def __init__(self,*args,**kwargs):
        self.moderator_ready=False
        super().__init__(*args,**kwargs)
        self.initial_stationary=super().checkpoint()
        before=self._rhs(0.,self.state)
        self.moderator_exchange=ConservativeMap(self.neutron_core.model.faces,self.fluid_grid.faces_m)
        fields=self._moderator_fields(self.state)
        self.moderator_reference=_Reference(tuple(fields['density_kg_m3']),tuple(fields['boron_ppm']),
            float(self.plant.network.volumes['cold_leg'].thermo.rho_kg_m3),float(self.plant.network.volumes['core'].boron_fraction*1e6))
        self.moderator_ready=True;after=self._rhs(0.,self.state)
        difference=float(np.max(abs(after-before)))
        if difference>1e-10:raise SolverError('Local moderator reference changes its own baseline')
        # Audit the new stationary Jacobian with references held fixed; the
        # inherited optimizer's rank belongs to the pre-feedback baseline.
        problem=_Problem(self,self.stationary_config);z=problem.start;residual=problem.residual(z);columns=[]
        for i in range(len(z)):
            step=1e-6*max(abs(float(z[i])),1.);up=z.copy();down=z.copy();up[i]+=step;down[i]-=step
            columns.append((problem.residual(up)-problem.residual(down))/(2*step))
        jac=np.column_stack(columns);jac/=np.maximum(np.linalg.norm(jac,axis=1),1e-30)[:,None]
        jac/=np.maximum(np.linalg.norm(jac,axis=0),1e-30)[None,:];singular=np.linalg.svd(jac,compute_uv=False)
        rank=int(sum(singular>singular[0]*1e-9))
        if rank!=len(z) or np.max(abs(residual))>1:raise SolverError('Frozen-reference moderator stationary audit failed')
        self.local_stationary_report=dict(baseline_rhs_max_difference=difference,constrained_rank=rank,constrained_unknowns=len(z),
            max_selected_tolerance_ratio=float(np.max(abs(residual))),equilibrated_singular_values=singular.tolist(),
            reference_frozen_during_differentiation=True,parameters_fit=False)
        self._rhs(0.,self.state);self._admit(self.state)

    def _moderator_fields(self,y):
        fluid=self._fluid(y)
        return moderator_fields(self.moderator_exchange,fluid[:,0],fluid[:,2],[v.volume_m3 for v in self.channel.volumes.values()])

    def _spatial_reactivity(self,projected_temperature):
        base=super()._spatial_reactivity(projected_temperature)
        if not self.moderator_ready:return base
        # ChannelPWRSession has synchronized the actual local M/U/B before
        # entering the fuel/neutron RHS. No old aggregate density is mapped.
        fluid=self.channel.pack()[:3*self.fluid_cells].reshape(self.fluid_cells,3)
        fields=moderator_fields(self.moderator_exchange,fluid[:,0],fluid[:,2],[v.volume_m3 for v in self.channel.volumes.values()])
        p=self.plant;ref=self.moderator_reference
        cold=float(p.network.volumes['cold_leg'].thermo.rho_kg_m3);bulk=float(p.network.volumes['core'].boron_fraction*1e6)
        density_delta=p.density_coef*((fields['density_kg_m3']-ref.density_kg_m3)-(cold-ref.cold_density_kg_m3))
        boron_delta=p.boron_coef*((fields['boron_ppm']-ref.boron_ppm)-(bulk-ref.bulk_boron_ppm))
        self.moderator_terms=dict(density_kg_m3=fields['density_kg_m3'].tolist(),boron_ppm=fields['boron_ppm'].tolist(),
            density_reactivity_correction=density_delta.tolist(),boron_reactivity_correction=boron_delta.tolist(),
            density_coefficient_per_kg_m3=float(p.density_coef),boron_coefficient_per_ppm=float(p.boron_coef),
            mass_relative_defect=fields['mass_relative_defect'],boron_relative_defect=fields['boron_relative_defect'],
            reference=ref.json(),law='Inherited baseline plus local moderator changes minus inherited bulk moderator changes',
            concentration_average='Water-mass weighted from mapped boron density and mapped water density',
            coefficients_family='illustrative incremental coefficients; no evaluated local nuclear data',validated=False)
        return base+density_delta+boron_delta

    def _rhs(self,t,y):
        rate=super()._rhs(t,y)
        if self.moderator_ready:
            self.fluid_terms['feedback_scope']='Local anchored density and mass-weighted boron; local fuel Doppler retained'
            self.core_terms['local_moderator_feedback_connected']=True
        return rate

    def snapshot(self):
        result=super().snapshot()
        if self.moderator_ready:
            result.update(model_id='pwr-local-moderator-research/v1',local_moderator=deepcopy(self.moderator_terms),
                local_moderator_source=identity(),local_moderator_stationarity=deepcopy(getattr(self,'local_stationary_report',None)),
                local_density_boron_feedback=True)
        return result

    def checkpoint(self):
        if not self.moderator_ready:return super().checkpoint()
        return seal(dict(format='pwr-local-moderator/v1',source=identity(),initial_stationary=deepcopy(self.initial_stationary),
            history=list(deepcopy(self.history)),snapshot=self.snapshot()))

    def parameter_registry(self):
        result=super().parameter_registry()
        if self.moderator_ready:
            def add(name,value,unit,source):
                result['parameters']['local_moderator.'+name]=dict(value=float(value),unit=unit,source=source,
                    family='illustrative',version='1',uncertainty=None)
            add('density_coefficient',self.plant.density_coef,'m^3/kg','Inherited plants.vver density_coef; synthetic incremental response, not evaluated local XS')
            add('boron_coefficient',self.plant.boron_coef,'1/ppm','Inherited plants.vver boron_coef; synthetic incremental response, not evaluated local XS')
            ref=self.moderator_reference;source='Constructed stationary t0 reference; frozen, not measured or fitted'
            add('cold_density_reference',ref.cold_density_kg_m3,'kg/m^3',source)
            add('bulk_boron_reference',ref.bulk_boron_ppm,'ppm',source)
            for i,(density,boron) in enumerate(zip(ref.density_kg_m3,ref.boron_ppm)):
                add(f'density_reference.{i}',density,'kg/m^3',source)
                add(f'boron_reference.{i}',boron,'ppm',source)
        return result

    @classmethod
    def restore(cls,saved):
        p=unseal(deepcopy(saved))
        if type(p) is not dict or set(p)!={'format','source','initial_stationary','history','snapshot'} or p['format']!='pwr-local-moderator/v1' or p['source']!=identity():raise ValueError('Local moderator checkpoint source/format')
        if type(p['history']) is not list or len(p['history'])>256:raise ValueError('Local moderator history budget')
        initial=unseal(p['initial_stationary'])
        if (type(initial) is not dict or set(initial)!={'format','source','stationary','initial_channel','history','snapshot'}
                or initial['format']!='pwr-stationary-channel/v1' or initial['source']!=stationary_identity() or initial['history']!=[]):
            raise ValueError('Unadvanced stationary reference checkpoint required')
        if type(initial['stationary']) is not dict or set(initial['stationary'])!=set(StationaryConfig.__dataclass_fields__):raise ValueError('Moderator stationary configuration')
        # Validate the cheap lifted seed first; the fresh constructor then
        # recomputes the stationary baseline once and exact comparison also
        # checks its saved report. No saved reference/solution is trusted.
        seed=ChannelPWRSession.restore(initial['initial_channel'])
        if seed.time!=0 or seed.history:raise ValueError('Moderator reference seed must be unadvanced')
        obj=cls(seed.config,cooling=seed.cooling_config,feed=seed.feed_config,backup=seed.backup_config,core=seed.core_config,
            fuel_grid=seed.fuel_grid,fluid_grid=seed.fluid_grid,max_step_s=seed.max_step_s,rtol=seed.rtol,stationary=StationaryConfig(**initial['stationary']))
        for row in p['history']:
            if type(row) is not dict or set(row)!={'duration_s','patch'}:raise ValueError('Moderator history fields')
            obj.advance(**row)
        if obj.checkpoint()!=saved:raise ValueError('Local moderator replay differs')
        return obj
