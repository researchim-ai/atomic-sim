"""Replaceable neutron-state owners and conservative lumped thermal projection.

The diffusion adapter is a synthetic one-group reflecting slab with six local
precursor families. It does not infer evaluated plant nuclear data.
"""
from copy import deepcopy
from dataclasses import asdict,dataclass
import hashlib
import math
from pathlib import Path
from typing import Protocol

import numpy as np
from atomic_model_spatial import Model,eigenvalue
from atomic_model_spatial.workflow import identity as spatial_identity
from physics.neutronics import point_kinetics
from atomic_model_multiphysics.provenance import seal,unseal
from .power import PWRPowerSession,BackupConfig
from .feed import FeedConfig
from .cooling import CoolingConfig
from .pwr_train import TrainConfig,physical_signature


def core_signature():
    return dict(station=physical_signature(),spatial=spatial_identity(),
        point_kinetics_sha256=hashlib.sha256(Path(point_kinetics.__file__).read_bytes()).hexdigest())


@dataclass(frozen=True)
class CoreConfig:
    kind:str='point'
    cells:int=1
    length_m:float=4.
    diffusion_m:float=.01
    nu_fission_per_m:float=10.
    axial_bias_reactivity:float=0.

    def __post_init__(self):
        if self.kind not in ('point','diffusion') or type(self.cells) is not int:raise ValueError('Explicit core kind and integer cell count')
        if (self.kind=='point' and self.cells!=1) or (self.kind=='diffusion' and not 2<=self.cells<=8):raise ValueError('Core state budget: point1 or diffusion2..8 cells')
        for key,low,high in [('length_m',.1,20.),('diffusion_m',1e-4,1.),('nu_fission_per_m',.1,100.),('axial_bias_reactivity',-1e-4,1e-4)]:
            value=getattr(self,key)
            if type(value) not in (int,float) or not math.isfinite(value) or not low<=value<=high:raise ValueError('Core coefficient domain:'+key)
        if self.kind=='point' and self.axial_bias_reactivity!=0.:raise ValueError('Axial bias needs a spatial core')


class NeutronCore(Protocol):
    """One owner of neutron states; project returns seven derived legacy aliases."""
    size:int
    def lift(self,point_state):...
    def project(self,state):...
    def rhs(self,state,reactivity):...
    def fission_by_cell(self,state):...


class PointCore:
    def __init__(self,kinetics):
        self.kinetics=deepcopy(kinetics);self.size=7;self.cells=1
    def lift(self,state):return np.array(state,dtype=float,copy=True)
    def project(self,state):return np.array(state,dtype=float,copy=True)
    def rhs(self,state,reactivity):return self.kinetics.derivatives(state[0],state[1:],reactivity,0.)
    def fission_by_cell(self,state):return np.array([self.kinetics.power_w(state[0])])


class DiffusionCore:
    def __init__(self,config,kinetics):
        self.config=config;self.kinetics=deepcopy(kinetics);n=config.cells
        self.cells=n;self.size=7*n;self.speed=1/(kinetics.generation_time_s*config.nu_fission_per_m)
        self.weights=np.full(n,1/n)
        centres=(np.arange(n)+.5)*config.length_m/n
        self.bias=config.axial_bias_reactivity*(2*centres/config.length_m-1)
        self.specification=dict(faces_m=np.linspace(0.,config.length_m,n+1).tolist(),diffusion_m=[[config.diffusion_m]*n],
            absorption_per_m=[[config.nu_fission_per_m]*n],nu_fission_per_m=[[config.nu_fission_per_m]*n],scatter_per_m=[[[0.]*n]],
            speed_m_s=[[self.speed]*n],chi_prompt=[[1.]*n],chi_delayed=[[1.]*n],beta=np.repeat(kinetics.beta[:,None],n,axis=1).tolist(),
            decay_per_s=kinetics.decay.tolist(),boundaries=['reflecting','reflecting'],
            provenance='Constructed synthetic critical reflecting one-group slab; beta/decay copied from declared point model; no evaluated plant cross sections')
        self.model=Model(self.specification);self.weights=self.model.width/config.length_m;self.operator=self.model.kinetics.toarray()
        self.eigenpair=eigenvalue(self.model,tolerance=1e-12)
    def lift(self,state):
        return np.r_[np.full(self.cells,state[0]),np.repeat(np.asarray(state[1:])[:,None]/self.speed,self.cells,axis=1).ravel()]
    def project(self,state):
        n=self.cells
        return np.r_[self.weights@state[:n],self.speed*(np.asarray(state[n:]).reshape(6,n)@self.weights)]
    def matrix(self,reactivity):
        if not math.isfinite(reactivity) or np.any(1-reactivity-self.bias<=0):raise ValueError('Positive finite synthetic absorption required')
        matrix=self.operator.copy()
        matrix[np.arange(self.cells),np.arange(self.cells)]+=(reactivity+self.bias)/self.kinetics.generation_time_s
        return matrix
    def rhs(self,state,reactivity):
        if not math.isfinite(reactivity) or np.any(1-reactivity-self.bias<=0):raise ValueError('Positive finite synthetic absorption required')
        rate=self.operator@state
        rate[:self.cells]+=(reactivity+self.bias)/self.kinetics.generation_time_s*state[:self.cells]
        return rate
    def inventory_balance(self,state,reactivity):
        weights=np.r_[self.model.width/self.speed,np.tile(self.model.width,6)]
        rate=self.rhs(state,reactivity)
        expected=float(np.sum(self.config.nu_fission_per_m*(reactivity+self.bias)*state[:self.cells]*self.model.width))
        actual=float(weights@rate)
        scale=max(float(abs(weights)@abs(state)),float(abs(weights)@abs(rate)),abs(expected),1e-30)
        return dict(rate=actual,expected_rate=expected,relative_defect=abs(actual-expected)/scale)
    def fission_by_cell(self,state):return self.kinetics.power_nominal_w*self.weights*np.asarray(state[:self.cells])


def build_core(config,kinetics)->NeutronCore:
    if type(config) is not CoreConfig:raise ValueError('CoreConfig required')
    return PointCore(kinetics) if config.kind=='point' else DiffusionCore(config,kinetics)


class PWRCoreSession(PWRPowerSession):
    def __init__(self,config=None,*,cooling=None,feed=None,backup=None,core=None,max_step_s=.1,rtol=1e-7):
        self.core_ready=False;self.core_config=core if core is not None else CoreConfig()
        if type(self.core_config) is not CoreConfig:raise ValueError('CoreConfig required')
        super().__init__(config,cooling=cooling,feed=feed,backup=backup,max_step_s=max_step_s,rtol=rtol)
        self.neutron_core=build_core(self.core_config,self.plant.kinetics);self.core_index=len(self.state)
        self.state=np.r_[self.state,self.neutron_core.lift(self.state[:7])]
        self.core_ready=True;self.state=self._reconstruct_state(self.state)
        self._rhs(0.,self.state);self.origin_energy=self._stored();self._admit(self.state)

    def _reconstruct_state(self,y):
        if self.core_ready:y[:7]=self.neutron_core.project(y[self.core_index:self.core_index+self.neutron_core.size])
        return y

    def _absolute_tolerances(self,atol):
        atol=super()._absolute_tolerances(atol)
        if self.core_ready:atol[self.core_index:self.core_index+self.neutron_core.size]=1e-10
        return atol

    def _extend_known(self,known,modes):
        super()._extend_known(known,modes)
        if self.core_ready:
            for index in range(7):known[index]=lambda t:0.
            modes.append(dict(name='legacy_neutron_aliases',kind='derived_from_core_owner',indices=list(range(7))))

    def _rhs(self,t,y):
        if not self.core_ready:return super()._rhs(t,y)
        full=self._reconstruct_state(np.array(y,dtype=float,copy=True));rate=super()._rhs(t,full)
        state=full[self.core_index:self.core_index+self.neutron_core.size];feedback=self.plant.reactivity();dot=self.neutron_core.rhs(state,feedback)
        rate[:7]=self.neutron_core.project(dot)
        profile=self.neutron_core.fission_by_cell(state);power=float(profile.sum())
        aliases=self.neutron_core.project(state);declared=float(self.plant.kinetics.power_w(aliases[0]))
        self.core_terms=dict(kind=self.core_config.kind,owned_state_count=self.neutron_core.size,owned_state=state.tolist(),
            derived_point_state=aliases.tolist(),fission_by_cell_w=profile.tolist(),fission_total_w=power,
            fission_projection_residual_w=power-declared,bulk_reactivity=feedback,
            thermal_projection='Sum fission source into existing prompt/decay3 heat accounting, then common lumped fuel; no local thermal feedback',
            state_ownership='Only appended core variables are integrated; legacy neutron slots0..6 are derived aliases',
            nuclear_data_family='illustrative',validated=False,plant_referenced=False)
        if isinstance(self.neutron_core,DiffusionCore):
            self.core_terms.update(relative_flux=state[:self.neutron_core.cells].tolist(),local_precursors=np.asarray(state[self.neutron_core.cells:]).reshape(6,self.neutron_core.cells).tolist(),
                neutron_precursor_inventory=self.neutron_core.inventory_balance(state,feedback),
                reference_eigenpair=self.neutron_core.eigenpair,reference_eigenpair_scope='Zero bulk reactivity and zero axial bias only',local_reactivity_bias=self.neutron_core.bias.tolist())
        return np.r_[rate,dot]

    def _admit(self,y):
        if self.core_ready:
            state=y[self.core_index:self.core_index+self.neutron_core.size]
            if state.shape!=(self.neutron_core.size,) or not np.isfinite(state).all() or np.any(state<0):raise ValueError('Core population/precursor domain')
            point=self.neutron_core.project(state)
            if np.max(abs(y[:7]-point)/np.maximum(abs(point),1.))>1e-12:raise ValueError('Core derived aliases differ from their owner')
        super()._admit(y)

    def parameter_registry(self):
        result=super().parameter_registry()
        if not self.core_ready:return result
        for key,value,unit in [('cells',self.neutron_core.cells,'1'),('generation_time',self.plant.kinetics.generation_time_s,'s')]:
            result['parameters']['core.'+key]=dict(value=float(value),unit=unit,source='Declared core interface configuration',family='illustrative',version='1',uncertainty=None)
        for index,(beta,decay) in enumerate(zip(self.plant.kinetics.beta,self.plant.kinetics.decay)):
            for name,value,unit in [('beta',beta,'1'),('decay',decay,'1/s')]:
                result['parameters'][f'core.{name}.{index}']=dict(value=float(value),unit=unit,source='Existing declared teaching six-family data; not plant-evaluated effective data',family='illustrative',version='1',uncertainty=None)
        if self.core_config.kind=='diffusion':
            result['parameters']['core.speed_m_s']=dict(value=float(self.neutron_core.speed),unit='m/s',source='Constructed to match the existing generation time at reference fission coefficient',family='illustrative',version='1',uncertainty=None)
            for key,unit in [('length_m','m'),('diffusion_m','m'),('nu_fission_per_m','1/m'),('axial_bias_reactivity','1')]:
                result['parameters']['core.'+key]=dict(value=float(getattr(self.core_config,key)),unit=unit,
                    source='Constructed synthetic one-group slab; not evaluated nuclear data',family='illustrative',version='1',uncertainty=None)
        return result

    def snapshot(self):
        result=super().snapshot()
        if self.core_ready:result.update(model_id='npp-pwr-train-replaceable-core',core=deepcopy(self.core_terms),core_config=asdict(self.core_config),core_source=core_signature(),replaceable_core_connected=True)
        return result

    def checkpoint(self):
        p=unseal(super().checkpoint());p['core_config']=asdict(self.core_config);p['core_source']=core_signature();return seal(p)

    @classmethod
    def restore(cls,saved):
        p=unseal(deepcopy(saved))
        if p['core_source']!=core_signature() or type(p['history']) is not list or len(p['history'])>256:raise ValueError('Core replay source/budget')
        s=cls(TrainConfig(**p['config']),cooling=CoolingConfig(**p['cooling_config']),feed=FeedConfig(**p['feed_config']),backup=BackupConfig(**p['backup_config']),
            core=CoreConfig(**p['core_config']),max_step_s=p['max_step_s'],rtol=p['rtol'])
        for row in p['history']:s.advance(**row)
        if s.checkpoint()!=saved:raise ValueError('Core replay differs')
        return s
