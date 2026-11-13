"""Two equivalent expansion stages, zero-hold-up separator and finite-wall reheat.

Intermediate pressure and efficiencies are declared cycle parameters, not
manufacturer stage/nozzle maps. Heat is removed from the existing SG wall.
"""
from copy import deepcopy
from dataclasses import asdict,dataclass
import math

import numpy as np

from physics.materials import iapws
from atomic_model_multiphysics.provenance import seal,unseal
from .steam import PWRSteamSession,SteamConfig,actual_expansion,steam_signature
from .film import FilmConfig
from .primary import PrimaryConfig
from .core import CoreConfig
from .power import BackupConfig
from .feed import FeedConfig
from .cooling import CoolingConfig
from .pwr_train import TrainConfig


def stages_signature():return dict(steam=steam_signature())


@dataclass(frozen=True)
class StageConfig:
    stage_count:int=2
    intermediate_pressure_pa:float=7e5
    hp_efficiency:float=.85
    lp_efficiency:float=.85
    separator_efficiency:float=1.
    reheat_ua_w_k:float=4e6
    minimum_hp_exit_quality:float=.84
    minimum_lp_exit_quality:float=.88

    def __post_init__(self):
        if type(self.stage_count) is not int or self.stage_count!=2:
            raise ValueError('Exactly two equivalent thermodynamic stages supported')
        for name,low,high in [('intermediate_pressure_pa',3e5,1.5e6),('hp_efficiency',.5,1.),
            ('lp_efficiency',.5,1.),('separator_efficiency',.9,1.),('reheat_ua_w_k',0.,1e7),
            ('minimum_hp_exit_quality',.7,1.),('minimum_lp_exit_quality',.8,1.)]:
            value=getattr(self,name)
            if type(value) not in (float,int) or not math.isfinite(value) or not low<=value<=high:
                raise ValueError('Stage parameter domain: '+name)


def staged_path(pin,hin,pout,flow,wall_temperature,config,*,separator_available=True,reheater_available=True):
    """Conservative steady ports with secant-cp effectiveness heat closure."""
    if type(config) is not StageConfig or type(separator_available) is not bool or type(reheater_available) is not bool:
        raise ValueError('Explicit staged path configuration/availability')
    if any(not math.isfinite(v) for v in (pin,hin,pout,flow,wall_temperature)) or flow<0:
        raise ValueError('Finite forward stage path required')
    mid=config.intermediate_pressure_pa
    if not 0<pout<mid<pin or not 300<=wall_temperature<=900:
        raise ValueError('Stage pressure order or wall temperature')
    if flow==0.:
        return dict(active=False,stage_count=2,active_stage_count=0,mass_flow_kg_s=0.,
            total_work_w=0.,reheat_w=0.,condenser_energy_w=0.,
            condenser_mean_enthalpy_j_kg=float(hin),mass_residual_kg_s=0.,energy_residual_w=0.,
            entropy_generation_w_k=dict(hp=0.,separator=0.,reheater=0.,lp=0.,drain_throttle=0.,condenser_mixing=0.),
            hp=None,separator=None,reheater=None,lp=None,condenser_mixture=None,
            lp_exit_quality=None,flow_status='No material stream; enthalpy is an unused zero-flow reference')
    hp=actual_expansion(pin,hin,mid,config.hp_efficiency)
    if hp['outlet_quality']<config.minimum_hp_exit_quality:
        raise ValueError('HP exhaust moisture limit')
    liquid,vapor=iapws.saturation_properties(mid)
    efficiency=config.separator_efficiency if separator_available else 0.
    drain=flow*(1-hp['outlet_quality'])*efficiency
    steam=flow-drain
    if steam<=0:raise ValueError('Separator steam stream exhausted')
    after_h=(flow*hp['outlet_enthalpy_j_kg']-drain*liquid.h_j_kg)/steam
    after=iapws.properties_ph(mid,after_h)
    hout=after_h;effectiveness=0.;capacity=0.;heat=0.
    if reheater_available and config.reheat_ua_w_k>0 and wall_temperature>after.T_k:
        reference=iapws.properties_pt(mid,wall_temperature)
        capacity=steam*(reference.h_j_kg-after_h)/(wall_temperature-after.T_k)
        if capacity<=0:raise ValueError('Positive reheater secant heat capacity required')
        effectiveness=-math.expm1(-config.reheat_ua_w_k/capacity)
        hout=after_h+effectiveness*(reference.h_j_kg-after_h)
        heat=steam*(hout-after_h)
    heated=iapws.properties_ph(mid,hout)
    if heated.T_k>max(wall_temperature,after.T_k)+1e-7 or heat<0:
        raise ValueError('Reheater temperature/heat direction')
    lp=actual_expansion(mid,hout,pout,config.lp_efficiency)
    if lp['outlet_quality']<config.minimum_lp_exit_quality:
        raise ValueError('LP exhaust moisture limit; separation/reheat insufficient')
    hp_work=flow*hp['work_j_kg'];lp_work=steam*lp['work_j_kg'];work=hp_work+lp_work
    incoming=drain*liquid.h_j_kg+steam*lp['outlet_enthalpy_j_kg']
    mixture=iapws.properties_ph(pout,incoming/flow)
    drain_out=iapws.properties_ph(pout,liquid.h_j_kg)
    hp_out=iapws.properties_ph(mid,hp['outlet_enthalpy_j_kg'])
    entropy=dict(hp=flow*(hp['outlet_entropy_j_kg_k']-hp['inlet_entropy_j_kg_k']),
        separator=drain*liquid.s_j_kg_k+steam*after.s_j_kg_k-flow*hp_out.s_j_kg_k,
        reheater=steam*(heated.s_j_kg_k-after.s_j_kg_k)-heat/wall_temperature,
        lp=steam*(lp['outlet_entropy_j_kg_k']-lp['inlet_entropy_j_kg_k']),
        drain_throttle=drain*(drain_out.s_j_kg_k-liquid.s_j_kg_k),
        condenser_mixing=flow*mixture.s_j_kg_k-drain*drain_out.s_j_kg_k-steam*lp['outlet_entropy_j_kg_k'])
    return dict(active=True,stage_count=2,active_stage_count=2,mass_flow_kg_s=float(flow),
        total_work_w=float(work),reheat_w=float(heat),condenser_energy_w=float(incoming),
        condenser_mean_enthalpy_j_kg=float(incoming/flow),mass_residual_kg_s=float(flow-drain-steam),
        energy_residual_w=float(flow*hin+heat-work-incoming),
        entropy_generation_w_k={k:float(v) for k,v in entropy.items()},
        hp=dict(hp,mass_flow_kg_s=float(flow),shaft_work_w=float(hp_work),outlet_pressure_pa=float(mid)),
        separator=dict(drain_flow_kg_s=float(drain),steam_flow_kg_s=float(steam),efficiency=float(efficiency),
            drain_enthalpy_j_kg=float(liquid.h_j_kg),steam_enthalpy_j_kg=float(after_h),steam_quality=float(after.quality),
            energy_residual_w=float(flow*hp['outlet_enthalpy_j_kg']-drain*liquid.h_j_kg-steam*after_h)),
        reheater=dict(available=reheater_available,heat_w=float(heat),effectiveness=float(effectiveness),
            capacity_rate_w_k=float(capacity),inlet_enthalpy_j_kg=float(after_h),outlet_enthalpy_j_kg=float(hout),
            inlet_temperature_k=float(after.T_k),outlet_temperature_k=float(heated.T_k),wall_temperature_k=float(wall_temperature),
            closure='Secant-cp effectiveness; finite existing SG-wall donor, no independent heat source'),
        lp=dict(lp,mass_flow_kg_s=float(steam),shaft_work_w=float(lp_work),inlet_pressure_pa=float(mid)),
        condenser_mixture=dict(enthalpy_j_kg=float(mixture.h_j_kg),quality=float(mixture.quality),
            entropy_j_kg_k=float(mixture.s_j_kg_k),outlet_kind='Combined LP exhaust and separator drain, not blade-exit wetness'),
        lp_exit_quality=float(lp['outlet_quality']),flow_status='Active equilibrium ports')


class PWRStagedSession(PWRSteamSession):
    def _maximum_time_s(self):return 7200.  # Explicit thermal/decay settling plus one-hour hold.
    def __init__(self,config=None,*,cooling=None,feed=None,backup=None,core=None,
                 primary=None,film=None,steam=None,stages=None,max_step_s=.1,rtol=1e-7):
        self.stages_ready=False;self.stage_config=stages if stages is not None else StageConfig()
        if type(self.stage_config) is not StageConfig:raise ValueError('StageConfig required')
        super().__init__(config,cooling=cooling,feed=feed,backup=backup,core=core,primary=primary,
                         film=film,steam=steam,max_step_s=max_step_s,rtol=rtol)
        self.controls.update(separator_available=True,reheater_available=True)
        self.stages_ready=True;self._rhs(0.,self.state);self._admit(self.state)

    def _path(self,y):
        point=self.header.thermo
        return staged_path(float(point.p_pa),float(point.h_j_kg),float(self.plant._condenser_pressure_pa),
            float(y[self.steam_index+2]),float(y[self.base_size-1]),self.stage_config,
            separator_available=self.controls['separator_available'],reheater_available=self.controls['reheater_available'])

    def _rhs(self,t,y):
        if self.stages_ready:y=self._reconstruct_state(np.array(y,dtype=float,copy=True))
        rate=super()._rhs(t,y)
        if not self.stages_ready:return rate
        path=self._path(y);p=self.plant;qt=float(y[self.steam_index+2]);heat=path['reheat_w']
        old_work=self.powers['steam_work_w'];old_energy=qt*self.steam_terms['outlet_enthalpy_j_kg']
        delta_energy=path['condenser_energy_w']-old_energy;delta_work=path['total_work_w']-old_work
        rate[self.base_size-2]+=delta_energy
        rate[self.base_size]+=delta_work
        rate[self.base_size-1]-=heat/p.sg_wall_capacity
        self.feed_terms['hotwell_energy_rate_w']+=delta_energy
        residual=delta_energy+delta_work-heat
        self.instantaneous_energy_defect_w+=residual
        self.powers.update(steam_work_w=path['total_work_w'],reheater_heat_w=heat)
        self.steam_terms['outlet_enthalpy_j_kg']=path['condenser_mean_enthalpy_j_kg']
        self.steam_terms['routing_energy_residual_w']+=residual
        self.steam_terms['expansion_entropy_w_k']=path['entropy_generation_w_k']['hp']+path['entropy_generation_w_k']['lp']
        self.staged_terms=dict(path,wall_temperature_rate_k_s=float(rate[self.base_size-1]),
            routing_energy_residual_w=float(residual),heat_donor='Existing finite SG wall',
            state_ownership='No extra steam/energy reservoir: steady separator/reheater ports, existing SG wall pays heat')
        p.live_expansion=dict(model_id='two-equivalent-stage-separator-reheater',stage_count=2,
            inlet_enthalpy_j_kg=float(self.header.thermo.h_j_kg),outlet_enthalpy_j_kg=path['condenser_mean_enthalpy_j_kg'],
            work_j_kg=path['total_work_w']/qt if qt>0 else 0.,added_heat_j_kg=heat/qt if qt>0 else 0.,
            outlet_quality=path['condenser_mixture']['quality'] if path['active'] else None,
            lp_outlet_quality=path['lp_exit_quality'],outlet_kind='Energy-equivalent condenser inlet including separator drain',
            active=path['active'])
        return rate

    def _admit(self,y):
        super()._admit(y)
        if not self.stages_ready:return
        path=self._path(y)
        if abs(path['energy_residual_w'])>.01 or abs(path['mass_residual_kg_s'])>1e-8:
            raise ValueError('Staged path mass/energy residual')
        if min(path['entropy_generation_w_k'].values())<-.01:
            raise ValueError('Staged path negative entropy production')

    def _set_controls(self,patch):
        names=('separator_available','reheater_available')
        if type(patch) is not dict or not patch or not set(patch)<=set(self.controls):raise ValueError('Staged command patch')
        if any(type(value) is not bool for key,value in patch.items() if key in names):raise ValueError('Boolean staged availability')
        regular={k:v for k,v in patch.items() if k not in names}
        if regular:super()._set_controls(regular)
        self.controls.update({k:v for k,v in patch.items() if k in names})
        if any(k in names for k in patch):self.events.append(dict(time_s=self.time,kind='staged_availability',requested={k:v for k,v in patch.items() if k in names}))

    def parameter_registry(self):
        result=super().parameter_registry()
        if self.stages_ready:
            for key,value in asdict(self.stage_config).items():
                unit='Pa' if key.endswith('_pa') else 'W/K' if key.endswith('_w_k') else '1'
                result['parameters']['stages.'+key]=dict(value=float(value),unit=unit,
                    source='Declared two equivalent stages, ideal separator and secant-cp reheater; no hardware moisture/efficiency map',
                    family='illustrative',version='1',uncertainty=None)
        return result

    def snapshot(self):
        result=super().snapshot()
        if self.stages_ready:
            result.update(model_id='npp-pwr-two-stage-reheated',staged_turbine=deepcopy(self.staged_terms),stage_config=asdict(self.stage_config))
            audit=result['steam_port_audit'];drop=audit.pop('enthalpy_drop_work_w')
            audit.update(fluid_enthalpy_drop_w=drop,reheater_heat_w=self.staged_terms['reheat_w'],
                         shaft_work_w=self.staged_terms['total_work_w'],
                         energy_residual_w=drop+self.staged_terms['reheat_w']-self.staged_terms['total_work_w'])
        return result

    def checkpoint(self):
        p=unseal(super().checkpoint());p.update(stage_config=asdict(self.stage_config),stages_source=stages_signature());return seal(p)

    @classmethod
    def restore(cls,saved):
        p=unseal(deepcopy(saved))
        if p['stages_source']!=stages_signature() or type(p['history']) is not list or len(p['history'])>256:raise ValueError('Staged replay source/budget')
        s=cls(TrainConfig(**p['config']),cooling=CoolingConfig(**p['cooling_config']),feed=FeedConfig(**p['feed_config']),backup=BackupConfig(**p['backup_config']),
            core=CoreConfig(**p['core_config']),primary=PrimaryConfig(**p['primary_config']),film=FilmConfig(**p['film_config']),
            steam=SteamConfig(**p['steam_config']),stages=StageConfig(**p['stage_config']),max_step_s=p['max_step_s'],rtol=p['rtol'])
        for row in p['history']:s.advance(**row)
        if s.checkpoint()!=saved:raise ValueError('Staged replay differs')
        return s
