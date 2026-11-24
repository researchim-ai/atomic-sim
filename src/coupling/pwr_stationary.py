"""Constrained normal-hot stationary initialization of the local-fluid PWR.

Constructs a new initial condition, not an energy-conserving transient from
the lifted seed. No constitutive parameter or pressure-control setting is fit.
"""
from copy import deepcopy
from dataclasses import dataclass,asdict
import hashlib
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from core.status import SolverError
from physics.materials.iapws import properties_pt,saturation_properties
from atomic_model_multiphysics.provenance import seal,unseal
from .pwr_channel import ChannelPWRSession,identity as channel_identity


def identity():
    return dict(channel=channel_identity(),initializer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


@dataclass(frozen=True)
class StationaryConfig:
    relative_power:float=1.
    max_nfev:int=400
    max_rhs_calls:int=60000

    def __post_init__(self):
        if type(self.relative_power) not in (int,float) or not np.isfinite(self.relative_power) or not .9<=self.relative_power<=1.05:raise ValueError('Stationary requested power0.9..1.05 required')
        if type(self.max_nfev) is not int or not 1<=self.max_nfev<=600:raise ValueError('Stationary optimizer budget1..600 required')
        if type(self.max_rhs_calls) is not int or not 64<=self.max_rhs_calls<=100000:raise ValueError('Stationary RHS budget64..100000 required')


def _mixture(volume,pressure,quality):
    liquid,vapor=saturation_properties(pressure)
    mass=volume/((1-quality)*liquid.v_m3_kg+quality*vapor.v_m3_kg)
    return mass,mass*((1-quality)*liquid.u_j_kg+quality*vapor.u_j_kg)


class _Problem:
    def __init__(self,session,config):
        self.session=deepcopy(session);self.config=config;s=self.session;p=s.plant
        if s.time!=0 or s.history or s.scram_latched or s.turbine_tripped:raise ValueError('Unadvanced untripped initial PWR required')
        if not s.controls['grid_available'] or s.controls['dhr_requested'] or s.controls['makeup_valve']!=0 or s.controls['drain_valve']!=0:raise ValueError('Normal powered closed-inventory initialization required')
        self.initial=s.state.copy();self.items=[];self.direct={};self.calls=0
        self.indices={n:13+3*i for i,n in enumerate(p.network.volume_names())}
        self.primary=('core','hot_leg','sg_primary','cold_leg','pressurizer');self.circulating=self.primary[:-1]
        idx=self.indices;y=self.initial
        self.primary_mass=sum(y[idx[n]] for n in self.primary)
        self.circulating_boron=sum(y[idx[n]+2] for n in self.circulating)
        self.secondary_mass=y[idx['sg_secondary']]+y[s.base_size-3]+y[s.feed_index]
        self.pressure_ref=p.p_pressurizer_ref_pa
        self.original_press_entropy=p.network.volumes['pressurizer'].mass_kg*p.network.volumes['pressurizer'].thermo.s_j_kg_k
        for name in ('hot_leg','sg_primary','cold_leg'):
            v=p.network.volumes[name];self.add(name+'.p',v.pressure_pa/1e6,14,17.5);self.add(name+'.t',v.temperature_k,510,600)
        for i,v in enumerate(s.channel.volumes.values()):
            self.add(f'fluid{i}.p',v.pressure_pa/1e6,14,17.5);self.add(f'fluid{i}.t',v.temperature_k,510,600)
        for name in ('pressurizer','sg_secondary'):
            v=p.network.volumes[name]
            lo=(self.pressure_ref-p.pressurizer_deadband_pa+100)/1e6 if name=='pressurizer' else 5.
            hi=(self.pressure_ref+p.pressurizer_deadband_pa-100)/1e6 if name=='pressurizer' else 6.95
            self.add(name+'.p',v.pressure_pa/1e6,lo,hi);self.add(name+'.x',v.thermo.quality,.01,.6)
        hw=p.condenser.state();self.add('hw.p',hw.p_pa/1e6,.003,.02);self.add('hw.x',hw.quality,.005,.2)
        self.own('rod',12,0.,1.);self.own('shaft',11,.8*y[11],1.2*y[11])
        self.own('rotor',s.base_size,.8*y[s.base_size],1.2*y[s.base_size]);self.own('sgwall',s.base_size-1,450,600)
        for j,name in enumerate(p.network.junction_names()):
            if name not in ('spray','dump','surge'):self.own('q.'+name,s.junction_offset+j,.5*y[s.junction_offset+j],1.5*y[s.junction_offset+j])
        for i in range(s.fluid_cells-1):self.own(f'qi{i}',s.internal_flow_index+i,.5*y[s.internal_flow_index+i],1.5*y[s.internal_flow_index+i])
        for i in range(s.neutron_core.cells):self.own(f'phi{i}',s.core_index+i,.2,2.)
        for i in range(s.fuel_cells):
            self.own(f'tf{i}',s.fuel_index+i,600,1600);self.own(f'tc{i}',s.cladding_index+i,510,605)
        for name,i in [('cool.c',s.cooling_index),('cool.h',s.cooling_index+1)]:self.own(name,i,.95*y[i],1.1*y[i])
        self.own('cool.q',s.cooling_index+2,.5*y[s.cooling_index+2],1.5*y[s.cooling_index+2])
        self.own('feed.m',s.feed_index,.9*y[s.feed_index],1.1*y[s.feed_index]);self.own('feed.u',s.feed_index+1,.5*y[s.feed_index+1],1.5*y[s.feed_index+1])
        self.own('cond.q',s.feed_index+6,.5*y[s.feed_index+6],1.5*y[s.feed_index+6])
        self.scale=np.maximum(np.abs([v[1] for v in self.items]),1.)
        self.start=np.array([v[1] for v in self.items])/self.scale
        self.low=np.array([v[2] for v in self.items])/self.scale;self.high=np.array([v[3] for v in self.items])/self.scale
        if np.any(self.start<=self.low) or np.any(self.start>=self.high):raise ValueError('Initial seed outside stationary search bounds')
        unknown=set(self.direct.values())
        for name in ('hot_leg','sg_primary','cold_leg','pressurizer','sg_secondary'):unknown.update((idx[name],idx[name]+1))
        unknown.update((s.base_size-3,s.base_size-2))
        for j in range(s.fluid_cells):unknown.update((s.fluid_index+3*j,s.fluid_index+3*j+1))
        surge=s.junction_offset+p.network.junction_names().index('surge')
        # Conservation anchors replace dependent mass rows. The inactive
        # pressurizer energy row is replaced by its surge momentum condition.
        self.equations=sorted((unknown-{12,idx['cold_leg'],idx['pressurizer'],idx['pressurizer']+1,idx['sg_secondary']})|{surge})
        self.tolerances=np.full(len(y),1e-7)
        self.tolerances[7:10]=1.  # Legacy heat-reservoir aliases also have J/s units.
        for name,i in idx.items():self.tolerances[i:i+3]=[1e-6,1.,1e-10]
        self.tolerances[s.base_size-3:s.base_size-1]=[1e-6,1.];self.tolerances[s.base_size]=1.
        for j,link in enumerate(p.network.junctions.values()):self.tolerances[s.junction_offset+j]=.01*link.area_m2/link.length_m if link.flow_law=='momentum' else 1e-6
        self.tolerances[s.cooling_index:s.cooling_index+3]=[1.,1.,1e-3]
        self.tolerances[s.feed_index:s.feed_index+7]=[1e-6,1.,1e-6,1.,1e-6,1.,1e-3]
        self.tolerances[s.decay_index:s.exchange_index]=1.
        for j in range(s.fluid_cells):self.tolerances[s.fluid_index+3*j:s.fluid_index+3*j+3]=[1e-6,1.,1e-10]
        for j,link in enumerate(s.channel.junctions.values()):self.tolerances[s.internal_flow_index+j]=.01*link.area_m2/link.length_m

    def add(self,name,value,lo,hi):self.items.append((name,float(value),float(lo),float(hi)))

    def own(self,name,index,lo,hi):
        self.add(name,self.initial[index],lo,hi);self.direct[name]=index

    def physical(self,z):
        s=self.session;p=s.plant;idx=self.indices;d=dict(zip([v[0] for v in self.items],z*self.scale));y=self.initial.copy()
        for name,i in self.direct.items():y[i]=d[name]
        for name in ('hot_leg','sg_primary','cold_leg'):
            t=properties_pt(d[name+'.p']*1e6,d[name+'.t']);i=idx[name]
            y[i]=p.network.volumes[name].volume_m3*t.rho_kg_m3;y[i+1]=y[i]*t.u_j_kg
        for j,v in enumerate(s.channel.volumes.values()):
            t=properties_pt(d[f'fluid{j}.p']*1e6,d[f'fluid{j}.t']);i=s.fluid_index+3*j
            y[i]=v.volume_m3*t.rho_kg_m3;y[i+1]=y[i]*t.u_j_kg
        for name in ('pressurizer','sg_secondary'):
            i=idx[name];y[i],y[i+1]=_mixture(p.network.volumes[name].volume_m3,d[name+'.p']*1e6,d[name+'.x'])
        y[s.base_size-3],y[s.base_size-2]=_mixture(p.condenser.volume_m3,d['hw.p']*1e6,d['hw.x'])
        y=s._reconstruct_state(y)
        fraction=self.circulating_boron/sum(y[idx[n]] for n in self.circulating)
        for name in ('hot_leg','sg_primary','cold_leg'):y[idx[name]+2]=fraction*y[idx[name]]
        for j in range(s.fluid_cells):y[s.fluid_index+3*j+2]=fraction*y[s.fluid_index+3*j]
        n=s.neutron_core.cells;flux=y[s.core_index:s.core_index+n];op=s.neutron_core.operator
        y[s.core_index+n:s.core_index+7*n]=-(op[n:,:n]@flux)/np.diag(op[n:,n:])
        fission=s.exchange.powers(s.neutron_core.fission_by_cell(y[s.core_index:s.core_index+7*n]))
        y[s.decay_index:s.exchange_index]=(p.decay.fractions[:,None]*fission/p.decay.decay_constants[:,None]).ravel()
        return s._reconstruct_state(y)

    def anchors(self,y):
        s=self.session;p=s.plant;idx=self.indices
        return [(sum(y[idx[n]] for n in self.primary)-self.primary_mass)/max(1e-6,self.primary_mass*1e-10),
            (y[idx['sg_secondary']]+y[s.base_size-3]+y[s.feed_index]-self.secondary_mass)/max(1e-6,self.secondary_mass*1e-10),
            (s.neutron_core.project(y[s.core_index:s.core_index+s.neutron_core.size])[0]-self.config.relative_power)/1e-10,
            (p.network.volumes['pressurizer'].pressure_pa-self.pressure_ref)/.01]

    def residual(self,z):
        self.calls+=1
        if self.calls>self.config.max_rhs_calls:raise SolverError('Stationary RHS work budget exceeded')
        y=self.physical(z);rate=self.session._rhs(0.,y)
        return np.r_[rate[self.equations]/self.tolerances[self.equations],self.anchors(y)]


def solve(session,config=None):
    """Return a validated initial state/report without changing the seed session."""
    if not isinstance(session,ChannelPWRSession):raise ValueError('ChannelPWRSession required')
    config=StationaryConfig() if config is None else config
    if type(config) is not StationaryConfig:raise ValueError('StationaryConfig required')
    problem=_Problem(session,config);s=problem.session
    result=least_squares(problem.residual,problem.start,bounds=(problem.low,problem.high),jac='3-point',diff_step=1e-6,
        x_scale='jac',ftol=1e-13,xtol=1e-13,gtol=1e-10,max_nfev=config.max_nfev)
    residual=problem.residual(result.x);state=problem.physical(result.x);rate=s._rhs(0.,state)
    physical_indices=[i for i in range(len(state)) if not s._passive_column(i)]
    physical_ratio=abs(rate[physical_indices])/problem.tolerances[physical_indices]
    # Every omitted physical equation, including inactive flows and conserved
    # modes, is checked independently of the optimizer's selected square system.
    if np.max(abs(residual))>1 or np.max(physical_ratio)>1:raise SolverError('Full stationary dimensional residual failed')
    s._admit(state)
    jac=result.jac.copy();jac/=np.maximum(np.linalg.norm(jac,axis=1),1e-30)[:,None];jac/=np.maximum(np.linalg.norm(jac,axis=0),1e-30)[None,:]
    singular=np.linalg.svd(jac,compute_uv=False);rank=int(sum(singular>singular[0]*1e-9))
    if rank!=len(problem.items):raise SolverError('Stationary constrained Jacobian rank failed')
    if s.plant._pressurizer_heater_w!=0 or s.plant.network.junctions['spray'].flow_target_kg_s!=0 or s.plant.network.junctions['dump'].flow_target_kg_s!=0:
        raise SolverError('Inactive pressure/steam branches changed during stationary initialization')
    report=dict(source=identity(),config=asdict(config),relative_power=config.relative_power,rhs_calls=problem.calls,nfev=int(result.nfev),
        constrained_unknowns=len(problem.items),constrained_rank=rank,rank_threshold=1e-9,
        equilibrated_singular_values=singular.tolist(),optimizer_termination=str(result.message),
        max_selected_tolerance_ratio=float(np.max(abs(residual))),max_physical_tolerance_ratio=float(np.max(physical_ratio)),
        physical_indices=physical_indices,physical_tolerances=problem.tolerances[physical_indices].tolist(),physical_rates=rate[physical_indices].tolist(),
        anchor_tolerance_ratios=problem.anchors(state),
        original_pressurizer_entropy_j_k=problem.original_press_entropy,
        final_pressurizer_entropy_j_k=float(s.plant.network.volumes['pressurizer'].mass_kg*s.plant.network.volumes['pressurizer'].thermo.s_j_kg_k),
        constraints=['Primary total mass','Secondary+hotwell+feed total mass','Circulating boron total with uniform stationary concentration',
            'Requested relative fission power','Existing pressurizer reference pressure','Fixed inactive tanks, backup and noncirculating boron'],
        reductions=['Exact stationary six-family neutron precursors','Exact stationary three-family heat reservoirs','Derived lumped aliases'],
        equilibrium_heat_history_assumed=True,parameter_changes=False,pressure_control_settings_changed=False,
        entropy_preserved=False,transient_preparation_simulated=False,validated=False,plant_referenced=False)
    return state.copy(),report


class StationaryChannelPWRSession(ChannelPWRSession):
    """Opt-in stationary normal-hot start; original lifted-start API is unchanged."""
    def __init__(self,*args,stationary=None,**kwargs):
        self.stationary_ready=False;self.stationary_config=StationaryConfig() if stationary is None else stationary
        if type(self.stationary_config) is not StationaryConfig:raise ValueError('StationaryConfig required')
        super().__init__(*args,**kwargs)
        self.initial_channel=super().checkpoint();initial_energy=self._stored()
        state,report=solve(self,self.stationary_config)
        self.state=state;self.controls['rod_target']=float(state[12]);self.initial_decay_j=float(self._local(state)[1].sum())
        self._rhs(0.,self.state);self._admit(self.state)
        self.origin_energy=self._stored();closed=self.plant.closed_inventories()
        self.origin_mass=closed['closed_mass_kg'];self.origin_boron=closed['boron_kg']
        self.origin_feed_mass=sum(self._tank(self.state,n)['mass_kg'] for n in self.feed_tanks)
        report.update(initial_energy_change_j=self.origin_energy-initial_energy,
            initial_rod_target=float(unseal(self.initial_channel)['snapshot']['controls']['rod_target']),final_rod_target=self.controls['rod_target'])
        self.stationary_report=report;self.stationary_ready=True
        self.channel_initialization.update(method='Constrained stationary full coupled physical RHS',stationary_channel_solve=True)

    def snapshot(self):
        result=super().snapshot()
        if self.stationary_ready:result.update(model_id='pwr-stationary-channel-research/v1',stationary_initialization=deepcopy(self.stationary_report))
        return result

    def checkpoint(self):
        if not self.stationary_ready:return super().checkpoint()
        return seal(dict(format='pwr-stationary-channel/v1',source=identity(),stationary=asdict(self.stationary_config),
            initial_channel=deepcopy(self.initial_channel),history=list(deepcopy(self.history)),snapshot=self.snapshot()))

    @classmethod
    def restore(cls,saved):
        p=unseal(deepcopy(saved))
        if type(p) is not dict or set(p)!={'format','source','stationary','initial_channel','history','snapshot'} or p['format']!='pwr-stationary-channel/v1' or p['source']!=identity():raise ValueError('Stationary PWR checkpoint source/format')
        if type(p['stationary']) is not dict or set(p['stationary'])!=set(StationaryConfig.__dataclass_fields__):raise ValueError('Stationary checkpoint configuration')
        if type(p['history']) is not list or len(p['history'])>256:raise ValueError('Stationary history budget')
        initial=ChannelPWRSession.restore(p['initial_channel'])
        if initial.time!=0 or initial.history:raise ValueError('Stationary initial checkpoint must be unadvanced')
        obj=cls(initial.config,cooling=initial.cooling_config,feed=initial.feed_config,backup=initial.backup_config,core=initial.core_config,
            fuel_grid=initial.fuel_grid,fluid_grid=initial.fluid_grid,max_step_s=initial.max_step_s,rtol=initial.rtol,stationary=StationaryConfig(**p['stationary']))
        for row in p['history']:
            if type(row) is not dict or set(row)!={'duration_s','patch'}:raise ValueError('Stationary history fields')
            obj.advance(**row)
        if obj.checkpoint()!=saved:raise ValueError('Stationary PWR replay differs')
        return obj
