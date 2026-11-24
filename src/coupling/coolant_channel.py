"""Horizontal single-phase compressible coolant channel with local wall storage.

Component qualification only: no PWR/StationEnv binding, boiling, CHF, pump,
axial conduction or turbulent-mixing model. Uses existing IF97 inventories and
donor-enthalpy/lumped-momentum network equations (no kinetic-energy storage).
"""
from copy import deepcopy
from dataclasses import asdict,dataclass
import hashlib
import math
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from core.status import DomainError,SolverError
from atomic_model_multiphysics.provenance import seal,unseal
from physics.hydraulics import network as hydraulics
from physics.materials import iapws
from . import conservative
from .conservative import ConservativeMap,vector


def identity():
    return dict(channel_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        network_sha256=hashlib.sha256(Path(hydraulics.__file__).read_bytes()).hexdigest(),
        eos_sha256=hashlib.sha256(Path(iapws.__file__).read_bytes()).hexdigest(),
        exchange_sha256=hashlib.sha256(Path(conservative.__file__).read_bytes()).hexdigest())


def contact(exchange,wall_k,coolant_k,conductance_w_k):
    """Overlap conductance pairs, preserving conservation and entropy signs.

    Mapping a net wall power alone would erase simultaneous opposite transfers
    when one wall overlaps hotter and colder coolant cells.
    """
    nw=len(exchange.source_widths);nc=len(exchange.target_widths)
    wall=vector(wall_k,nw,positive=True);coolant=vector(coolant_k,nc,positive=True)
    ua=vector(conductance_w_k,nw,positive=True)
    try:
        with np.errstate(over='raise',invalid='raise'):
            conductance=exchange.power_matrix*ua[None,:]
            pairs=conductance*(wall[None,:]-coolant[:,None])
            entropy=conductance*(wall[None,:]-coolant[:,None])**2/(wall[None,:]*coolant[:,None])
    except FloatingPointError as exc:raise DomainError('Thermal contact exceeds finite range') from exc
    if not np.isfinite(pairs).all() or not np.isfinite(entropy).all():raise DomainError('Nonfinite thermal contact')
    return dict(pair_heat_w=pairs,wall_heat_out_w=pairs.sum(axis=0),coolant_heat_in_w=pairs.sum(axis=1),entropy_w_k=entropy)


@dataclass(frozen=True)
class Port:
    pressure_pa:float=15.5e6
    temperature_k:float=550.
    boron_fraction:float=.0005

    def __post_init__(self):
        for name,lo,hi in [('pressure_pa',1e6,20e6),('temperature_k',300.,600.),('boron_fraction',0.,.01)]:
            _number(getattr(self,name),lo,hi,name)
        state=iapws.properties_pt(self.pressure_pa,self.temperature_k)
        if state.region!=1:raise ValueError('Compressed-liquid IF97 region1 port required')


def _number(value,lo,hi,name):
    if type(value) not in (int,float) or not math.isfinite(value) or not lo<=value<=hi:raise ValueError('Channel parameter domain: '+name)


@dataclass(frozen=True)
class ChannelConfig:
    faces_m:tuple=(0.,1.,2.,4.)
    wall_faces_m:tuple=(0.,.7,2.2,4.)
    area_m2:float=.02
    loss_coefficient:float=2.
    wall_capacity_j_k:float=2e5
    wall_coolant_ua_w_k:float=2e4
    initial_pressure_pa:float=15.5e6
    initial_temperature_k:float=550.
    initial_wall_temperature_k:float=550.
    initial_boron_fraction:float=.0005
    initial_flow_kg_s:float=0.

    def __post_init__(self):
        if type(self.faces_m) is not tuple or type(self.wall_faces_m) is not tuple:raise ValueError('Immutable channel/wall faces required')
        if not 2<=len(self.faces_m)<=9 or not 2<=len(self.wall_faces_m)<=17:raise ValueError('Channel1..8/wall1..16 cells required')
        exchange=ConservativeMap(self.wall_faces_m,self.faces_m)
        if exchange.source_faces[0]!=0 or exchange.source_faces[-1]>20:raise ValueError('Channel length in (0,20] m required')
        for name,lo,hi in [('area_m2',.001,1.),('loss_coefficient',0.,100.),('wall_capacity_j_k',1e3,1e8),
                ('wall_coolant_ua_w_k',1.,1e7),('initial_wall_temperature_k',300.,600.),('initial_flow_kg_s',-100.,100.)]:
            _number(getattr(self,name),lo,hi,name)
        Port(self.initial_pressure_pa,self.initial_temperature_k,self.initial_boron_fraction)
        if self.initial_wall_temperature_k>=iapws.saturation_temperature_k(self.initial_pressure_pa*1e-6):
            raise ValueError('Initial wall must remain below saturation')


class _ChannelVolume(hydraulics.Volume):
    """Use fixed inverse-EOS seeds, keeping RHS independent of probe history.

    The existing volume caches only exact inventory keys. Changing its Newton
    guesses between Jacobian probes can otherwise leave tiny pressure defects
    at an identical resting state. No physical inventory is altered here.
    """
    def sync(self):
        key=(self.mass_kg/self.volume_m3,self.internal_energy_j/self.mass_kg)
        if getattr(self,'_thermo_key',None)!=key:
            self.pressure_pa=self.seed_pressure_pa;self.temperature_k=self.seed_temperature_k
        return super().sync()


class CoolantChannel:
    """Independent local fluid M/U/B and face momentum with transactional advance."""
    def __init__(self,config=None,*,left=None,right=None,max_step_s=.002,rtol=1e-8):
        self.config=ChannelConfig() if config is None else config
        if type(self.config) is not ChannelConfig:raise ValueError('ChannelConfig required')
        self.left=Port() if left is None else left;self.right=Port() if right is None else right
        if type(self.left) is not Port or type(self.right) is not Port:raise ValueError('Explicit liquid ports required')
        self._initial_ports=dict(left=asdict(self.left),right=asdict(self.right))
        _number(max_step_s,1e-5,.1,'max_step_s');_number(rtol,1e-11,1e-5,'rtol')
        self.max_step_s=float(max_step_s);self.rtol=float(rtol);c=self.config
        self.exchange=ConservativeMap(c.wall_faces_m,c.faces_m)
        self.cells=len(c.faces_m)-1;self.wall_cells=len(c.wall_faces_m)-1
        widths=self.exchange.target_widths;centres=(np.asarray(c.faces_m[:-1])+c.faces_m[1:])/2
        distances=np.diff(np.r_[c.faces_m[0],centres,c.faces_m[-1]])
        volumes={}
        for i,width in enumerate(widths):
            name=f'cell{i}';volumes[name]=_ChannelVolume(name,c.area_m2*width,0.)
            volumes[name].seed_pressure_pa=c.initial_pressure_pa;volumes[name].seed_temperature_k=c.initial_temperature_k
            volumes[name].set_liquid(c.initial_pressure_pa,c.initial_temperature_k,c.initial_boron_fraction*1e6)
        links={}
        for i,distance in enumerate(distances):
            links[f'face{i}']=hydraulics.Junction(f'face{i}','left' if i==0 else f'cell{i-1}',
                'right' if i==self.cells else f'cell{i}',c.area_m2,float(distance),
                c.loss_coefficient*float(distance)/c.faces_m[-1],mass_flow_kg_s=c.initial_flow_kg_s)
        self.network=hydraulics.FlowNetwork(volumes,links)
        self._ports();self.wall_index=4*self.cells+1;self.ledger_index=self.wall_index+self.wall_cells
        weights=self.exchange.source_widths/c.wall_faces_m[-1]
        self.wall_capacities=c.wall_capacity_j_k*weights;self.conductances=c.wall_coolant_ua_w_k*weights
        self.state=np.r_[self.network.pack(),np.full(self.wall_cells,c.initial_wall_temperature_k),np.zeros(3)]
        self.time=0.;self.history=[];self.solver_statistics=None
        self.origin=self._inventories(self.state);self._rhs(0.,self.state);self._admit(self.state)

    def _ports(self):
        for name,port in [('left',self.left),('right',self.right)]:
            state=iapws.properties_pt(port.pressure_pa,port.temperature_k)
            self.network.boundaries[name]=hydraulics.BoundaryState(port.pressure_pa,state.h_j_kg,state.rho_kg_m3,0.,port.boron_fraction)

    def _inventories(self,y):
        return np.array([y[:3*self.cells:3].sum(),y[1:3*self.cells:3].sum()+self.wall_capacities@y[self.wall_index:self.ledger_index],y[2:3*self.cells:3].sum()])

    def _sync(self,y):
        if y.shape!=(self.ledger_index+3,) or not np.isfinite(y).all():raise DomainError('Finite channel state with exact shape required')
        mass=y[:3*self.cells:3];boron=y[2:3*self.cells:3]
        if np.any(mass<=0) or np.any(boron<0) or np.any(boron>.01*mass):raise DomainError('Local mass/boron domain')
        self.network.unpack(y[:self.wall_index]);self.network.refresh_properties()
        wall=y[self.wall_index:self.ledger_index]
        for volume in self.network.volumes.values():
            if volume.thermo.region!=1 or not 1e6<=volume.pressure_pa<=20e6 or not 300<=volume.temperature_k<=600:
                raise DomainError('Local coolant requires liquid region1,1..20 MPa,300..600 K')
        saturation=np.array([iapws.saturation_temperature_k(v.pressure_pa*1e-6) for v in self.network.volumes.values()])
        overlap=self.exchange.overlap>0
        if np.any(wall<300) or np.any(overlap & (wall[None,:]>=saturation[:,None])):
            raise DomainError('Each local wall must remain below saturation in every overlapping coolant cell')

    def _rhs(self,t,y,powers=None):
        self._sync(y)
        powers=np.zeros(self.wall_cells) if powers is None else powers
        wall=y[self.wall_index:self.ledger_index]
        coolant=np.array([v.temperature_k for v in self.network.volumes.values()])
        thermal=contact(self.exchange,wall,coolant,self.conductances)
        for v,heat in zip(self.network.volumes.values(),thermal['coolant_heat_in_w']):v.heat_w=float(heat)
        rate=self.network.derivatives(y[:self.wall_index])
        wall_rate=(powers-thermal['wall_heat_out_w'])/self.wall_capacities
        # Inward signs at the left face; outward signs at the right face.
        boundary=np.array([0.,float(np.sum(powers)),0.])
        ports={}
        for name,face,sign,adjacent in [('left','face0',1.,'cell0'),('right',f'face{self.cells}',-1.,f'cell{self.cells-1}')]:
            q=self.network.junctions[face].mass_flow_kg_s
            incoming=sign*q>=0
            donor=self.network.boundaries[name] if incoming else self.network.volumes[adjacent]
            h=donor.enthalpy_j_kg if incoming else donor.thermo.h_j_kg
            b=donor.boron_fraction
            flux=np.array([sign*q,sign*q*h,sign*q*b]);boundary+=flux
            ports[name]=dict(mass_into_kg_s=float(flux[0]),enthalpy_into_w=float(flux[1]),boron_into_kg_s=float(flux[2]),donor='boundary' if incoming else adjacent)
        actual=np.array([rate[:3*self.cells:3].sum(),rate[1:3*self.cells:3].sum()+self.wall_capacities@wall_rate,rate[2:3*self.cells:3].sum()])
        self.terms=dict(wall_temperature_k=wall.tolist(),coolant_temperature_k=coolant.tolist(),
            pressure_pa=[v.pressure_pa for v in self.network.volumes.values()],
            density_kg_m3=[v.thermo.rho_kg_m3 for v in self.network.volumes.values()],
            face_flow_kg_s=y[3*self.cells:self.wall_index].tolist(),face_acceleration_kg_s2=rate[3*self.cells:].tolist(),
            pair_heat_w=thermal['pair_heat_w'].tolist(),wall_heat_out_w=thermal['wall_heat_out_w'].tolist(),
            coolant_heat_in_w=thermal['coolant_heat_in_w'].tolist(),conductive_entropy_w_k=float(thermal['entropy_w_k'].sum()),
            instantaneous_residual=(actual-boundary).tolist(),boundary_rates=boundary.tolist(),ports=ports)
        return np.r_[rate,wall_rate,boundary]

    def _admit(self,y):self._sync(y)

    def advance(self,duration_s,powers_w=None,*,left=None,right=None):
        _number(duration_s,1e-6,.1,'duration_s')
        if self.time+duration_s>10 or len(self.history)>=256:raise ValueError('Channel10 s/256 interval budget')
        power=vector(np.zeros(self.wall_cells) if powers_w is None else powers_w,self.wall_cells)
        if np.any(power<0) or np.any(power>1e8):raise ValueError('Nonnegative wall power at most1e8 W/cell required')
        trial=deepcopy(self)
        for name,value in [('left',left),('right',right)]:
            if value is not None:
                if type(value) is not Port:raise ValueError('Port required')
                setattr(trial,name,value)
        trial._ports();calls=0
        def fun(t,y):
            nonlocal calls
            calls+=1
            if calls>30000:raise SolverError('Channel RHS work budget exceeded')
            return trial._rhs(t,y,power)
        def jac(t,y):
            base=fun(t,y);result=np.zeros((len(y),len(y)))
            for i in range(trial.ledger_index):
                step=1e-7*max(abs(float(y[i])),1.)
                up=y.copy();down=y.copy();up[i]+=step;down[i]-=step
                try:above=fun(t,up)
                except DomainError:above=None
                try:below=fun(t,down)
                except DomainError:below=None
                if above is None and below is None:raise SolverError('No admitted channel Jacobian probe')
                result[:,i]=(above-below)/(2*step) if above is not None and below is not None else (above-base)/step if above is not None else (base-below)/step
            fun(t,y)
            return result
        atol=np.full(len(trial.state),1e-9);atol[1:3*trial.cells:3]=.001
        atol[trial.wall_index:trial.ledger_index]=1e-8;atol[trial.ledger_index:]=[1e-8,.001,1e-10]
        solution=solve_ivp(fun,(trial.time,trial.time+duration_s),trial.state.copy(),method='Radau',rtol=trial.rtol,atol=atol,max_step=trial.max_step_s,jac=jac)
        if not solution.success or solution.t[-1]!=trial.time+duration_s:raise SolverError('Channel integration did not reach endpoint')
        trial.state=solution.y[:,-1].copy();trial.time=float(solution.t[-1]);trial._admit(trial.state)
        trial.history.append(dict(duration_s=duration_s,powers_w=power.tolist(),left=asdict(trial.left),right=asdict(trial.right)))
        trial.solver_statistics=dict(method='Radau',rhs_evaluations=calls,accepted_steps=len(solution.t)-1)
        result=trial.snapshot(powers_w=power)
        if max(result['balance_relative'])>1e-9:raise SolverError('Channel cumulative mass/energy/boron balance exceeded')
        self.__dict__.update(trial.__dict__);return result

    def snapshot(self,*,powers_w=None):
        if powers_w is None and self.history:powers_w=self.history[-1]['powers_w']
        power=vector(np.zeros(self.wall_cells) if powers_w is None else powers_w,self.wall_cells)
        self._rhs(self.time,self.state,power);inventory=self._inventories(self.state)
        residual=inventory-self.origin-self.state[self.ledger_index:]
        scale=np.maximum(np.maximum(abs(self.origin),abs(self.state[self.ledger_index:])),[1.,1.,1e-9])
        return dict(model_id='single-phase-coolant-channel/v1',source=identity(),time_s=self.time,
            state=self.state.tolist(),terms=deepcopy(self.terms),inventory=inventory.tolist(),balance_residual=residual.tolist(),
            balance_relative=(abs(residual)/scale).tolist(),solver_statistics=deepcopy(self.solver_statistics),
            distributed_fluid_inventories=True,distributed_momentum=True,pwr_connected=False,validated=False,
            energy_scope='Internal fluid energy and wall storage; inherited low-Mach network approximation excludes flow kinetic energy',
            initial_condition='Uniform constructed liquid inventory and declared initial face flows; no stationary solve')

    def checkpoint(self):
        power=None if not self.history else self.history[-1]['powers_w']
        config=asdict(self.config)
        for name in ('faces_m','wall_faces_m'):config[name]=list(config[name])
        return seal(dict(format='coolant-channel/v1',source=identity(),config=config,
            initial_ports=deepcopy(self._initial_ports),
            max_step_s=self.max_step_s,rtol=self.rtol,history=deepcopy(self.history),snapshot=self.snapshot(powers_w=power)))

    @classmethod
    def restore(cls,saved):
        p=unseal(deepcopy(saved))
        keys={'format','source','config','initial_ports','max_step_s','rtol','history','snapshot'}
        if type(p) is not dict or set(p)!=keys or p['format']!='coolant-channel/v1' or p['source']!=identity():raise ValueError('Channel checkpoint source/format')
        if type(p['config']) is not dict or set(p['config'])!=set(ChannelConfig.__dataclass_fields__):raise ValueError('Channel config fields')
        if type(p['initial_ports']) is not dict or set(p['initial_ports'])!={'left','right'}:raise ValueError('Channel port fields')
        def port(data):
            if type(data) is not dict or set(data)!=set(Port.__dataclass_fields__):raise ValueError('Channel port configuration fields')
            return Port(**data)
        if type(p['history']) is not list or len(p['history'])>256:raise ValueError('Channel history budget')
        config=dict(p['config'])
        for key in ('faces_m','wall_faces_m'):
            if type(config[key]) not in (list,tuple):raise ValueError('Channel mesh fields')
            config[key]=tuple(config[key])
        obj=cls(ChannelConfig(**config),left=port(p['initial_ports']['left']),right=port(p['initial_ports']['right']),max_step_s=p['max_step_s'],rtol=p['rtol'])
        for row in p['history']:
            if type(row) is not dict or set(row)!={'duration_s','powers_w','left','right'}:raise ValueError('Channel history record')
            obj.advance(row['duration_s'],row['powers_w'],left=port(row['left']),right=port(row['right']))
        if obj.checkpoint()!=saved:raise ValueError('Channel checkpoint replay differs')
        return obj
