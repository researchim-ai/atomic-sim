"""Bounded affine power-block assembly with unique state ownership and ports.

This eliminates declared affine power laws into an ODE. It is not a general
nonlinear DAE, fluid enthalpy compiler or arbitrary mechanical network solver.
"""
from copy import deepcopy
from dataclasses import dataclass
import math

import numpy as np
from atomic_model_workspace.common import digest
from .ports import Port,UNITS,validate_connections


def number(value):
    if type(value) not in (int,float) or not math.isfinite(value):raise ValueError('Finite scalar required')
    return float(value)


def name(value):
    if type(value) is not str or not value or len(value)>80:raise ValueError('Bounded nonempty name required')
    return value


@dataclass(frozen=True)
class Inventory:
    owner:str
    state:str
    kind:str
    capacity:float
    ports:tuple
    def __post_init__(self):
        name(self.owner);name(self.state)
        if self.kind not in ('thermal','mechanical') or number(self.capacity)<=0:raise ValueError('Inventory family/capacity')
        if not isinstance(self.ports,tuple) or not 1<=len(self.ports)<=16 or len(self.ports)!=len(set(self.ports)):raise ValueError('Unique bounded inventory ports required')
        for p in self.ports:name(p)
        if self.kind=='mechanical' and self.capacity!=1.:raise ValueError('Mechanical state must be kinetic energy in joules')


@dataclass(frozen=True)
class Boundary:
    owner:str
    kind:str
    def __post_init__(self):
        name(self.owner)
        if self.kind not in ('thermal','electrical_power'):raise ValueError('Boundary family')


class PowerBlock:
    """All port powers positive from block into its connected inventory/boundary.

    Each port has (kind, {state: W/state-unit coefficient}, constant W).
    Blocks must conserve energy independently before connections are compiled.
    """
    def __init__(self,owner,ports):
        name(owner)
        if type(ports) is not dict or not 1<=len(ports)<=16:raise ValueError('Power block port budget')
        owned={}
        for port,value in ports.items():
            name(port)
            if not isinstance(value,(tuple,list)) or len(value)!=3:raise ValueError('Port power declaration')
            kind,terms,constant=value
            if kind not in ('thermal','mechanical','electrical_power') or type(terms) is not dict or len(terms)>64:raise ValueError('Power family/terms')
            owned[port]=(kind,{name(k):number(v) for k,v in terms.items()},number(constant))
        keys={k for _,terms,_ in owned.values() for k in terms}
        sums=[math.fsum(v[1].get(k,0.) for v in owned.values()) for k in sorted(keys)]
        sums.append(math.fsum(v[2] for v in owned.values()))
        scale=max([1.]+[abs(x) for _,terms,c in owned.values() for x in [c,*terms.values()]])
        if any(abs(x)>1e-12*scale for x in sums):raise ValueError('Nonconservative power block')
        self._owner=owner;self._ports=owned
    @property
    def owner(self):return self._owner
    def declaration(self):return deepcopy(self._ports)


class AffineAssembly:
    def __init__(self,inventories,boundaries,blocks,connections):
        if not isinstance(inventories,(list,tuple)) or not 1<=len(inventories)<=64:raise ValueError('Inventory budget1..64')
        if not isinstance(boundaries,(list,tuple)) or not 1<=len(boundaries)<=32:raise ValueError('Boundary budget1..32')
        if not isinstance(blocks,(list,tuple)) or not 1<=len(blocks)<=64:raise ValueError('Block budget1..64')
        if any(type(x) is not Inventory for x in inventories) or any(type(x) is not Boundary for x in boundaries) or any(type(x) is not PowerBlock for x in blocks):
            raise ValueError('Declared component types required')
        owners=[x.owner for x in [*inventories,*boundaries,*blocks]]
        states=[x.state for x in inventories]
        if len(set(owners))!=len(owners) or len(set(states))!=len(states):raise ValueError('Duplicate component/state owner')
        nodes={x.owner:x for x in [*inventories,*boundaries]}
        laws={x.owner:dict(sorted(x.declaration().items())) for x in sorted(blocks,key=lambda x:x.owner)}
        ports=[]
        for x in inventories:
            for p in x.ports:ports.append(Port(x.owner,p,x.kind,*UNITS[x.kind]))
        for x in boundaries:ports.append(Port(x.owner,'port',x.kind,*UNITS[x.kind]))
        for owner,declaration in laws.items():
            for port,(kind,_,_) in declaration.items():ports.append(Port(owner,port,kind,*UNITS[kind]))
        validate_connections(ports,connections)
        target={}
        for pair in connections:
            a,b=map(tuple,pair)
            if a[0] in nodes and b[0] in laws:a,b=b,a
            if a[0] not in laws or b[0] not in nodes:raise ValueError('Connect one power block to one inventory/boundary')
            target[a]=b[0]
        n=len(inventories);indices={x.state:i for i,x in enumerate(inventories)}
        a=np.zeros((n,n));b=np.zeros(n);external={x.owner:(np.zeros(n),0.) for x in boundaries}
        inventory_index={x.owner:i for i,x in enumerate(inventories)}
        for owner,declaration in laws.items():
            attached={nodes[target[(owner,p)]].state for p in declaration if isinstance(nodes[target[(owner,p)]],Inventory)}
            for p,(_,terms,constant) in declaration.items():
                if not set(terms)<=attached:raise ValueError('Power law references an unknown or unconnected state')
                vector=np.zeros(n)
                for state,v in terms.items():vector[indices[state]]=v
                receiver=target[(owner,p)]
                if receiver in inventory_index:
                    i=inventory_index[receiver];capacity=inventories[i].capacity
                    a[i]+=vector/capacity;b[i]+=constant/capacity
                else:
                    old,offset=external[receiver];external[receiver]=(old+vector,offset+constant)
        capacities=np.array([x.capacity for x in inventories])
        defect=capacities@a+sum((v[0] for v in external.values()),start=np.zeros(n))
        constant=float(capacities@b+math.fsum(v[1] for v in external.values()))
        scale=max(1.,float(np.max(abs(a)))*float(np.max(capacities)),float(np.max(abs(b)))*float(np.max(capacities)))
        if not np.isfinite(a).all() or not np.isfinite(b).all() or max(float(np.max(abs(defect))),abs(constant))>1e-11*scale:
            raise ValueError('Global assembly balance failure')
        self._a=a;self._b=b;self._external=external
        self._manifest=dict(states=[dict(index=i,owner=x.owner,name=x.state,unit='K' if x.kind=='thermal' else 'J',energy_scale=x.capacity) for i,x in enumerate(inventories)],
            boundaries=[dict(owner=x.owner,kind=x.kind) for x in boundaries],
            blocks={owner:{port:[kind,terms,constant] for port,(kind,terms,constant) in declaration.items()} for owner,declaration in laws.items()},
            connections=sorted([sorted([list(v) for v in pair]) for pair in connections]),scope='Affine power laws; ODE elimination only')
        self._manifest['sha256']=digest(self._manifest)
    def operators(self):return self._a.copy(),self._b.copy()
    def boundary_power(self,owner,state):
        y=np.asarray(state)
        if owner not in self._external or y.shape!=self._b.shape or y.dtype.kind not in 'if' or not np.isfinite(y).all():raise ValueError('Boundary/state admission')
        v,c=self._external[owner];return float(v@y+c)
    def manifest(self):return deepcopy(self._manifest)
    def residual(self,state,rate):
        y=np.asarray(state);d=np.asarray(rate)
        if y.shape!=self._b.shape or d.shape!=y.shape or y.dtype.kind not in 'if' or d.dtype.kind not in 'if' or not np.isfinite(y).all() or not np.isfinite(d).all():raise ValueError('Finite state/rate required')
        return d-self._a@y-self._b
    def steady(self,scales):
        scales=np.asarray(scales)
        if scales.shape!=self._b.shape or scales.dtype.kind not in 'if' or not np.isfinite(scales).all() or np.any(scales<=0):raise ValueError('Positive equation scales required')
        scales=scales.astype(float,copy=True)
        scaled=self._a/scales[:,None]
        if np.linalg.matrix_rank(scaled)!=len(self._b):raise ValueError('Rank-deficient assembly')
        y=np.linalg.solve(scaled,-self._b/scales)
        residual=float(np.max(abs(self.residual(y,np.zeros_like(y)))/scales))
        if not np.isfinite(y).all() or residual>1e-8:raise ValueError('Inconsistent assembled stationary state')
        return y,residual
