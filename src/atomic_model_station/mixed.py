"""Bounded index-1 residual blocks, scaled Newton and implicit Euler stepping."""
from dataclasses import dataclass
import math
import numpy as np
from core.residual import ResidualBlock


@dataclass(frozen=True)
class Variable:
    owner:str
    name:str
    unit:str
    differential:bool
    scale:float
    lower:float
    upper:float
    def __post_init__(self):
        if any(type(s) is not str or not s or len(s)>80 for s in (self.owner,self.name,self.unit)) or type(self.differential) is not bool:raise ValueError('Variable declaration')
        if any(type(v) not in (int,float) or not math.isfinite(v) for v in (self.scale,self.lower,self.upper)) or self.scale<=0 or self.lower>=self.upper:raise ValueError('Variable bounds/scale')
    @property
    def key(self):return self.owner+'.'+self.name


class ResidualComponent:
    def __init__(self,owner,variables,imports,equation_scales,evaluate):
        if type(owner) is not str or not owner or not isinstance(variables,(list,tuple)) or not variables or any(type(v) is not Variable or v.owner!=owner for v in variables):raise ValueError('Owned component variables required')
        if type(imports) is not dict or len(imports)>64 or not callable(evaluate):raise ValueError('Explicit imports/callback required')
        scales=np.asarray(equation_scales)
        if scales.shape!=(len(variables),) or scales.dtype.kind not in 'if' or not np.isfinite(scales).all() or np.any(scales<=0):raise ValueError('Equation scales')
        self.owner=owner;self.variables=tuple(variables);self.imports=dict(imports);self.scales=scales.astype(float,copy=True);self.evaluate=evaluate


class MixedSystem(ResidualBlock):
    def __init__(self,components,validate=None):
        if not isinstance(components,(tuple,list)) or not 1<=len(components)<=32:raise ValueError('Component budget')
        if any(type(c) is not ResidualComponent for c in components) or len({c.owner for c in components})!=len(components):raise ValueError('Unique residual component owners')
        self.components=tuple(components);self.variables=tuple(v for c in components for v in c.variables)
        if not 1<=len(self.variables)<=64 or len({v.key for v in self.variables})!=len(self.variables):raise ValueError('Unique bounded variable map')
        self.index={v.key:i for i,v in enumerate(self.variables)}
        self.size=len(self.variables)
        for c in components:
            for key,unit in c.imports.items():
                if key not in self.index or self.variables[self.index[key]].unit!=unit:raise ValueError('Unknown/unit-incompatible component import')
        self.mask=np.array([v.differential for v in self.variables]);self.scales=np.array([v.scale for v in self.variables]);self.eq_scales=np.concatenate([c.scales for c in components])
        self.extra_validate=validate
    def admit(self,y):
        y=np.asarray(y)
        if y.shape!=(len(self.variables),) or y.dtype.kind not in 'if' or not np.isfinite(y).all() or any(not v.lower<=x<=v.upper for v,x in zip(self.variables,y,strict=True)):raise ValueError('Mixed state outside declared domain')
        if self.extra_validate is not None:self.extra_validate(y)
    def residual(self,t,y,rate,u):
        if type(t) not in (int,float) or not math.isfinite(t):raise ValueError('Finite residual time')
        self.admit(y);rate=np.asarray(rate)
        if rate.shape!=self.scales.shape or rate.dtype.kind not in 'if' or not np.isfinite(rate).all():raise ValueError('Finite mixed rate required')
        values={v.key:float(y[i]) for i,v in enumerate(self.variables)};rates={v.key:float(rate[i]) for i,v in enumerate(self.variables)};result=[]
        for c in self.components:
            keys={v.key for v in c.variables}|set(c.imports)
            r=np.asarray(c.evaluate(t,{k:values[k] for k in keys},{k:rates[k] for k in keys},dict(u)))
            if r.shape!=(len(c.variables),) or r.dtype.kind not in 'if' or not np.isfinite(r).all():raise ValueError('Invalid component residual')
            result.extend(r.tolist())
        return np.array(result)
    def manifest(self):
        from dataclasses import asdict
        return dict(variables=[dict(index=i,**asdict(v)) for i,v in enumerate(self.variables)],
            components=[dict(owner=c.owner,imports=dict(c.imports),equation_scales=c.scales.tolist()) for c in self.components],scope='Index-1 residual blocks; no serialized executable callbacks')
    def diagnostics(self,t,y,rate,u):
        values=self.residual(t,y,rate,u);cursor=0;result={}
        for c in self.components:
            count=len(c.variables);local=values[cursor:cursor+count]
            result[c.owner]=dict(residual=local.tolist(),scaled_max=float(np.max(abs(local)/c.scales)),
                event_variables=[v.key for v in c.variables])
            cursor+=count
        return result
    def _newton(self,fun,guess):
        raw=np.asarray(guess)
        if raw.dtype.kind not in 'if' or not np.isfinite(raw).all():raise ValueError('Finite real Newton guess')
        x=np.array(guess,dtype=float,copy=True)
        for iteration in range(24):
            r=fun(x)
            if not np.isfinite(r).all():raise ValueError('Nonfinite Newton residual')
            jac=np.empty((len(r),len(x)))
            for j in range(len(x)):
                h=1e-6*max(1.,abs(x[j]));up=x.copy();down=x.copy();up[j]+=h;down[j]-=h
                try:jac[:,j]=(fun(up)-fun(down))/(2*h)
                except ValueError:
                    try:jac[:,j]=(fun(up)-r)/h
                    except ValueError:jac[:,j]=(r-fun(down))/h
            rank=int(np.linalg.matrix_rank(jac))
            if rank!=len(x):raise ValueError('Rank-deficient scaled initialization/step Jacobian')
            norm=float(np.max(abs(r)))
            if norm<1e-11:return x,dict(iterations=iteration,rank=rank,residual=norm)
            delta=np.linalg.solve(jac,-r);accepted=False
            for k in range(16):
                candidate=x+delta*(.5**k)
                try:rr=fun(candidate)
                except ValueError:continue
                if np.isfinite(rr).all() and np.max(abs(rr))<norm:x=candidate;accepted=True;break
            if not accepted:raise ValueError('Damped Newton cannot find an admissible step')
        raise ValueError('Mixed Newton budget exhausted')
    def steady(self,guess,u):
        x,receipt=self._newton(lambda z:self.residual(0.,z*self.scales,np.zeros(len(z)),u)/self.eq_scales,np.asarray(guess)/self.scales)
        return x*self.scales,receipt
    def consistent(self,t,state,u):
        self.admit(state);state=np.array(state,dtype=float,copy=True);alg=np.flatnonzero(~self.mask);diff=np.flatnonzero(self.mask)
        # Unknown algebraic values and differential rates; differential stocks fixed.
        def unpack(z):
            y=state.copy();rate=np.zeros_like(y);y[alg]=z[:len(alg)]*self.scales[alg];rate[diff]=z[len(alg):]*self.scales[diff]
            return y,rate
        guess=np.r_[state[alg]/self.scales[alg],np.zeros(len(diff))]
        x,receipt=self._newton(lambda z:self.residual(t,*unpack(z),u)/self.eq_scales,guess)
        y,rate=unpack(x)
        if not np.array_equal(y[diff],state[diff]):raise ValueError('Initialization changed differential inventories')
        return y,rate,receipt
    def step(self,t,state,h,u):
        if type(t) not in (int,float) or not math.isfinite(t):raise ValueError('Finite step time')
        if type(h) not in (int,float) or not math.isfinite(h) or h<=0:raise ValueError('Positive finite step')
        old=np.asarray(state)
        def rate(y):return np.where(self.mask,(y-old)/h,0.)
        x,receipt=self._newton(lambda z:self.residual(t+h,z*self.scales,rate(z*self.scales),u)/self.eq_scales,old/self.scales)
        y=x*self.scales;return y,rate(y),receipt
