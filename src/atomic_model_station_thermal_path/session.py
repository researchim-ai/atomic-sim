"""Atomic segment-boundary continuation with global physical budget bounds.

Each BDF segment starts a new solver history. Physical state and accumulated
quadratures are serialized exactly; uninterrupted BDF equivalence is numerical,
not bitwise. Bounds compose local sampled defects by the triangle inequality.
"""
import copy
import math
import numpy as np

from atomic_model_multiphysics.provenance import seal, unseal
from atomic_model_station_regular.model import _finite_real
from atomic_model_station_stokes_pressure import recover_pressure
from atomic_model_station_thermal import ThermalResult
from atomic_model_station_thermal_resolved import (
    ResolvedThermalSystem, resolve_thermal, audit_resolved,
    resolved_velocity_mode, resolved_scalar_mode,
)
from .provenance import runtime_identity


class TrajectoryAccuracyError(ValueError):
    def __init__(self, receipt):
        super().__init__('Thermal trajectory refused without state change')
        self.receipt = receipt


def _vector(value, size, name):
    if type(value) is not list or len(value)!=size or any(type(x) not in (int,float) for x in value):
        raise ValueError('Complete numeric '+name+' required')
    result=np.asarray(value,dtype=float)
    if not np.all(np.isfinite(result)):raise ValueError('Finite '+name+' required')
    return result


def _encode(q):
    return dict(real=q.real.tolist(),imag=q.imag.tolist())


def _decode(value,size):
    if type(value) is not dict or set(value)!={'real','imag'}:raise ValueError('Complex state envelope required')
    result=np.empty(size,dtype=complex)
    result.real=_vector(value['real'],size,'real state')
    result.imag=_vector(value['imag'],size,'imaginary state')
    return result


def _clone(system,q=None):
    q=system.initial if q is None else q
    v,t=system.split(q)
    return ResolvedThermalSystem(list(zip(system.models,v)),list(zip(system.scalars,t)),
        viscosity=float(system.viscosity),diffusivity=float(system.diffusivity),
        buoyancy=float(system.buoyancy),sources=system.sources)


def _with_state(system,q):
    """Reuse private prepared operators; copy and validate every physical coefficient."""
    v,t=system.split(q)
    obj=copy.copy(system)
    obj.velocity=[(m,np.array(a,complex,copy=True)) for m,a in zip(system.models,v)]
    obj.temperature=[(m,np.array(a,complex,copy=True)) for m,a in zip(system.scalars,t)]
    obj.initial=np.concatenate([a for m,a in obj.velocity+obj.temperature])
    for m,a in obj.velocity+obj.temperature:a.setflags(write=False)
    obj.initial.setflags(write=False)
    obj.calculate(obj.initial)
    return obj


def _combination(ledger):
    return np.array([ledger[0]-ledger[1],ledger[2]-ledger[3],ledger[5]-ledger[4]])


def _rollup(history,origin):
    ledger=np.zeros(6);endpoint=np.zeros(3);bound=np.zeros(3)
    for index,row in enumerate(history):
        guard=8*np.finfo(float).eps*(1+np.sum(abs(origin))+np.sum(abs(ledger))+np.sum(abs(np.asarray(row['observables']))))
        bound=np.maximum(bound,abs(endpoint)+np.asarray(row['local_maximum'])+guard)
        ledger=np.array([math.fsum(r['integrals'][i] for r in history[:index+1]) for i in range(6)])
        endpoint=np.asarray(row['observables'])-origin+_combination(ledger)
        bound=np.maximum(bound,abs(endpoint))
    return ledger,endpoint,bound


class ThermalTrajectory:
    def __init__(self,system,*,start_time=0.,max_segment=.1,rtol=1e-10,atol=1e-13,
                 heat_tolerance=1e-10,energy_tolerance=1e-9,wall_tolerance=1e-8):
        if type(system) is not ResolvedThermalSystem:raise ValueError('ResolvedThermalSystem required')
        _finite_real(start_time,'trajectory origin',0.,1e6)
        _finite_real(max_segment,'maximum segment',1e-8,1.)
        _finite_real(rtol,'trajectory relative tolerance',1e-13,1e-6)
        _finite_real(atol,'trajectory absolute tolerance',1e-15,1e-8)
        _finite_real(heat_tolerance,'path heat tolerance',1e-14,1e-3)
        _finite_real(energy_tolerance,'path energy tolerance',1e-14,1e-3)
        if wall_tolerance is not None:_finite_real(wall_tolerance,'path sampled wall tolerance',1e-14,1e-3)
        self._system=_clone(system)
        self._origin=self._system.initial.copy();self._origin.setflags(write=False)
        self._origin_observables=self._system.observables(self._origin)
        self._origin_observables.setflags(write=False)
        self._origin_time=float(start_time);self._time=float(start_time)
        self._policy=dict(max_segment=float(max_segment),rtol=float(rtol),atol=float(atol),
            heat_tolerance=float(heat_tolerance),energy_tolerance=float(energy_tolerance),
            wall_tolerance=None if wall_tolerance is None else float(wall_tolerance))
        self._history=[];self._identity=runtime_identity()

    @property
    def time(self):return self._time

    @property
    def segments(self):return copy.deepcopy(self._history)

    def status(self):
        ledger,endpoint,bound=_rollup(self._history,self._origin_observables)
        return dict(time=self.time,origin_time=self._origin_time,accepted_segments=len(self._history),
            integrated_rates=ledger.tolist(),endpoint_global_defects=endpoint.tolist(),
            sampled_global_defect_upper_bounds=bound.tolist(),
            budget_order=['kinetic','scalar_quadratic','physical_heat'],
            conservative_composition_bound=True,continuous_error_bound=False,
            bdf_history_serialized=False,station_usable=False,validated=False)

    def endpoint(self):
        if self._identity!=runtime_identity():raise ValueError('Thermal trajectory source changed')
        system=self._system;q=system.initial
        dq,_,fields=system.calculate(q)
        v,t=system.split(q);dv,dt=system.split(dq);pressure={};weak=0.
        for i,(key,m,u,d,theta) in enumerate(zip(system.keys,system.models,v,dv,t)):
            force=-fields['nonlinear'][key].copy()
            force[2]+=system.buoyancy*(system.scalars[i].basis@theta)
            reaction=m.load(force)-m.mass_full@m.coefficients(d)-system.viscosity*m.stiffness_full@m.coefficients(u)
            field,proof=recover_pressure(m,reaction);pressure[key]=field
            weak=max(weak,proof['weak_momentum_relative_residual'])
        status=self.status()
        report=dict(time=self.time,start_time=self._origin_time,equation_fingerprint=system.equation_fingerprint,
            viscosity=system.viscosity,diffusivity=system.diffusivity,buoyancy=system.buoyancy,
            maximum_physical_heat_budget_residual=status['sampled_global_defect_upper_bounds'][2],
            heat_budget_is_composed_upper_bound=True,maximum_weak_pressure_residual=weak,
            station_usable=False,validated=False)
        pairs=lambda models,values:[(m,z.copy()) for m,z in zip(models,values)]
        return ThermalResult(pairs(system.models,v),pairs(system.scalars,t),pairs(system.models,dv),pairs(system.scalars,dt),pressure,report)

    def audit(self):
        return audit_resolved(self._system,self.endpoint(),radius=np.linspace(.005,1.,129))

    def advance_to(self,target,*,max_segments=256):
        _finite_real(target,'trajectory target',0.,1e6)
        if target-self.time<1e-8:raise ValueError('Forward trajectory interval >=1e-8 required')
        if type(max_segments) is not int or not 1<=max_segments<=256:raise ValueError('Segment budget1..256 required')
        if self._identity!=runtime_identity():raise ValueError('Thermal trajectory source changed')
        candidate=copy.copy(self)
        candidate._system=_with_state(self._system,self._system.initial)
        candidate._history=copy.deepcopy(self._history)
        candidate._policy=copy.deepcopy(self._policy)
        candidate._identity=copy.deepcopy(self._identity)
        before=len(candidate._history)
        for _ in range(max_segments):
            if candidate.time==target:break
            if len(candidate._history)>=256:raise TrajectoryAccuracyError(dict(reason='total segment budget',time=candidate.time))
            slack=8*float(np.spacing(max(1.,abs(target),abs(candidate.time))))
            end=(target if target-candidate.time<=candidate._policy['max_segment']+slack
                 else candidate.time+candidate._policy['max_segment'])
            if end<=candidate.time:raise TrajectoryAccuracyError(dict(reason='time cannot advance',time=candidate.time))
            result=resolve_thermal(candidate._system,start_time=candidate.time,end_time=end,
                rtol=candidate._policy['rtol'],atol=candidate._policy['atol'])
            audit=audit_resolved(candidate._system,result,radius=np.linspace(.005,1.,129))
            report=result.report
            state=np.concatenate([q for m,q in result.velocity+result.temperature])
            row=dict(start_time=candidate.time,time=float(end),state_sha256=seal(_encode(state))['sha256'],integrals=report['integrated_rates'],
                local_maximum=[report['maximum_kinetic_budget_residual'],report['maximum_variance_budget_residual'],report['maximum_physical_heat_budget_residual']],
                observables=report['final'],nfev=report['nfev'],solver_steps=report['solver_steps'],
                wall=dict(momentum=audit['maximum_wall_momentum_residual'],thermal=audit['maximum_wall_thermal_residual'],heat_rate=audit['physical_heat_rate_residual']))
            proposed=candidate._history+[row]
            _,_,bound=_rollup(proposed,candidate._origin_observables)
            policy=candidate._policy
            wall=max(row['wall']['momentum'],row['wall']['thermal'])
            if (np.any(bound[:2]>policy['energy_tolerance']) or bound[2]>policy['heat_tolerance']
                    or (policy['wall_tolerance'] is not None and wall>policy['wall_tolerance'])):
                raise TrajectoryAccuracyError(dict(reason='global budget or sampled segment-end wall',time=end,
                    sampled_global_defect_upper_bounds=bound.tolist(),wall=row['wall'],policy=copy.deepcopy(policy)))
            candidate._system=_with_state(candidate._system,state)
            candidate._time=float(end);candidate._history=proposed
        if candidate.time!=target:
            raise TrajectoryAccuracyError(dict(reason='interval segment budget',target=target,time=candidate.time))
        if candidate._identity!=runtime_identity():raise ValueError('Thermal trajectory source changed')
        summary=candidate.status();summary['advanced_segments']=len(candidate._history)-before
        self.__dict__.update(candidate.__dict__)
        return summary

    def checkpoint(self):
        if self._identity!=runtime_identity():raise ValueError('Thermal trajectory source changed')
        system=self._system
        def spec(m):return dict(k=m.k,m=m.m,alpha=m.alpha,order=m.order,quadrature=m.quadrature)
        basis=[dict(velocity=spec(v),temperature=spec(t),velocity_fingerprint=v.fingerprint,
                    temperature_fingerprint=t.fingerprint) for v,t in zip(system.models,system.scalars)]
        sources=[dict(key=list(key),coefficients=_encode(system.sources[key])) for key in system.keys]
        status=self.status()
        return seal(dict(format='thermal-trajectory-boundary/v1',identity=copy.deepcopy(self._identity),
            basis=basis,sources=sources,parameters=dict(viscosity=system.viscosity,diffusivity=system.diffusivity,buoyancy=system.buoyancy),
            origin_time=self._origin_time,time=self.time,origin=_encode(self._origin),current=_encode(system.initial),
            policy=copy.deepcopy(self._policy),history=copy.deepcopy(self._history),
            integrated_rates=status['integrated_rates'],global_bounds=status['sampled_global_defect_upper_bounds']))

    @classmethod
    def restore(cls,checkpoint):
        p=unseal(checkpoint)
        expected={'format','identity','basis','sources','parameters','origin_time','time','origin','current','policy','history','integrated_rates','global_bounds'}
        if type(p) is not dict or set(p)!=expected or p['format']!='thermal-trajectory-boundary/v1':raise ValueError('Complete thermal trajectory checkpoint required')
        if p['identity']!=runtime_identity():raise ValueError('Thermal checkpoint source/environment differs')
        if type(p['basis']) is not list or not 1<=len(p['basis'])<=61:raise ValueError('Bounded thermal checkpoint basis required')
        models=[];scalars=[]
        for row in p['basis']:
            if type(row) is not dict or set(row)!={'velocity','temperature','velocity_fingerprint','temperature_fingerprint'}:raise ValueError('Complete paired basis record required')
            for name,factory,collection in (('velocity',resolved_velocity_mode,models),('temperature',resolved_scalar_mode,scalars)):
                spec=row[name]
                if type(spec) is not dict or set(spec)!={'k','m','alpha','order','quadrature'}:raise ValueError('Complete basis specification required')
                model=factory(**spec)
                if model.fingerprint!=row[name+'_fingerprint']:raise ValueError('Thermal checkpoint basis identity differs')
                collection.append(model)
        n=sum(m.dimension for m in models+scalars)
        origin=_decode(p['origin'],n);current=_decode(p['current'],n)
        if type(p['sources']) is not list or len(p['sources'])!=len(models):raise ValueError('Complete polynomial sources required')
        sources={}
        for row in p['sources']:
            if type(row) is not dict or set(row)!={'key','coefficients'}:raise ValueError('Complete source record required')
            key=row['key']
            if type(key) is not list or len(key)!=2 or any(type(x) is not int for x in key):raise ValueError('Integer source keys required')
            key=tuple(key)
            if key in sources:raise ValueError('Duplicate source key')
            encoded=row['coefficients']
            if type(encoded) is not dict or set(encoded)!={'real','imag'} or type(encoded['real']) is not list or not 1<=len(encoded['real'])<=17:raise ValueError('Bounded complex source coefficients required')
            sources[key]=_decode(encoded,len(encoded['real']))
        keys=[(m.k,m.m) for m in models]
        if keys!=sorted(set(keys)) or set(sources)!=set(keys):raise ValueError('Canonical checkpoint basis/source order required')
        if type(p['parameters']) is not dict or set(p['parameters'])!={'viscosity','diffusivity','buoyancy'}:raise ValueError('Complete physical parameters required')
        if type(p['policy']) is not dict or set(p['policy'])!={'max_segment','rtol','atol','heat_tolerance','energy_tolerance','wall_tolerance'}:raise ValueError('Complete trajectory policy required')
        cuts=np.cumsum([0]+[m.dimension for m in models+scalars])
        def system(q):
            values=[q[cuts[i]:cuts[i+1]] for i in range(len(cuts)-1)];nv=len(models)
            return ResolvedThermalSystem(list(zip(models,values[:nv])),list(zip(scalars,values[nv:])),sources=sources,**p['parameters'])
        obj=cls(system(origin),start_time=p['origin_time'],**p['policy'])
        _finite_real(p['time'],'checkpoint time',obj._origin_time,1e6)
        history=p['history']
        if type(history) is not list or len(history)>256:raise ValueError('Bounded thermal history required')
        previous=obj._origin_time
        for row in history:
            if type(row) is not dict or set(row)!={'start_time','time','state_sha256','integrals','local_maximum','observables','nfev','solver_steps','wall'}:raise ValueError('Complete thermal history record required')
            _finite_real(row['time'],'history time',previous,1e6)
            _finite_real(row['start_time'],'history start',obj._origin_time,1e6)
            if row['start_time']!=previous or row['time']<=previous or row['time']-previous>obj._policy['max_segment']+8*np.spacing(max(1.,row['time'])):raise ValueError('Contiguous bounded thermal history required')
            if type(row['state_sha256']) is not str or len(row['state_sha256'])!=64 or any(x not in '0123456789abcdef' for x in row['state_sha256']):raise ValueError('State digest required')
            for name,size in (('integrals',6),('local_maximum',3),('observables',3)):_vector(row[name],size,name)
            if any(row['integrals'][i]<0 for i in (0,2)):raise ValueError('Nonnegative integrated dissipation required')
            if any(x<0 for x in row['local_maximum']) or any(row['observables'][i]<0 for i in (0,1)):raise ValueError('Nonnegative budget magnitudes required')
            for name in ('nfev','solver_steps'):
                if type(row[name]) is not int or not 1<=row[name]<=10**8:raise ValueError('Bounded thermal solver counters required')
            if type(row['wall']) is not dict or set(row['wall'])!={'momentum','thermal','heat_rate'}:raise ValueError('Complete sampled wall evidence required')
            for value in row['wall'].values():_finite_real(value,'wall residual',0.,1e9)
            wall=obj._policy['wall_tolerance']
            if wall is not None and max(row['wall']['momentum'],row['wall']['thermal'])>wall:raise ValueError('Checkpoint contains refused wall segment')
            previous=row['time']
        if previous!=p['time']:raise ValueError('Checkpoint time/history differ')
        obj._system=_with_state(obj._system,current)
        if history:
            if history[-1]['state_sha256']!=seal(_encode(current))['sha256']:raise ValueError('Checkpoint endpoint state digest differs')
            if not np.array_equal(obj._system.observables(current),np.array(history[-1]['observables'])):raise ValueError('Checkpoint endpoint observables differ')
        elif not np.array_equal(current,origin):raise ValueError('Empty history must preserve origin state')
        obj._history=copy.deepcopy(history);obj._time=float(p['time'])
        ledger,_,bound=_rollup(history,obj._origin_observables)
        if not np.array_equal(ledger,_vector(p['integrated_rates'],6,'integrated rates')) or not np.array_equal(bound,_vector(p['global_bounds'],3,'global bounds')):raise ValueError('Checkpoint cumulative budgets differ')
        if np.any(bound[:2]>obj._policy['energy_tolerance']) or bound[2]>obj._policy['heat_tolerance']:raise ValueError('Checkpoint contains refused global budget')
        return obj
