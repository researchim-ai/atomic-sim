"""Accept the actual two-half-step solution; no Richardson state correction.

The error estimate is local. Independent reference/convergence checks are
required for any claim about global temporal accuracy.
"""
from pathlib import Path
import numpy as np
from atomic_model_multiphysics.provenance import seal,unseal,sha
from atomic_model_station_regular.model import _finite_real
from atomic_model_station_nonlinear import NonlinearSession,NonlinearConvergenceError


class AccuracyError(ValueError):
    def __init__(self,attempts,step,error_ratio,reason):
        super().__init__('Adaptive step refused without state change: '+reason)
        self.attempts=attempts;self.step=step;self.error_ratio=error_ratio


def _norm(states):
    value=sum((1 if (m.k,m.m)==(0,0) else 2)*np.real(np.vdot(q,m.mass@q)) for m,q in states)
    if not np.isfinite(value) or value<0:raise ValueError('Finite nonnegative velocity norm required')
    return float(np.sqrt(value))


def _difference(first,second):
    if len(first)!=len(second):raise ValueError('Same error-estimator spaces required')
    delta=[]
    for (a,q),(b,p) in zip(first,second):
        if a.fingerprint!=b.fingerprint:raise ValueError('Same error-estimator basis required')
        delta.append((a,q-p))
    return _norm(delta)


class AdaptiveSession:
    def __init__(self,core,*,atol=1e-11,rtol=1e-8,min_step=1e-7,max_step=.0025,max_rejections=16):
        if type(core) is not NonlinearSession:raise ValueError('NonlinearSession required')
        _finite_real(atol,'absolute accuracy',1e-14,1e-3);_finite_real(rtol,'relative accuracy',1e-12,1e-2)
        _finite_real(min_step,'minimum step',2e-10,1.);_finite_real(max_step,'maximum step',min_step,1.)
        if type(max_rejections) is not int or not 0<=max_rejections<=30:raise ValueError('Bounded rejection budget required')
        self._core=NonlinearSession.restore(core.checkpoint())
        self.atol=float(atol);self.rtol=float(rtol);self.min_step=float(min_step);self.max_step=float(max_step);self.max_rejections=max_rejections
        self.next_step=self.max_step;self.accepted=0;self.rejected=0

    @property
    def modes(self):return self._core.modes

    @property
    def time(self):return self._core.time

    def step(self,max_duration,force=None):
        _finite_real(max_duration,'maximum duration',self.min_step,1e3)
        if force is not None and type(force) is not dict and not callable(force):raise ValueError('Force mapping or deterministic callback(time_mid) required')
        h=min(max_duration,self.next_step,self.max_step);initial=self._core.checkpoint();initial_norm=_norm(self.modes)
        def value(time):return force(time) if callable(force) else force
        for attempt in range(self.max_rejections+1):
            try:
                full=NonlinearSession.restore(initial);full.advance(h,value(self.time+h/2))
                fine=NonlinearSession.restore(initial);fine.advance(h/2,value(self.time+h/4))
                last_start=fine.checkpoint()
                last,pressure=fine.advance(h/2,value(self.time+3*h/4))
                error=_difference(full.modes,fine.modes)/3
                budget=self.atol+self.rtol*max(initial_norm,_norm(fine.modes));ratio=error/budget
                if not np.isfinite(ratio):raise ValueError('Nonfinite adaptive error estimate')
            except NonlinearConvergenceError:
                ratio=None;error=None;budget=None;fine=None
            if ratio is not None and ratio<=1:
                factor=2. if ratio==0 else min(2.,max(.5,.9*ratio**(-1/3)))
                proposal=min(self.max_step,max(self.min_step,h*factor))
                record=dict(time=fine.time,accepted_duration=h,error_estimate=error,error_budget=budget,error_ratio=ratio,
                    rejected_trials=attempt,next_step=proposal,actual_half_step=h/2,pressure_time=last['midpoint_time'],
                    core_last_step=last,core_last_step_start=last_start,richardson_state_correction=False,
                    spatial_convergence_qualified=False,strong_wall_momentum_qualified=False,station_usable=False,validated=False)
                self._core=fine;self.next_step=proposal;self.accepted+=1;self.rejected+=attempt
                return record,pressure
            if attempt==self.max_rejections:raise AccuracyError(attempt+1,h,ratio,'rejection budget exhausted')
            factor=.5 if ratio is None else min(.8,max(.2,.8*ratio**(-1/3)))
            proposed=h*factor
            if proposed<self.min_step:
                if h<=self.min_step*(1+1e-12):raise AccuracyError(attempt+1,h,ratio,'minimum step reached')
                proposed=self.min_step
            if proposed>=h:raise AccuracyError(attempt+1,h,ratio,'step cannot decrease')
            h=proposed
        raise AssertionError('Unreachable adaptive controller state')

    def advance_to(self,target,force=None,*,max_steps=10000):
        _finite_real(target,'target time',0.,1e6)
        if target-self.time<self.min_step:raise ValueError('Target must advance at least minimum step')
        if type(max_steps) is not int or not 1<=max_steps<=20000:raise ValueError('Bounded interval step budget required')
        candidate=AdaptiveSession.restore(self.checkpoint());history=[];last=None;pressure=None
        for i in range(max_steps):
            remaining=target-candidate.time
            slack=max(1e-15,8*float(np.spacing(max(abs(target),abs(candidate.time)))))
            if remaining<=slack and history:break
            if remaining<candidate.min_step:raise AccuracyError(i,remaining,None,'endpoint remainder below minimum step')
            last,pressure=candidate.step(remaining,force)
            history.append({name:last[name] for name in ('time','accepted_duration','error_estimate','error_budget','error_ratio','rejected_trials','actual_half_step','pressure_time')})
        slack=max(1e-15,8*float(np.spacing(max(abs(target),abs(candidate.time)))))
        if abs(target-candidate.time)>slack:raise AccuracyError(len(history),target-candidate.time,None,'interval step budget exhausted')
        result=dict(time=candidate.time,target_time=target,endpoint_difference=target-candidate.time,
            accepted_steps=len(history),rejected_trials=candidate.rejected-self.rejected,
            maximum_local_error_ratio=max(row['error_ratio'] for row in history),steps=history,last_step=last,
            global_error_bound_claimed=False,station_usable=False,validated=False)
        self._core=candidate._core;self.next_step=candidate.next_step;self.accepted=candidate.accepted;self.rejected=candidate.rejected
        return result,pressure

    def checkpoint(self):
        return seal(dict(format='adaptive-nonlinear-step-doubling/v1',controller_sha256=sha(Path(__file__)),
            atol=self.atol,rtol=self.rtol,min_step=self.min_step,max_step=self.max_step,max_rejections=self.max_rejections,
            next_step=self.next_step,accepted=self.accepted,rejected=self.rejected,core=self._core.checkpoint()))

    @classmethod
    def restore(cls,checkpoint):
        p=unseal(checkpoint)
        expected={'format','controller_sha256','atol','rtol','min_step','max_step','max_rejections','next_step','accepted','rejected','core'}
        if type(p) is not dict or set(p)!=expected or p['format']!='adaptive-nonlinear-step-doubling/v1':raise ValueError('Complete adaptive checkpoint required')
        if p['controller_sha256']!=sha(Path(__file__)):raise ValueError('Adaptive controller identity differs')
        obj=cls(NonlinearSession.restore(p['core']),atol=p['atol'],rtol=p['rtol'],min_step=p['min_step'],max_step=p['max_step'],max_rejections=p['max_rejections'])
        _finite_real(p['next_step'],'next step',obj.min_step,obj.max_step)
        for name in ('accepted','rejected'):
            if type(p[name]) is not int or not 0<=p[name]<=10**9:raise ValueError('Bounded adaptive counters required')
        obj.next_step=p['next_step'];obj.accepted=p['accepted'];obj.rejected=p['rejected'];return obj
