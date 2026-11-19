"""Atomic implicit-midpoint multi-mode viscous/advection dynamics.

Pressure is reconstructed in retained weak momentum only. This is not a
strong-wall, turbulent, thermal, or plant qualification.
"""
from pathlib import Path
import platform
import numpy as np
import scipy
from scipy.linalg import cho_factor,cho_solve
from atomic_model_multiphysics.provenance import sha,seal,unseal
from atomic_model_station_regular import regular_mode,RegularMode
from atomic_model_station_regular.model import _finite_real
from atomic_model_station_advection import signed_spectrum,convolution_advection,fft_advection,nonlinear_power
from atomic_model_station_advection import kernel
from atomic_model_station_stokes_pressure import recover_step_pressure
from atomic_model_station_stokes_pressure import field as pressure_module
from atomic_model_station_wall import operators as velocity_module


class NonlinearConvergenceError(ValueError):
    def __init__(self,iterations,residual):
        super().__init__('Nonlinear midpoint iteration did not converge; candidate not committed')
        self.iterations=iterations;self.residual=residual


def _policy():
    return dict(session=sha(Path(__file__)),advection=sha(kernel.__file__),pressure=sha(pressure_module.__file__),
        velocity=sha(velocity_module.__file__),numpy=np.__version__,scipy=scipy.__version__,python=platform.python_version())


class NonlinearSession:
    def __init__(self,modes,*,viscosity=.1,method='fft',tolerance=1e-12,max_iterations=40):
        _finite_real(viscosity,'viscosity',0.,1e3);_finite_real(tolerance,'nonlinear tolerance',1e-13,1e-10)
        if type(max_iterations) is not int or not 1<=max_iterations<=100 or method not in ('fft','convolution'):raise ValueError('Bounded explicit nonlinear solver policy required')
        if type(modes) not in (list,tuple) or not 1<=len(modes)<=41:raise ValueError('Bounded canonical mode sequence required')
        self._models={};self._q={}
        for model,q in modes:
            if type(model) is not RegularMode:raise ValueError('RegularMode required')
            key=(model.k,model.m)
            if key in self._models:raise ValueError('Duplicate mode')
            model.coefficients(q);self._models[key]=model;self._q[key]=np.array(q,dtype=complex,copy=True)
        if (0,0) not in self._models or sum(m.dimension for m in self._models.values())>256:raise ValueError('Mean mode and bounded reduced dimension required')
        self._keys=sorted(self._models)
        self._models={key:self._models[key] for key in self._keys};self._q={key:self._q[key] for key in self._keys}
        first=self._models[self._keys[0]]
        self.radius=first.radius;self.weights=first.weights;self.alpha=first.alpha
        for model in self._models.values():
            if model.alpha!=self.alpha or not np.array_equal(model.radius,self.radius) or not np.array_equal(model.weights,self.weights):raise ValueError('Common exact radial quadrature and alpha required')
        velocity,_=signed_spectrum(self._states(self._q),self.radius,self.alpha)
        self._targets=sorted(velocity);self.viscosity=float(viscosity);self.method=method;self.tolerance=float(tolerance);self.max_iterations=max_iterations
        self.time=0.;self.work=0.;self.dissipation=0.;self.advection_work=0.;self.initial_energy=self._energy(self._q)
        if not np.isfinite(self.initial_energy):raise ValueError('Finite initial kinetic energy required')

    def _states(self,q):return [(self._models[key],q[key]) for key in self._keys]

    @property
    def modes(self):return [(model,q.copy()) for model,q in self._states(self._q)]

    def _multiple(self,key):return 1 if key==(0,0) else 2

    def _energy(self,q):
        return float(sum(self._multiple(key)*.5*np.real(np.vdot(q[key],self._models[key].mass@q[key])) for key in self._keys))

    def _advection(self,q):
        velocity,radial=signed_spectrum(self._states(q),self.radius,self.alpha)
        if self.method=='fft':
            nonlinear,proof=fft_advection(self.radius,self.alpha,velocity,radial,self._targets)
            proof={key:proof[key] for key in ('grid','required_grid','alias_free_for_requested_targets','radial_points','input_modes','output_modes','axis_sampled')}
        else:
            nonlinear=convolution_advection(self.radius,self.alpha,velocity,radial,self._targets)
            proof=dict(alias_free_for_requested_targets=True,method='exact mode convolution')
        loads={key:self._models[key].load(nonlinear[key]) for key in self._keys}
        return loads,nonlinear_power(velocity,nonlinear,self.weights),proof

    def advance(self,dt,force=None):
        _finite_real(dt,'step',1e-10,1e3)
        next_time=self.time+dt
        if not np.isfinite(next_time) or next_time<=self.time:raise ValueError('Step cannot advance finite clock')
        if force is not None and (type(force) is not dict or set(force)!=set(self._keys)):raise ValueError('Complete canonical generalized force mapping required')
        external={}
        for key in self._keys:
            model=self._models[key];a=np.zeros(3*model.order,complex) if force is None else np.asarray(force[key],dtype=complex)
            if a.shape!=(3*model.order,) or not np.all(np.isfinite(a)):raise ValueError('Finite complete generalized force per mode required')
            external[key]=a.copy()
        before={key:q.copy() for key,q in self._q.items()};candidate={key:q.copy() for key,q in before.items()}
        factors={};base={}
        for key,model in self._models.items():
            factors[key]=cho_factor(model.mass+.5*dt*self.viscosity*model.stiffness)
            base[key]=(model.mass-.5*dt*self.viscosity*model.stiffness)@before[key]
        residual=float('inf');effective=None
        for iteration in range(1,self.max_iterations+1):
            mid={key:(before[key]+candidate[key])/2 for key in self._keys}
            loads,power,fft_proof=self._advection(mid)
            effective={key:external[key]-loads[key] for key in self._keys};residual=0.
            for key,model in self._models.items():
                acceleration=model.mass@(candidate[key]-before[key])/dt
                viscous=self.viscosity*model.stiffness@mid[key]
                forcing=model.transform.conj().T@effective[key]
                scale=max(1.,float(np.linalg.norm(acceleration)),float(np.linalg.norm(viscous)),float(np.linalg.norm(forcing)))
                residual=max(residual,float(np.linalg.norm(acceleration+viscous-forcing))/scale)
            if not np.isfinite(residual):raise ValueError('Nonfinite nonlinear residual')
            if residual<=self.tolerance:break
            if iteration==self.max_iterations:raise NonlinearConvergenceError(iteration,residual)
            candidate={key:cho_solve(factors[key],base[key]+dt*self._models[key].transform.conj().T@effective[key]) for key in self._keys}
        energy=self._energy(candidate);old_energy=self._energy(before)
        delta_work=float(dt*sum(self._multiple(key)*np.real(np.vdot(self._models[key].coefficients(mid[key]),external[key])) for key in self._keys))
        delta_diss=float(dt*self.viscosity*sum(self._multiple(key)*np.real(np.vdot(mid[key],self._models[key].stiffness@mid[key])) for key in self._keys))
        delta_advection=dt*power['advection_power'];balance=energy-old_energy-delta_work+delta_diss+delta_advection
        work=self.work+delta_work;dissipation=self.dissipation+delta_diss;advection_work=self.advection_work+delta_advection
        total=energy-self.initial_energy-work+dissipation+advection_work
        if not np.all(np.isfinite([energy,delta_work,delta_diss,delta_advection,balance,work,dissipation,total])):raise ValueError('Nonfinite nonlinear energy ledger')
        scale=max(1.,energy,old_energy,abs(work),dissipation)
        if abs(balance)>1e-9*scale or abs(total)>1e-9*scale or power['absolute_advection_power']>1e-9*max(1.,energy**1.5):raise ValueError('Nonlinear energy or advection conservation gate failed')
        fields={};proofs=[]
        for key,model in self._models.items():
            field,proof=recover_step_pressure(model,before[key],candidate[key],dt,effective[key],self.viscosity)
            fields[key]=field;proofs.append(dict(k=key[0],m=key[1],proof=proof))
        receipt=dict(time=next_time,midpoint_time=self.time+dt/2,iterations=iteration,nonlinear_relative_residual=residual,
            kinetic_energy=energy,external_work=work,viscous_dissipation=dissipation,advection_work=advection_work,
            step_energy_residual=balance,total_energy_residual=total,advection_power=power['advection_power'],
            alias_control=fft_proof,pressures=proofs,nonlinear_pressure_recovered=True,
            pressure_scope='Retained weak momentum including midpoint advection',strong_wall_momentum_qualified=False,
            thermal_model_supported=False,station_usable=False,validated=False)
        self._q=candidate;self.time=next_time;self.work=work;self.dissipation=dissipation;self.advection_work=advection_work
        return receipt,fields

    def checkpoint(self):
        records=[dict(spec=model.spec,fingerprint=model.fingerprint,q_real=q.real.tolist(),q_imag=q.imag.tolist()) for model,q in self._states(self._q)]
        return seal(dict(format='multimode-midpoint-galerkin/v1',policy=_policy(),modes=records,viscosity=self.viscosity,
            method=self.method,tolerance=self.tolerance,max_iterations=self.max_iterations,time=self.time,work=self.work,
            dissipation=self.dissipation,advection_work=self.advection_work,initial_energy=self.initial_energy))

    @classmethod
    def restore(cls,checkpoint):
        p=unseal(checkpoint)
        required={'format','policy','modes','viscosity','method','tolerance','max_iterations','time','work','dissipation','advection_work','initial_energy'}
        if type(p) is not dict or set(p)!=required or p['format']!='multimode-midpoint-galerkin/v1':raise ValueError('Complete nonlinear checkpoint required')
        if p['policy']!=_policy():raise ValueError('Nonlinear implementation or environment differs')
        if type(p['modes']) is not list or not 1<=len(p['modes'])<=41:raise ValueError('Bounded checkpoint mode list required')
        modes=[]
        for row in p['modes']:
            if type(row) is not dict or set(row)!={'spec','fingerprint','q_real','q_imag'}:raise ValueError('Complete mode checkpoint required')
            model=regular_mode(**row['spec'])
            if model.fingerprint!=row['fingerprint']:raise ValueError('Nonlinear velocity-space fingerprint differs')
            real=np.asarray(row['q_real'],dtype=float);imag=np.asarray(row['q_imag'],dtype=float)
            if real.shape!=(model.dimension,) or imag.shape!=real.shape:raise ValueError('Complete nonlinear coefficient vectors required')
            modes.append((model,real+1j*imag))
        obj=cls(modes,viscosity=p['viscosity'],method=p['method'],tolerance=p['tolerance'],max_iterations=p['max_iterations'])
        for name in ('time','work','dissipation','advection_work','initial_energy'):
            v=p[name]
            if type(v) not in (int,float) or not np.isfinite(v) or (name in ('time','dissipation','initial_energy') and v<0):raise ValueError('Finite consistent nonlinear ledger required')
        total=obj._energy(obj._q)-p['initial_energy']-p['work']+p['dissipation']+p['advection_work']
        if abs(total)>1e-9*max(1.,obj._energy(obj._q),abs(p['work']),p['dissipation']):raise ValueError('Nonlinear checkpoint energy budget inconsistent')
        obj.time=p['time'];obj.work=p['work'];obj.dissipation=p['dissipation'];obj.advection_work=p['advection_work'];obj.initial_energy=p['initial_energy']
        return obj
