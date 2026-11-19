"""Implicit-midpoint constrained Stokes dynamics with work/dissipation ledger."""
from pathlib import Path
import numpy as np
from scipy.linalg import solve,lstsq
from atomic_model_multiphysics.provenance import seal,unseal,sha
from .model import RegularMode,regular_mode,_finite_real


def _energy(model,q):return float(.5*np.real(np.vdot(q,model.mass@q)))


class StokesSession:
    def __init__(self,model,q,*,viscosity=.1):
        if type(model) is not RegularMode:raise ValueError('RegularMode required')
        _finite_real(viscosity,'viscosity',0.,1e3)
        model.coefficients(q)
        self.model=model;self.q=np.array(q,dtype=complex,copy=True);self.viscosity=float(viscosity)
        self.time=0.;self.work=0.;self.dissipation=0.;self.initial_energy=_energy(model,self.q)

    def advance(self,dt,force=None):
        _finite_real(dt,'time step',1e-10,1e3)
        model=self.model;nu=self.viscosity
        f=np.zeros(3*model.order,complex) if force is None else np.asarray(force,dtype=complex)
        if f.shape!=(3*model.order,) or not np.all(np.isfinite(f)):raise ValueError('Finite full-space generalized force required')
        before=self.q.copy();a=model.mass+.5*dt*nu*model.stiffness
        rhs=(model.mass-.5*dt*nu*model.stiffness)@before+dt*model.transform.conj().T@f
        candidate=solve(a,rhs,assume_a='pos');mid=(before+candidate)/2
        delta_work=float(dt*np.real(np.vdot(model.transform@mid,f)))
        delta_dissipation=float(dt*nu*np.real(np.vdot(mid,model.stiffness@mid)))
        energy=_energy(model,candidate);balance=energy-_energy(model,before)-delta_work+delta_dissipation
        full_before=model.transform@before;full_after=model.transform@candidate
        reaction=f-model.mass_full@(full_after-full_before)/dt-.5*nu*model.stiffness_full@(full_after+full_before)
        dual=lstsq(model.constraints.conj().T,reaction,cond=1e-12,lapack_driver='gelsy')[0]
        reaction_error=float(np.linalg.norm(reaction-model.constraints.conj().T@dual))
        relative_reaction=reaction_error/max(1.,float(np.linalg.norm(f)),float(np.linalg.norm(reaction)))
        wall=float(np.max(abs(model.evaluate(candidate,[1.]))));continuity=float(np.max(abs(model.analytic_divergence(candidate,model.radius))))
        if not np.all(np.isfinite([energy,delta_work,delta_dissipation,balance,relative_reaction,wall,continuity,self.work+delta_work,self.dissipation+delta_dissipation])) or not np.all(np.isfinite(candidate)) or abs(balance)>1e-9*max(1.,energy,abs(delta_work)) or relative_reaction>1e-8 or wall>1e-9 or continuity>1e-8:
            raise ValueError('Constrained Stokes candidate failed balance, reaction or kinematic gate')
        next_time=self.time+dt
        if not np.isfinite(next_time) or next_time<=self.time:raise ValueError('Time step cannot advance finite clock')
        self.q=candidate;self.time=next_time;self.work+=delta_work;self.dissipation+=delta_dissipation
        return dict(time=self.time,kinetic_energy=energy,work=self.work,dissipation=self.dissipation,
            step_energy_residual=balance,total_energy_residual=energy-self.initial_energy-self.work+self.dissipation,
            constraint_reaction_relative_residual=relative_reaction,wall_maximum=wall,analytic_divergence_maximum=continuity,
            constraint_reaction_work=float(dt*np.real(np.vdot(model.transform@mid,reaction))),
            physical_pressure_reconstructed=False,nonlinear_advection_supported=False,station_usable=False,validated=False)

    def checkpoint(self):
        return seal(dict(format='regular-constrained-stokes/v1',spec=self.model.spec,fingerprint=self.model.fingerprint,integrator_sha256=sha(Path(__file__)),
            q_real=self.q.real.tolist(),q_imag=self.q.imag.tolist(),viscosity=self.viscosity,time=self.time,
            work=self.work,dissipation=self.dissipation,initial_energy=self.initial_energy))

    @classmethod
    def restore(cls,checkpoint):
        p=unseal(checkpoint)
        expected={'format','spec','fingerprint','integrator_sha256','q_real','q_imag','viscosity','time','work','dissipation','initial_energy'}
        if type(p) is not dict or set(p)!=expected:raise ValueError('Complete regular Stokes checkpoint schema required')
        if p['format']!='regular-constrained-stokes/v1':raise ValueError('Regular Stokes checkpoint format differs')
        if p['integrator_sha256']!=sha(Path(__file__)):raise ValueError('Stokes integrator identity differs')
        model=regular_mode(**p['spec'])
        if p['fingerprint']!=model.fingerprint:raise ValueError('Model or numerical environment identity differs')
        real=np.asarray(p['q_real']);imag=np.asarray(p['q_imag'])
        if real.shape!=(model.dimension,) or imag.shape!=real.shape:raise ValueError('Complete checkpoint coefficient vectors required')
        session=cls(model,real+1j*imag,viscosity=p['viscosity'])
        for name in ('time','work','dissipation','initial_energy'):
            value=p[name]
            if type(value) not in (int,float) or not np.isfinite(value) or (name!='work' and value<0):raise ValueError('Finite physical checkpoint ledger required')
        energy=_energy(model,session.q)
        if abs(energy-p['initial_energy']-p['work']+p['dissipation'])>1e-9*max(1.,energy,abs(p['work'])):
            raise ValueError('Checkpoint energy ledger inconsistent')
        session.time=p['time'];session.work=p['work'];session.dissipation=p['dissipation'];session.initial_energy=p['initial_energy']
        return session
