"""Bounded nondimensional Boussinesq research system in a vertical pipe.

The anomaly is zero at an isothermal wall. Gravity/buoyancy points along
+z; axial data are periodic. No variable-density EOS or plant claims.
"""
import hashlib

import numpy as np
from numpy.polynomial import Legendre
from scipy.integrate import solve_ivp
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse import bmat, block_diag, csc_matrix

from atomic_model_station_regular import RegularMode
from atomic_model_station_regular.model import _finite_real
from atomic_model_station_refined.products import signed_modes, expanded_fft, direct_products
from atomic_model_station_stokes_pressure import recover_pressure
from atomic_model_station_thermal.scalar import ScalarMode
from atomic_model_station_thermal.system import ThermalResult
from .products import scalar_products


class ResolvedThermalSystem:
    """Copied initial data, static polynomial sources, common resolved quadrature."""

    def __init__(self, velocity, temperature, *, viscosity=.05, diffusivity=.03,
                 buoyancy=.2, sources=None):
        _finite_real(viscosity, 'viscosity', 1e-6, 10.)
        _finite_real(diffusivity, 'thermal diffusivity', 1e-6, 10.)
        _finite_real(buoyancy, 'signed buoyancy', -10., 10.)
        for state, cls in ((velocity, RegularMode), (temperature, ScalarMode)):
            if (type(state) not in (list, tuple) or not 1 <= len(state) <= 61
                    or any(type(pair) not in (list, tuple) or len(pair) != 2
                           or type(pair[0]) is not cls for pair in state)):
                raise ValueError('Bounded typed velocity and temperature pairs required')
        self.velocity = sorted([(m, np.array(q, complex, copy=True)) for m,q in velocity], key=lambda p:(p[0].k,p[0].m))
        self.temperature = sorted([(m, np.array(q, complex, copy=True)) for m,q in temperature], key=lambda p:(p[0].k,p[0].m))
        self.models = [m for m,q in self.velocity]
        self.scalars = [m for m,q in self.temperature]
        self.keys = [(m.k,m.m) for m in self.models]
        if (len(set(self.keys)) != len(self.keys) or (0,0) not in self.keys
                or self.keys != [(m.k,m.m) for m in self.scalars]):
            raise ValueError('Unique identical canonical modes including the mean required')
        self.radius, self.weights, self.alpha = self.models[0].radius, self.models[0].weights, self.models[0].alpha
        for m in self.models + self.scalars:
            if (abs(m.k)>5 or not 0<=m.m<=5 or (m.m==0 and m.k<0)
                    or m.alpha != self.alpha or m.quadrature<64
                    or not np.array_equal(m.radius,self.radius)
                    or not np.array_equal(m.weights,self.weights)):
                raise ValueError('Common resolved quadrature, alpha and bounded canonical keys required')
        if any(not 2 <= m.order <= 32 for m in self.models) or any(not 2 <= m.order <= 48 for m in self.scalars):
            raise ValueError('Resolved velocity order 2..32 and temperature order 2..48 required')
        signed_modes(self.velocity,self.radius,self.alpha)
        self.cv = np.cumsum([0]+[m.dimension for m in self.models])
        self.ct = self.cv[-1]+np.cumsum([0]+[m.dimension for m in self.scalars])
        self.nv, self.n = int(self.cv[-1]), int(self.ct[-1])
        if self.n > 6144:
            raise ValueError('Coupled coefficient budget exceeded')
        for m,q in self.temperature:
            m.coefficients(q)
            if m.k==m.m==0 and np.max(abs(q.imag))>1e-13:
                raise ValueError('Real mean temperature required')
        self.initial = np.concatenate([q for m,q in self.velocity+self.temperature])
        self.viscosity, self.diffusivity, self.buoyancy = viscosity,diffusivity,buoyancy
        self.mult = [1 if key==(0,0) else 2 for key in self.keys]
        self.vb = [np.einsum('cni,ij->cnj',m.basis,m.transform) for m in self.models]
        from atomic_model_station_regular.model import _basis
        self.vr = [np.einsum('cni,ij->cnj',_basis(self.radius,m.m,m.order)[1],m.transform) for m in self.models]
        self.vf = [cho_factor(m.mass) for m in self.models]
        self.tf = [cho_factor(m.mass) for m in self.scalars]
        self.lv = [-viscosity*cho_solve(f,m.stiffness) for f,m in zip(self.vf,self.models)]
        self.lt = [-diffusivity*cho_solve(f,m.stiffness) for f,m in zip(self.tf,self.scalars)]
        self.coupling = [buoyancy*cho_solve(f, b[2].conj().T@(self.weights[:,None]*s.basis))
                         for f,b,s in zip(self.vf,self.vb,self.scalars)]
        self.sources = {}
        if sources is not None and (type(sources) is not dict or not set(sources)<=set(self.keys)):
            raise ValueError('Sources must map retained keys to radial Legendre coefficients')
        for key in self.keys:
            coefficients = np.array((sources or {}).get(key,[0.]),complex,copy=True)
            if (coefficients.ndim!=1 or not 1<=len(coefficients)<=17
                    or not np.all(np.isfinite(coefficients))
                    or (key==(0,0) and np.max(abs(coefficients.imag))>1e-13)):
                raise ValueError('Bounded finite polynomial sources and real mean required')
            coefficients.setflags(write=False)
            self.sources[key] = coefficients
        signature = hashlib.sha256(repr((viscosity,diffusivity,buoyancy)).encode())
        for key,m,scalar in zip(self.keys,self.models,self.scalars):
            signature.update((m.fingerprint+scalar.fingerprint).encode())
            signature.update(self.sources[key].tobytes())
        self.equation_fingerprint = signature.hexdigest()
        self.source_values = {key:self.source(key,self.radius) for key in self.keys}
        self.source_load = [cho_solve(f,s.basis.conj().T@(self.weights*self.source_values[key]))
                            for key,s,f in zip(self.keys,self.scalars,self.tf)]
        self.mean = self.keys.index((0,0))
        self.heat_row = self.weights @ self.scalars[self.mean].basis
        sm = self.scalars[self.mean]
        # Wall derivative is analytic; it is not inferred from the heat ledger.
        self.wall_row = np.array([sm.jets(np.eye(sm.order)[i],[1.])[1][0] for i in range(sm.order)])
        linear = block_diag(self.lv+self.lt,format='lil')
        for i,c in enumerate(self.coupling):
            linear[self.cv[i]:self.cv[i+1],self.ct[i]:self.ct[i+1]] = c
        linear = csc_matrix(linear)
        real = bmat([[linear.real,-linear.imag],[linear.imag,linear.real]],format='csc')
        self.jacobian = bmat([[real,None],[None,csc_matrix((6,6))]],format='csc')
        for _,q in self.velocity+self.temperature:
            q.setflags(write=False)
        self.initial.setflags(write=False)
        self.calculate(self.initial)  # Validate all initial spectra before any integration.

    def source(self,key,radius):
        r = np.asarray(radius,dtype=float)
        return r**key[1]*Legendre(self.sources[key],domain=[0,1])(r*r)

    def split(self,q):
        a = np.asarray(q,dtype=complex)
        if a.shape!=(self.n,) or not np.all(np.isfinite(a)):
            raise ValueError('Complete finite coupled state required')
        return ([a[self.cv[i]:self.cv[i+1]] for i in range(len(self.models))],
                [a[self.ct[i]:self.ct[i+1]] for i in range(len(self.models))])

    def calculate(self,q,*,method='fft'):
        vq,tq = self.split(q)
        velocity,radial,theta,thermal_radial = {},{},{},{}
        for key,b,dr,s,u,t in zip(self.keys,self.vb,self.vr,self.scalars,vq,tq):
            velocity[key],radial[key] = b@u,dr@u
            theta[key],thermal_radial[key] = s.basis@t,s.radial@t
            if key!=(0,0):
                opposite=(-key[0],-key[1])
                for mapping in (velocity,radial,theta,thermal_radial):mapping[opposite]=mapping[key].conj()
        if method=='fft':
            nonlin = expanded_fft(self.radius,self.alpha,velocity,radial,sorted(velocity))
        elif method=='direct':
            nonlin = direct_products(self.radius,self.alpha,velocity,radial)
        else:raise ValueError('Transport method must be fft or direct')
        transport = scalar_products(self.radius,self.alpha,velocity,theta,thermal_radial,method=method)
        dv,dt = [],[]
        for i,(key,s,u,t) in enumerate(zip(self.keys,self.scalars,vq,tq)):
            load = np.einsum('cni,n,cn->i',self.vb[i].conj(),self.weights,nonlin[key])
            dv.append(self.lv[i]@u+self.coupling[i]@t-cho_solve(self.vf[i],load))
            dt.append(self.lt[i]@t+self.source_load[i]-cho_solve(self.tf[i],s.basis.conj().T@(self.weights*transport[key])))
        diss_v = self.viscosity*sum(a*np.real(np.vdot(u,m.stiffness@u)) for a,m,u in zip(self.mult,self.models,vq))
        diss_t = self.diffusivity*sum(a*np.real(np.vdot(t,s.stiffness@t)) for a,s,t in zip(self.mult,self.scalars,tq))
        work = self.buoyancy*sum(a*np.real(np.vdot(velocity[key][2],self.weights*theta[key])) for a,key in zip(self.mult,self.keys))
        source_variance = sum(a*np.real(np.vdot(theta[key],self.weights*self.source_values[key])) for a,key in zip(self.mult,self.keys))
        source_heat = float(np.real(self.weights@self.source_values[(0,0)]))
        wall_out = float(-self.diffusivity*np.real(self.wall_row@tq[self.mean]))
        adv_power = sum(a*np.real(np.vdot(velocity[key],self.weights*nonlin[key])) for a,key in zip(self.mult,self.keys))
        adv_variance = sum(a*np.real(np.vdot(theta[key],self.weights*transport[key])) for a,key in zip(self.mult,self.keys))
        rates = np.array([diss_v,work,diss_t,source_variance,source_heat,wall_out],float)
        return np.concatenate(dv+dt),rates,dict(velocity=velocity,temperature=theta,nonlinear=nonlin,transport=transport,
            advection_power=float(adv_power),advection_variance=float(adv_variance))

    def observables(self,q):
        vq,tq=self.split(q)
        energy=.5*sum(a*np.real(np.vdot(u,m.mass@u)) for a,m,u in zip(self.mult,self.models,vq))
        variance=.5*sum(a*np.real(np.vdot(t,s.mass@t)) for a,s,t in zip(self.mult,self.scalars,tq))
        heat=np.real(self.heat_row@tq[self.mean])
        return np.array([energy,variance,heat],float)


def resolve_thermal(system,*,end_time=.1,start_time=0.,rtol=1e-10,atol=1e-13,method='BDF',products='fft'):
    if type(system) is not ResolvedThermalSystem:raise ValueError('ResolvedThermalSystem required')
    _finite_real(start_time,'thermal start',0.,1e6)
    _finite_real(end_time,'thermal end',0.,1e6)
    _finite_real(rtol,'thermal relative tolerance',1e-13,1e-6)
    _finite_real(atol,'thermal absolute tolerance',1e-15,1e-8)
    if end_time<=start_time or method not in ('BDF','DOP853') or products not in ('fft','direct'):
        raise ValueError('Forward thermal interval and supported solvers required')
    n=system.n
    def rhs(time,y):
        dq,rates,_=system.calculate(y[:n]+1j*y[n:2*n],method=products)
        return np.r_[dq.real,dq.imag,rates]
    options=dict(jac=system.jacobian) if method=='BDF' else {}
    solution=solve_ivp(rhs,(start_time,end_time),np.r_[system.initial.real,system.initial.imag,np.zeros(6)],
                       method=method,rtol=rtol,atol=atol,max_step=min(.01,end_time-start_time),**options)
    if not solution.success or solution.t[-1]!=end_time or not np.all(np.isfinite(solution.y)):
        raise ValueError('Coupled thermal integration failed')
    initial=system.observables(system.initial)
    maximum=np.zeros(3)
    for j in range(len(solution.t)):
        state=solution.y[:n,j]+1j*solution.y[n:2*n,j]
        values=system.observables(state)-initial
        ledger=solution.y[2*n:,j]
        values+=np.array([ledger[0]-ledger[1],ledger[2]-ledger[3],ledger[5]-ledger[4]])
        maximum=np.maximum(maximum,abs(values))
    if np.any(maximum[:2]>1e-9*np.maximum(1.,abs(initial[:2]))):
        raise ValueError('Coupled kinetic or scalar variance budget failed')
    final=solution.y[:n,-1]+1j*solution.y[n:2*n,-1]
    dq,rates,fields=system.calculate(final,method=products)
    vq,tq=system.split(final);dv,dt=system.split(dq)
    pressure={};max_pressure=0.
    for i,(key,m,u,d,t) in enumerate(zip(system.keys,system.models,vq,dv,tq)):
        force=-fields['nonlinear'][key].copy()
        force[2]+=system.buoyancy*(system.scalars[i].basis@t)
        reaction=m.load(force)-m.mass_full@m.coefficients(d)-system.viscosity*m.stiffness_full@m.coefficients(u)
        field,proof=recover_pressure(m,reaction)
        pressure[key]=field
        max_pressure=max(max_pressure,proof['weak_momentum_relative_residual'])
    report=dict(equation_fingerprint=system.equation_fingerprint,time=float(end_time),start_time=float(start_time),viscosity=system.viscosity,diffusivity=system.diffusivity,
        buoyancy=system.buoyancy,method=method,products=products,rtol=rtol,atol=atol,nfev=int(solution.nfev),
        solver_steps=len(solution.t)-1,initial=initial.tolist(),final=system.observables(final).tolist(),
        integrated_rates=solution.y[2*n:,-1].tolist(),rate_order=['viscous_dissipation','buoyant_work','thermal_dissipation','variance_source','heat_source','outward_wall_heat'],
        maximum_kinetic_budget_residual=float(maximum[0]),maximum_variance_budget_residual=float(maximum[1]),
        maximum_physical_heat_budget_residual=float(maximum[2]),physical_heat_gate_enforced=False,
        maximum_weak_pressure_residual=max_pressure,endpoint_advection_power=fields['advection_power'],endpoint_advection_variance=fields['advection_variance'],
        endpoint_rates=rates.tolist(),real_split=True,linear_buoyancy_in_jacobian=True,nonlinear_jacobian_included=False,
        boundary='theta=0, no slip; periodic axial; buoyancy along +z',temperature='dimensionless anomaly',
        thermal_variance_is_internal_energy=False,bit_exact_restart=False,station_usable=False,validated=False)
    pairs=lambda models,values:[(m,q.copy()) for m,q in zip(models,values)]
    return ThermalResult(pairs(system.models,vq),pairs(system.scalars,tq),pairs(system.models,dv),pairs(system.scalars,dt),pressure,report)
