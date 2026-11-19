"""Unforced high-resolution reference evolution with stiff, real-split BDF.

The sparse Jacobian uses the exact viscous block and ledger derivative;
nonlinear Jacobian terms are omitted (quasi-Newton). Actual RHS residuals
and independent temporal refinement, not this approximation, determine accuracy.
"""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from scipy.integrate import solve_ivp
from scipy.sparse import block_diag,bmat,csc_matrix,csr_matrix
from atomic_model_station_regular import RegularMode
from atomic_model_station_regular.model import _basis,_finite_real
from atomic_model_station_advection import nonlinear_power
from .products import signed_modes,expanded_fft,direct_products
from atomic_model_station_wall import velocity_jets
from atomic_model_station_stokes_pressure import recover_pressure,PressureField


@dataclass(frozen=True)
class RefinedResult:
    modes: list
    derivatives: list
    pressure: dict
    report: dict


def evolve_refined(modes,*,end_time=.1,viscosity=.05,rtol=1e-10,atol=1e-13,start_time=0.):
    _finite_real(start_time,'start time',0.,1e6);_finite_real(end_time,'end time',0.,1e6)
    if end_time<=start_time:raise ValueError('Forward refined interval required')
    _finite_real(viscosity,'viscosity',0.,1e3);_finite_real(rtol,'relative tolerance',1e-13,1e-6);_finite_real(atol,'absolute tolerance',1e-15,1e-8)
    if type(modes) not in (list,tuple) or not 1<=len(modes)<=61:raise ValueError('Bounded canonical modes required')
    if any(type(pair) not in (list,tuple) or len(pair)!=2 or type(pair[0]) is not RegularMode for pair in modes):raise ValueError('Regular model/coefficient pairs required')
    states=sorted([(m,np.array(q,dtype=complex,copy=True)) for m,q in modes],key=lambda pair:(pair[0].k,pair[0].m));models=[m for m,q in states]
    if any(type(m) is not RegularMode or m.order>24 for m in models):raise ValueError('Validated regular/refined models required')
    if len({(m.k,m.m) for m in models})!=len(models) or not any(m.k==m.m==0 for m in models):raise ValueError('Unique modes including mean required')
    cuts=np.cumsum([0]+[m.dimension for m in models]);n=int(cuts[-1])
    if n>3072:raise ValueError('Refined coefficient budget exceeded')
    radius=models[0].radius;weights=models[0].weights;alpha=models[0].alpha
    if any(m.alpha!=alpha or not np.array_equal(m.radius,radius) or not np.array_equal(m.weights,weights) for m in models):raise ValueError('Common exact quadrature and alpha required')
    signed_modes(states,radius,alpha)
    factors=[cho_factor(m.mass) for m in models]
    linear=[-viscosity*cho_solve(f,m.stiffness) for m,f in zip(models,factors)]
    b=[np.einsum('cni,ij->cnj',m.basis,m.transform) for m in models]
    dr=[np.einsum('cni,ij->cnj',_basis(radius,m.m,m.order)[1],m.transform) for m in models]
    targets=sorted({key for m in models for key in [(m.k,m.m),(-m.k,-m.m)]})
    multiplicities=[1 if m.k==m.m==0 else 2 for m in models]
    lr=block_diag([csc_matrix(a.real) for a in linear],format='csc');li=block_diag([csc_matrix(a.imag) for a in linear],format='csc')
    real_linear=bmat([[lr,-li],[li,lr]],format='csc')
    def split(q):return [q[cuts[i]:cuts[i+1]] for i in range(len(models))]
    def calculate(q):
        values=split(q);v={};radial={}
        for i,(m,z) in enumerate(zip(models,values)):
            key=(m.k,m.m);u=np.einsum('cni,i->cn',b[i],z);d=np.einsum('cni,i->cn',dr[i],z)
            v[key]=u;radial[key]=d
            if key!=(0,0):v[(-key[0],-key[1])]=u.conj();radial[(-key[0],-key[1])]=d.conj()
        nonlinear=expanded_fft(radius,alpha,v,radial,targets)
        derivative=[]
        for i,(m,z) in enumerate(zip(models,values)):
            load=np.einsum('cni,n,cn->i',b[i].conj(),weights,nonlinear[(m.k,m.m)])
            derivative.append(linear[i]@z-cho_solve(factors[i],load))
        dissipation=float(viscosity*sum(a*np.real(np.vdot(z,m.stiffness@z)) for a,m,z in zip(multiplicities,models,values)))
        return values,np.concatenate(derivative),nonlinear,dissipation,v
    def rhs(time,y):
        _,dq,_,dissipation,_=calculate(y[:n]+1j*y[n:2*n])
        return np.r_[dq.real,dq.imag,dissipation]
    def jac(time,y):
        q=y[:n]+1j*y[n:2*n];row=np.zeros(2*n)
        for i,(a,m,z) in enumerate(zip(multiplicities,models,split(q))):
            gradient=2*viscosity*a*(m.stiffness@z)
            row[cuts[i]:cuts[i+1]]=gradient.real;row[n+cuts[i]:n+cuts[i+1]]=gradient.imag
        return bmat([[real_linear,None],[csr_matrix(row[None,:]),csc_matrix((1,1))]],format='csc')
    q0=np.concatenate([np.asarray(q,dtype=complex) for m,q in states])
    solution=solve_ivp(rhs,(start_time,end_time),np.r_[q0.real,q0.imag,0.],method='BDF',jac=jac,rtol=rtol,atol=atol,max_step=min(.01,end_time-start_time))
    if not solution.success or solution.t[-1]!=end_time or not np.all(np.isfinite(solution.y)):raise ValueError('Refined stiff solve failed')
    def energy(q):return float(.5*sum(a*np.real(np.vdot(z,m.mass@z)) for a,m,z in zip(multiplicities,models,split(q))))
    initial_energy=energy(q0);maximum_budget=0.
    for i in range(len(solution.t)):
        value=energy(solution.y[:n,i]+1j*solution.y[n:2*n,i])-initial_energy+solution.y[-1,i]
        maximum_budget=max(maximum_budget,abs(float(value)))
    if maximum_budget>1e-9*max(1.,initial_energy):raise ValueError('Refined energy ledger failed')
    q=solution.y[:n,-1]+1j*solution.y[n:2*n,-1];values,dq,nonlinear,dissipation,v=calculate(q);derivatives=split(dq)
    pressures={};maximum_pressure=0.
    for m,z,zd in zip(models,values,derivatives):
        key=(m.k,m.m);reaction=-m.load(nonlinear[key])-m.mass_full@m.coefficients(zd)-viscosity*m.stiffness_full@m.coefficients(z)
        field,proof=recover_pressure(m,reaction);pressures[key]=field;maximum_pressure=max(maximum_pressure,proof['weak_momentum_relative_residual'])
    power=nonlinear_power(v,nonlinear,weights)
    report=dict(time=float(solution.t[-1]),start_time=float(start_time),rtol=rtol,atol=atol,viscosity=viscosity,
        nfev=int(solution.nfev),njev=int(solution.njev),nlu=int(solution.nlu),accepted_solver_steps=len(solution.t)-1,
        complex_coefficients=n,real_split=True,sparse_viscous_quasi_newton=True,initial_energy=initial_energy,final_energy=energy(q),
        integrated_dissipation=float(solution.y[-1,-1]),maximum_energy_budget_residual=maximum_budget,
        maximum_weak_pressure_residual=maximum_pressure,advection_power=power['advection_power'],
        nonlinear_full_pde_qualified=False,thermal_model_supported=False,station_usable=False,validated=False)
    return RefinedResult([(m,z.copy()) for m,z in zip(models,values)],[(m,z.copy()) for m,z in zip(models,derivatives)],pressures,report)


def endpoint_momentum(result,*,radius):
    if type(result) is not RefinedResult:raise ValueError('RefinedResult required')
    if not result.modes or len(result.modes)!=len(result.derivatives) or type(result.pressure) is not dict or set(result.pressure)!={(m.k,m.m) for m,q in result.modes}:raise ValueError('Complete endpoint modes, derivatives and pressures required')
    sample=np.asarray(radius,dtype=float)
    if sample.ndim!=1 or len(sample)<2 or sample[-1]!=1.:raise ValueError('Positive samples through wall required')
    alpha=result.modes[0][0].alpha;v,dr=signed_modes(result.modes,sample,alpha)
    # Independent direct mode convolution, including the unretained product tail.
    nonlinear=direct_products(sample,alpha,v,dr);maximum=0.;wall=0.;rows={}
    for (m,q),(dm,derivative) in zip(result.modes,result.derivatives):
        if m.fingerprint!=dm.fingerprint:raise ValueError('Endpoint derivative basis differs')
        field=result.pressure[(m.k,m.m)]
        if type(field) is not PressureField or field.model.fingerprint!=m.fingerprint:raise ValueError('Endpoint pressure basis differs')
        _,_,lap=velocity_jets(m,q,sample);_,gradient=field.evaluate(sample)
        residual=m.evaluate(derivative,sample)+nonlinear[(m.k,m.m)]+gradient-result.report['viscosity']*lap
        if not np.all(np.isfinite(residual)):raise ValueError('Nonfinite endpoint momentum')
        absolute=float(np.max(abs(residual)));boundary=float(np.max(abs(residual[:,-1])))
        maximum=max(maximum,absolute);wall=max(wall,boundary)
        rows[str((m.k,m.m))]=dict(maximum=absolute,wall=boundary)
    return dict(time=result.report['time'],maximum_retained_residual=maximum,maximum_wall_residual=wall,rows=rows,
        unretained_nonlinear_maximum=max((float(np.max(abs(value))) for key,value in nonlinear.items() if key not in v),default=0.),
        unretained_pressure_reconstructed=False,nonlinear_advection_included=True,station_usable=False,validated=False)
