"""Absolute-stock reference with its own HLLC/RH arithmetic; shared EOS only.

The production face function, centered-energy layout and RK stepper are unused.
"""
import numpy as np
from scipy.integrate import solve_ivp
from atomic_model_hem_qualified.reference import Assembly as Geometry
from atomic_model_hem_qualified import eos


class Assembly(Geometry):
    def __init__(self,request,scheme='hllc'):
        if scheme not in ('hllc','hllc-local'):raise ValueError('Reference scheme required')
        self.scheme=scheme
        super().__init__(request)
        if request['friction'] or any(e['friction']!='none' for e in request['model']['edges']) or any(w['conductance_w_k'] or w['ambient_ua_w_k'] or w['heat_power_w'] for w in self.walls):
            raise ValueError('Adiabatic liquid reference only')
        if any(self.controls['fluid'].values()):raise ValueError('Unforced liquid reference only')

    def rhs(self,time,y):
        self.evaluations+=1
        if self.evaluations>20000:raise ValueError('Independent HLLC reference budget')
        states=self.thermos(y)
        if any(s.region!=1 for s in states):raise ValueError('Reference left liquid chart')
        rate=np.zeros_like(y)
        for k,(a,b,area) in enumerate(self.faces):
            def state(i):
                rho=float(y[i]/self.volumes[i]);v=float(y[self.n+i]/y[i])
                U=np.array([rho,rho*v,y[2*self.n+i]/self.volumes[i]])
                return rho,v,states[i].pressure_pa,U,np.array([U[1],U[1]*v+states[i].pressure_pa,v*(U[2]+states[i].pressure_pa)])
            rl,vl,pl,ul,fl=state(a);rr,vr,pr,ur,fr=state(b)
            if pl==pr and vl==vr:flux=fl if vl>=0 else fr
            else:
                bound=self.wave
                if self.scheme=='hllc-local':
                    bound=1.01*max(abs(vl)+eos.characteristic_speeds(states[a])[1],abs(vr)+eos.characteristic_speeds(states[b])[1])
                    if bound>self.wave:raise ValueError('Reference exceeds global cap')
                left,right=-bound,bound
                ml=rl*(left-vl);mr=rr*(right-vr)
                contact=(pr-pl+ml*vl-mr*vr)/(ml-mr)
                if not left<contact<right:raise ValueError('Reference wave fan invalid')
                if contact>=0:rho,v,p,U,F,s=rl,vl,pl,ul,fl,left
                else:rho,v,p,U,F,s=rr,vr,pr,ur,fr,right
                pressure=p+rho*(s-v)*(contact-v)
                density=rho*(s-v)/(s-contact)
                # Rankine-Hugoniot energy expression in absolute coordinates.
                energy=((s-v)*U[2]-p*v+pressure*contact)/(s-contact)
                if density<=0 or pressure<=0:raise ValueError('Reference star invalid')
                admitted=eos.rho_u(float(density),float(energy/density-.5*contact**2),float(pressure),states[a if contact>=0 else b].temperature_k)
                if admitted.region!=1:raise ValueError('Reference star left liquid chart')
                star=np.array([density,density*contact,energy])
                flux=F+s*(star-U)
            flux=flux*area
            rate[self.face_start+3*k:self.face_start+3*k+3]=flux
            for i,sign in ((a,-1),(b,1)):
                rate[i]+=sign*flux[0];rate[2*self.n+i]+=sign*flux[2]
                if i<self.nodes:rate[self.tail]-=sign*flux[1]
                else:rate[self.n+i]+=sign*flux[1]
        if not np.isfinite(rate).all():raise ValueError('Nonfinite reference rate')
        return rate


def solve(request,times,method='DOP853',*,max_step=5e-6,scheme='hllc'):
    if method not in ('DOP853','Radau'):raise ValueError('Declared reference integrators only')
    assembly=Assembly(request,scheme=scheme);y=assembly.y0.copy();observations=[assembly.observation(0.,y)]
    for a,b in zip(times,times[1:]):
        result=solve_ivp(assembly.rhs,(a,b),y,method=method,rtol=2e-11,atol=1e-10,max_step=max_step)
        if not result.success or not np.isfinite(result.y).all():raise ValueError('Reference integration refused')
        y=result.y[:,-1];observations.append(assembly.observation(b,y))
    return dict(method=method,observations=observations,evaluations=assembly.evaluations)
